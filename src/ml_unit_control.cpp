#include "ml_unit_control.h"

#if defined(WZ_ML_EXPERIMENT)

#include "lib/framework/crc.h"
#include "lib/framework/fixedpoint.h"
#include "lib/framework/frame.h"
#include "lib/framework/trig.h"
#include "lib/gamelib/gtime.h"
#include "lib/framework/wzapp.h"
#include "lib/ivis_opengl/ivisdef.h"
#include "lib/netplay/netplay.h"
#include "lib/netplay/sync_debug.h"

#include "action.h"
#include "ai.h"
#include "combat.h"
#include "campaigninfo.h"
#include "droid.h"
#include "game_world.h"
#include "map.h"
#include "move.h"
#include "multiplay.h"
#include "objmem.h"
#include "projectile.h"
#include "random.h"
#include "stats.h"
#include "template.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iterator>
#include <limits>
#include <map>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace wzml
{
namespace
{

constexpr uint32_t WZML_VERSION = 1;
constexpr uint32_t INPUT_SIZE = 200;
constexpr uint32_t HIDDEN_SIZE = 256;
constexpr uint32_t OUTPUT_SIZE = 12;
constexpr int32_t Q15_MAX = 32767;
constexpr int32_t ENGINE_ANGLE_PER_ACTION_STEP = 16;

enum class Backend
{
	Off,
	Scripted,
	Native,
};

struct NativeModel
{
	std::vector<float> weights1;
	std::vector<float> bias1;
	std::vector<float> weights2;
	std::vector<float> bias2;
	std::vector<float> weights3;
	std::vector<float> bias3;
	std::string expectedStatsHash;
	bool ready = false;
};

struct SlotState
{
	std::array<uint32_t, 7> allyIds{};
	std::array<uint32_t, 8> enemyIds{};
	bool assigned = false;
};

Backend backend = Backend::Off;
uint32_t controlledPlayers = 0;
std::string traceFilePath;
std::string resolvedStatsFilePath;
std::ofstream traceFile;
NativeModel nativeModel;
uint32_t cachedGameTime = std::numeric_limits<uint32_t>::max();
std::unordered_map<uint32_t, QuantizedActionV1> cachedActions;
std::unordered_map<uint32_t, ObservationV1> cachedObservations;
std::unordered_map<uint32_t, nlohmann::ordered_json> cachedGoldenStates;
std::unordered_map<uint32_t, Position> cachedTargetPositions;
std::unordered_map<uint32_t, SlotState> slotsByDroid;
std::unordered_map<uint32_t, uint32_t> previousBody;
std::unordered_map<uint32_t, QuantizedActionV1> previousAction;
bool resolvedStatsWritten = false;
bool resolvedStatsVerified = false;
bool fatalError = false;

struct CombatTestState
{
	bool enabled = false;
	bool initialized = false;
	uint32_t seed = 1;
	uint32_t durationTicks = 90 * GAME_TICKS_PER_SEC;
	uint32_t startTime = 0;
	std::string reportPath;
	std::array<uint32_t, 4> droidIds{};
	bool rangeInvariant = false;
	bool parallelMovement = false;
	bool pairedCombat = false;
	int forcedHitRoll = -1;
	uint32_t fireStopTicks = std::numeric_limits<uint32_t>::max();
	uint32_t hitRateSamples = 0;
	uint32_t hitRateHits = 0;
	std::string hitRateBand;
	int hitRateChance = -1;
};

CombatTestState combatTest;

bool failExperiment(const std::string &message)
{
	if (!fatalError)
	{
		fatalError = true;
		debug(LOG_ERROR, "WZML fatal error: %s", message.c_str());
		wzQuit(2);
	}
	return false;
}

int16_t q15Ratio(int64_t value, int64_t scale)
{
	if (scale <= 0)
	{
		return 0;
	}
	const int64_t result = value * Q15_MAX / scale;
	return static_cast<int16_t>(std::max<int64_t>(-Q15_MAX, std::min<int64_t>(Q15_MAX, result)));
}

int16_t q15Unsigned(uint64_t value, uint64_t maximum)
{
	if (maximum == 0)
	{
		return 0;
	}
	return static_cast<int16_t>(std::min<uint64_t>(Q15_MAX, value * Q15_MAX / maximum));
}

DROID *findDroid(uint32_t id)
{
	for (uint32_t player = 0; player < MAX_PLAYERS; ++player)
	{
		for (DROID *candidate : gameWorld.objects.droids[player])
		{
			if (candidate != nullptr && candidate->id == id && !candidate->died)
			{
				return candidate;
			}
		}
	}
	return nullptr;
}

bool isValidEnemy(const DROID *observer, const DROID *candidate)
{
	return candidate != nullptr && !candidate->died && candidate->player != observer->player
	       && !aiCheckAlliances(observer->player, candidate->player)
	       && candidate->visible[observer->player] == UBYTE_MAX;
}

SlotState &getSlots(const DROID *observer)
{
	SlotState &state = slotsByDroid[observer->id];
	if (state.assigned)
	{
		return state;
	}

	std::vector<uint32_t> allyIds;
	std::vector<uint32_t> enemyIds;
	for (uint32_t player = 0; player < MAX_PLAYERS; ++player)
	{
		for (DROID *candidate : gameWorld.objects.droids[player])
		{
			if (candidate == nullptr || candidate->died || candidate->id == observer->id)
			{
				continue;
			}
			if (candidate->player == observer->player || aiCheckAlliances(observer->player, candidate->player))
			{
				allyIds.push_back(candidate->id);
			}
			else
			{
				enemyIds.push_back(candidate->id);
			}
		}
	}
	std::sort(allyIds.begin(), allyIds.end());
	std::sort(enemyIds.begin(), enemyIds.end());
	for (size_t i = 0; i < std::min(allyIds.size(), state.allyIds.size()); ++i)
	{
		state.allyIds[i] = allyIds[i];
	}
	for (size_t i = 0; i < std::min(enemyIds.size(), state.enemyIds.size()); ++i)
	{
		state.enemyIds[i] = enemyIds[i];
	}
	state.assigned = true;
	return state;
}

void putRelativeObject(std::array<int16_t, OBSERVATION_V1_SIZE> &out, size_t offset,
		const DROID *observer, const DROID *other, size_t width)
{
	if (other == nullptr || other->died)
	{
		return;
	}
	const int64_t dx = static_cast<int64_t>(other->pos.x) - observer->pos.x;
	const int64_t dy = static_cast<int64_t>(other->pos.y) - observer->pos.y;
	const int32_t localX = iCosR(observer->rot.direction, static_cast<int32_t>(dx)) - iSinR(observer->rot.direction, static_cast<int32_t>(dy));
	const int32_t localY = iSinR(observer->rot.direction, static_cast<int32_t>(dx)) + iCosR(observer->rot.direction, static_cast<int32_t>(dy));
	const int64_t rangeScale = 16 * TILE_UNITS;
	out[offset + 0] = q15Ratio(localX, rangeScale);
	out[offset + 1] = q15Ratio(localY, rangeScale);
	out[offset + 2] = q15Unsigned(other->body, std::max<UDWORD>(other->originalBody, 1));
	out[offset + 3] = q15Ratio(angleDelta(other->rot.direction - observer->rot.direction), 32768);
	out[offset + 4] = q15Unsigned(other->sMove.speed, std::max<UDWORD>(other->baseSpeed, 1));
	out[offset + 5] = Q15_MAX;
	if (width > 6)
	{
		out[offset + 6] = q15Ratio(other->pos.z - observer->pos.z, 4 * TILE_UNITS);
	}
	if (width > 7)
	{
		out[offset + 7] = other->numWeaps > 0 ? Q15_MAX : 0;
	}
	if (width > 8)
	{
		out[offset + 8] = other->visible[observer->player] == UBYTE_MAX ? Q15_MAX : 0;
	}
	if (width > 9)
	{
		out[offset + 9] = other->died ? 0 : Q15_MAX;
	}
	if (width > 10)
	{
		out[offset + 10] = q15Unsigned(other->id & 0x7fff, 0x7fff);
	}
}

uint32_t readU32(const std::vector<uint8_t> &bytes, size_t &offset)
{
	if (offset + 4 > bytes.size())
	{
		return 0;
	}
	const uint32_t value = static_cast<uint32_t>(bytes[offset])
	                     | static_cast<uint32_t>(bytes[offset + 1]) << 8
	                     | static_cast<uint32_t>(bytes[offset + 2]) << 16
	                     | static_cast<uint32_t>(bytes[offset + 3]) << 24;
	offset += 4;
	return value;
}

bool readFloats(const std::vector<uint8_t> &bytes, size_t &offset, size_t count, std::vector<float> &output)
{
	if (offset + count * sizeof(float) > bytes.size())
	{
		return false;
	}
	output.resize(count);
	std::memcpy(output.data(), bytes.data() + offset, count * sizeof(float));
	offset += count * sizeof(float);
	return true;
}

bool loadNativeModel(const std::string &manifestPath)
{
	try
	{
		std::ifstream manifestFile(manifestPath);
		if (!manifestFile)
		{
			debug(LOG_ERROR, "WZML: cannot open manifest: %s", manifestPath.c_str());
			return false;
		}
		nlohmann::json manifest;
		manifestFile >> manifest;
		if (manifest.value("contract", std::string()) != "warzone-tactical-v1")
		{
			debug(LOG_ERROR, "WZML: manifest has the wrong contract");
			return false;
		}
		std::filesystem::path modelPath = manifest.at("wzml_path").get<std::string>();
		if (modelPath.is_relative())
		{
			modelPath = std::filesystem::path(manifestPath).parent_path() / modelPath;
		}
		std::ifstream modelFile(modelPath, std::ios::binary);
		if (!modelFile)
		{
			debug(LOG_ERROR, "WZML: cannot open model: %s", modelPath.string().c_str());
			return false;
		}
		std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(modelFile)), std::istreambuf_iterator<char>());
		if (sha256Sum(bytes.data(), bytes.size()).toString() != manifest.at("wzml_sha256").get<std::string>())
		{
			debug(LOG_ERROR, "WZML: model hash does not match the manifest");
			return false;
		}
		if (bytes.size() < 24 || std::memcmp(bytes.data(), "WZMLV1\0", 8) != 0)
		{
			debug(LOG_ERROR, "WZML: invalid model header");
			return false;
		}
		size_t offset = 8;
		const uint32_t version = readU32(bytes, offset);
		const uint32_t input = readU32(bytes, offset);
		const uint32_t hidden1 = readU32(bytes, offset);
		const uint32_t hidden2 = readU32(bytes, offset);
		const uint32_t output = readU32(bytes, offset);
		if (version != WZML_VERSION || input != INPUT_SIZE || hidden1 != HIDDEN_SIZE
		    || hidden2 != HIDDEN_SIZE || output != OUTPUT_SIZE)
		{
			debug(LOG_ERROR, "WZML: unsupported model dimensions");
			return false;
		}
		if (!readFloats(bytes, offset, INPUT_SIZE * HIDDEN_SIZE, nativeModel.weights1)
		    || !readFloats(bytes, offset, HIDDEN_SIZE, nativeModel.bias1)
		    || !readFloats(bytes, offset, HIDDEN_SIZE * HIDDEN_SIZE, nativeModel.weights2)
		    || !readFloats(bytes, offset, HIDDEN_SIZE, nativeModel.bias2)
		    || !readFloats(bytes, offset, HIDDEN_SIZE * OUTPUT_SIZE, nativeModel.weights3)
		    || !readFloats(bytes, offset, OUTPUT_SIZE, nativeModel.bias3)
		    || offset != bytes.size())
		{
			debug(LOG_ERROR, "WZML: invalid model data size");
			return false;
		}
		nativeModel.expectedStatsHash = manifest.at("resolved_stats_sha256").get<std::string>();
		nativeModel.ready = true;
		return true;
	}
	catch (const std::exception &error)
	{
		debug(LOG_ERROR, "WZML: model load failed: %s", error.what());
		return false;
	}
}

void denseTanh(const std::vector<float> &input, const std::vector<float> &weights,
		const std::vector<float> &bias, size_t outputSize, std::vector<float> &output)
{
	output.assign(outputSize, 0.0f);
	for (size_t row = 0; row < outputSize; ++row)
	{
		double sum = static_cast<double>(bias[row]);
		const size_t base = row * input.size();
		for (size_t column = 0; column < input.size(); ++column)
		{
			sum += static_cast<double>(weights[base + column]) * static_cast<double>(input[column]);
		}
		output[row] = static_cast<float>(std::tanh(sum));
	}
}

std::array<float, OUTPUT_SIZE> runNative(const ObservationV1 &observation)
{
	std::vector<float> input(INPUT_SIZE);
	for (size_t i = 0; i < input.size(); ++i)
	{
		input[i] = std::ldexp(static_cast<float>(observation.q15[i]), -15);
	}
	std::vector<float> hidden1;
	std::vector<float> hidden2;
	denseTanh(input, nativeModel.weights1, nativeModel.bias1, HIDDEN_SIZE, hidden1);
	denseTanh(hidden1, nativeModel.weights2, nativeModel.bias2, HIDDEN_SIZE, hidden2);

	std::array<float, OUTPUT_SIZE> output{};
	for (size_t row = 0; row < OUTPUT_SIZE; ++row)
	{
		double sum = static_cast<double>(nativeModel.bias3[row]);
		const size_t base = row * HIDDEN_SIZE;
		for (size_t column = 0; column < HIDDEN_SIZE; ++column)
		{
			sum += static_cast<double>(nativeModel.weights3[base + column]) * static_cast<double>(hidden2[column]);
		}
		output[row] = static_cast<float>(sum);
	}
	return output;
}

bool nativeAction(const ObservationV1 &observation, QuantizedActionV1 &action)
{
	const std::array<float, OUTPUT_SIZE> output = runNative(observation);
	for (size_t index = 0; index < output.size(); ++index)
	{
		if (!std::isfinite(output[index]))
		{
			return failExperiment("model output " + std::to_string(index) + " is not finite");
		}
	}
	action.headingDeltaQ = static_cast<int16_t>(std::max(-2048, std::min(2047,
		static_cast<int>(std::lround(std::tanh(output[0]) * 2048.0f)))));
	action.speedFractionQ = static_cast<uint16_t>(std::max(0, std::min(256,
		static_cast<int>(std::lround((std::tanh(output[1]) + 1.0f) * 128.0f)))));
	int best = -1;
	float bestLogit = output[2]; // Slot -1, "none", is always valid.
	for (size_t slot = 0; slot < TARGET_SLOT_COUNT; ++slot)
	{
		if (observation.targetMask[slot] && output[3 + slot] > bestLogit)
		{
			bestLogit = output[3 + slot];
			best = static_cast<int>(slot);
		}
	}
	action.targetSlotQ = static_cast<int8_t>(best);
	action.fireQ = output[11] > 0.0f ? 1 : 0;
	return true;
}

DROID *targetForSlot(const DROID *observer, int slot)
{
	if (slot < 0 || slot >= static_cast<int>(TARGET_SLOT_COUNT))
	{
		return nullptr;
	}
	DROID *target = findDroid(getSlots(observer).enemyIds[slot]);
	return isValidEnemy(observer, target) ? target : nullptr;
}

nlohmann::ordered_json serializeGoldenUnit(const DROID *observer, const DROID *unit, uint32_t assignedId)
{
	nlohmann::ordered_json output;
	output["assigned_id"] = assignedId;
	if (unit == nullptr || unit->died)
	{
		output["unit"] = nullptr;
		return output;
	}
	output["unit"] = {
		{"id", unit->id},
		{"player", unit->player},
		{"position", {unit->pos.x, unit->pos.y, unit->pos.z}},
		{"body", unit->body},
		{"original_body", unit->originalBody},
		{"base_speed", unit->baseSpeed},
		{"speed", unit->sMove.speed},
		{"body_heading", unit->rot.direction},
		{"weapon_count", unit->numWeaps},
		{"visible", unit->visible[observer->player] == UBYTE_MAX},
	};
	return output;
}

nlohmann::ordered_json serializeGoldenState(const DROID *observer)
{
	const WEAPON_STATS *weapon = observer->getWeaponStats(0);
	const PROPULSION_STATS *propulsion = observer->getPropulsionStats();
	const BODY_STATS *body = observer->getBodyStats();
	const QuantizedActionV1 oldAction = previousAction.count(observer->id) ? previousAction[observer->id] : QuantizedActionV1{};
	const uint32_t oldBody = previousBody.count(observer->id) ? previousBody[observer->id] : observer->body;
	Vector3i muzzleBase = observer->pos;
	calcDroidMuzzleBaseLocation(observer, &muzzleBase, 0);
	nlohmann::ordered_json state;
	state["game_time"] = gameTime;
	state["map"] = {{"width", gameWorld.map.width}, {"height", gameWorld.map.height}};
	state["self"] = {
		{"id", observer->id},
		{"player", observer->player},
		{"position", {observer->pos.x, observer->pos.y, observer->pos.z}},
		{"body", observer->body},
		{"original_body", observer->originalBody},
		{"previous_body", oldBody},
		{"base_speed", observer->baseSpeed},
		{"calculated_speed", moveCalcDroidSpeed(const_cast<DROID *>(observer))},
		{"speed", observer->sMove.speed},
		{"move_direction", observer->sMove.moveDir},
		{"move_status", observer->sMove.Status},
		{"body_heading", observer->rot.direction},
		{"pitch", observer->rot.pitch},
		{"roll", observer->rot.roll},
		{"turret_heading", observer->asWeaps[0].rot.direction},
		{"turret_pitch", observer->asWeaps[0].rot.pitch},
		{"muzzle_base_position", {muzzleBase.x, muzzleBase.y, muzzleBase.z}},
		{"weapon_last_fired", observer->asWeaps[0].lastFired},
		{"fire_pause", weaponFirePause(*weapon, observer->player)},
		{"turn_speed", propulsion->turnSpeed},
		{"spin_speed", propulsion->spinSpeed},
		{"weapon_long_range", proj_GetLongRange(*weapon, observer->player)},
		{"weapon_minimum_range", proj_GetMinRange(*weapon, observer->player)},
		{"weapon_damage", weapon->upgrade[observer->player].damage},
		{"direct_weapon", proj_Direct(weapon)},
		{"minimum_elevation", weapon->minElevation},
		{"maximum_elevation", weapon->maxElevation},
		{"armour", body->upgrade[observer->player].armour},
		{"fire_on_move", weapon->fireOnMove},
	};
	state["previous_action"] = {oldAction.headingDeltaQ, oldAction.speedFractionQ, oldAction.targetSlotQ, oldAction.fireQ};
	state["allies"] = nlohmann::ordered_json::array();
	state["enemies"] = nlohmann::ordered_json::array();
	SlotState &slots = getSlots(observer);
	for (uint32_t id : slots.allyIds)
	{
		state["allies"].push_back(serializeGoldenUnit(observer, findDroid(id), id));
	}
	for (uint32_t id : slots.enemyIds)
	{
		state["enemies"].push_back(serializeGoldenUnit(observer, findDroid(id), id));
	}
	return state;
}

QuantizedActionV1 scriptedAction(const ObservationV1 &observation)
{
	int nearestSlot = -1;
	int64_t nearestDistance = std::numeric_limits<int64_t>::max();
	for (size_t slot = 0; slot < TARGET_SLOT_COUNT; ++slot)
	{
		if (!observation.targetMask[slot])
		{
			continue;
		}
		const int64_t dx = observation.q15[98 + slot * 11];
		const int64_t dy = observation.q15[99 + slot * 11];
		const int64_t distance = dx * dx + dy * dy;
		if (distance < nearestDistance)
		{
			nearestDistance = distance;
			nearestSlot = static_cast<int>(slot);
		}
	}

	QuantizedActionV1 action;
	if (nearestSlot < 0)
	{
		return action;
	}
	const int32_t localX = observation.q15[98 + nearestSlot * 11];
	const int32_t localY = observation.q15[99 + nearestSlot * 11];
	const uint16_t relativeHeading = iAtan2(localX, localY);
	action.headingDeltaQ = static_cast<int16_t>(angleDelta(relativeHeading) / ENGINE_ANGLE_PER_ACTION_STEP);
	action.targetSlotQ = static_cast<int8_t>(nearestSlot);
	const int32_t holdRangeQ15 = observation.q15[13] * 3 / 4;
	action.speedFractionQ = nearestDistance > static_cast<int64_t>(holdRangeQ15) * holdRangeQ15 ? 256 : 0;
	const int32_t turretDirection = static_cast<int32_t>(observation.q15[7]) * 32768 / Q15_MAX;
	const int turretError = angleDelta(relativeHeading - turretDirection);
	action.fireQ = std::abs(turretError) <= DEG(1) ? 1 : 0;
	return action;
}

void writeTrace(DROID *droid, const ObservationV1 &observation, const QuantizedActionV1 &action,
		DROID *target, bool aligned, bool shotFired, int hitResult, uint32_t lastFiredBefore,
		uint32_t targetBodyBefore, int32_t targetDistance,
		const nlohmann::ordered_json &goldenState)
{
	if (!traceFile.is_open())
	{
		return;
	}
	nlohmann::ordered_json row;
	row["game_time"] = gameTime;
	row["droid_id"] = droid->id;
	row["player"] = droid->player;
	row["position"] = {droid->pos.x, droid->pos.y, droid->pos.z};
	row["body_heading"] = droid->rot.direction;
	row["turret_heading"] = droid->asWeaps[0].rot.direction;
	row["turret_pitch"] = droid->asWeaps[0].rot.pitch;
	row["speed"] = droid->sMove.speed;
	row["body"] = droid->body;
	const WEAPON_STATS *weaponStats = droid->getWeaponStats(0);
	const uint32_t firePause = std::max(1, weaponFirePause(*weaponStats, droid->player));
	row["weapon_last_fired"] = droid->asWeaps[0].lastFired;
	row["weapon_last_fired_before"] = lastFiredBefore;
	row["cooldown"] = gameTime - droid->asWeaps[0].lastFired >= firePause ? 0 : firePause - (gameTime - droid->asWeaps[0].lastFired);
	row["shot_eligible"] = gameTime - droid->asWeaps[0].lastFired >= firePause;
	row["target_id"] = target != nullptr ? static_cast<int64_t>(target->id) : -1;
	row["target_distance"] = targetDistance;
	row["target_body_before"] = target != nullptr ? targetBodyBefore : 0;
	row["target_body_after"] = target != nullptr ? target->body : 0;
	row["minimum_range"] = proj_GetMinRange(*weaponStats, droid->player);
	row["long_range"] = proj_GetLongRange(*weaponStats, droid->player);
	row["turret_aligned"] = aligned;
	row["requested_fire"] = action.fireQ != 0;
	row["shot_fired"] = shotFired;
	row["shot_hit"] = hitResult;
	row["action"] = {action.headingDeltaQ, action.speedFractionQ, action.targetSlotQ, action.fireQ};
	row["observation_sha256"] = sha256Sum(observation.q15.data(), observation.q15.size() * sizeof(int16_t)).toString();
	row["observation_q15"] = observation.q15;
	row["target_mask"] = observation.targetMask;
	row["golden_state"] = goldenState;
	traceFile << row.dump() << '\n';
}

bool checkResolvedStats(DROID *droid)
{
	if (fatalError)
	{
		return false;
	}
	if (resolvedStatsVerified)
	{
		return true;
	}
	const std::string stats = resolvedStatsV1(droid);
	const std::string hash = sha256Sum(stats.data(), stats.size()).toString();
	if (!resolvedStatsWritten)
	{
		resolvedStatsWritten = true;
		if (!resolvedStatsFilePath.empty())
		{
			std::ofstream output(resolvedStatsFilePath, std::ios::binary);
			output << stats;
			if (!output)
			{
				return failExperiment("cannot write resolved stats: " + resolvedStatsFilePath);
			}
		}
	}
	if (backend == Backend::Native && nativeModel.expectedStatsHash != hash)
	{
		return failExperiment("resolved stats hash mismatch (engine " + hash
			+ ", model " + nativeModel.expectedStatsHash + ")");
	}
	resolvedStatsVerified = true;
	debug(LOG_INFO, "WZML: resolved stats SHA-256 %s", hash.c_str());
	return true;
}

uint32_t nextScenarioRandom(uint32_t &state)
{
	state ^= state << 13;
	state ^= state >> 17;
	state ^= state << 5;
	return state;
}

bool spawnCombatTestDroid(const DROID_TEMPLATE &unitTemplate, uint32_t player,
		const Vector2i &position, uint16_t direction, size_t idSlot)
{
	DROID *droid = reallyBuildDroid(gameWorld, &unitTemplate, Position(position.x, position.y, 0),
		player, false, Rotation(direction, 0, 0));
	if (droid == nullptr)
	{
		return false;
	}
	addDroid(droid, gameWorld.objects.droids);
	for (uint32_t viewer = 0; viewer < MAX_PLAYERS; ++viewer)
	{
		droid->visible[viewer] = UBYTE_MAX;
		droid->seenThisTick[viewer] = UBYTE_MAX;
	}
	combatTest.droidIds[idSlot] = droid->id;
	return true;
}

bool initializeCombatTest()
{
	// The combat-test seed controls all synchronized engine randomness as well as
	// the mirrored spawn layout. This is required for repeatable CRC traces.
	gameSRand(combatTest.seed);
	resetSyncDebug();

	freeAllStructs(gameWorld);
	freeAllFeatures(gameWorld);
	for (uint32_t player : {0u, 1u})
	{
		while (!gameWorld.objects.droids[player].empty())
		{
			vanishDroid(gameWorld.objects.droids[player].front(), gameWorld.objects);
		}
	}

	DROID_TEMPLATE unitTemplate;
	unitTemplate.name = WzString::fromUtf8("ML Viper Mini-Rocket Pod");
	unitTemplate.droidType = DROID_WEAPON;
	const int body = getCompFromName(COMP_BODY, WzString::fromUtf8("Body1REC"));
	const int propulsion = getCompFromName(COMP_PROPULSION, WzString::fromUtf8("tracked01"));
	const int weapon = getCompFromName(COMP_WEAPON, WzString::fromUtf8("Rocket-Pod"));
	if (body < 0 || propulsion < 0 || weapon < 0
	    || body > std::numeric_limits<uint8_t>::max() || propulsion > std::numeric_limits<uint8_t>::max())
	{
		debug(LOG_ERROR, "WZML: required combat-test components are not available");
		return false;
	}
	unitTemplate.asParts[COMP_BODY] = static_cast<uint8_t>(body);
	unitTemplate.asParts[COMP_PROPULSION] = static_cast<uint8_t>(propulsion);
	unitTemplate.numWeaps = 1;
	unitTemplate.asWeaps[0] = static_cast<uint32_t>(weapon);

	combatTest.hitRateHits = 0;
	combatTest.hitRateChance = -1;
	if (combatTest.hitRateSamples > 0)
	{
		const WEAPON_STATS &weaponStats = asWeaponStats[weapon];
		if (combatTest.hitRateBand == "short")
		{
			combatTest.hitRateChance = weaponStats.upgrade[0].shortHitChance;
		}
		else if (combatTest.hitRateBand == "long")
		{
			combatTest.hitRateChance = weaponStats.upgrade[0].hitChance;
		}
		else
		{
			debug(LOG_ERROR, "WZML: hit_rate_band must be short or long when hit_rate_samples is nonzero");
			return false;
		}
		for (uint32_t sample = 0; sample < combatTest.hitRateSamples; ++sample)
		{
			combatTest.hitRateHits += static_cast<uint32_t>(gameRand(100) <= combatTest.hitRateChance);
		}
	}

	uint32_t randomState = combatTest.seed == 0 ? 1 : combatTest.seed;
	const int32_t centerX = world_coord(gameWorld.map.width) / 2;
	const int32_t centerY = world_coord(gameWorld.map.height) / 2;
	int landTexture = 0;
	for (int texture = 0; texture < MAX_TILE_TEXTURES; ++texture)
	{
		if (terrainTypes[texture] == TER_ROAD)
		{
			landTexture = texture;
			break;
		}
	}
	const int32_t centerTileX = map_coord(centerX);
	const int32_t centerTileY = map_coord(centerY);
	for (int32_t y = std::max(1, centerTileY - 12); y <= std::min(gameWorld.map.height - 2, centerTileY + 12); ++y)
	{
		for (int32_t x = std::max(1, centerTileX - 12); x <= std::min(gameWorld.map.width - 2, centerTileX + 12); ++x)
		{
			MAPTILE *tile = mapTile(gameWorld.map, x, y);
			tile->height = 128;
			tile->waterLevel = 0;
			tile->texture = TileNumber_texture(tile->texture) | landTexture;
			// The source map can retain cliff and water flags in its derived
			// blocking maps. Clear these flags with the terrain change so this
			// test area is also flat for the movement system.
			auxClearBlocking(gameWorld.map, x, y, AIR_BLOCKED | FEATURE_BLOCKED | WATER_BLOCKED);
			auxSetBlocking(gameWorld.map, x, y, LAND_BLOCKED);
			auxClearAll(gameWorld.map, x, y, AUXBITS_NONPASSABLE | AUXBITS_OUR_BUILDING | AUXBITS_BLOCKING);
			markTileDirty(x, y);
		}
	}
	std::array<Vector2i, 4> positions;
	std::array<uint16_t, 4> headings;
	if (combatTest.parallelMovement)
	{
		positions = {
			Vector2i(centerX - 4 * TILE_UNITS, centerY - 6 * TILE_UNITS),
			Vector2i(centerX - 4 * TILE_UNITS, centerY - 2 * TILE_UNITS),
			Vector2i(centerX - 4 * TILE_UNITS, centerY + 2 * TILE_UNITS),
			Vector2i(centerX - 4 * TILE_UNITS, centerY + 6 * TILE_UNITS),
		};
		headings = {DEG(90), DEG(90), DEG(90), DEG(90)};
	}
	else if (combatTest.pairedCombat)
	{
		positions = {
			Vector2i(centerX - 3 * TILE_UNITS, centerY - 3 * TILE_UNITS),
			Vector2i(centerX - 3 * TILE_UNITS, centerY + 3 * TILE_UNITS),
			Vector2i(centerX + 3 * TILE_UNITS, centerY - 3 * TILE_UNITS),
			Vector2i(centerX + 3 * TILE_UNITS, centerY + 3 * TILE_UNITS),
		};
		headings = {DEG(90), DEG(90), DEG(270), DEG(270)};
	}
	else if (combatTest.rangeInvariant)
	{
		const int32_t distance = proj_GetLongRange(*unitTemplate.getWeaponStats(0), 0) + 1;
		const int32_t leftX = centerX - distance / 2;
		const int32_t rightX = leftX + distance;
		positions = {
			Vector2i(leftX, centerY - 2 * TILE_UNITS),
			Vector2i(leftX, centerY + 2 * TILE_UNITS),
			Vector2i(rightX, centerY - 2 * TILE_UNITS),
			Vector2i(rightX, centerY + 2 * TILE_UNITS),
		};
		headings = {DEG(90), DEG(90), DEG(270), DEG(270)};
	}
	else
	{
		const int32_t xOffset = 4 * TILE_UNITS + static_cast<int32_t>(nextScenarioRandom(randomState) % TILE_UNITS);
		const int32_t yOffset = TILE_UNITS + static_cast<int32_t>(nextScenarioRandom(randomState) % TILE_UNITS);
		const int32_t yJitter = static_cast<int32_t>(nextScenarioRandom(randomState) % TILE_UNITS) - TILE_UNITS / 2;
		const uint16_t headingJitter = static_cast<uint16_t>(nextScenarioRandom(randomState) % 8193) - 4096;
		positions = {
			Vector2i(centerX - xOffset, centerY - yOffset + yJitter),
			Vector2i(centerX - xOffset, centerY + yOffset + yJitter),
			Vector2i(centerX + xOffset, centerY + yOffset - yJitter),
			Vector2i(centerX + xOffset, centerY - yOffset - yJitter),
		};
		headings = {
			static_cast<uint16_t>(DEG(90) + headingJitter),
			static_cast<uint16_t>(DEG(90) - headingJitter),
			static_cast<uint16_t>(DEG(270) + headingJitter),
			static_cast<uint16_t>(DEG(270) - headingJitter),
		};
	}
	if (!spawnCombatTestDroid(unitTemplate, 0, positions[0], headings[0], 0)
	    || !spawnCombatTestDroid(unitTemplate, 0, positions[1], headings[1], 1)
	    || !spawnCombatTestDroid(unitTemplate, 1, positions[2], headings[2], 2)
	    || !spawnCombatTestDroid(unitTemplate, 1, positions[3], headings[3], 3))
	{
		return false;
	}

	combatTest.startTime = gameTime;
	combatTest.initialized = true;
	slotsByDroid.clear();
	cachedGameTime = std::numeric_limits<uint32_t>::max();
	debug(LOG_INFO, "WZML: combat test seed %u started at game time %u", combatTest.seed, combatTest.startTime);
	return true;
}

void finishCombatTest(const char *result, uint32_t team0Alive, uint32_t team1Alive)
{
	nlohmann::ordered_json report;
	report["contract"] = "warzone-tactical-v1";
	report["seed"] = combatTest.seed;
	report["start_game_time"] = combatTest.startTime;
	report["end_game_time"] = gameTime;
	report["elapsed_ticks"] = gameTime - combatTest.startTime;
	report["result"] = result;
	report["team_0_alive"] = team0Alive;
	report["team_1_alive"] = team1Alive;
	report["droid_ids"] = combatTest.droidIds;
	report["hit_rate_samples"] = combatTest.hitRateSamples;
	report["hit_rate_hits"] = combatTest.hitRateHits;
	report["hit_rate_band"] = combatTest.hitRateBand;
	report["hit_rate_chance"] = combatTest.hitRateChance;
	if (!combatTest.reportPath.empty())
	{
		std::ofstream output(combatTest.reportPath, std::ios::binary | std::ios::trunc);
		output << report.dump(2) << '\n';
	}
	debug(LOG_INFO, "WZML: combat test result: %s", result);
	combatTest.enabled = false;
	wzQuit(0);
}

bool buildActionBatch()
{
	cachedActions.clear();
	cachedObservations.clear();
	cachedGoldenStates.clear();
	cachedTargetPositions.clear();
	for (uint32_t player = 0; player < MAX_PLAYERS; ++player)
	{
		for (DROID *droid : gameWorld.objects.droids[player])
		{
			if (droid != nullptr && !droid->died)
			{
				cachedTargetPositions.emplace(droid->id, droid->pos);
			}
		}
	}
	std::vector<DROID *> controlled;
	for (uint32_t player = 0; player < MAX_PLAYERS; ++player)
	{
		if ((controlledPlayers & (1u << player)) == 0)
		{
			continue;
		}
		for (DROID *droid : gameWorld.objects.droids[player])
		{
			if (controlsDroid(droid))
			{
				controlled.push_back(droid);
			}
		}
	}
	std::sort(controlled.begin(), controlled.end(), [](const DROID *a, const DROID *b) { return a->id < b->id; });
	for (DROID *droid : controlled)
	{
		ObservationV1 observation = buildObservationV1(droid);
		QuantizedActionV1 action;
		if (backend == Backend::Native)
		{
			if (!nativeAction(observation, action))
			{
				return false;
			}
		}
		else
		{
			action = scriptedAction(observation);
		}
		cachedObservations.emplace(droid->id, observation);
		if (traceFile.is_open())
		{
			cachedGoldenStates.emplace(droid->id, serializeGoldenState(droid));
		}
		cachedActions.emplace(droid->id, action);
	}
	cachedGameTime = gameTime;
	return true;
}

} // namespace

bool configure(const std::string &policy, const std::string &players,
		const std::string &tracePath, const std::string &resolvedStatsPath)
{
	reset();
	traceFilePath = tracePath;
	resolvedStatsFilePath = resolvedStatsPath;

	std::stringstream playerList(players.empty() ? "0" : players);
	std::string item;
	while (std::getline(playerList, item, ','))
	{
		try
		{
			const unsigned player = static_cast<unsigned>(std::stoul(item));
			if (player >= MAX_PLAYERS)
			{
				return false;
			}
			controlledPlayers |= 1u << player;
		}
		catch (...)
		{
			return false;
		}
	}

	if (policy == "off" || policy.empty())
	{
		backend = Backend::Off;
	}
	else if (policy == "scripted")
	{
		backend = Backend::Scripted;
	}
	else if (policy.rfind("native:", 0) == 0)
	{
		backend = Backend::Native;
		if (!loadNativeModel(policy.substr(7)))
		{
			backend = Backend::Off;
			return false;
		}
	}
	else
	{
		return false;
	}

	if (!traceFilePath.empty())
	{
		traceFile.open(traceFilePath, std::ios::out | std::ios::trunc);
		if (!traceFile)
		{
			return false;
		}
	}
	return true;
}

bool configureCombatTest(const std::string &specPath)
{
	try
	{
		std::ifstream input(specPath);
		if (!input)
		{
			debug(LOG_ERROR, "WZML: cannot open combat-test spec: %s", specPath.c_str());
			return false;
		}
		nlohmann::json spec;
		input >> spec;
		combatTest = CombatTestState{};
		combatTest.enabled = true;
		combatTest.seed = spec.value("seed", 1u);
		combatTest.durationTicks = spec.value("duration_ticks", 90u * GAME_TICKS_PER_SEC);
		combatTest.reportPath = spec.value("report_path", std::string());
		combatTest.rangeInvariant = spec.value("range_invariant", false);
		combatTest.parallelMovement = spec.value("parallel_movement", false);
		combatTest.pairedCombat = spec.value("paired_combat", false);
		combatTest.forcedHitRoll = spec.value("forced_hit_roll", -1);
		combatTest.fireStopTicks = spec.value("fire_stop_ticks", std::numeric_limits<uint32_t>::max());
		combatTest.hitRateSamples = spec.value("hit_rate_samples", 0u);
		combatTest.hitRateBand = spec.value("hit_rate_band", std::string());
		return combatTest.durationTicks > 0
		       && combatTest.forcedHitRoll >= -1 && combatTest.forcedHitRoll <= 99
		       && (combatTest.hitRateSamples == 0
		           || combatTest.hitRateBand == "short" || combatTest.hitRateBand == "long");
	}
	catch (const std::exception &error)
	{
		debug(LOG_ERROR, "WZML: invalid combat-test spec: %s", error.what());
		return false;
	}
}

void combatTestUpdate()
{
	if (!combatTest.enabled)
	{
		return;
	}
	if (!combatTest.initialized)
	{
		if (!initializeCombatTest())
		{
			combatTest.enabled = false;
			wzQuit(2);
		}
		return;
	}

	uint32_t team0Alive = 0;
	uint32_t team1Alive = 0;
	for (size_t index = 0; index < combatTest.droidIds.size(); ++index)
	{
		DROID *droid = findDroid(combatTest.droidIds[index]);
		if (droid == nullptr)
		{
			continue;
		}
		for (uint32_t viewer = 0; viewer < MAX_PLAYERS; ++viewer)
		{
			droid->visible[viewer] = UBYTE_MAX;
			droid->seenThisTick[viewer] = UBYTE_MAX;
		}
		(index < 2 ? team0Alive : team1Alive)++;
	}
	if (team0Alive == 0 && team1Alive == 0)
	{
		finishCombatTest("draw", team0Alive, team1Alive);
	}
	else if (team0Alive == 0)
	{
		finishCombatTest("team_1_win", team0Alive, team1Alive);
	}
	else if (team1Alive == 0)
	{
		finishCombatTest("team_0_win", team0Alive, team1Alive);
	}
	else if (gameTime - combatTest.startTime >= combatTest.durationTicks)
	{
		finishCombatTest("unfinished", team0Alive, team1Alive);
	}
}

void combatTestApplyFullVisibility()
{
	if (!combatTest.enabled || !combatTest.initialized)
	{
		return;
	}
	for (uint32_t id : combatTest.droidIds)
	{
		DROID *droid = findDroid(id);
		if (droid == nullptr)
		{
			continue;
		}
		for (uint32_t viewer = 0; viewer < MAX_PLAYERS; ++viewer)
		{
			droid->visible[viewer] = UBYTE_MAX;
			droid->seenThisTick[viewer] = UBYTE_MAX;
		}
	}
}

void reset()
{
	if (traceFile.is_open())
	{
		traceFile.close();
	}
	backend = Backend::Off;
	controlledPlayers = 0;
	traceFilePath.clear();
	resolvedStatsFilePath.clear();
	nativeModel = NativeModel{};
	cachedGameTime = std::numeric_limits<uint32_t>::max();
	cachedActions.clear();
	cachedObservations.clear();
	cachedGoldenStates.clear();
	cachedTargetPositions.clear();
	slotsByDroid.clear();
	previousBody.clear();
	previousAction.clear();
	resolvedStatsWritten = false;
	resolvedStatsVerified = false;
	fatalError = false;
	combatTest = CombatTestState{};
}

bool controlsDroid(const DROID *psDroid)
{
	if (backend == Backend::Off || psDroid == nullptr || psDroid->died || psDroid->player >= MAX_PLAYERS)
	{
		return false;
	}
	if (NetPlay.bComms)
	{
		return false;
	}
	if ((controlledPlayers & (1u << psDroid->player)) == 0 || psDroid->numWeaps == 0)
	{
		return false;
	}
	const PROPULSION_STATS *propulsion = psDroid->getPropulsionStats();
	return psDroid->droidType != DROID_PERSON && !psDroid->isCyborg() && !psDroid->isTransporter()
	       && propulsion != nullptr && propulsion->propulsionType != PROPULSION_TYPE_LIFT;
}

ObservationV1 buildObservationV1(const DROID *psDroid)
{
	ObservationV1 observation;
	if (psDroid == nullptr)
	{
		return observation;
	}
	const WEAPON_STATS *weapon = psDroid->numWeaps > 0 ? psDroid->getWeaponStats(0) : nullptr;
	const PROPULSION_STATS *propulsion = psDroid->getPropulsionStats();
	const BODY_STATS *body = psDroid->getBodyStats();
	const uint32_t firePause = weapon != nullptr ? weaponFirePause(*weapon, psDroid->player) : 1;
	const uint32_t sinceFired = gameTime - psDroid->asWeaps[0].lastFired;
	const uint32_t currentBody = psDroid->body;
	const uint32_t oldBody = previousBody.count(psDroid->id) ? previousBody[psDroid->id] : currentBody;
	const QuantizedActionV1 oldAction = previousAction.count(psDroid->id) ? previousAction[psDroid->id] : QuantizedActionV1{};
	DROID *selectedTarget = targetForSlot(psDroid, oldAction.targetSlotQ);
	int32_t turretError = 0;
	bool turretAligned = false;
	if (selectedTarget != nullptr)
	{
		const uint16_t targetHeading = iAtan2(selectedTarget->pos.xy() - psDroid->pos.xy());
		turretError = angleDelta(targetHeading - psDroid->rot.direction - psDroid->asWeaps[0].rot.direction);
		turretAligned = turretError == 0;
		const int32_t minimumRange = proj_GetMinRange(*weapon, psDroid->player);
		if (proj_Direct(weapon) && objPosDiffSq(psDroid, selectedTarget) > minimumRange * minimumRange)
		{
			Vector3i muzzleBase = psDroid->pos;
			calcDroidMuzzleBaseLocation(psDroid, &muzzleBase, 0);
			const Vector3i delta = selectedTarget->pos - muzzleBase;
			const int32_t horizontalDistance = iHypot(delta.x, delta.y);
			const int32_t targetPitch = std::max(
				static_cast<int32_t>(DEG(weapon->minElevation)),
				std::min(
					static_cast<int32_t>(DEG(weapon->maxElevation)),
					angleDelta(iAtan2(delta.z, horizontalDistance))));
			turretAligned = turretAligned && targetPitch == angleDelta(psDroid->asWeaps[0].rot.pitch);
		}
	}

	// Self, weapon, and turret block: 20 values.
	observation.q15[0] = q15Unsigned(psDroid->body, std::max<UDWORD>(psDroid->originalBody, 1));
	observation.q15[1] = q15Unsigned(psDroid->sMove.speed, std::max<UDWORD>(psDroid->baseSpeed, 1));
	observation.q15[2] = q15Unsigned(moveCalcDroidSpeed(const_cast<DROID *>(psDroid)), std::max<UDWORD>(psDroid->baseSpeed, 1));
	observation.q15[3] = iSinR(psDroid->rot.direction, Q15_MAX);
	observation.q15[4] = iCosR(psDroid->rot.direction, Q15_MAX);
	observation.q15[5] = q15Unsigned(std::min(sinceFired, firePause), std::max<uint32_t>(firePause, 1));
	observation.q15[6] = sinceFired >= firePause ? Q15_MAX : 0;
	observation.q15[7] = q15Ratio(angleDelta(psDroid->asWeaps[0].rot.direction), 32768);
	observation.q15[8] = q15Ratio(turretError, 32768);
	observation.q15[9] = selectedTarget != nullptr && turretAligned ? Q15_MAX : 0;
	observation.q15[10] = q15Unsigned(oldBody > currentBody ? oldBody - currentBody : 0, std::max<UDWORD>(psDroid->originalBody, 1));
	observation.q15[11] = propulsion != nullptr ? q15Unsigned(propulsion->turnSpeed, std::max(propulsion->turnSpeed, propulsion->spinSpeed)) : 0;
	observation.q15[12] = propulsion != nullptr ? q15Unsigned(propulsion->spinSpeed, std::max(propulsion->turnSpeed, propulsion->spinSpeed)) : 0;
	observation.q15[13] = weapon != nullptr ? q15Unsigned(proj_GetLongRange(*weapon, psDroid->player), 16 * TILE_UNITS) : 0;
	observation.q15[14] = weapon != nullptr ? q15Unsigned(weapon->upgrade[psDroid->player].damage, 1000) : 0;
	observation.q15[15] = body != nullptr ? q15Unsigned(body->upgrade[psDroid->player].armour, 1000) : 0;
	observation.q15[16] = weapon != nullptr && weapon->fireOnMove ? Q15_MAX : 0;
	observation.q15[17] = q15Ratio(angleDelta(psDroid->rot.pitch), 32768);
	observation.q15[18] = q15Ratio(angleDelta(psDroid->rot.roll), 32768);
	observation.q15[19] = Q15_MAX;

	// Boundary rays: 8 values. They use the map rectangle in V1 flat arenas.
	const int32_t worldWidth = world_coord(gameWorld.map.width);
	const int32_t worldHeight = world_coord(gameWorld.map.height);
	for (size_t ray = 0; ray < 8; ++ray)
	{
		const uint16_t direction = psDroid->rot.direction + static_cast<uint16_t>(ray * 8192);
		const int32_t dx = iSinR(direction, 16 * TILE_UNITS);
		const int32_t dy = iCosR(direction, 16 * TILE_UNITS);
		int32_t fraction = Q15_MAX;
		if (dx > 0) fraction = std::min(fraction, static_cast<int32_t>((static_cast<int64_t>(worldWidth - 1 - psDroid->pos.x) * Q15_MAX) / dx));
		if (dx < 0) fraction = std::min(fraction, static_cast<int32_t>((static_cast<int64_t>(psDroid->pos.x - 1) * Q15_MAX) / -dx));
		if (dy > 0) fraction = std::min(fraction, static_cast<int32_t>((static_cast<int64_t>(worldHeight - 1 - psDroid->pos.y) * Q15_MAX) / dy));
		if (dy < 0) fraction = std::min(fraction, static_cast<int32_t>((static_cast<int64_t>(psDroid->pos.y - 1) * Q15_MAX) / -dy));
		observation.q15[20 + ray] = static_cast<int16_t>(std::max(0, std::min(Q15_MAX, fraction)));
	}

	SlotState &slots = getSlots(psDroid);
	for (size_t slot = 0; slot < slots.allyIds.size(); ++slot)
	{
		putRelativeObject(observation.q15, 28 + slot * 10, psDroid, findDroid(slots.allyIds[slot]), 10);
	}
	for (size_t slot = 0; slot < slots.enemyIds.size(); ++slot)
	{
		DROID *enemy = findDroid(slots.enemyIds[slot]);
		putRelativeObject(observation.q15, 98 + slot * 11, psDroid, enemy, 11);
		observation.targetMask[slot] = isValidEnemy(psDroid, enemy) ? 1 : 0;
	}

	// Recovery block is zero until repair facilities enter the contract.
	// Time and previous action block starts at index 190.
	observation.q15[190] = q15Unsigned(gameTime % GAME_TICKS_PER_SEC, GAME_TICKS_PER_SEC - 1);
	observation.q15[191] = q15Ratio(oldAction.headingDeltaQ, 2048);
	observation.q15[192] = q15Unsigned(oldAction.speedFractionQ, 256);
	observation.q15[193] = oldAction.targetSlotQ < 0 ? -Q15_MAX : q15Unsigned(oldAction.targetSlotQ + 1, 8);
	observation.q15[194] = oldAction.fireQ ? Q15_MAX : 0;
	observation.q15[195] = iSinR(static_cast<uint16_t>((static_cast<uint64_t>(gameTime) * 65536) / (90 * GAME_TICKS_PER_SEC)), Q15_MAX);
	observation.q15[196] = iCosR(static_cast<uint16_t>((static_cast<uint64_t>(gameTime) * 65536) / (90 * GAME_TICKS_PER_SEC)), Q15_MAX);
	observation.q15[197] = q15Unsigned(psDroid->player, MAX_PLAYERS - 1);
	observation.q15[198] = q15Unsigned(psDroid->id & 0x7fff, 0x7fff);
	observation.q15[199] = Q15_MAX;
	return observation;
}

UpdateResult updateDroid(DROID *psDroid)
{
	if (!controlsDroid(psDroid))
	{
		return UpdateResult::NoAction;
	}
	if (fatalError)
	{
		return UpdateResult::FatalError;
	}
	if (cachedGameTime != std::numeric_limits<uint32_t>::max() && gameTime < cachedGameTime)
	{
		slotsByDroid.clear();
		previousBody.clear();
		previousAction.clear();
		resolvedStatsWritten = false;
		resolvedStatsVerified = false;
	}
	if (psDroid->numWeaps != 1)
	{
		failExperiment("Contract V1 requires exactly one weapon per ML droid");
		return UpdateResult::FatalError;
	}
	if (!checkResolvedStats(psDroid))
	{
		return UpdateResult::FatalError;
	}
	if (cachedGameTime != gameTime)
	{
		if (!buildActionBatch())
		{
			return UpdateResult::FatalError;
		}
	}
	const auto actionIt = cachedActions.find(psDroid->id);
	const auto observationIt = cachedObservations.find(psDroid->id);
	if (actionIt == cachedActions.end() || observationIt == cachedObservations.end())
	{
		failExperiment("the action batch does not contain the controlled droid");
		return UpdateResult::FatalError;
	}
	QuantizedActionV1 action = actionIt->second;
	action.headingDeltaQ = std::max<int16_t>(-2048, std::min<int16_t>(2047, action.headingDeltaQ));
	action.speedFractionQ = std::min<uint16_t>(256, action.speedFractionQ);
	if (combatTest.initialized && gameTime - combatTest.startTime >= combatTest.fireStopTicks)
	{
		action.fireQ = 0;
	}
	DROID *target = targetForSlot(psDroid, action.targetSlotQ);
	if (target == nullptr)
	{
		action.targetSlotQ = -1;
		action.fireQ = 0;
	}
	setDroidActionTarget(psDroid, target, 0);
	const uint32_t lastFiredBefore = psDroid->asWeaps[0].lastFired;
	const uint32_t targetBodyBefore = target != nullptr ? target->body : 0;
	Position currentTargetPosition(0, 0, 0);
	bool targetPositionWasOverridden = false;
	if (target != nullptr)
	{
		const auto positionIt = cachedTargetPositions.find(target->id);
		if (positionIt != cachedTargetPositions.end())
		{
			currentTargetPosition = target->pos;
			target->pos = positionIt->second;
			targetPositionWasOverridden = true;
		}
	}
	const int32_t targetDistance = target != nullptr ? iHypot((target->pos - psDroid->pos).xy()) : -1;
	const bool aligned = target != nullptr && actionTargetTurret(psDroid, target, &psDroid->asWeaps[0]);
	bool shotFired = false;
	int hitResult = -1;
	if (aligned && action.fireQ)
	{
		const uint16_t aimedPitch = psDroid->asWeaps[0].rot.pitch;
		combSetMlTestHitRoll(combatTest.forcedHitRoll);
		shotFired = combFire(&psDroid->asWeaps[0], psDroid, target, 0);
		hitResult = shotFired ? combGetMlTestHitResult() : -1;
		// Projectile launch can replace the turret pitch with a randomized
		// miss trajectory. Keep the pitch that actionTargetTurret selected so
		// the next policy update has deterministic turret state.
		psDroid->asWeaps[0].rot.pitch = aimedPitch;
	}
	if (targetPositionWasOverridden)
	{
		target->pos = currentTargetPosition;
	}
	const uint16_t desiredHeading = psDroid->rot.direction + static_cast<uint16_t>(action.headingDeltaQ * ENGINE_ANGLE_PER_ACTION_STEP);
	const SDWORD desiredSpeed = static_cast<SDWORD>((static_cast<uint64_t>(moveCalcDroidSpeed(psDroid)) * action.speedFractionQ) / 256);
	moveUpdateDroidDirect(psDroid, desiredSpeed, desiredHeading);
	if (traceFile.is_open())
	{
		const auto goldenIt = cachedGoldenStates.find(psDroid->id);
		if (goldenIt == cachedGoldenStates.end())
		{
			failExperiment("the trace batch does not contain the controlled droid golden state");
			return UpdateResult::FatalError;
		}
		writeTrace(psDroid, observationIt->second, action, target, aligned, shotFired, hitResult,
			lastFiredBefore, targetBodyBefore, targetDistance, goldenIt->second);
	}
	previousBody[psDroid->id] = psDroid->body;
	previousAction[psDroid->id] = action;
	return UpdateResult::Applied;
}

std::string resolvedStatsV1(const DROID *psDroid)
{
	nlohmann::ordered_json root;
	root["contract"] = "warzone-tactical-v1";
	root["bMultiPlayer"] = bMultiPlayer;
	root["campaign_tweaks"] = {
		{"autosaves_only", getCamTweakOption_AutosavesOnly()},
		{"fast_experience", getCamTweakOption_FastExp()},
		{"heavily_damaged_penalty", getCamTweakOption_heavilyDamagedPenalty()},
		{"no_experience", getCamTweakOption_NoExp()},
		{"ps1_modifiers", getCamTweakOption_PS1Modifiers()},
	};
	const BODY_STATS *body = psDroid->getBodyStats();
	const PROPULSION_STATS *propulsion = psDroid->getPropulsionStats();
	const WEAPON_STATS *weapon = psDroid->getWeaponStats(0);
	Vector3i bodyConnector(0, 0, 0);
	if (body->pIMD != nullptr && !body->pIMD->connectors.empty())
	{
		bodyConnector = body->pIMD->connectors[0];
	}
	const int turretRotationRate = actionTurretRotationRate(*weapon, false);
	const uint32_t effectiveDamage = calcDamage(weaponDamage(*weapon, psDroid->player), weapon->weaponEffect, psDroid);
	const uint32_t targetArmour = objArmour(psDroid, weapon->weaponClass);
	const int64_t damageAfterArmour = std::max<int64_t>(
		static_cast<int64_t>(effectiveDamage) - targetArmour,
		static_cast<int64_t>(effectiveDamage) * weapon->upgrade[psDroid->player].minimumDamage / 100);
	const uint32_t effectiveHitDamage = static_cast<uint32_t>(std::max<int64_t>(MIN_WEAPON_DAMAGE, damageAfterArmour));
	root["body"] = {
		{"id", body->id.toUtf8()},
		{"health", psDroid->originalBody},
		{"armour", body->upgrade[psDroid->player].armour},
		{"thermal", body->upgrade[psDroid->player].thermal},
		{"weight", body->weight},
	};
	root["propulsion"] = {
		{"id", propulsion->id.toUtf8()},
		{"base_speed", psDroid->baseSpeed},
		{"max_speed", propulsion->maxSpeed},
		{"acceleration", propulsion->acceleration},
		{"deceleration", propulsion->deceleration},
		{"skid_deceleration", propulsion->skidDeceleration},
		{"turn_speed", propulsion->turnSpeed},
		{"spin_speed", propulsion->spinSpeed},
		{"spin_angle", propulsion->spinAngle},
		{"calculated_max_speed", moveCalcDroidSpeed(const_cast<DROID *>(psDroid))},
		{"propulsion_type", propulsion->propulsionType},
	};
	root["weapon"] = {
		{"id", weapon->id.toUtf8()},
		{"damage", weapon->upgrade[psDroid->player].damage},
		{"effective_damage_before_armour", effectiveDamage},
		{"effective_hit_damage_zero_experience", effectiveHitDamage},
		{"short_range", proj_GetShortRange(*weapon, psDroid->player)},
		{"long_range", proj_GetLongRange(*weapon, psDroid->player)},
		{"minimum_range", proj_GetMinRange(*weapon, psDroid->player)},
		{"short_hit_chance", weapon->upgrade[psDroid->player].shortHitChance},
		{"long_hit_chance", weapon->upgrade[psDroid->player].hitChance},
		{"fire_pause", weaponFirePause(*weapon, psDroid->player)},
		{"reload_time", weapon->upgrade[psDroid->player].reloadTime},
		{"minimum_damage_percent", weapon->upgrade[psDroid->player].minimumDamage},
		{"projectile_speed", weapon->flightSpeed},
		{"rotate", weapon->rotate},
		{"weight", weapon->weight},
		{"turret_rotation_rate", turretRotationRate},
		{"turret_pitch_rate", std::max(gameTimeAdjustedIncrement(DEG(90)), DEG(1))},
		{"muzzle_base_connector", {bodyConnector.x, bodyConnector.y, bodyConnector.z}},
		{"direct", proj_Direct(weapon)},
		{"max_elevation", weapon->maxElevation},
		{"min_elevation", weapon->minElevation},
		{"fire_on_move", weapon->fireOnMove},
		{"movement_model", weapon->movementModel},
		{"flags", weapon->flags.to_string()},
	};
	nlohmann::ordered_json bodyUpgrades = nlohmann::ordered_json::array();
	nlohmann::ordered_json weaponUpgrades = nlohmann::ordered_json::array();
	for (uint32_t player = 0; player < MAX_PLAYERS; ++player)
	{
		bodyUpgrades.push_back({
			{"player", player},
			{"armour", body->upgrade[player].armour},
			{"thermal", body->upgrade[player].thermal},
			{"hitpoints", body->upgrade[player].hitpoints},
			{"hitpoint_percent", body->upgrade[player].hitpointPct},
		});
		weaponUpgrades.push_back({
			{"player", player},
			{"short_range", weapon->upgrade[player].shortRange},
			{"max_range", weapon->upgrade[player].maxRange},
			{"minimum_range", weapon->upgrade[player].minRange},
			{"short_hit_chance", weapon->upgrade[player].shortHitChance},
			{"long_hit_chance", weapon->upgrade[player].hitChance},
			{"fire_pause", weapon->upgrade[player].firePause},
			{"reload_time", weapon->upgrade[player].reloadTime},
			{"damage", weapon->upgrade[player].damage},
		});
	}
	root["upgrade_tables"] = {
		{"body", std::move(bodyUpgrades)},
		{"weapon", std::move(weaponUpgrades)},
	};
	return root.dump(2) + "\n";
}

} // namespace wzml

#else

namespace wzml
{
bool configure(const std::string &, const std::string &, const std::string &, const std::string &) { return false; }
bool configureCombatTest(const std::string &) { return false; }
void combatTestUpdate() {}
void combatTestApplyFullVisibility() {}
bool controlsDroid(const DROID *) { return false; }
UpdateResult updateDroid(DROID *) { return UpdateResult::NoAction; }
void reset() {}
ObservationV1 buildObservationV1(const DROID *) { return {}; }
std::string resolvedStatsV1(const DROID *) { return {}; }
} // namespace wzml

#endif
