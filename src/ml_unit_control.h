/*
 * Experimental local learned control for ground combat droids.
 * This interface is not part of network game state or the stable game API.
 */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

struct DROID;

namespace wzml
{

constexpr size_t OBSERVATION_V1_SIZE = 200;
constexpr size_t TARGET_SLOT_COUNT = 8;

struct QuantizedActionV1
{
	int16_t headingDeltaQ = 0;       // 1/4096 turn, range -2048..2047.
	uint16_t speedFractionQ = 0;     // Range 0..256. Reverse is not valid.
	int8_t targetSlotQ = -1;         // -1 is none. Enemy slots are 0..7.
	uint8_t fireQ = 0;               // Range 0..1.
};

struct ObservationV1
{
	std::array<int16_t, OBSERVATION_V1_SIZE> q15{};
	std::array<uint8_t, TARGET_SLOT_COUNT> targetMask{};
};

enum class UpdateResult
{
	Applied,
	NoAction,
	FatalError,
};

// Policy values: "off", "scripted", or "native:<manifest path>".
bool configure(const std::string &policy, const std::string &players,
		const std::string &tracePath, const std::string &resolvedStatsPath);
bool configureCombatTest(const std::string &specPath);
void combatTestUpdate();
void combatTestApplyFullVisibility();
bool controlsDroid(const DROID *psDroid);
UpdateResult updateDroid(DROID *psDroid);
void reset();

ObservationV1 buildObservationV1(const DROID *psDroid);
std::string resolvedStatsV1(const DROID *psDroid);

} // namespace wzml
