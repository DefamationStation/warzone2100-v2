"""Vectorized integer simulator for the Warzone tactical V1 contract."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .contracts import (
    ALLY_OFFSET,
    CONTEXT_OFFSET,
    ENEMY_OFFSET,
    OBSERVATION_SIZE,
    Q15_SCALE,
    RAY_OFFSET,
)

GAME_TICKS_PER_SEC = 1000
UPDATE_TICKS = 100
ANGLE_UNITS = 65536
ACTION_ANGLE_STEP = 16
EXTRA_PRECISION = 256
TILE_UNITS = 128
MAX_UNITS = 16
MAX_PLAYERS = 11
PROJECTILES_PER_UNIT = 32
Q15_MAX = 32767
V1_DROID_RADIUS = 40


def _degrees(value: int) -> int:
    return value * ANGLE_UNITS // 360


@dataclass(frozen=True, slots=True)
class ResolvedStats:
    base_speed: int
    calculated_max_speed: int
    acceleration: int
    deceleration: int
    skid_deceleration: int
    turn_speed: int
    spin_speed: int
    spin_angle: int
    health: int
    armour: int
    damage: int
    hit_damage: int
    short_range: int
    long_range: int
    min_range: int
    short_hit_chance: int
    long_hit_chance: int
    fire_pause: int
    projectile_speed: int
    turret_rotation_rate: int
    turret_pitch_rate: int
    muzzle_connector_x: int
    muzzle_connector_y: int
    muzzle_connector_z: int
    min_elevation: int
    max_elevation: int
    direct_weapon: bool
    fire_on_move: bool

    @classmethod
    def load(cls, path: Path) -> "ResolvedStats":
        data = json.loads(path.read_text(encoding="utf-8"))
        body = data["body"]
        propulsion = data["propulsion"]
        weapon = data["weapon"]
        connector = weapon.get("muzzle_base_connector", [0, 10, 23])
        return cls(
            base_speed=int(propulsion["base_speed"]),
            calculated_max_speed=int(propulsion.get("calculated_max_speed", propulsion["base_speed"])),
            acceleration=int(propulsion["acceleration"]),
            deceleration=int(propulsion["deceleration"]),
            skid_deceleration=int(propulsion["skid_deceleration"]),
            turn_speed=int(propulsion["turn_speed"]),
            spin_speed=int(propulsion["spin_speed"]),
            spin_angle=int(propulsion["spin_angle"]),
            health=int(body["health"]),
            armour=int(body["armour"]),
            damage=int(weapon["damage"]),
            hit_damage=int(
                weapon.get(
                    "effective_hit_damage_zero_experience",
                    max(1, int(weapon["damage"]) - int(body["armour"])),
                )
            ),
            short_range=int(weapon["short_range"]),
            long_range=int(weapon["long_range"]),
            min_range=int(weapon.get("minimum_range", 0)),
            short_hit_chance=int(weapon["short_hit_chance"]),
            long_hit_chance=int(weapon["long_hit_chance"]),
            fire_pause=int(weapon["fire_pause"]),
            projectile_speed=int(weapon["projectile_speed"]),
            turret_rotation_rate=int(
                weapon.get(
                    "turret_rotation_rate",
                    max(
                        _degrees(1),
                        _degrees(int(weapon.get("turret_rotation_degrees_per_second", 180)))
                        * UPDATE_TICKS
                        // GAME_TICKS_PER_SEC,
                    ),
                )
            ),
            turret_pitch_rate=int(weapon.get("turret_pitch_rate", max(_degrees(1), _degrees(90) * UPDATE_TICKS // GAME_TICKS_PER_SEC))),
            muzzle_connector_x=int(connector[0]),
            muzzle_connector_y=int(connector[1]),
            muzzle_connector_z=int(connector[2]),
            min_elevation=_degrees(int(weapon.get("min_elevation", -60))),
            max_elevation=_degrees(int(weapon.get("max_elevation", 90))),
            direct_weapon=bool(weapon.get("direct", True)),
            fire_on_move=bool(weapon["fire_on_move"]),
        )


def trunc_div(numerator: Tensor, denominator: Tensor | int) -> Tensor:
    """Use C++ signed integer division, which truncates toward zero."""

    denominator_tensor = torch.as_tensor(denominator, dtype=numerator.dtype, device=numerator.device)
    quotient = torch.div(numerator.abs(), denominator_tensor.abs(), rounding_mode="floor")
    sign = torch.sign(numerator) * torch.sign(denominator_tensor)
    return quotient * sign


def quantise_fraction(numerator: Tensor, denominator: int, new_time: Tensor, old_time: Tensor) -> Tensor:
    """Match gtime.h quantiseFraction with int64 intermediate values."""

    return trunc_div(new_time * numerator, denominator) - trunc_div(old_time * numerator, denominator)


def angle_delta(angle: Tensor) -> Tensor:
    wrapped = torch.bitwise_and(angle, 0xFFFF)
    return torch.where(wrapped >= 0x8000, wrapped - 0x10000, wrapped)


def integer_floor_sqrt(value: Tensor) -> Tensor:
    """Return exact floor square roots for the non-negative V1 integer range."""

    root = torch.sqrt(value.to(torch.float32)).to(torch.int64)
    root -= (root * root > value).to(torch.int64)
    root += ((root + 1) * (root + 1) <= value).to(torch.int64)
    return root


class IntegerTrig:
    """GPU copies of the deterministic engine trigonometric tables."""

    def __init__(self, device: torch.device) -> None:
        index = np.arange(0x4001, dtype=np.float64)
        sin_table = (65536.0 * np.sin(index * (math.pi / 0x8000)) + 0.5).astype(np.int64)
        sin_table -= index != 0
        atan_index = np.arange(0x2001, dtype=np.float64)
        atan_table = (0x8000 / math.pi * np.arctan(atan_index / 0x2000) + 0.5).astype(np.int64)
        self.sin_table = torch.from_numpy(sin_table).to(device)
        self.atan_table = torch.from_numpy(atan_table).to(device)

    def sin(self, angle: Tensor) -> Tensor:
        angle = torch.bitwise_and(angle, 0xFFFF)
        quadrant = torch.bitwise_right_shift(angle, 14)
        remainder = torch.bitwise_and(angle, 0x3FFF)
        reverse = (quadrant == 1) | (quadrant == 3)
        table_index = torch.where(reverse, 0x4000 - remainder, remainder)
        sign = torch.where(quadrant < 2, 1, -1)
        return sign * (self.sin_table[table_index] + (table_index != 0).to(torch.int64))

    def cos(self, angle: Tensor) -> Tensor:
        return self.sin(angle + 0x4000)

    def sin_r(self, angle: Tensor, radius: Tensor) -> Tensor:
        return trunc_div(radius * self.sin(angle), 65536)

    def cos_r(self, angle: Tensor, radius: Tensor) -> Tensor:
        return trunc_div(radius * self.cos(angle), 65536)

    def atan2(self, sine: Tensor, cosine: Tensor) -> Tensor:
        result = torch.zeros_like(sine)
        nonzero = (sine != 0) | (cosine != 0)
        case = (sine < 0).to(torch.int64) * 2 + (cosine < 0).to(torch.int64)
        j = torch.where(case == 0, sine, torch.where(case == 1, -cosine, torch.where(case == 2, cosine, -sine)))
        k = torch.where(case == 0, cosine, torch.where(case == 1, sine, torch.where(case == 2, -sine, -cosine)))
        base = torch.where(case == 0, 0, torch.where(case == 1, 0x4000, torch.where(case == 2, 0xC000, 0x8000)))
        j_safe = torch.clamp(j, min=1)
        k_safe = torch.clamp(k, min=1)
        first = j < k
        first_index = torch.div(j * 0x2000 + torch.div(k, 2, rounding_mode="floor"), k_safe, rounding_mode="floor")
        second_index = torch.div(k * 0x2000 + torch.div(j, 2, rounding_mode="floor"), j_safe, rounding_mode="floor")
        first_index.clamp_(0, 0x2000)
        second_index.clamp_(0, 0x2000)
        value = torch.where(first, base + self.atan_table[first_index], base + 0x4000 - self.atan_table[second_index])
        return torch.where(nonzero, torch.bitwise_and(value, 0xFFFF), result)


class TacticalSimulator:
    """Fixed-buffer, vectorized tactical arenas with integer game state."""

    def __init__(
        self,
        arenas: int,
        stats: ResolvedStats,
        device: str | torch.device = "cuda",
        max_units: int = MAX_UNITS,
        seed: int = 1,
    ) -> None:
        if max_units > MAX_UNITS or max_units < 2:
            raise ValueError("Contract V1 supports 2 to 16 units")
        self.device = torch.device(device)
        self.arenas = arenas
        self.max_units = max_units
        self.stats = stats
        self.trig = IntegerTrig(self.device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        shape = (arenas, max_units)
        self.game_time = torch.zeros((arenas, 1), dtype=torch.int64, device=self.device)
        self.position_x = torch.zeros(shape, dtype=torch.int64, device=self.device)
        self.position_y = torch.zeros(shape, dtype=torch.int64, device=self.device)
        self.position_z = torch.zeros(shape, dtype=torch.int64, device=self.device)
        self.direction = torch.zeros(shape, dtype=torch.int64, device=self.device)
        self.move_direction = torch.zeros(shape, dtype=torch.int64, device=self.device)
        self.speed = torch.zeros(shape, dtype=torch.int64, device=self.device)
        self.turret_direction = torch.zeros(shape, dtype=torch.int64, device=self.device)
        self.turret_pitch = torch.zeros(shape, dtype=torch.int64, device=self.device)
        self.turret_aligned = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.body = torch.zeros(shape, dtype=torch.int64, device=self.device)
        self.previous_body = torch.zeros(shape, dtype=torch.int64, device=self.device)
        self.alive = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.team = torch.zeros(shape, dtype=torch.int64, device=self.device)
        self.unit_id = torch.zeros(shape, dtype=torch.int64, device=self.device)
        self.last_fired = torch.full(shape, -stats.fire_pause, dtype=torch.int64, device=self.device)
        projectile_shape = (arenas, max_units, PROJECTILES_PER_UNIT)
        self.projectile_impact_time = torch.full(projectile_shape, -1, dtype=torch.int64, device=self.device)
        self.projectile_target = torch.full(projectile_shape, -1, dtype=torch.int64, device=self.device)
        self.projectile_damage = torch.zeros(projectile_shape, dtype=torch.int64, device=self.device)
        self.projectile_drop_count = torch.zeros((arenas,), dtype=torch.int64, device=self.device)
        self.previous_action = torch.zeros((arenas, max_units, 4), dtype=torch.int64, device=self.device)
        self.enemy_slots = torch.zeros((arenas, max_units, 8), dtype=torch.int64, device=self.device)
        self.enemy_slot_assigned = torch.zeros((arenas, max_units, 8), dtype=torch.bool, device=self.device)
        self.ally_slots = torch.zeros((arenas, max_units, 7), dtype=torch.int64, device=self.device)
        self.ally_slot_assigned = torch.zeros((arenas, max_units, 7), dtype=torch.bool, device=self.device)
        self.reset()

    def reset(self, mask: Tensor | None = None) -> None:
        if mask is None:
            mask = torch.ones(self.arenas, dtype=torch.bool, device=self.device)
        mask = mask.to(device=self.device, dtype=torch.bool)
        # Use fixed-size work. A CUDA tensor must not select a Python branch or
        # a dynamic allocation size in the simulator update path.
        count = self.arenas
        phase = torch.randint(0, GAME_TICKS_PER_SEC, (count, 1), generator=self.generator, device=self.device)
        self.game_time.copy_(torch.where(mask.unsqueeze(-1), phase, self.game_time))
        active_units = min(self.max_units, 4)
        self.alive[mask] = False
        self.alive[mask, :active_units] = True
        teams = torch.tensor([0, 0, 1, 1], dtype=torch.int64, device=self.device)[:active_units]
        self.team[mask, :active_units] = teams
        ids = torch.arange(1, active_units + 1, dtype=torch.int64, device=self.device)
        self.unit_id[mask, :active_units] = ids
        center = 32 * TILE_UNITS
        positions_x = torch.tensor([-4, -4, 4, 4], device=self.device)[:active_units] * TILE_UNITS + center
        positions_y = torch.tensor([-1, 1, 1, -1], device=self.device)[:active_units] * TILE_UNITS + center
        self.position_x[mask, :active_units] = positions_x
        self.position_y[mask, :active_units] = positions_y
        self.position_z[mask, :active_units] = TILE_UNITS
        directions = torch.tensor([_degrees(90), _degrees(90), _degrees(270), _degrees(270)], device=self.device)[:active_units]
        self.direction[mask, :active_units] = directions
        self.move_direction[mask, :active_units] = directions
        self.speed[mask] = 0
        self.turret_direction[mask] = 0
        self.turret_pitch[mask] = 0
        self.turret_aligned[mask] = False
        self.body[mask] = 0
        self.body[mask, :active_units] = self.stats.health
        self.previous_body[mask] = self.body[mask]
        self.last_fired[mask] = self.game_time[mask] - self.stats.fire_pause
        self.projectile_impact_time[mask] = -1
        self.projectile_target[mask] = -1
        self.projectile_damage[mask] = 0
        self.projectile_drop_count[mask] = 0
        self.previous_action[mask] = 0
        unit = torch.arange(self.max_units, device=self.device)
        candidates = unit.view(1, 1, -1).expand(count, self.max_units, -1)
        selected_team = self.team
        hostile = selected_team.unsqueeze(-1) != selected_team.unsqueeze(-2)
        exists = self.alive.unsqueeze(1).expand(-1, self.max_units, -1)
        sentinel = torch.full_like(candidates, self.max_units)
        assigned = torch.where(hostile & exists, candidates, sentinel).sort(dim=-1).values
        if self.max_units < 8:
            padding = torch.full(
                (count, self.max_units, 8 - self.max_units),
                self.max_units,
                dtype=torch.int64,
                device=self.device,
            )
            assigned = torch.cat((assigned, padding), dim=-1)
        assigned = assigned[..., :8]
        self.enemy_slot_assigned[mask] = (assigned < self.max_units)[mask]
        self.enemy_slots[mask] = assigned.clamp(max=self.max_units - 1)[mask]
        self_index = unit.view(1, -1, 1)
        allied = (selected_team.unsqueeze(-1) == selected_team.unsqueeze(-2)) & (candidates != self_index)
        ally_assigned = torch.where(allied & exists, candidates, sentinel).sort(dim=-1).values
        if self.max_units < 7:
            padding = torch.full(
                (count, self.max_units, 7 - self.max_units),
                self.max_units,
                dtype=torch.int64,
                device=self.device,
            )
            ally_assigned = torch.cat((ally_assigned, padding), dim=-1)
        ally_assigned = ally_assigned[..., :7]
        self.ally_slot_assigned[mask] = (ally_assigned < self.max_units)[mask]
        self.ally_slots[mask] = ally_assigned.clamp(max=self.max_units - 1)[mask]

    def _enemy_indices_and_mask(self) -> tuple[Tensor, Tensor]:
        candidates = self.enemy_slots
        target_alive = torch.gather(
            self.alive.unsqueeze(1).expand(-1, self.max_units, -1), 2, candidates
        )
        target_team = torch.gather(
            self.team.unsqueeze(1).expand(-1, self.max_units, -1), 2, candidates
        )
        mask = (
            self.enemy_slot_assigned
            & self.alive.unsqueeze(-1)
            & target_alive
            & (target_team != self.team.unsqueeze(-1))
        )
        return candidates, mask

    def build_observation(
        self,
        enemy_data: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        observation = torch.zeros(
            (self.arenas, self.max_units, OBSERVATION_SIZE),
            dtype=torch.int16,
            device=self.device,
        )
        max_speed = max(self.stats.base_speed, 1)
        health = max(self.stats.health, 1)
        observation[..., 0] = torch.clamp(self.body * Q15_MAX // health, 0, Q15_MAX).to(torch.int16)
        observation[..., 1] = torch.clamp(self.speed * Q15_MAX // max_speed, 0, Q15_MAX).to(torch.int16)
        observation[..., 2] = min(Q15_MAX, self.stats.calculated_max_speed * Q15_MAX // max_speed)
        observation[..., 3] = self.trig.sin_r(self.direction, torch.full_like(self.direction, Q15_MAX)).to(torch.int16)
        observation[..., 4] = self.trig.cos_r(self.direction, torch.full_like(self.direction, Q15_MAX)).to(torch.int16)
        cooldown = torch.clamp(self.game_time - self.last_fired, 0, self.stats.fire_pause)
        observation[..., 5] = (cooldown * Q15_MAX // max(self.stats.fire_pause, 1)).to(torch.int16)
        observation[..., 6] = ((self.game_time - self.last_fired) >= self.stats.fire_pause).to(torch.int16) * Q15_MAX
        observation[..., 7] = trunc_div(angle_delta(self.turret_direction) * Q15_MAX, 32768).to(torch.int16)
        if enemy_data is None:
            enemy_indices, target_mask = self._enemy_indices_and_mask()
        else:
            enemy_indices, target_mask = enemy_data
        previous_slot = self.previous_action[..., 2]
        previous_valid_slot = (previous_slot >= 0) & (previous_slot < 8)
        selected_slot = previous_slot.clamp(0, 7)
        selected_target = torch.gather(enemy_indices, 2, selected_slot.unsqueeze(-1)).squeeze(-1)
        selected_valid = previous_valid_slot & torch.gather(
            target_mask, 2, selected_slot.unsqueeze(-1)
        ).squeeze(-1)
        selected_x = torch.gather(self.position_x, 1, selected_target)
        selected_y = torch.gather(self.position_y, 1, selected_target)
        selected_z = torch.gather(self.position_z, 1, selected_target)
        selected_heading = self.trig.atan2(
            selected_x - self.position_x, selected_y - self.position_y
        )
        turret_error = angle_delta(selected_heading - self.direction - self.turret_direction)
        turret_error = torch.where(selected_valid, turret_error, 0)
        selected_pitch, selected_pitch_required = self._target_pitch_and_required(
            selected_x, selected_y, selected_z
        )
        selected_pitch_aligned = ~selected_pitch_required | (
            angle_delta(selected_pitch - self.turret_pitch) == 0
        )
        observation[..., 8] = trunc_div(turret_error * Q15_MAX, 32768).to(torch.int16)
        observation[..., 9] = (
            selected_valid & (turret_error == 0) & selected_pitch_aligned
        ).to(torch.int16) * Q15_MAX
        recent_damage = torch.clamp(self.previous_body - self.body, min=0)
        observation[..., 10] = (recent_damage * Q15_MAX // health).to(torch.int16)
        turn_denominator = max(self.stats.turn_speed, self.stats.spin_speed, 1)
        observation[..., 11] = self.stats.turn_speed * Q15_MAX // turn_denominator
        observation[..., 12] = self.stats.spin_speed * Q15_MAX // turn_denominator
        observation[..., 13] = min(Q15_MAX, self.stats.long_range * Q15_MAX // (16 * TILE_UNITS))
        observation[..., 14] = min(Q15_MAX, self.stats.damage * Q15_MAX // 1000)
        observation[..., 15] = min(Q15_MAX, self.stats.armour * Q15_MAX // 1000)
        observation[..., 16] = Q15_MAX if self.stats.fire_on_move else 0
        observation[..., 19] = self.alive.to(torch.int16) * Q15_MAX

        world_width = 64 * TILE_UNITS
        world_height = 64 * TILE_UNITS
        for ray in range(8):
            ray_direction = self.direction + ray * 8192
            ray_radius = torch.full_like(self.direction, 16 * TILE_UNITS)
            ray_dx = self.trig.sin_r(ray_direction, ray_radius)
            ray_dy = self.trig.cos_r(ray_direction, ray_radius)
            fraction = torch.full_like(self.direction, Q15_MAX)
            fraction = torch.where(
                ray_dx > 0,
                torch.minimum(fraction, (world_width - 1 - self.position_x) * Q15_MAX // ray_dx.clamp(min=1)),
                fraction,
            )
            fraction = torch.where(
                ray_dx < 0,
                torch.minimum(fraction, (self.position_x - 1) * Q15_MAX // (-ray_dx).clamp(min=1)),
                fraction,
            )
            fraction = torch.where(
                ray_dy > 0,
                torch.minimum(fraction, (world_height - 1 - self.position_y) * Q15_MAX // ray_dy.clamp(min=1)),
                fraction,
            )
            fraction = torch.where(
                ray_dy < 0,
                torch.minimum(fraction, (self.position_y - 1) * Q15_MAX // (-ray_dy).clamp(min=1)),
                fraction,
            )
            observation[..., RAY_OFFSET + ray] = fraction.clamp(0, Q15_MAX).to(torch.int16)

        def fill_slots(indices: Tensor, assigned: Tensor, offset: int, width: int) -> None:
            expanded_x = self.position_x.unsqueeze(1).expand(-1, self.max_units, -1)
            expanded_y = self.position_y.unsqueeze(1).expand(-1, self.max_units, -1)
            expanded_body = self.body.unsqueeze(1).expand(-1, self.max_units, -1)
            expanded_direction = self.direction.unsqueeze(1).expand(-1, self.max_units, -1)
            expanded_speed = self.speed.unsqueeze(1).expand(-1, self.max_units, -1)
            expanded_id = self.unit_id.unsqueeze(1).expand(-1, self.max_units, -1)
            target_alive = torch.gather(
                self.alive.unsqueeze(1).expand(-1, self.max_units, -1), 2, indices
            )
            valid = assigned & target_alive
            target_x = torch.gather(expanded_x, 2, indices)
            target_y = torch.gather(expanded_y, 2, indices)
            target_body = torch.gather(expanded_body, 2, indices)
            target_direction = torch.gather(expanded_direction, 2, indices)
            target_speed = torch.gather(expanded_speed, 2, indices)
            target_id = torch.gather(expanded_id, 2, indices)
            dx = target_x - self.position_x.unsqueeze(-1)
            dy = target_y - self.position_y.unsqueeze(-1)
            local_x = self.trig.cos_r(self.direction.unsqueeze(-1), dx) - self.trig.sin_r(
                self.direction.unsqueeze(-1), dy
            )
            local_y = self.trig.sin_r(self.direction.unsqueeze(-1), dx) + self.trig.cos_r(
                self.direction.unsqueeze(-1), dy
            )
            for slot in range(indices.shape[-1]):
                base = offset + slot * width
                slot_valid = valid[..., slot]
                local_x_q = trunc_div(local_x[..., slot] * Q15_MAX, 16 * TILE_UNITS).clamp(
                    -Q15_MAX, Q15_MAX
                )
                local_y_q = trunc_div(local_y[..., slot] * Q15_MAX, 16 * TILE_UNITS).clamp(
                    -Q15_MAX, Q15_MAX
                )
                direction_q = trunc_div(
                    angle_delta(target_direction[..., slot] - self.direction) * Q15_MAX, 32768
                ).clamp(-Q15_MAX, Q15_MAX)
                observation[..., base + 0] = torch.where(slot_valid, local_x_q, 0).to(torch.int16)
                observation[..., base + 1] = torch.where(slot_valid, local_y_q, 0).to(torch.int16)
                observation[..., base + 2] = torch.where(
                    slot_valid, target_body[..., slot] * Q15_MAX // health, 0
                ).to(torch.int16)
                observation[..., base + 3] = torch.where(slot_valid, direction_q, 0).to(torch.int16)
                observation[..., base + 4] = torch.where(
                    slot_valid, target_speed[..., slot] * Q15_MAX // max_speed, 0
                ).to(torch.int16)
                observation[..., base + 5] = slot_valid.to(torch.int16) * Q15_MAX
                if width > 6:
                    observation[..., base + 6] = 0
                if width > 7:
                    observation[..., base + 7] = slot_valid.to(torch.int16) * Q15_MAX
                if width > 8:
                    observation[..., base + 8] = slot_valid.to(torch.int16) * Q15_MAX
                if width > 9:
                    observation[..., base + 9] = slot_valid.to(torch.int16) * Q15_MAX
                if width > 10:
                    observation[..., base + 10] = torch.where(
                        slot_valid, (torch.bitwise_and(target_id[..., slot], 0x7FFF) * Q15_MAX // 0x7FFF), 0
                    ).to(torch.int16)

        fill_slots(self.ally_slots, self.ally_slot_assigned, ALLY_OFFSET, 10)
        fill_slots(enemy_indices, self.enemy_slot_assigned, ENEMY_OFFSET, 11)

        observation[..., CONTEXT_OFFSET] = (self.game_time % GAME_TICKS_PER_SEC * Q15_MAX // (GAME_TICKS_PER_SEC - 1)).to(torch.int16)
        observation[..., CONTEXT_OFFSET + 1] = trunc_div(
            self.previous_action[..., 0] * Q15_MAX, 2048
        ).clamp(-Q15_MAX, Q15_MAX).to(torch.int16)
        observation[..., CONTEXT_OFFSET + 2] = (self.previous_action[..., 1] * Q15_MAX // 256).to(torch.int16)
        observation[..., CONTEXT_OFFSET + 3] = torch.where(
            self.previous_action[..., 2] < 0,
            -Q15_MAX,
            (self.previous_action[..., 2] + 1) * Q15_MAX // 8,
        ).to(torch.int16)
        observation[..., CONTEXT_OFFSET + 4] = self.previous_action[..., 3].to(torch.int16) * Q15_MAX
        phase = torch.bitwise_and(self.game_time * 65536 // 90000, 0xFFFF)
        observation[..., CONTEXT_OFFSET + 5] = self.trig.sin_r(
            phase, torch.full_like(phase, Q15_MAX)
        ).to(torch.int16)
        observation[..., CONTEXT_OFFSET + 6] = self.trig.cos_r(
            phase, torch.full_like(phase, Q15_MAX)
        ).to(torch.int16)
        observation[..., CONTEXT_OFFSET + 7] = (self.team * Q15_MAX // (MAX_PLAYERS - 1)).to(torch.int16)
        observation[..., CONTEXT_OFFSET + 8] = (
            torch.bitwise_and(self.unit_id, 0x7FFF) * Q15_MAX // 0x7FFF
        ).to(torch.int16)
        observation[..., CONTEXT_OFFSET + 9] = Q15_MAX
        target_mask = target_mask & self.alive.unsqueeze(-1)
        return observation, target_mask

    def _move(self, action: Tensor) -> None:
        desired_direction = torch.bitwise_and(self.direction + action[..., 0] * ACTION_ANGLE_STEP, 0xFFFF)
        desired_speed = self.stats.calculated_max_speed * action[..., 1] // 256
        difference = angle_delta(desired_direction - self.direction)
        spin_angle = max(_degrees(self.stats.spin_angle), 1)
        desired_speed = torch.clamp(trunc_div(desired_speed * (spin_angle - difference.abs()), spin_angle), min=0)
        spin_speed = self.stats.base_speed * self.stats.spin_speed
        turn_speed = self.stats.base_speed * self.stats.turn_speed
        turn_rate = torch.minimum(
            torch.full_like(difference, spin_speed),
            turn_speed + trunc_div((spin_speed - turn_speed) * difference.abs(), spin_angle),
        )
        new_time = self.game_time + UPDATE_TICKS
        max_change = quantise_fraction(turn_rate, GAME_TICKS_PER_SEC, new_time, self.game_time)
        facing = torch.bitwise_and(self.direction + torch.clamp(difference, -max_change, max_change), 0xFFFF)

        relative_move = torch.bitwise_and(facing - self.move_direction, 0xFFFF)
        normal_speed = self.trig.cos_r(relative_move, self.speed)
        accel = quantise_fraction(torch.full_like(normal_speed, self.stats.acceleration), GAME_TICKS_PER_SEC, new_time, self.game_time)
        decel = quantise_fraction(torch.full_like(normal_speed, self.stats.deceleration), GAME_TICKS_PER_SEC, new_time, self.game_time)
        normal_speed = torch.where(
            normal_speed < desired_speed,
            torch.minimum(normal_speed + accel, desired_speed),
            torch.maximum(normal_speed - decel, desired_speed),
        )
        old_angle_difference = angle_delta(facing - self.move_direction).abs()
        perpendicular = self.trig.sin_r(old_angle_difference, self.speed)
        skid = quantise_fraction(torch.full_like(perpendicular, self.stats.skid_deceleration), GAME_TICKS_PER_SEC, new_time, self.game_time)
        perpendicular = torch.clamp(perpendicular - skid, min=0)
        final_speed = integer_floor_sqrt(normal_speed * normal_speed + perpendicular * perpendicular)
        relative_direction = self.trig.atan2(perpendicular, normal_speed)
        signed_difference = angle_delta(facing - self.move_direction)
        move_direction = torch.where(signed_difference < 0, facing + relative_direction, facing - relative_direction)
        move_direction = torch.where(perpendicular == 0, facing, move_direction)
        move_direction = torch.bitwise_and(move_direction, 0xFFFF)

        active = self.alive
        self.direction = torch.where(active, facing, self.direction)
        self.move_direction = torch.where(active, move_direction, self.move_direction)
        self.speed = torch.where(active, final_speed, self.speed)
        high_precision_move = final_speed * EXTRA_PRECISION
        dx = self.trig.sin_r(move_direction, high_precision_move)
        dy = self.trig.cos_r(move_direction, high_precision_move)
        self._move_positions_with_droid_slide(dx, dy, new_time)

    def _move_positions_with_droid_slide(
        self, movement_x: Tensor, movement_y: Tensor, new_time: Tensor
    ) -> None:
        """Apply the V1 light-droid slide in engine droid-update order."""

        arena = torch.arange(self.arenas, device=self.device)
        # The engine visits players in ascending order. New droids are at the
        # front of each player list, so their IDs are in descending order.
        dead_key = torch.full_like(self.unit_id, (MAX_PLAYERS + 1) << 32)
        order_key = self.team * (1 << 32) + ((1 << 32) - 1 - self.unit_id)
        order_key = torch.where(self.alive, order_key, dead_key)
        update_order = torch.argsort(order_key, dim=1)
        candidate_index = torch.arange(self.max_units, device=self.device).view(1, -1)
        collision_radius_squared = (2 * V1_DROID_RADIUS) ** 2
        old_time = self.game_time[:, 0]
        next_time = new_time[:, 0]

        for order_slot in range(self.max_units):
            unit = update_order[:, order_slot]
            active = self.alive[arena, unit]
            raw_x = movement_x[arena, unit]
            raw_y = movement_y[arena, unit]
            step_x = quantise_fraction(
                raw_x, GAME_TICKS_PER_SEC * EXTRA_PRECISION, next_time, old_time
            )
            step_y = quantise_fraction(
                raw_y, GAME_TICKS_PER_SEC * EXTRA_PRECISION, next_time, old_time
            )
            own_x = self.position_x[arena, unit]
            own_y = self.position_y[arena, unit]
            x_difference = own_x.unsqueeze(-1) + step_x.unsqueeze(-1) - self.position_x
            y_difference = own_y.unsqueeze(-1) + step_y.unsqueeze(-1) - self.position_y
            distance_squared = x_difference * x_difference + y_difference * y_difference
            behind = (
                x_difference * step_x.unsqueeze(-1) + y_difference * step_y.unsqueeze(-1)
            ) >= 0
            collision = (
                self.alive
                & (candidate_index != unit.unsqueeze(-1))
                & ~behind
                & (distance_squared < collision_radius_squared)
                & active.unsqueeze(-1)
            )
            collision_count = collision.sum(dim=-1)
            obstacle = collision.to(torch.int64).argmax(dim=-1)
            obstacle_x = self.position_x[arena, obstacle]
            obstacle_y = self.position_y[arena, obstacle]
            obstruction_x = own_x - obstacle_x
            obstruction_y = own_y - obstacle_y
            already_away = obstruction_x * raw_x + obstruction_y * raw_y >= 0
            dot = obstruction_y * raw_x - obstruction_x * raw_y
            positive_side = dot >= 0
            tangent_x = torch.where(positive_side, obstruction_y, -obstruction_y)
            tangent_y = torch.where(positive_side, -obstruction_x, obstruction_x)
            tangent_dot = dot.abs()
            tangent_magnitude_squared = torch.clamp(
                tangent_x * tangent_x + tangent_y * tangent_y, min=1
            )
            slide_x = trunc_div(tangent_x * tangent_dot, tangent_magnitude_squared)
            slide_y = trunc_div(tangent_y * tangent_dot, tangent_magnitude_squared)
            one_collision = (collision_count == 1) & ~already_away
            adjusted_x = torch.where(one_collision, slide_x, raw_x)
            adjusted_y = torch.where(one_collision, slide_y, raw_y)
            adjusted_x = torch.where(collision_count > 1, 0, adjusted_x)
            adjusted_y = torch.where(collision_count > 1, 0, adjusted_y)
            position_x = quantise_fraction(
                adjusted_x, GAME_TICKS_PER_SEC * EXTRA_PRECISION, next_time, old_time
            )
            position_y = quantise_fraction(
                adjusted_y, GAME_TICKS_PER_SEC * EXTRA_PRECISION, next_time, old_time
            )
            self.position_x[arena, unit] = torch.where(active, own_x + position_x, own_x)
            self.position_y[arena, unit] = torch.where(active, own_y + position_y, own_y)

    def _target_pitch_and_required(
        self, target_x: Tensor, target_y: Tensor, target_z: Tensor
    ) -> tuple[Tensor, Tensor]:
        connector_x = torch.full_like(self.position_x, self.stats.muzzle_connector_x)
        connector_y = torch.full_like(self.position_y, self.stats.muzzle_connector_y)
        muzzle_x = trunc_div(
            self.position_x * 65536
            + self.trig.cos(self.direction) * connector_x
            - self.trig.sin(self.direction) * connector_y,
            65536,
        )
        muzzle_y = trunc_div(
            self.position_y * 65536
            - self.trig.sin(self.direction) * connector_x
            - self.trig.cos(self.direction) * connector_y,
            65536,
        )
        muzzle_z = self.position_z + self.stats.muzzle_connector_z
        pitch_dx = target_x - muzzle_x
        pitch_dy = target_y - muzzle_y
        pitch_distance = integer_floor_sqrt(pitch_dx * pitch_dx + pitch_dy * pitch_dy)
        target_pitch = angle_delta(self.trig.atan2(target_z - muzzle_z, pitch_distance))
        target_pitch.clamp_(self.stats.min_elevation, self.stats.max_elevation)
        pitch_required = self.stats.direct_weapon & (
            (target_x - self.position_x) ** 2 + (target_y - self.position_y) ** 2
            > self.stats.min_range**2
        )
        return target_pitch, pitch_required

    def _combat(
        self,
        action: Tensor,
        forced_roll: Tensor | None,
        enemy_indices: Tensor,
        target_mask: Tensor,
    ) -> Tensor:
        requested = action[..., 2]
        valid_slot = (requested >= 0) & (requested < 8)
        slot = requested.clamp(0, 7)
        target = torch.gather(enemy_indices, 2, slot.unsqueeze(-1)).squeeze(-1)
        valid_target = valid_slot & torch.gather(target_mask, 2, slot.unsqueeze(-1)).squeeze(-1) & self.alive
        target_x = torch.gather(self.position_x, 1, target)
        target_y = torch.gather(self.position_y, 1, target)
        target_z = torch.gather(self.position_z, 1, target)
        bearing = self.trig.atan2(target_x - self.position_x, target_y - self.position_y)
        error = angle_delta(bearing - (self.direction + self.turret_direction))
        next_turret_direction = torch.bitwise_and(
            self.turret_direction
            + torch.clamp(error, -self.stats.turret_rotation_rate, self.stats.turret_rotation_rate),
            0xFFFF,
        )
        self.turret_direction = torch.where(valid_target, next_turret_direction, self.turret_direction)
        yaw_aligned = angle_delta(bearing - (self.direction + self.turret_direction)) == 0
        target_pitch, pitch_required = self._target_pitch_and_required(
            target_x, target_y, target_z
        )
        pitch_error = angle_delta(target_pitch - self.turret_pitch)
        next_turret_pitch = torch.bitwise_and(
            self.turret_pitch
            + torch.clamp(
                pitch_error,
                -self.stats.turret_pitch_rate,
                self.stats.turret_pitch_rate,
            ),
            0xFFFF,
        )
        self.turret_pitch = torch.where(
            valid_target & pitch_required, next_turret_pitch, self.turret_pitch
        )
        pitch_aligned = ~pitch_required | (angle_delta(target_pitch - self.turret_pitch) == 0)
        aligned = yaw_aligned & pitch_aligned
        self.turret_aligned = valid_target & aligned
        ready = (self.game_time - self.last_fired) >= self.stats.fire_pause
        moving_ok = self.stats.fire_on_move | (self.speed == 0)
        dx = target_x - self.position_x
        dy = target_y - self.position_y
        distance = integer_floor_sqrt(dx * dx + dy * dy)
        in_range = (distance <= self.stats.long_range) & (distance >= self.stats.min_range)
        fire = valid_target & aligned & ready & moving_ok & in_range & (action[..., 3] != 0)
        new_last_fired = torch.maximum(
            self.game_time - UPDATE_TICKS + 1,
            self.last_fired + self.stats.fire_pause,
        )
        self.last_fired = torch.where(fire, new_last_fired, self.last_fired)

        hit_chance = torch.where(distance <= self.stats.short_range, self.stats.short_hit_chance, self.stats.long_hit_chance)
        if forced_roll is None:
            roll = torch.randint(0, 100, fire.shape, generator=self.generator, device=self.device)
        else:
            roll = forced_roll.to(device=self.device, dtype=torch.int64)
        hit = fire & (roll <= hit_chance)
        delay = torch.maximum(
            torch.full_like(distance, UPDATE_TICKS),
            trunc_div(distance * GAME_TICKS_PER_SEC, max(self.stats.projectile_speed, 1)),
        )
        free = self.projectile_impact_time < 0
        free_index = free.to(torch.int64).argmax(dim=-1)
        has_free = free.any(dim=-1)
        enqueue = hit & has_free
        self.projectile_drop_count += (hit & ~has_free).sum(dim=1)
        destination = free_index.unsqueeze(-1)
        impact = torch.where(enqueue, new_last_fired + delay, -1)
        self.projectile_impact_time.scatter_(2, destination, torch.where(enqueue, impact, torch.gather(self.projectile_impact_time, 2, destination).squeeze(-1)).unsqueeze(-1))
        self.projectile_target.scatter_(2, destination, torch.where(enqueue, target, torch.gather(self.projectile_target, 2, destination).squeeze(-1)).unsqueeze(-1))
        self.projectile_damage.scatter_(2, destination, torch.where(enqueue, self.stats.hit_damage, torch.gather(self.projectile_damage, 2, destination).squeeze(-1)).unsqueeze(-1))

        target_alive = torch.gather(
            self.alive.unsqueeze(1).expand(-1, self.max_units, -1),
            2,
            enemy_indices,
        )
        return target_mask & self.alive.unsqueeze(-1) & target_alive

    def _apply_due_projectiles(self) -> None:
        # The ML droid update and its trace run before proj_UpdateAll in the
        # engine update. Damage from this update is first visible to the policy
        # on the next 100 ms update.
        due = (self.projectile_impact_time >= 0) & (
            self.projectile_impact_time
            <= self.game_time.unsqueeze(-1) - UPDATE_TICKS
        )
        damage = self.projectile_damage * due
        target_index = self.projectile_target.clamp(min=0)
        received = torch.zeros_like(self.body)
        received.scatter_add_(1, target_index.flatten(start_dim=1), damage.flatten(start_dim=1))
        self.body = torch.clamp(self.body - received, min=0)
        self.alive &= self.body > 0
        self.projectile_impact_time.masked_fill_(due, -1)
        self.projectile_target.masked_fill_(due, -1)
        self.projectile_damage.masked_fill_(due, 0)

    def step(
        self,
        action: Tensor,
        forced_roll: Tensor | None = None,
        *,
        build_observation: bool = True,
    ) -> tuple[Tensor, Tensor] | None:
        if action.shape != (self.arenas, self.max_units, 4):
            raise ValueError("action must have shape [arenas, units, 4]")
        action = action.to(device=self.device, dtype=torch.int64).clone()
        action[..., 0].clamp_(-2048, 2047)
        action[..., 1].clamp_(0, 256)
        action[..., 2].clamp_(-1, 7)
        action[..., 3].clamp_(0, 1)
        self.previous_body.copy_(self.body)
        self._apply_due_projectiles()
        enemy_indices, target_mask = self._enemy_indices_and_mask()
        next_target_mask = self._combat(action, forced_roll, enemy_indices, target_mask)
        self._move(action)
        self.previous_action.copy_(action)
        self.game_time += UPDATE_TICKS
        if not build_observation:
            return None
        return self.build_observation((enemy_indices, next_target_mask))

    @property
    def unit_decisions(self) -> int:
        return self.arenas * self.max_units
