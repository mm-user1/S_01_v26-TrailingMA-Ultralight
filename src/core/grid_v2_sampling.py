"""Deterministic, O(K) sampling primitives for budgeted Grid V2 plans."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


GRID_V2_PLANNING_POLICY_VERSION = "grid_v2_planning_v1"
GRID_V2_SAMPLER_VERSION = "balanced_discrete_lhs_pcg64_v1"
GRID_V2_ALLOCATOR_VERSION = "ordered_block_allocator_v1"
_UINT64_RANGE = 1 << 64


@dataclass(frozen=True)
class AxisSamplingDiagnostics:
    axis_name: str
    level_count: int
    requested_count: int
    phase: int
    minimum_level_frequency: int
    maximum_level_frequency: int


@dataclass(frozen=True)
class BlockSamplingDiagnostics:
    full_count: int
    requested_count: int
    delivered_count: int
    primary_unique_count: int
    collision_count: int
    topup_count: int
    topup_attempt_count: int
    shortfall_count: int
    axis_diagnostics: tuple[AxisSamplingDiagnostics, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_grid_v2_seed(seed: Any) -> int:
    """Return a seed accepted by the versioned PCG64 raw-stream contract."""

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("Grid V2 seed must be an integer.")
    value = int(seed)
    if value < 0 or value >= _UINT64_RANGE:
        raise ValueError("Grid V2 seed must be between 0 and 2**64 - 1.")
    return value


def derive_named_seed(global_seed: int, context: Mapping[str, Any]) -> int:
    """Derive a stable 128-bit named substream seed using canonical JSON."""

    payload = {
        "planning_policy_version": GRID_V2_PLANNING_POLICY_VERSION,
        "sampler_version": GRID_V2_SAMPLER_VERSION,
        "global_seed": validate_grid_v2_seed(global_seed),
        "context": _jsonable(context),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(encoded, digest_size=16).digest(), "little")


def stable_randbelow(bit_generator: np.random.PCG64, upper_bound: int) -> int:
    """Draw uniformly from ``range(upper_bound)`` using raw uint64 rejection."""

    bound = int(upper_bound)
    if bound <= 0 or bound > _UINT64_RANGE:
        raise ValueError("upper_bound must be in [1, 2**64].")
    rejection_limit = _UINT64_RANGE - (_UINT64_RANGE % bound)
    while True:
        raw = int(bit_generator.random_raw())
        if raw < rejection_limit:
            return raw % bound


def stable_fisher_yates(size: int, bit_generator: np.random.PCG64) -> tuple[int, ...]:
    """Return a stable Fisher-Yates permutation driven only by raw PCG64."""

    count = int(size)
    if count < 0:
        raise ValueError("Permutation size must be non-negative.")
    values = list(range(count))
    for index in range(count - 1, 0, -1):
        swap_index = stable_randbelow(bit_generator, index + 1)
        values[index], values[swap_index] = values[swap_index], values[index]
    return tuple(values)


def mixed_radix_index(codes: Sequence[int], radices: Sequence[int]) -> int:
    if len(codes) != len(radices):
        raise ValueError("Mixed-radix code and radix lengths differ.")
    index = 0
    for code, radix in zip(codes, radices):
        radix_value = int(radix)
        code_value = int(code)
        if radix_value <= 0 or code_value < 0 or code_value >= radix_value:
            raise ValueError("Mixed-radix code is outside its domain.")
        index = index * radix_value + code_value
    return index


def decode_mixed_radix(index: int, radices: Sequence[int]) -> tuple[int, ...]:
    full_count = math.prod(int(radix) for radix in radices)
    value = int(index)
    if value < 0 or value >= full_count:
        raise ValueError("Mixed-radix index is outside the full block.")
    codes = [0] * len(radices)
    for position in range(len(radices) - 1, -1, -1):
        radix = int(radices[position])
        codes[position] = value % radix
        value //= radix
    return tuple(codes)


def sample_block_codes(
    radices: Sequence[int],
    requested_count: int,
    *,
    global_seed: int,
    strategy_id: str,
    strategy_version: str,
    block_id: str,
    axis_names: Sequence[str],
) -> tuple[tuple[tuple[int, ...], ...], BlockSamplingDiagnostics]:
    """Sample one discrete Cartesian block in O(K) memory and time."""

    normalized_radices = tuple(int(radix) for radix in radices)
    normalized_axis_names = tuple(str(name) for name in axis_names)
    if len(normalized_radices) != len(normalized_axis_names):
        raise ValueError("Every sampled radix must have an axis name.")
    if any(radix <= 0 for radix in normalized_radices):
        raise ValueError("Sampled axis domains must be non-empty.")
    full_count = math.prod(normalized_radices)
    budget = int(requested_count)
    if budget < 0 or budget > full_count:
        raise ValueError("Sampled block budget must be between zero and its full count.")
    if budget == 0:
        diagnostics = BlockSamplingDiagnostics(
            full_count=full_count,
            requested_count=0,
            delivered_count=0,
            primary_unique_count=0,
            collision_count=0,
            topup_count=0,
            topup_attempt_count=0,
            shortfall_count=0,
            axis_diagnostics=(),
        )
        return (), diagnostics

    base_context = {
        "strategy_id": str(strategy_id),
        "strategy_version": str(strategy_version),
        "block_id": str(block_id),
    }
    axis_columns: list[tuple[int, ...]] = []
    axis_diagnostics: list[AxisSamplingDiagnostics] = []
    for axis_name, level_count in zip(normalized_axis_names, normalized_radices):
        phase_rng = np.random.PCG64(
            derive_named_seed(global_seed, {**base_context, "axis": axis_name, "purpose": "phase"})
        )
        phase = stable_randbelow(phase_rng, level_count)
        base_codes = tuple(
            ((row * level_count + phase) // budget) % level_count
            for row in range(budget)
        )
        permutation_rng = np.random.PCG64(
            derive_named_seed(global_seed, {**base_context, "axis": axis_name, "purpose": "permutation"})
        )
        permutation = stable_fisher_yates(budget, permutation_rng)
        column = tuple(base_codes[position] for position in permutation)
        axis_columns.append(column)
        frequencies = [0] * level_count
        for code in column:
            frequencies[code] += 1
        axis_diagnostics.append(
            AxisSamplingDiagnostics(
                axis_name=axis_name,
                level_count=level_count,
                requested_count=budget,
                phase=phase,
                minimum_level_frequency=min(frequencies),
                maximum_level_frequency=max(frequencies),
            )
        )

    primary = tuple(tuple(column[row] for column in axis_columns) for row in range(budget))
    selected_by_index: dict[int, tuple[int, ...]] = {}
    for codes in primary:
        selected_by_index.setdefault(mixed_radix_index(codes, normalized_radices), codes)
    primary_unique_count = len(selected_by_index)

    topup_count = 0
    topup_attempt_count = 0
    if primary_unique_count < budget:
        topup_rng = np.random.PCG64(
            derive_named_seed(global_seed, {**base_context, "purpose": "mixed_radix_topup"})
        )
        start = stable_randbelow(topup_rng, full_count)
        if full_count == 1:
            step = 1
        else:
            step = 1
            for _ in range(64):
                candidate_step = stable_randbelow(topup_rng, full_count - 1) + 1
                if math.gcd(candidate_step, full_count) == 1:
                    step = candidate_step
                    break
        for attempt in range(budget):
            if len(selected_by_index) >= budget:
                break
            topup_attempt_count += 1
            index = (step * attempt + start) % full_count
            if index in selected_by_index:
                continue
            selected_by_index[index] = decode_mixed_radix(index, normalized_radices)
            topup_count += 1

    ordered_codes = tuple(selected_by_index[index] for index in sorted(selected_by_index))
    delivered = len(ordered_codes)
    diagnostics = BlockSamplingDiagnostics(
        full_count=full_count,
        requested_count=budget,
        delivered_count=delivered,
        primary_unique_count=primary_unique_count,
        collision_count=budget - primary_unique_count,
        topup_count=topup_count,
        topup_attempt_count=topup_attempt_count,
        shortfall_count=budget - delivered,
        axis_diagnostics=tuple(axis_diagnostics),
    )
    return ordered_codes, diagnostics


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
