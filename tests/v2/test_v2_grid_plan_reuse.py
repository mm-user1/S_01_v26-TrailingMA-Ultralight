from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from core.grid_v2 import (
    GridV2PlanReuseCache,
    GridV2Settings,
    _GridV2CachedPlan,
    _PLAN_REUSE_RUNTIME_PARAM_NAMES,
    _pack_table_config_arrays,
    build_grid_v2_plan,
)
from runtime_test_helpers import canonical_v2_runtime_declarations


def _tiny_v2_config() -> dict:
    return {
        "id": "grid_reuse_fixture",
        "version": "test",
        "engine": "v2",
        "execution": {
            "entryOrder": "market_next_open",
            "stop": "atr_swing",
            "sizing": "risk_per_trade",
            "maxDays": True,
            "margin": "off",
            "boundary": "strict_close",
            "target": "rr",
            "trail": "none",
            "priceRounding": "none",
        },
        "parameters": {
            **canonical_v2_runtime_declarations(),
            "signalMode": {
                "type": "select",
                "default": "A",
                "options": ["A", "B"],
                "role": "signal",
                "optimize": {"enabled": True},
            },
            "threshold": {
                "type": "float",
                "default": 1.0,
                "gridValues": [1.0, 2.0],
                "role": "signal",
                "optimize": {"enabled": True},
            },
            "stopX": {
                "type": "float",
                "default": 2.0,
                "role": "execution",
                "optimize": {"enabled": False},
            },
            "stopLP": {
                "type": "int",
                "default": 2,
                "role": "execution",
                "optimize": {"enabled": False},
            },
            "stopMaxPct": {
                "type": "float",
                "default": 6.0,
                "role": "execution",
                "optimize": {"enabled": False},
            },
            "stopRR": {
                "type": "float",
                "default": 2.0,
                "role": "execution",
                "optimize": {"enabled": False},
            },
            "riskPerTrade": {
                "type": "float",
                "default": 2.0,
                "role": "execution",
                "optimize": {"enabled": False},
            },
            "contractSize": {
                "type": "float",
                "default": 0.01,
                "role": "execution",
                "optimize": {"enabled": False},
            },
            "stopMaxDays": {
                "type": "int",
                "default": 4,
                "role": "execution",
                "optimize": {"enabled": False},
            },
        },
    }


def _window_params(start: str, end: str, **extra) -> dict:
    params = {
        "dateFilter": True,
        "start": start,
        "end": end,
        "stopX": 2.0,
    }
    params.update(extra)
    return params


def test_plan_reuse_rebases_runtime_dates_without_mutating_cached_table():
    cache = GridV2PlanReuseCache()
    settings = GridV2Settings(enabled_axes=("threshold",))

    first = cache.get_or_build(
        _tiny_v2_config(),
        settings=settings,
        base_params=_window_params("2025-01-01T00:00:00Z", "2025-02-01T00:00:00Z"),
    )
    first_table = first.plan.candidate_table
    first_params = first_table.params_for_index(0)
    first_metadata = deepcopy(first.plan.metadata)

    second = cache.get_or_build(
        _tiny_v2_config(),
        settings=settings,
        base_params=_window_params("2025-02-01T00:00:00Z", "2025-03-01T00:00:00Z"),
    )
    second_table = second.plan.candidate_table
    second_params = second_table.params_for_index(0)

    assert first.hit is False
    assert second.hit is True
    assert second.stats.build_count == 1
    assert second.stats.hit_count == 1
    assert second.stats.miss_count == 1
    assert second.plan is not first.plan
    assert second_table is not first_table
    assert second_table.variant_codes is first_table.variant_codes
    assert second_table.axis_value_codes is first_table.axis_value_codes
    assert first_params["start"] == "2025-01-01T00:00:00Z"
    assert first_table.params_for_index(0)["start"] == "2025-01-01T00:00:00Z"
    assert second_params["start"] == "2025-02-01T00:00:00Z"
    assert second_params["end"] == "2025-03-01T00:00:00Z"
    assert second_params["dateFilter"] is True
    assert first.plan.metadata == first_metadata
    assert second.plan.metadata["diagnostics"] == first.plan.metadata["diagnostics"]
    assert second.plan.metadata["validation_warnings"] == ()
    assert "diagnostics" not in first.plan.metadata["planning"]
    assert "validation_warnings" not in first.plan.metadata["planning"]
    assert "diagnostics" not in second.plan.metadata["planning"]
    assert "validation_warnings" not in second.plan.metadata["planning"]
    assert second.runtime_rebase_seconds >= 0.0
    assert second.plan_build_seconds == pytest.approx(0.0)


def test_rebase_removes_stale_nested_diagnostics_and_recomputes_top_level():
    cache = GridV2PlanReuseCache()
    settings = GridV2Settings(enabled_axes=("threshold",))
    first = cache.get_or_build(
        _tiny_v2_config(),
        settings=settings,
        base_params=_window_params("2025-01-01T00:00:00Z", "2025-02-01T00:00:00Z"),
    )
    assert first.cache_key is not None
    entry = cache._entries[first.cache_key]
    stale_metadata = deepcopy(entry.plan.metadata)
    stale_metadata["diagnostics"] = ({"code": "STALE"},)
    stale_metadata["validation_warnings"] = ("stale",)
    stale_metadata["planning"]["diagnostics"] = ({"code": "STALE_NESTED"},)
    stale_metadata["planning"]["validation_warnings"] = ("stale nested",)
    stale_plan = replace(entry.plan, metadata=stale_metadata)
    cache._entries[first.cache_key] = _GridV2CachedPlan(
        plan=stale_plan,
        identity_signature=entry.identity_signature,
    )

    rebased = cache.get_or_build(
        _tiny_v2_config(),
        settings=settings,
        base_params=_window_params("2025-02-01T00:00:00Z", "2025-03-01T00:00:00Z"),
    )
    fresh = build_grid_v2_plan(
        _tiny_v2_config(),
        settings=settings,
        base_params=_window_params("2025-02-01T00:00:00Z", "2025-03-01T00:00:00Z"),
    )

    assert rebased.hit is True
    assert stale_plan.metadata["planning"]["diagnostics"] == (
        {"code": "STALE_NESTED"},
    )
    assert rebased.plan.metadata["diagnostics"] == fresh.metadata["diagnostics"]
    assert rebased.plan.metadata["validation_warnings"] == fresh.metadata[
        "validation_warnings"
    ]
    assert "diagnostics" not in rebased.plan.metadata["planning"]
    assert "validation_warnings" not in rebased.plan.metadata["planning"]
    assert rebased.plan.plan_fingerprint == first.plan.plan_fingerprint


def test_plan_reuse_identity_excludes_runtime_but_rebasable_set_stays_three_dates():
    cache = GridV2PlanReuseCache()
    settings = GridV2Settings(enabled_axes=("threshold",))

    cache.get_or_build(
        _tiny_v2_config(),
        settings=settings,
        base_params=_window_params("2025-01-01T00:00:00Z", "2025-02-01T00:00:00Z"),
    )
    date_only = cache.get_or_build(
        _tiny_v2_config(),
        settings=settings,
        base_params=_window_params("2025-02-01T00:00:00Z", "2025-03-01T00:00:00Z"),
    )
    warmup_only = cache.get_or_build(
        _tiny_v2_config(),
        settings=settings,
        base_params=_window_params(
            "2025-02-01T00:00:00Z", "2025-03-01T00:00:00Z", warmupBars=5000
        ),
    )
    non_date_change = cache.get_or_build(
        _tiny_v2_config(),
        settings=settings,
        base_params=_window_params(
            "2025-03-01T00:00:00Z",
            "2025-04-01T00:00:00Z",
            stopX=3.0,
        ),
    )

    assert date_only.hit is True
    assert warmup_only.hit is True
    assert _PLAN_REUSE_RUNTIME_PARAM_NAMES == {"dateFilter", "start", "end"}
    assert (
        warmup_only.plan.candidate_table.semantic_keys_by_row
        == date_only.plan.candidate_table.semantic_keys_by_row
    )
    assert non_date_change.hit is False
    assert non_date_change.stats.build_count == 2
    assert non_date_change.stats.hit_count == 2
    assert non_date_change.stats.miss_count == 2


def test_old_schema_cache_entry_misses_safely():
    cache = GridV2PlanReuseCache()
    config = _tiny_v2_config()
    settings = GridV2Settings(enabled_axes=("threshold",))
    first = cache.get_or_build(config, settings=settings)
    assert first.cache_key is not None
    cache._entries[first.cache_key] = _GridV2CachedPlan(
        plan=first.plan,
        identity_signature="grid_v2_plan_identity_v2",
    )

    second = cache.get_or_build(config, settings=settings)

    assert second.hit is False
    assert second.stats.build_count == 2
    assert second.plan.metadata["planning"]["plan_identity_schema_version"] == (
        "grid_v2_plan_identity_v3"
    )


def test_rebased_plan_matches_fresh_window_plan_for_params_and_packed_dates():
    cache = GridV2PlanReuseCache()
    settings = GridV2Settings(enabled_axes=("threshold",))
    config = _tiny_v2_config()
    cache.get_or_build(
        config,
        settings=settings,
        base_params=_window_params("2025-01-01T00:00:00Z", "2025-02-01T00:00:00Z"),
    )
    second_params = _window_params(
        "2025-02-01T00:00:00Z",
        "2025-03-01T00:00:00Z",
    )
    rebased = cache.get_or_build(config, settings=settings, base_params=second_params).plan
    fresh = build_grid_v2_plan(config, settings=settings, base_params=second_params)
    indices = (0, rebased.deduped_candidate_count - 1)

    assert rebased.deduped_candidate_count == fresh.deduped_candidate_count
    assert rebased.per_variant_counts == fresh.per_variant_counts
    for index in indices:
        assert rebased.candidate_table.params_for_index(index) == fresh.candidate_table.params_for_index(index)
        assert rebased.candidate_table.semantic_key_for_index(index) == fresh.candidate_table.semantic_key_for_index(index)
        assert rebased.candidate_table.canonical_identity_for_index(index) == fresh.candidate_table.canonical_identity_for_index(index)

    rebased_arrays = _pack_table_config_arrays(rebased, indices, trade_start_idx=0)
    fresh_arrays = _pack_table_config_arrays(fresh, indices, trade_start_idx=0)
    assert rebased_arrays["use_date_filter"].tolist() == fresh_arrays["use_date_filter"].tolist()
    assert rebased_arrays["start_ns"].tolist() == fresh_arrays["start_ns"].tolist()
    assert rebased_arrays["end_ns"].tolist() == fresh_arrays["end_ns"].tolist()
