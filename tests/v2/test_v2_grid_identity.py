from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from core.grid_v2 import GRID_V2_ENGINE_VERSION, GridV2Settings, build_grid_v2_plan
from core.optuna_engine import OptimizationConfig

from strategies import get_strategy_config
from strategies.s06_r_trend_v02_b2.strategy import load_config


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "raw" / "OKX_SUIUSDT.P, 30 2025.01.01-2026.02.01.csv"
TRADING_START = "2025-08-01T00:00:00+00:00"
TRADING_END = "2025-12-01T00:00:00+00:00"
GRID_PARAMS = (
    "stopX",
    "stopRR",
    "stopLP",
    "stopMaxPct",
    "stopMaxDays",
    "trailRR",
    "trailMAType",
    "trailMALength",
    "trailMAOffsetEx",
)


def _fast_grid():
    from strategies.s06_r_trend_v02 import fast_grid

    return fast_grid


def _v1_config() -> OptimizationConfig:
    return OptimizationConfig(
        csv_file=str(DATA_PATH),
        strategy_id="s06_r_trend_v02",
        enabled_params={name: True for name in GRID_PARAMS} | {"thresholdOS": False, "thresholdOB": False},
        param_ranges={},
        param_types={
            "thresholdOS": "int",
            "thresholdOB": "int",
            "stopX": "float",
            "stopRR": "float",
            "stopLP": "int",
            "stopMaxPct": "float",
            "stopMaxDays": "int",
            "trailRR": "float",
            "trailMAType": "select",
            "trailMALength": "int",
            "trailMAOffsetEx": "float",
        },
        fixed_params={
            "dateFilter": True,
            "start": TRADING_START,
            "end": TRADING_END,
            "entryMode": "Reversal @ Triangle",
            "enableLong": True,
            "enableShort": True,
            "fastLength": 21,
            "fastSmoothing": 7,
            "slowLength": 112,
            "slowSmoothing": 3,
            "thresholdOS": 20,
            "thresholdOB": 20,
            "stopX": 2.0,
            "stopRR": 3.0,
            "stopLP": 2,
            "stopMaxPct": 6.0,
            "stopMaxDays": 6,
            "riskPerTrade": 2.0,
            "contractSize": 0.01,
            "useTrailMA": True,
            "trailRR": 1.0,
            "trailMAType": "SMA",
            "trailMALength": 150,
            "trailMAOffsetEx": 0.0,
            "initialCapital": 100.0,
            "commissionPct": 0.05,
        },
        warmup_bars=1000,
        optimization_mode="grid",
        objectives=["net_profit_pct"],
        grid_enabled_modes=["bracket", "trail"],
        grid_budget=1,
    )


def _v2_base_params() -> dict:
    fixed = dict(_v1_config().fixed_params)
    fixed["fastSmooth"] = fixed.pop("fastSmoothing")
    fixed["slowSmooth"] = fixed.pop("slowSmoothing")
    return fixed


def _v1_candidates():
    fast_grid = _fast_grid()
    config = _v1_config()
    space = fast_grid.build_parameter_space(config)
    allocation = fast_grid.build_allocation(config, space, None)
    return fast_grid.generate_candidates(config, space, allocation, seed=123).candidates


def _canonical_from_v1(candidate) -> tuple:
    fast_grid = _fast_grid()
    return (
        candidate.mode,
        tuple((name, candidate.params[name]) for name in fast_grid.MODE_AXES[candidate.mode]),
    )


def _canonical_from_v2(candidate) -> tuple:
    fast_grid = _fast_grid()
    return (
        candidate.variant_name,
        tuple((name, candidate.params[name]) for name in fast_grid.MODE_AXES[candidate.variant_name]),
    )


@pytest.mark.slow
def test_full_s06_default_identity_space_maps_one_to_one_with_v1_fast_grid():
    v1 = _v1_candidates()
    v2 = build_grid_v2_plan(load_config(), base_params=_v2_base_params())

    assert len(v1) == 48_480
    assert v2.deduped_candidate_count == 48_480
    assert v2.per_variant_counts == {"bracket": 480, "trail": 48_000}

    v1_keys = [_canonical_from_v1(candidate) for candidate in v1]
    v2_keys = [_canonical_from_v2(candidate) for candidate in v2.candidates]
    assert v2_keys == v1_keys
    assert len(set(v2_keys)) == 48_480

    sample_indices = [0, 1, 479, 480, 20_000, 48_479]
    assert [v2.candidates[index].candidate_id for index in sample_indices] == [
        index + 1 for index in sample_indices
    ]


@pytest.mark.slow
def test_v2_semantic_keys_exclude_runtime_and_inactive_variant_params():
    plan = build_grid_v2_plan(load_config(), base_params=_v2_base_params())

    bracket_payload = json.loads(plan.candidates[0].semantic_key)
    assert "dateFilter" not in bracket_payload["params"]
    assert "start" not in bracket_payload["params"]
    assert "end" not in bracket_payload["params"]
    assert "trailMAType" not in bracket_payload["params"]
    assert "trailRR" not in bracket_payload["params"]
    assert "stopRR" in bracket_payload["params"]

    trail_payload = json.loads(plan.candidates[480].semantic_key)
    assert "stopRR" not in trail_payload["params"]
    assert "trailMAType" in trail_payload["params"]
    assert "trailRR" in trail_payload["params"]


def test_candidate_table_lazily_decodes_identity_subset_without_full_legacy_tuple():
    plan = build_grid_v2_plan(load_config(), base_params=_v2_base_params())
    table = plan.candidate_table

    assert plan._candidates_cache is None
    assert table.legacy_candidates_materialized_count == 0
    assert table.params_by_row is None
    assert table.params_materialized_count == 0
    assert table.semantic_keys_materialized_count == plan.deduped_candidate_count
    assert table.canonical_identities_materialized_count == 0

    subset = (0, 479, 480, 18_435, 48_479)
    for index in subset:
        candidate = plan.candidate_for_index(index)
        assert candidate.candidate_id == index + 1
        assert dict(table.params_for_index(index)) == dict(candidate.params)
        assert table.active_names_for_index(index) == candidate.active_param_names
        assert table.inactive_names_for_index(index) == candidate.inactive_param_names
        assert table.axis_names_for_index(index) == candidate.axis_param_names
        assert table.semantic_payload_for_index(index) == candidate.semantic_payload
        assert table.semantic_key_for_index(index) == candidate.semantic_key
        assert table.canonical_identity_for_index(index) == candidate.canonical_identity

    assert table.legacy_candidates_materialized_count == len(subset)
    assert table.params_materialized_count == len(subset)
    assert table.semantic_keys_materialized_count == plan.deduped_candidate_count
    assert table.canonical_identities_materialized_count == len(subset)
    assert plan._candidates_cache is None


def test_candidate_table_lazy_params_match_legacy_values_for_deterministic_rows():
    plan = build_grid_v2_plan(load_config(), base_params=_v2_base_params())
    table = plan.candidate_table
    assert table.params_by_row is None

    for candidate_id in (1, 480, 481, 18_436, 48_480):
        index = candidate_id - 1
        decoded = dict(table.params_for_index(index))
        candidate = plan.candidate_for_index(index)
        assert decoded == dict(candidate.params)
        assert candidate.candidate_id == candidate_id

    assert table.params_materialized_count == 5


@pytest.mark.slow
def test_candidate_table_compatibility_candidates_materialize_on_explicit_access():
    plan = build_grid_v2_plan(load_config(), base_params=_v2_base_params())

    candidates = plan.candidates

    assert len(candidates) == 48_480
    assert plan._candidates_cache is candidates
    assert plan.candidate_table.legacy_candidates_materialized_count == 48_480
    assert candidates[0].candidate_id == 1
    assert candidates[-1].candidate_id == 48_480
    assert json.loads(candidates[0].semantic_key)["engine"] == GRID_V2_ENGINE_VERSION


@pytest.mark.slow
def test_select_subset_helper_does_not_affect_identity_or_candidate_order():
    base_params = _v2_base_params()
    first = build_grid_v2_plan(
        load_config(),
        base_params={**base_params, "trailMAType_options": ["SMA", "HMA"]},
    )
    reordered = build_grid_v2_plan(
        load_config(),
        base_params={**base_params, "trailMAType_options": ["HMA", "SMA"]},
    )

    assert first.deduped_candidate_count == 24_480
    assert first.per_variant_counts == {"bracket": 480, "trail": 24_000}
    assert first.parameter_domains["trailMAType"].values == ("SMA", "HMA")
    assert reordered.parameter_domains["trailMAType"].values == ("SMA", "HMA")
    assert [candidate.canonical_identity for candidate in first.candidates] == [
        candidate.canonical_identity for candidate in reordered.candidates
    ]
    assert [candidate.semantic_key for candidate in first.candidates] == [
        candidate.semantic_key for candidate in reordered.candidates
    ]
    assert len({candidate.semantic_key for candidate in first.candidates}) == first.deduped_candidate_count
    assert all(
        not str(key).endswith("_options")
        for candidate in first.candidates
        for key in candidate.params
    )
    assert all(
        "trailMAType_options" not in json.loads(candidate.semantic_key)["params"]
        for candidate in first.candidates
    )


def _collapse_config():
    config = {
        "id": "collapse_fixture",
        "version": "test",
        "engine": "v2",
        "execution": {
            "entryOrder": "market_next_open",
            "stop": "atr_swing",
            "sizing": "risk_per_trade",
            "maxDays": True,
            "margin": "off",
            "boundary": "strict_close",
            "priceRounding": "none",
            "variantSelector": {
                "param": "selector",
                "mapping": {"false": "with_target", "true": "without_target"},
            },
            "variants": {
                "with_target": {"target": "rr", "trail": "none", "trailActivation": "none"},
                "without_target": {"target": "none", "trail": "ma", "trailActivation": "rr"},
            },
        },
        "parameters": {
            "selector": {"type": "bool", "default": False, "role": "execution", "optimize": {"enabled": False}},
            "signal": {"type": "int", "default": 1, "role": "signal", "optimize": {"enabled": False}},
            "stopRR": {
                "type": "float",
                "default": 1.0,
                "role": "execution",
                "optimize": {"enabled": True, "gridValues": [1.0, 2.0]},
            },
            "stopX": {"type": "float", "default": 2.0, "role": "execution", "optimize": {"enabled": False}},
            "stopLP": {"type": "int", "default": 2, "role": "execution", "optimize": {"enabled": False}},
            "stopMaxPct": {"type": "float", "default": 10.0, "role": "execution", "optimize": {"enabled": False}},
            "trailRR": {"type": "float", "default": 1.0, "role": "execution", "optimize": {"enabled": False}},
            "trailMAType": {"type": "select", "default": "SMA", "role": "execution", "optimize": {"enabled": False}},
            "trailMALength": {"type": "int", "default": 150, "role": "execution", "optimize": {"enabled": False}},
            "trailMAOffsetEx": {"type": "float", "default": 0.0, "role": "execution", "optimize": {"enabled": False}},
            "riskPerTrade": {"type": "float", "default": 2.0, "role": "execution", "optimize": {"enabled": False}},
            "contractSize": {"type": "float", "default": 0.01, "role": "execution", "optimize": {"enabled": False}},
            "stopMaxDays": {"type": "int", "default": 4, "role": "execution", "optimize": {"enabled": False}},
        },
    }
    return config


def test_inactive_axis_dedup_collapses_to_first_deterministic_candidate():
    plan = build_grid_v2_plan(
        _collapse_config(),
        GridV2Settings(include_inactive_axes_for_dedup=True),
    )

    assert plan.raw_candidate_count == 4
    assert plan.enumerated_candidate_count == 4
    assert plan.deduped_candidate_count == 3
    assert plan.per_variant_counts == {"with_target": 2, "without_target": 1}
    assert [candidate.candidate_id for candidate in plan.candidates] == [1, 2, 3]
    assert plan.candidates[-1].params["stopRR"] == 1.0


@pytest.mark.slow
def test_tz64a_request_runtime_row_digests_and_identity_pins():
    from core.grid_v2 import build_grid_v2_plan
    from ui import server_services

    strategy_id = "s06_r_trend_v02_b2"
    config = get_strategy_config(strategy_id)
    context = server_services._resolve_strategy_context(
        [("strategy_id", True, strategy_id)]
    )

    def ordered_row_digest(plan):
        digest = hashlib.sha256()
        for index in range(len(plan.candidate_table)):
            candidate = plan.candidate_for_index(index)
            row = {
                "candidate_id": candidate.candidate_id,
                "variant_name": candidate.variant_name,
                "grid_mode_name": candidate.grid_mode_name,
                "params": dict(candidate.params),
            }
            digest.update(
                (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode(
                    "utf-8"
                )
            )
        return digest.hexdigest()

    def semantic_digest(plan):
        digest = hashlib.sha256()
        for key in plan.candidate_table.semantic_keys_by_row or ():
            digest.update((key + "\n").encode("utf-8"))
        return digest.hexdigest()

    blank_payload, blank_runtime = server_services._normalize_v2_optimizer_payload(
        context,
        {
            "optimization_mode": "grid",
            "fixed_params": {"dateFilter": False, "start": "", "end": ""},
        },
        warmup_members=[],
    )
    dated_payload, dated_runtime = server_services._normalize_v2_optimizer_payload(
        context,
        {
            "optimization_mode": "grid",
            "fixed_params": {
                "dateFilter": True,
                "start": "2025-05-01T00:00",
                "end": "2025-11-20T00:00",
            }
        },
        warmup_members=[],
    )
    alias_payload, _alias_runtime = server_services._normalize_v2_optimizer_payload(
        context,
        {"optimization_mode": "grid", "fixed_params": {"dateFilter": "0"}},
        warmup_members=[],
    )
    date_only_payload, _date_only_runtime = (
        server_services._normalize_v2_optimizer_payload(
            context,
            {
                "optimization_mode": "grid",
                "fixed_params": {
                    "dateFilter": True,
                    "start": "2025-06-01",
                    "end": "2025-06-30",
                }
            },
            warmup_members=[],
        )
    )
    assert blank_payload["fixed_params"] == {
        "dateFilter": False,
        "start": None,
        "end": None,
    }
    assert dated_payload["fixed_params"] == {
        "dateFilter": True,
        "start": "2025-05-01T00:00:00Z",
        "end": "2025-11-20T00:00:00Z",
    }
    assert alias_payload["fixed_params"] == {"dateFilter": False}
    assert date_only_payload["fixed_params"] == {
        "dateFilter": True,
        "start": "2025-06-01T00:00:00Z",
        "end": "2025-06-30T23:59:59.999999Z",
    }
    assert blank_runtime.values["warmupBars"] == dated_runtime.values["warmupBars"] == 1000

    cases = [
        (None, "f6a6258a07f21102ae60a91a239b960a23a5c59f15790423ab3a3874c00bfa6f"),
        ({"dateFilter": False}, "e0ba87a4dbf66a843d462d0f73d8b1c991841b247acbb6051d999bd5767b5f7c"),
        (
            {"dateFilter": False, "start": "", "end": ""},
            "8c2ca227ef140b31700f358d16a84246f34d7edb0f46b63f139bef003a8a25a5",
        ),
        (
            blank_payload["fixed_params"],
            "c5645dfc07084855c4618053b12c58b55859637425fedafe849dd9f751f01ff6",
        ),
        (
            {
                "dateFilter": True,
                "start": "2025-05-01T00:00",
                "end": "2025-11-20T00:00",
            },
            "73d4665bffa3a4df39d84e71cce5778d50cae0cb43ecb1f6f340010d20c68784",
        ),
        (
            dated_payload["fixed_params"],
            "54a7709b31efc592bf6d329d45b7a0dd2bab38a78d6a7bdf1673d89f1829e19d",
        ),
        (
            {
                "dateFilter": True,
                "start": "2025-06-01",
                "end": "2025-06-30",
            },
            "b1900647b0afefb3f8f11b5150426ef04c4dc8164221f51eb260e19f1c46339c",
        ),
        (
            date_only_payload["fixed_params"],
            "9faf81224522a95727fb39961e3a7ea9da010c3dc4655e99ce84be11edd24e68",
        ),
    ]
    for base_params, expected_row_digest in cases:
        plan = build_grid_v2_plan(config, base_params=base_params)
        assert len(plan.candidate_table) == 48_480
        assert plan.per_variant_counts == {"bracket": 480, "trail": 48_000}
        assert ordered_row_digest(plan) == expected_row_digest
        assert semantic_digest(plan) == (
            "fc55d174e835e7196ae5fcf21427d318dc364241f6b10560aa32545e6910a08f"
        )
        assert plan.plan_fingerprint == (
            "0f8d001c380df5ee95d34ca4e25910c674e20e9e8f34886a1bd2f1c261f019b2"
        )

    from core.param_identity import create_display_param_id

    trading_params = {"maType": "EMA", "maLength": 50}
    raw_display_id = create_display_param_id(
        trading_params,
        fixed_params={
            "dateFilter": True,
            "start": "2025-06-01",
            "end": "2025-06-30",
            "warmupBars": 1000,
        },
    )
    canonical_display_id = create_display_param_id(
        trading_params,
        fixed_params={
            **date_only_payload["fixed_params"],
            "warmupBars": 5000,
        },
    )
    assert raw_display_id == canonical_display_id
