from __future__ import annotations

import hashlib

import numpy as np
import pytest

from core.grid_engine import normalize_grid_v2_planning_policy
from core.grid_v2 import (
    GridV2PlanReuseCache,
    GridV2Settings,
    _can_use_table_config_packer,
    build_grid_v2_plan,
    preview_grid_v2_counts,
)
from core.grid_v2_sampling import (
    GRID_V2_SAMPLER_VERSION,
    derive_named_seed,
    sample_block_codes,
    stable_fisher_yates,
    stable_randbelow,
)
from strategies.s06_r_trend_v02_b2.strategy import load_config


def _small_config() -> dict:
    return {
        "id": "sampled_fixture",
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
            "signalA": {
                "type": "int",
                "default": 0,
                "role": "signal",
                "optimize": {"enabled": True, "min": 0, "max": 9, "step": 1},
            },
            "signalB": {
                "type": "select",
                "default": "x",
                "options": ["x", "y", "z"],
                "role": "signal",
                "optimize": {"enabled": True},
            },
            "dateFilter": {"type": "bool", "default": False, "role": "runtime"},
            "start": {"type": "select", "default": "", "options": [""], "role": "runtime"},
            "end": {"type": "select", "default": "", "options": [""], "role": "runtime"},
            "stopX": {"type": "float", "default": 2.0, "role": "execution"},
            "riskPerTrade": {"type": "float", "default": 2.0, "role": "execution"},
            "contractSize": {"type": "float", "default": 0.01, "role": "execution"},
            "stopMaxDays": {"type": "int", "default": 4, "role": "execution"},
        },
    }


def _sampled_settings(*, budget: int = 11, seed: int = 42, workers: int = 1) -> GridV2Settings:
    return GridV2Settings(
        planning_policy="sampled",
        requested_budget=budget,
        seed=seed,
        allocation_method="proportional_space",
        compiled_workers=workers,
    )


def _semantic_digest(plan) -> str:
    digest = hashlib.sha256()
    for key in plan.candidate_table.semantic_keys_by_row or ():
        digest.update((key + "\n").encode("utf-8"))
    return digest.hexdigest()


def test_planning_policy_normalizer_accepts_documented_aliases_and_rejects_unknown():
    assert normalize_grid_v2_planning_policy(None) == "full"
    assert normalize_grid_v2_planning_policy("FULL-ENUMERATION-V2") == "full"
    assert normalize_grid_v2_planning_policy("budgeted") == "sampled"
    with pytest.raises(ValueError, match="full.*sampled"):
        normalize_grid_v2_planning_policy("random")


def test_versioned_raw_pcg64_primitives_have_pinned_golden_outputs():
    assert derive_named_seed(42, {"strategy_id": "s", "block_id": "b", "purpose": "x"}) == (
        58_752_243_257_758_174_588_505_173_281_051_738_598
    )
    assert stable_fisher_yates(10, np.random.PCG64(123)) == (8, 4, 9, 5, 7, 0, 3, 6, 2, 1)
    assert stable_randbelow(np.random.PCG64(42), 17) == 12


def test_balanced_discrete_lhs_is_sorted_balanced_and_topup_is_bounded():
    codes, diagnostics = sample_block_codes(
        (3, 4, 5),
        11,
        global_seed=42,
        strategy_id="s",
        strategy_version="v",
        block_id="b",
        axis_names=("a", "b", "c"),
    )
    assert codes == tuple(sorted(codes, key=lambda row: row[0] * 20 + row[1] * 5 + row[2]))
    assert len(codes) == len(set(codes)) == 11
    assert all(axis.maximum_level_frequency - axis.minimum_level_frequency <= 1 for axis in diagnostics.axis_diagnostics)

    topped_up, topup = sample_block_codes(
        (2, 2),
        3,
        global_seed=2,
        strategy_id="s",
        strategy_version="v",
        block_id="b",
        axis_names=("a", "b"),
    )
    assert len(topped_up) == 3
    assert topup.collision_count == topup.topup_count == 1
    assert topup.topup_attempt_count <= topup.requested_count
    assert topup.shortfall_count == 0


def test_sampled_plan_is_exact_deterministic_worker_invariant_and_seed_sensitive():
    config = _small_config()
    first = build_grid_v2_plan(config, _sampled_settings())
    repeated = build_grid_v2_plan(config, _sampled_settings(workers=7))
    changed_seed = build_grid_v2_plan(config, _sampled_settings(seed=43))

    assert first.full_raw_candidate_count == 30
    assert first.planned_candidate_count == first.deduped_candidate_count == 11
    assert first.candidate_table.raw_candidate_count == 30
    assert first.candidate_table.enumerated_candidate_count == 11
    assert first.per_variant_counts == {"default": 11}
    assert first.plan_fingerprint == repeated.plan_fingerprint
    assert first.candidate_table.semantic_keys_by_row == repeated.candidate_table.semantic_keys_by_row
    assert first.candidate_table.semantic_keys_by_row != changed_seed.candidate_table.semantic_keys_by_row
    assert [first.candidate_for_index(index).candidate_id for index in range(11)] == list(range(1, 12))
    assert first.per_block_counts["default"]["generation_mode"] == GRID_V2_SAMPLER_VERSION


def test_sampled_semantic_keys_are_unchanged_and_saturated_budget_uses_exact_full_path():
    config = _small_config()
    full = build_grid_v2_plan(config)
    sampled = build_grid_v2_plan(config, _sampled_settings())
    saturated = build_grid_v2_plan(config, _sampled_settings(budget=30, seed=999))

    assert set(sampled.candidate_table.semantic_keys_by_row or ()).issubset(
        set(full.candidate_table.semantic_keys_by_row or ())
    )
    assert saturated.candidate_table.semantic_keys_by_row == full.candidate_table.semantic_keys_by_row
    assert saturated.candidate_table.axis_value_codes.tolist() == full.candidate_table.axis_value_codes.tolist()
    assert saturated.metadata["planning"]["effective_policy"] == "full"
    assert saturated.metadata["planning"]["effective_policy_reason"] == "budget_covers_full_space"
    assert saturated.plan_fingerprint == full.plan_fingerprint


def test_preview_reports_full_requested_planned_and_coverage_without_building_population():
    preview = preview_grid_v2_counts(_small_config(), _sampled_settings())
    assert preview.full_raw_candidate_count == preview.full_valid_candidate_count == 30
    assert preview.requested_budget == preview.planned_candidate_count == 11
    assert preview.effective_planning_policy == "sampled"
    assert preview.coverage_pct == pytest.approx(11 / 30 * 100.0)
    assert preview.per_block_counts["default"]["planned_count"] == 11


def test_sampled_planning_scales_with_delivered_population_not_full_space():
    config = _small_config()
    for name in ("signalA", "signalC", "signalD", "signalE"):
        config["parameters"][name] = {
            "type": "int",
            "default": 0,
            "role": "signal",
            "optimize": {"enabled": True, "min": 0, "max": 99, "step": 1},
        }
    # signalB contributes another factor of three: N = 300,000,000.
    settings = _sampled_settings(budget=50_000)
    preview = preview_grid_v2_counts(config, settings)
    plan = build_grid_v2_plan(config, settings)
    diagnostics = plan.metadata["planning"]["sampling_diagnostics"]["default"]

    assert preview.full_raw_candidate_count == 300_000_000
    assert preview.planned_candidate_count == 50_000
    assert len(plan.candidate_table) == 50_000
    assert plan.candidate_table.axis_value_codes.shape == (50_000, 5)
    assert len(plan.candidate_table.semantic_keys_by_row or ()) == 50_000
    assert diagnostics["primary_unique_count"] <= 50_000
    assert diagnostics["primary_unique_count"] + diagnostics["collision_count"] == 50_000
    assert diagnostics["topup_attempt_count"] <= 50_000
    assert diagnostics["delivered_count"] == 50_000


def test_full_s06_population_order_and_semantic_digest_are_byte_compatible():
    plan = build_grid_v2_plan(load_config())
    assert plan.deduped_candidate_count == 48_480
    assert plan.per_variant_counts == {"bracket": 480, "trail": 48_000}
    assert _semantic_digest(plan) == "f1aa8bba4099d07f4d0d2865a72bf6537767329f3fdd9b324f07986b8b8369e4"


def test_sampled_plan_reuse_rebases_runtime_and_invalidates_seed_and_budget():
    config = _small_config()
    cache = GridV2PlanReuseCache()
    first = cache.get_or_build(
        config,
        settings=_sampled_settings(),
        base_params={"dateFilter": True, "start": "2025-01-01Z", "end": "2025-02-01Z"},
    )
    runtime = cache.get_or_build(
        config,
        settings=_sampled_settings(),
        base_params={"dateFilter": True, "start": "2025-02-01Z", "end": "2025-03-01Z"},
    )
    seed = cache.get_or_build(config, settings=_sampled_settings(seed=43))
    budget = cache.get_or_build(config, settings=_sampled_settings(budget=12))
    assert first.hit is False
    assert runtime.hit is True
    assert runtime.plan.plan_fingerprint == first.plan.plan_fingerprint
    assert runtime.plan.candidate_table.axis_value_codes is first.plan.candidate_table.axis_value_codes
    assert (
        runtime.plan.metadata["planning"]["sampling_diagnostics"]
        == first.plan.metadata["planning"]["sampling_diagnostics"]
    )
    assert seed.hit is False
    assert budget.hit is False


def test_full_plan_reuse_ignores_nonoperative_seed_and_sampled_table_packer_is_disabled():
    config = _small_config()
    cache = GridV2PlanReuseCache()
    first = cache.get_or_build(config, settings=GridV2Settings(seed=1))
    second = cache.get_or_build(config, settings=GridV2Settings(seed=999))
    sampled = build_grid_v2_plan(config, _sampled_settings())

    class Hooks:
        normalize_params = None

    assert first.hit is False
    assert second.hit is True
    assert _can_use_table_config_packer(sampled, Hooks(), tuple(range(11))) is False


def test_sampled_safety_rejects_inactive_axes_dependency_parent_and_invalid_seed():
    config = _small_config()
    assert build_grid_v2_plan(config, _sampled_settings(seed=(1 << 64) - 1)).planned_candidate_count == 11
    with pytest.raises(ValueError, match="include_inactive"):
        build_grid_v2_plan(
            config,
            GridV2Settings(
                planning_policy="sampled",
                requested_budget=5,
                include_inactive_axes_for_dedup=True,
            ),
        )
    dependent = _small_config()
    dependent["parameters"]["switch"] = {
        "type": "bool",
        "default": True,
        "role": "signal",
        "optimize": {"enabled": True},
    }
    dependent["parameters"]["child"] = {
        "type": "int",
        "default": 1,
        "role": "signal",
        "depends_on": "switch",
        "optimize": {"enabled": False},
    }
    with pytest.raises(ValueError, match="switch.*child"):
        build_grid_v2_plan(dependent, _sampled_settings())
    with pytest.raises(ValueError, match="seed"):
        build_grid_v2_plan(config, _sampled_settings(seed=-1))
    with pytest.raises(ValueError, match="seed"):
        build_grid_v2_plan(config, _sampled_settings(seed=1 << 64))
