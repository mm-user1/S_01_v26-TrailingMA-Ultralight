from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from core.engine_v2.profile import ProfileValidationError
from core.engine_v2.runtime_contract import V2_RUNTIME_CONTRACT_VERSION
from core.grid_engine import (
    allocate_ordered_block_budgets,
    normalize_grid_v2_planning_policy,
    rank_grid_results,
)
from core.grid_v2 import (
    GRID_V2_PLAN_IDENTITY_SCHEMA_VERSION,
    GRID_V2_SEMANTIC_IDENTITY_SCHEMA_VERSION,
    GridV2PlanReuseCache,
    GridV2Settings,
    _can_use_table_config_packer,
    _grid_v2_plan_fingerprint,
    _reconcile_grid_v2_automatic_allocation,
    _stable_json,
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
from core.optuna_engine import OptimizationResult
from strategies import get_strategy_config
from strategies.s06_r_trend_v02_b2.strategy import load_config
from runtime_test_helpers import canonical_v2_runtime_declarations


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
            **canonical_v2_runtime_declarations(),
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
            "stopX": {"type": "float", "default": 2.0, "role": "execution"},
            "stopLP": {"type": "int", "default": 2, "role": "execution"},
            "stopMaxPct": {"type": "float", "default": 6.0, "role": "execution"},
            "stopRR": {"type": "float", "default": 2.0, "role": "execution"},
            "riskPerTrade": {"type": "float", "default": 2.0, "role": "execution"},
            "contractSize": {"type": "float", "default": 0.01, "role": "execution"},
            "tickSize": {"type": "float", "default": 0.01, "role": "execution"},
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


def _ordered_row_digest(plan) -> str:
    digest = hashlib.sha256()
    for index in range(len(plan.candidate_table)):
        candidate = plan.candidate_for_index(index)
        payload = {
            "candidate_id": candidate.candidate_id,
            "variant_name": candidate.variant_name,
            "grid_mode_name": candidate.grid_mode_name,
            "params": dict(candidate.params),
        }
        digest.update(
            (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def test_planning_policy_normalizer_accepts_documented_aliases_and_rejects_unknown():
    assert normalize_grid_v2_planning_policy(None) == "full"
    assert normalize_grid_v2_planning_policy("FULL-ENUMERATION-V2") == "full"
    assert normalize_grid_v2_planning_policy("budgeted") == "sampled"
    with pytest.raises(ValueError, match="full.*sampled"):
        normalize_grid_v2_planning_policy("random")


def test_grid_v2_direct_defaults_match_public_contract():
    settings = GridV2Settings()
    assert settings.requested_budget == 200_000
    assert settings.allocation_method == "auto_sqrt_space"


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
    assert first.metadata["diagnostics"] == repeated.metadata["diagnostics"]
    assert first.metadata["validation_warnings"] == repeated.metadata[
        "validation_warnings"
    ]
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
    for plan in (full, saturated):
        planning = plan.metadata["planning"]
        assert planning["effective_allocation_method"] == "full_enumeration_v2"
        assert planning["effective_seed"] is None
        assert planning["effective_min_quota"] is None
        assert planning["effective_manual_percents"] == {}
        assert planning["plan_identity_schema_version"] == GRID_V2_PLAN_IDENTITY_SCHEMA_VERSION
        assert planning["semantic_identity_schema_version"] == GRID_V2_SEMANTIC_IDENTITY_SCHEMA_VERSION
        assert planning["runtime_contract_version"] == V2_RUNTIME_CONTRACT_VERSION
    assert preview_grid_v2_counts(config).effective_allocation_method == "full_enumeration_v2"
    assert (
        preview_grid_v2_counts(config, _sampled_settings(budget=30, seed=999)).effective_allocation_method
        == "full_enumeration_v2"
    )


def test_preview_reports_full_requested_planned_and_coverage_without_building_population():
    preview = preview_grid_v2_counts(_small_config(), _sampled_settings())
    assert preview.full_raw_candidate_count == preview.full_valid_candidate_count == 30
    assert preview.requested_budget == preview.planned_candidate_count == 11
    assert preview.effective_planning_policy == "sampled"
    assert preview.coverage_pct == pytest.approx(11 / 30 * 100.0)
    assert preview.per_block_counts["default"]["planned_count"] == 11
    assert preview.effective_allocation_method == "proportional_space"
    assert preview.plan_identity_schema_version == "grid_v2_plan_identity_v3"
    assert preview.semantic_identity_schema_version == "grid_v2_semantic_identity_v2"
    assert preview.runtime_contract_version == "v2_runtime_contract_v1"


def test_runtime_values_are_not_domains_and_do_not_change_candidate_identity_rows():
    config = _small_config()
    settings = _sampled_settings()
    first = build_grid_v2_plan(
        config,
        settings,
        base_params={
            "dateFilter": True,
            "start": "2025-01-01T00:00:00Z",
            "end": "2025-02-01T00:00:00Z",
            "warmupBars": 100,
        },
    )
    second = build_grid_v2_plan(
        config,
        settings,
        base_params={
            "dateFilter": False,
            "start": "2025-03-01T00:00:00Z",
            "end": "2025-04-01T00:00:00Z",
            "warmupBars": 5000,
        },
    )

    assert set(first.parameter_domains).isdisjoint(
        {"dateFilter", "start", "end", "warmupBars"}
    )
    assert first.candidate_table.axis_value_codes.tolist() == second.candidate_table.axis_value_codes.tolist()
    assert first.candidate_table.semantic_keys_by_row == second.candidate_table.semantic_keys_by_row
    assert first.plan_fingerprint == second.plan_fingerprint
    for key in first.candidate_table.semantic_keys_by_row or ():
        assert set(json.loads(key)["params"]).isdisjoint(
            {"dateFilter", "start", "end", "warmupBars"}
        )


@pytest.mark.parametrize(
    ("strategy_id", "settings", "base_params", "inactive_name"),
    [
        (
            "s06_r_trend_v02_b2",
            GridV2Settings(enabled_variants=("trail",), enabled_axes=("stopRR",)),
            {},
            "stopRR",
        ),
        (
            "s03_reversal_v11_regime_er_b2",
            GridV2Settings(enabled_axes=("emergencySlPct",)),
            {"useEmergencySL": False},
            "emergencySlPct",
        ),
    ],
)
def test_bound_selected_run_inactive_axes_warn_without_multiplying_candidates(
    strategy_id, settings, base_params, inactive_name
):
    config = get_strategy_config(strategy_id)
    preview = preview_grid_v2_counts(config, settings, base_params)
    plan = build_grid_v2_plan(config, settings, base_params)
    warnings = [
        diagnostic
        for diagnostic in preview.diagnostics
        if diagnostic["code"] == "V2_INACTIVE_ENABLED_AXIS"
    ]

    assert preview.planned_candidate_count == plan.planned_candidate_count == 1
    assert [item["path"] for item in warnings] == [f"parameters.{inactive_name}"]
    assert preview.diagnostics == plan.metadata["diagnostics"]
    assert preview.validation_warnings == plan.metadata["validation_warnings"]
    assert "diagnostics" not in plan.metadata["planning"]
    assert "validation_warnings" not in plan.metadata["planning"]
    assert inactive_name not in json.loads(plan.candidate_table.semantic_key_for_index(0))["params"]


def test_s03_bool_group_axes_are_covered_by_logical_blocks_not_inactive():
    config = get_strategy_config("s03_reversal_v11_regime_er_b2")
    preview = preview_grid_v2_counts(config)

    assert list(preview.per_block_counts) == ["cc_only", "tbands_only", "both"]
    assert preview.per_variant_counts == {
        "cc_only": 71_280,
        "tbands_only": 198_000,
        "both": 7_128_000,
    }
    assert not any(
        item["code"] == "V2_INACTIVE_ENABLED_AXIS"
        and item["path"] in {"parameters.useCloseCount", "parameters.useTBands"}
        for item in preview.diagnostics
    )


def test_single_value_bool_group_axes_still_define_the_only_selected_block():
    config = copy.deepcopy(get_strategy_config("s03_reversal_v11_regime_er_b2"))
    config["parameters"]["useCloseCount"]["gridValues"] = [True]
    config["parameters"]["useTBands"]["gridValues"] = [False]

    preview = preview_grid_v2_counts(config)

    assert list(preview.per_block_counts) == ["cc_only"]
    assert not any(
        item["code"] == "V2_INACTIVE_ENABLED_AXIS"
        and item["path"] in {"parameters.useCloseCount", "parameters.useTBands"}
        for item in preview.diagnostics
    )


@pytest.mark.parametrize(
    ("mutation", "path", "fragment"),
    [
        (
            lambda config: config["optimization_rules"]["bool_groups"][0].update(
                params=["dateFilter", "useTBands"]
            ),
            "optimization_rules.bool_groups[0].params",
            "reserved/runtime",
        ),
        (
            lambda config: config["optimization_rules"]["bool_groups"][0].update(
                params=["missing", "useTBands"]
            ),
            "optimization_rules.bool_groups[0].params",
            "not declared",
        ),
        (
            lambda config: config["optimization_rules"]["bool_groups"][0].update(
                params=["useTBands", "useTBands"]
            ),
            "optimization_rules.bool_groups[0].params",
            "unique",
        ),
        (
            lambda config: config["optimization_rules"]["bool_groups"][0].update(
                params=["", "useTBands"]
            ),
            "optimization_rules.bool_groups[0].params",
            "non-empty strings",
        ),
        (
            lambda config: config["optimization_rules"]["bool_groups"][0].update(
                params=["maLength3", "useTBands"]
            ),
            "optimization_rules.bool_groups[0].params",
            "type 'bool'",
        ),
        (
            lambda config: config["optimization_rules"]["bool_groups"][0].update(
                params="useCloseCount"
            ),
            "optimization_rules.bool_groups[0].params",
            "exactly two",
        ),
        (
            lambda config: config["optimization_rules"]["bool_groups"][0].update(
                mode="all_true"
            ),
            "optimization_rules.bool_groups[0].mode",
            "at_least_one_true",
        ),
        (
            lambda config: config["optimization_rules"]["bool_groups"].append(
                copy.deepcopy(config["optimization_rules"]["bool_groups"][0])
            ),
            "optimization_rules.bool_groups",
            "exactly one",
        ),
    ],
)
def test_invalid_bool_groups_fail_with_structured_diagnostics(
    mutation, path, fragment
):
    config = copy.deepcopy(get_strategy_config("s03_reversal_v11_regime_er_b2"))
    mutation(config)

    with pytest.raises(ProfileValidationError) as exc_info:
        preview_grid_v2_counts(config)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.severity == "error"
    assert diagnostic.code == "V2_INVALID_BOOL_GROUP"
    assert diagnostic.strategy_id == "s03_reversal_v11_regime_er_b2"
    assert diagnostic.path == path
    assert fragment in diagnostic.message


def test_absent_optimization_rules_is_valid_but_explicit_null_is_structured():
    absent = _small_config()
    absent.pop("optimization_rules", None)
    assert preview_grid_v2_counts(absent).planned_candidate_count == 30

    explicit_null = _small_config()
    explicit_null["optimization_rules"] = None
    with pytest.raises(ProfileValidationError) as exc_info:
        preview_grid_v2_counts(explicit_null)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "V2_INVALID_BOOL_GROUP"
    assert diagnostic.strategy_id == "sampled_fixture"
    assert diagnostic.path == "optimization_rules"
    assert "must be a mapping" in diagnostic.message


def test_absent_logical_modes_preserves_deterministic_fallback_names():
    config = copy.deepcopy(get_strategy_config("s03_reversal_v11_regime_er_b2"))
    del config["optimization_rules"]["bool_groups"][0]["logical_modes"]

    preview = preview_grid_v2_counts(config)

    assert list(preview.per_block_counts) == ["tbands_only", "cc_only", "both"]
    assert list(preview_grid_v2_counts(config).per_block_counts) == list(
        preview.per_block_counts
    )


@pytest.mark.parametrize(
    ("case", "path", "fragment"),
    [
        ("not_mapping", "optimization_rules.bool_groups[0].logical_modes", "must be a mapping"),
        ("empty_key", "optimization_rules.bool_groups[0].logical_modes", "keys must be non-empty"),
        ("entry_not_mapping", "optimization_rules.bool_groups[0].logical_modes.cc_only", "must be a mapping"),
        ("values_not_mapping", "optimization_rules.bool_groups[0].logical_modes.cc_only.values", "must be a mapping"),
        ("missing_name", "optimization_rules.bool_groups[0].logical_modes.cc_only.values", "missing=['useTBands']"),
        ("unknown_name", "optimization_rules.bool_groups[0].logical_modes.cc_only.values", "unknown=['unknown']"),
        ("invalid_bool", "optimization_rules.bool_groups[0].logical_modes.cc_only.values.useCloseCount", "not a valid boolean"),
        ("invalid_numeric_bool", "optimization_rules.bool_groups[0].logical_modes.cc_only.values.useCloseCount", "not a valid boolean"),
        ("all_false", "optimization_rules.bool_groups[0].logical_modes.cc_only.values", "all-false"),
        ("duplicate", "optimization_rules.bool_groups[0].logical_modes.tbands_only.values", "duplicate value combinations"),
        ("invalid_label", "optimization_rules.bool_groups[0].logical_modes.cc_only.label", "label must be a non-empty string"),
        ("partial", "optimization_rules.bool_groups[0].logical_modes", "define every supported"),
    ],
)
def test_malformed_logical_modes_are_structured(case, path, fragment):
    config = copy.deepcopy(get_strategy_config("s03_reversal_v11_regime_er_b2"))
    group = config["optimization_rules"]["bool_groups"][0]
    logical_modes = group["logical_modes"]
    if case == "not_mapping":
        group["logical_modes"] = None
    elif case == "empty_key":
        group["logical_modes"] = {"": logical_modes["cc_only"]}
    elif case == "entry_not_mapping":
        logical_modes["cc_only"] = []
    elif case == "values_not_mapping":
        logical_modes["cc_only"]["values"] = []
    elif case == "missing_name":
        del logical_modes["cc_only"]["values"]["useTBands"]
    elif case == "unknown_name":
        logical_modes["cc_only"]["values"]["unknown"] = True
    elif case == "invalid_bool":
        logical_modes["cc_only"]["values"]["useCloseCount"] = "maybe"
    elif case == "invalid_numeric_bool":
        logical_modes["cc_only"]["values"]["useCloseCount"] = 2
    elif case == "all_false":
        logical_modes["cc_only"]["values"] = {
            "useCloseCount": False,
            "useTBands": False,
        }
    elif case == "duplicate":
        logical_modes["tbands_only"]["values"] = copy.deepcopy(
            logical_modes["cc_only"]["values"]
        )
    elif case == "invalid_label":
        logical_modes["cc_only"]["label"] = ""
    elif case == "partial":
        del logical_modes["both"]

    with pytest.raises(ProfileValidationError) as exc_info:
        preview_grid_v2_counts(config)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.severity == "error"
    assert diagnostic.code == "V2_INVALID_BOOL_GROUP"
    assert diagnostic.strategy_id == "s03_reversal_v11_regime_er_b2"
    assert diagnostic.path == path
    assert fragment in diagnostic.message


@pytest.mark.parametrize(
    ("modes", "match"),
    [
        (("stale",), "unknown"),
        (("trail", " trail "), "duplicate normalized"),
        ((), "at least one"),
    ],
)
def test_user_facing_grid_modes_fail_closed_before_planning(modes, match):
    with pytest.raises(ValueError, match=match):
        build_grid_v2_plan(load_config(), GridV2Settings(enabled_variants=modes))


def test_reserved_runtime_fields_cannot_be_grid_axes_or_option_subsets():
    with pytest.raises(ValueError, match="runtime"):
        build_grid_v2_plan(
            _small_config(), GridV2Settings(enabled_axes=("warmupBars",))
        )
    with pytest.raises(ValueError, match="runtime.*option"):
        build_grid_v2_plan(
            _small_config(), base_params={"start_options": ["2025-01-01"]}
        )


def test_preview_and_plan_report_the_same_normalized_variant_and_block_order():
    settings = GridV2Settings(enabled_variants=("trail", "bracket"))
    preview = preview_grid_v2_counts(load_config(), settings)
    plan = build_grid_v2_plan(load_config(), settings)

    # Profile declaration order remains authoritative after validation.
    assert list(preview.per_variant_counts) == plan.metadata["grid_mode_order"] == [
        "bracket", "trail"
    ]
    assert preview.mode_labels == plan.metadata["grid_mode_labels"]
    assert preview.diagnostics == plan.metadata["diagnostics"]
    assert preview.validation_warnings == plan.metadata["validation_warnings"]
    assert "diagnostics" not in plan.metadata["planning"]
    assert "validation_warnings" not in plan.metadata["planning"]


def test_streaming_fingerprint_matches_small_canonical_reference_without_sequence_copy():
    plan = build_grid_v2_plan(_small_config(), _sampled_settings())
    identity = dict(plan.metadata["planning"]["identity"])
    identity.pop("requested_policy", None)
    identity.pop("effective_policy_reason", None)
    header = {
        "effective_policy": "sampled",
        "plan_identity_schema_version": GRID_V2_PLAN_IDENTITY_SCHEMA_VERSION,
        "planning_identity": identity,
    }
    keys = plan.candidate_table.semantic_keys_by_row or ()
    canonical = _stable_json(header) + "".join(f"\n{key}" for key in keys)
    expected = hashlib.blake2b(canonical.encode("utf-8"), digest_size=32).hexdigest()

    class OnePassKeys:
        def __iter__(self):
            yield from keys

        def __len__(self):
            raise AssertionError("streaming fingerprint must not size or copy the key stream")

    assert plan.plan_fingerprint == expected
    assert _grid_v2_plan_fingerprint(
        planning_identity=identity,
        effective_policy="sampled",
        semantic_keys=OnePassKeys(),
    ) == expected


def test_plan_fingerprint_is_invariant_to_compiled_worker_count():
    one_worker = build_grid_v2_plan(_small_config(), _sampled_settings(workers=1))
    many_workers = build_grid_v2_plan(_small_config(), _sampled_settings(workers=6))

    assert one_worker.plan_fingerprint == many_workers.plan_fingerprint
    assert _semantic_digest(one_worker) == _semantic_digest(many_workers)


def test_v2_allocation_reconciles_only_automatic_zero_targets_in_declared_order():
    initial = allocate_ordered_block_budgets(
        ("small", "large_a", "large_b"),
        {"small": 1, "large_a": 100, "large_b": 100},
        3,
        method="proportional_space",
    )
    assert initial.mode_budgets == {"small": 0, "large_a": 2, "large_b": 1}
    reconciled = _reconcile_grid_v2_automatic_allocation(
        initial,
        block_order=("small", "large_a", "large_b"),
    )
    assert reconciled.mode_budgets == {"small": 1, "large_a": 1, "large_b": 1}
    assert reconciled.actual_budget == sum(reconciled.mode_budgets.values()) == 3
    assert all(reconciled.mode_budgets[name] <= reconciled.mode_space_sizes[name] for name in reconciled.mode_budgets)


def test_s06_sampled_allocation_edges_and_previously_successful_outputs_are_stable():
    config = load_config()

    def counts(budget, method, manual=()):
        preview = preview_grid_v2_counts(
            config,
            GridV2Settings(
                planning_policy="sampled",
                requested_budget=budget,
                allocation_method=method,
                manual_percents=manual,
            ),
        )
        return dict(preview.per_variant_counts)

    assert counts(50, "proportional_space") == {"bracket": 1, "trail": 49}
    assert counts(2, "auto_sqrt_space") == {"bracket": 1, "trail": 1}
    assert counts(1_000, "manual", (("bracket", 0.0), ("trail", 100.0))) == {
        "bracket": 0,
        "trail": 1_000,
    }
    assert counts(1, "manual", (("bracket", 0.0), ("trail", 100.0))) == {
        "bracket": 0,
        "trail": 1,
    }
    assert counts(1_000, "manual", (("bracket", 50.0), ("trail", 50.0))) == {
        "bracket": 480,
        "trail": 520,
    }
    assert counts(1_000, "auto_sqrt_space") == {"bracket": 173, "trail": 827}
    assert counts(1_000, "proportional_space") == {"bracket": 10, "trail": 990}
    with pytest.raises(ValueError, match="increase the budget or disable blocks"):
        counts(1, "auto_sqrt_space")


def test_s06_manual_allocation_validation_remains_fail_closed():
    config = load_config()

    def preview(manual, *, enabled_variants=None):
        return preview_grid_v2_counts(
            config,
            GridV2Settings(
                planning_policy="sampled",
                requested_budget=100,
                allocation_method="manual",
                manual_percents=manual,
                enabled_variants=enabled_variants,
            ),
        )

    with pytest.raises(ValueError, match="ghost"):
        preview((("bracket", 90.0), ("trail", 0.0), ("ghost", 10.0)))
    with pytest.raises(ValueError, match="trail"):
        preview((("bracket", 50.0), ("trail", 50.0)), enabled_variants=("bracket",))
    with pytest.raises(ValueError, match="sum to 100"):
        preview((("bracket", 40.0), ("trail", 50.0)))
    with pytest.raises(ValueError, match="non-negative"):
        preview((("bracket", -1.0), ("trail", 101.0)))
    with pytest.raises(ValueError, match="finite"):
        preview((("bracket", float("inf")), ("trail", 0.0)))
    with pytest.raises(ValueError, match="disabled mode 'empty'"):
        allocate_ordered_block_budgets(
            ("active", "empty"),
            {"active": 10, "empty": 0},
            5,
            method="manual",
            manual_percents={"active": 50.0, "empty": 50.0},
        )


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


def test_full_s06_population_order_and_semantic_identity_v2_digest_are_pinned():
    plan = build_grid_v2_plan(load_config())
    assert plan.deduped_candidate_count == 48_480
    assert plan.per_variant_counts == {"bracket": 480, "trail": 48_000}
    # Historical semantic_identity_v1 digest before warmupBars left identity:
    # f1aa8bba4099d07f4d0d2865a72bf6537767329f3fdd9b324f07986b8b8369e4
    assert GRID_V2_SEMANTIC_IDENTITY_SCHEMA_VERSION == "grid_v2_semantic_identity_v2"
    assert _semantic_digest(plan) == "fc55d174e835e7196ae5fcf21427d318dc364241f6b10560aa32545e6910a08f"
    assert plan.plan_fingerprint == "0f8d001c380df5ee95d34ca4e25910c674e20e9e8f34886a1bd2f1c261f019b2"


def test_tz62_sampled_s06_and_s03_rows_and_fingerprints_are_unchanged():
    settings = GridV2Settings(
        planning_policy="sampled",
        requested_budget=11,
        seed=42,
        allocation_method="auto_sqrt_space",
    )
    s06 = build_grid_v2_plan(get_strategy_config("s06_r_trend_v02_b2"), settings)
    s03 = build_grid_v2_plan(
        get_strategy_config("s03_reversal_v11_regime_er_b2"), settings
    )

    assert s06.per_variant_counts == {"bracket": 2, "trail": 9}
    assert _ordered_row_digest(s06) == (
        "7366aa9916b6c5bfec168295fa37388d3ba56c571b116d92efee35e4ad1d14d5"
    )
    assert s06.plan_fingerprint == (
        "f2aa08f49666fc0f3ae75fda89421df5b5a20ffd2c4a987a8df53c2b1584db17"
    )
    assert s03.per_variant_counts == {
        "cc_only": 2,
        "tbands_only": 2,
        "both": 7,
    }
    assert _ordered_row_digest(s03) == (
        "61e1cfc5d9eb53ed1859d05bc9f99e3654727d43f80eb8c910a4a5a4aa12a51f"
    )
    assert s03.plan_fingerprint == (
        "2f769df30fb03f981e6edd0e07b73b9d929627751fa867a9a6e1b3c759791670"
    )


def test_semantic_identity_migration_preserves_tied_metric_rank_order():
    plan = build_grid_v2_plan(load_config(), _sampled_settings())

    def result(candidate_id, semantic_key):
        item = OptimizationResult(
            params={},
            net_profit_pct=1.0,
            max_drawdown_pct=1.0,
            total_trades=1,
            winning_trades=1,
            losing_trades=0,
            win_rate=100.0,
            avg_win=1.0,
            avg_loss=0.0,
            gross_profit=1.0,
            gross_loss=0.0,
            max_consecutive_losses=0,
            optuna_trial_number=candidate_id,
        )
        item.candidate_id = candidate_id
        item.semantic_key = semantic_key
        return item

    current_results = []
    historical_results = []
    for index, current_key in enumerate(plan.candidate_table.semantic_keys_by_row or (), 1):
        payload = json.loads(current_key)
        payload.pop("semantic_identity_schema_version")
        payload["params"]["warmupBars"] = 1000
        historical_key = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        current_results.append(result(index, current_key))
        historical_results.append(result(index, historical_key))

    current = rank_grid_results(
        current_results,
        objectives=["net_profit_pct"],
        primary_objective=None,
        constraints=[],
    )
    historical = rank_grid_results(
        historical_results,
        objectives=["net_profit_pct"],
        primary_objective=None,
        constraints=[],
    )

    assert [item.candidate_id for item in current] == [
        item.candidate_id for item in historical
    ]
    assert [item.grid_rank for item in current] == list(range(1, 12))


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


def test_plan_reuse_invalidates_normalized_modes_and_matches_fresh_plans():
    config = _small_config()
    settings = _sampled_settings()
    cache = GridV2PlanReuseCache()
    strict = cache.get_or_build(config, settings=settings)

    boundary_config = copy.deepcopy(config)
    boundary_config["execution"]["boundary"] = "none"
    boundary = cache.get_or_build(boundary_config, settings=settings)
    boundary_fresh = build_grid_v2_plan(boundary_config, settings)
    runtime = cache.get_or_build(
        boundary_config,
        settings=settings,
        base_params={"dateFilter": True, "start": "2025-03-01Z", "end": "2025-04-01Z"},
    )

    rounding_config = copy.deepcopy(boundary_config)
    rounding_config["execution"]["priceRounding"] = "tick_outward"
    rounding = cache.get_or_build(rounding_config, settings=settings)
    rounding_fresh = build_grid_v2_plan(rounding_config, settings)

    assert strict.hit is False
    assert boundary.hit is False
    assert runtime.hit is True
    assert rounding.hit is False
    for cached, fresh in ((boundary.plan, boundary_fresh), (rounding.plan, rounding_fresh)):
        assert cached.plan_fingerprint == fresh.plan_fingerprint
        assert cached.candidate_table.semantic_keys_by_row == fresh.candidate_table.semantic_keys_by_row
        assert cached.candidate_table.mode_tuples_by_variant == fresh.candidate_table.mode_tuples_by_variant
        assert cached.profile == fresh.profile
    assert boundary.plan.plan_fingerprint != strict.plan.plan_fingerprint
    assert rounding.plan.plan_fingerprint != boundary.plan.plan_fingerprint


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
