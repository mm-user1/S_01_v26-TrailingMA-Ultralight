from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from core.grid_v2 import (
    GridV2Settings,
    GridV2StrategyHooks,
    build_grid_v2_plan,
    execute_grid_v2_candidates,
    preview_grid_v2_counts,
)
from strategies import get_strategy_config
from strategies.s06_r_trend_v02_b2 import strategy as old_strategy
from strategies.s06_r_trend_v06_4_a2_b2 import strategy as new_strategy
from tools.strategy_lab.config import semantic_key_digest
from s06_v064a2_test_helpers import (
    BASELINE_ROOT,
    REFERENCE_IDS,
    merged_reference_params,
    prepared_reference_dataset,
    entry_alignment,
    tradingview_comparison,
)


CONFIG = get_strategy_config("s06_r_trend_v06_4_a2_b2")
FULL_PINS = {
    ("bracket",): (480, "c047759d16a99bedd9235f0ea81ffbb3867e98a755db3e816404a4d4b2c36641", "34e698625c33fceac8c4e16e802b601966b2f948bbf74b08172a00e8a96a9a25"),
    ("r_trail",): (3600, "b6a4f1a3ed7c5c60caa653fa61f17f20b32535a38f822a763d4f549a1011b8fe", "cede8b2fbffa58b76d85f7183b9c1a25a0f0d9e554c36334d15290f21b8131c8"),
    ("chandelier",): (6000, "6d0b9481bfee338ccc33dd440aaeb664082b73b12d60be1896bb9e71ad4fb418", "96922bfd26960e16d12fc0ceb58d2b345d0e3c0d2e5a29734f5733608adadd51"),
    ("fixed_af_sar",): (2400, "d1e1f0641181a00af5e711acf2a94d663013f51c3218f7828a753bf2fceb1d42", "592ebea0136435074f3e751c9686285faef464339bd8d4af5f2b49cbcd6d485f"),
    None: (12480, "4d7173e4c4b2bb26a68195e32fb6b8a27b47c7098af7bcaecd158d1431337036", "73244c7dc697e11b33e721956b2b3a9c45098f055a64e95f26b921934763578d"),
}
SAMPLED_PINS = {
    479: ("0e677c01511ebad018d5aeb6a7417b543dfae4e0f835283b056b38847d8a98ed", "9375ca7cafa39f186679f7187f1ea6ecf352028fea5ad67e63ae6622a898b257"),
    480: ("6c1742c3f12a7285151f845f876002a8c3f01972bd5b2cc756ecf9976b5451a6", "4ab2e1b9d210942785d7244fd3f896acb90d14660ab32a6e235a36ba3bc2055a"),
    481: ("9712e20dca83d7956e1988a4c11812f83447ff506c8c47ea10c4f297c0044f47", "e275c25940d46ba1ae051dd37e460e790a298a7e3b90616082e77cebfe92c860"),
    2399: ("1a346199963d3618f715c9b6e7edd338219db7d8c4f54a0e3d97857eb11cb87c", "cfad7aa12fcc8bcec6a573f1c667069e31e28e253c9fba183a9bbdcc8d3f17ef"),
    2400: ("8ab32e73ce1ab53c15423fa246215089763d9779af6105a3c98bb19508048382", "db4d13293a086b77b42a6ef44569b624acfe2fbddd766b51724623726d361d23"),
    2401: ("4bba30d0c865c58e7b8883e8363f2cb6b21265d13ed8e2df0c39f97ec693d067", "e427fdf120b4c222bb98da09d73741558c748ea764fb070860dfe675d3c884c9"),
    5999: ("54ddbc79f373953db48eebe04c61bc5aec1d02c72dd0a2e8a1aae1a2e73af5fd", "095ea4f2c7d53313368af76c032a2d99f81355ece14f092563482b73eb7a1b8d"),
    6000: ("377d6e734ee22aa487731544bc1d52022b16b15da17d7c923ec414cbe6c6a90c", "060e0c8aa77a22eaec16a44d79ca8849685d172d8253399886b3daf3f4ee88b1"),
    6001: ("37b80bb4e3c2b6f944fea2044c9e8c1e8fb972c2e5e3d19991fc909bc902956f", "36b5d02bda2f95e51baf3cddefe26213103f7c957cd5e1d1ca0b326d8609b585"),
    12479: ("676d51bc6531ce4c2a0b6e5f4aa5d1fa40784adcb750a4897ec45699872975e3", "8751482804daf607ba4e49e63cbf2f659be108c63b0dac2cb480406b8402e8b7"),
}


@pytest.mark.parametrize("variants", list(FULL_PINS))
def test_full_counts_identities_uniqueness_and_preview_parity(variants):
    settings = GridV2Settings(enabled_variants=variants)
    plan = build_grid_v2_plan(CONFIG, settings)
    count, digest, fingerprint = FULL_PINS[variants]
    preview = preview_grid_v2_counts(CONFIG, settings)
    assert plan.deduped_candidate_count == preview.planned_candidate_count == count
    assert semantic_key_digest(plan) == digest
    assert plan.plan_fingerprint == fingerprint
    keys = plan.candidate_table.semantic_keys_by_row
    assert keys is not None and len(keys) == len(set(keys)) == count


def test_r_trail_and_chandelier_combined_count_is_unchanged():
    settings = GridV2Settings(enabled_variants=("r_trail", "chandelier"))
    plan = build_grid_v2_plan(CONFIG, settings)
    preview = preview_grid_v2_counts(CONFIG, settings)

    assert plan.deduped_candidate_count == preview.planned_candidate_count == 9_600


def test_stop_lp_current_grid_domain_remains_exactly_two_and_four():
    declaration = CONFIG["parameters"]["stopLP"]
    assert (declaration["min"], declaration["max"], declaration["step"]) == (2, 4, 2)
    assert declaration["optimize"] == {"enabled": True, "min": 2, "max": 4, "step": 2}

    plan = build_grid_v2_plan(CONFIG, GridV2Settings(enabled_variants=("bracket",)))
    assert plan.parameter_domains["stopLP"].values == (2, 4)


@pytest.mark.parametrize("budget", list(SAMPLED_PINS))
def test_sampled_boundary_membership_and_order_are_pinned(budget):
    settings = GridV2Settings(planning_policy="sampled", requested_budget=budget, seed=42)
    first = build_grid_v2_plan(CONFIG, settings)
    repeated = build_grid_v2_plan(CONFIG, settings)
    assert (semantic_key_digest(first), first.plan_fingerprint) == SAMPLED_PINS[budget]
    assert first.candidate_table.semantic_keys_by_row == repeated.candidate_table.semantic_keys_by_row
    assert len(set(first.candidate_table.semantic_keys_by_row or ())) == budget


@pytest.mark.parametrize(
    ("variant", "trail_mode", "inactive_fields"),
    [
        ("bracket", "Off (Bracket)", ("trailRR", "trailDistanceR", "chandelierATRLength", "chandelierATRMult", "sarSpeed")),
        ("r_trail", "R Trail", ("stopRR", "chandelierATRLength", "chandelierATRMult", "sarSpeed")),
        ("chandelier", "Chandelier Exit", ("stopRR", "trailDistanceR", "sarSpeed")),
        ("fixed_af_sar", "Fixed-AF SAR", ("stopRR", "trailDistanceR", "chandelierATRLength", "chandelierATRMult")),
    ],
)
def test_single_variant_grid_makes_every_inactive_fixed_value_identity_inert(
    variant, trail_mode, inactive_fields
):
    settings = GridV2Settings(enabled_variants=(variant,), enabled_axes=())
    clean = build_grid_v2_plan(CONFIG, settings, {"trailMode": trail_mode})
    stale = build_grid_v2_plan(
        CONFIG,
        settings,
        {"trailMode": trail_mode, **dict.fromkeys(inactive_fields, "inactive-bad")},
    )

    assert stale.deduped_candidate_count == clean.deduped_candidate_count == 1
    assert stale.plan_fingerprint == clean.plan_fingerprint
    assert semantic_key_digest(stale) == semantic_key_digest(clean)
    assert stale.candidate_for_index(0).semantic_key == clean.candidate_for_index(0).semantic_key
    assert stale.candidate_for_index(0).params == clean.candidate_for_index(0).params


@pytest.mark.parametrize(
    ("variants", "field"),
    [
        (("bracket", "r_trail"), "stopRR"),
        (("bracket", "r_trail"), "trailRR"),
        (("bracket", "r_trail"), "trailDistanceR"),
        (("r_trail", "chandelier"), "chandelierATRLength"),
        (("r_trail", "chandelier"), "chandelierATRMult"),
        (("r_trail", "fixed_af_sar"), "sarSpeed"),
    ],
)
def test_combined_grid_rejects_malformed_field_active_in_any_selected_variant(
    variants, field
):
    with pytest.raises(ValueError, match=field):
        build_grid_v2_plan(
            CONFIG,
            GridV2Settings(enabled_variants=variants, enabled_axes=()),
            {field: "active-bad"},
        )


@pytest.mark.parametrize("value", [2, 4, 2.0, 4.0, "2", "2.0"])
def test_grid_fixed_stop_lp_accepts_exact_integral_forms(value):
    plan = build_grid_v2_plan(
        CONFIG,
        GridV2Settings(enabled_variants=("bracket",), enabled_axes=()),
        {"trailMode": "Off (Bracket)", "stopLP": value},
    )
    assert plan.parameter_domains["stopLP"].values == (int(float(value)),)


@pytest.mark.parametrize(
    "value",
    [2.9, "2.9", True, False, float("nan"), float("inf"), float("-inf"), "bad"],
)
def test_grid_fixed_stop_lp_rejects_nonintegral_forms(value):
    with pytest.raises(ValueError, match="stopLP"):
        build_grid_v2_plan(
            CONFIG,
            GridV2Settings(enabled_variants=("bracket",), enabled_axes=()),
            {"trailMode": "Off (Bracket)", "stopLP": value},
        )


def test_all_frozen_external_assets_retain_declared_hashes():
    manifest = json.loads((BASELINE_ROOT / "dataset.json").read_text(encoding="utf-8"))
    declared = list(manifest["asset_hashes"])
    declared.extend(manifest["pine_sources"].values())
    declared.append(manifest["market_data"])
    repo = BASELINE_ROOT.parents[2]
    for item in declared:
        if not isinstance(item, dict) or "path" not in item:
            continue
        assert hashlib.sha256((repo / item["path"]).read_bytes()).hexdigest() == item["sha256"]


def _assert_float(actual, expected):
    if math.isnan(float(expected)):
        assert math.isnan(float(actual))
    elif math.isinf(float(expected)):
        assert actual == expected
    else:
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("reference_id", REFERENCE_IDS)
def test_each_external_configuration_has_reference_compiled_and_mtm_parity(reference_id):
    frame, trade_start_idx = prepared_reference_dataset()
    base = merged_reference_params(reference_id)
    settings = GridV2Settings(enabled_axes=(), prefer_compiled=True)
    compiled_plan = build_grid_v2_plan(CONFIG, settings, base)
    reference_plan = build_grid_v2_plan(
        CONFIG, GridV2Settings(enabled_axes=(), prefer_compiled=False), base
    )
    hooks = GridV2StrategyHooks.from_strategy(new_strategy)
    compiled = execute_grid_v2_candidates(
        compiled_plan, frame, trade_start_idx, hooks, (0,), compute_max_drawdown_mtm=True
    ).rows[0]
    reference = execute_grid_v2_candidates(
        reference_plan, frame, trade_start_idx, hooks, (0,), compute_max_drawdown_mtm=True
    ).rows[0]
    for field in (
        "net_profit_pct", "max_drawdown_pct", "romad", "profit_factor", "win_rate_pct",
        "gross_profit", "gross_loss", "final_balance", "max_drawdown_mtm_pct",
    ):
        _assert_float(getattr(compiled, field), getattr(reference, field))
    for field in ("total_trades", "winning_trades", "losing_trades", "max_consecutive_losses"):
        assert getattr(compiled, field) == getattr(reference, field)
    for field, value in compiled.guardrail_summary.items():
        assert value == reference.guardrail_summary[field]


TV_EXPECTED = {
    REFERENCE_IDS[0]: {
        "counts": (52, 52),
        "alignment": [("equal", 0, 52, 0, 52)],
        "unmatched_merlin": [],
        "exit_mismatches": [(43, 43, "2025-11-03T15:30:00Z", "2025-11-03T15:00:00Z")],
        "nonzero": {"entry_price": 0, "exit_price": 30, "quantity": 26, "net_pnl": 52},
        "max_at": {"entry_price": (1, 1), "exit_price": (43, 43), "quantity": (52, 52), "net_pnl": (43, 43)},
        "above": {
            "entry_price": [], "exit_price": [(43, 43)],
            "quantity": [(number, number) for number in range(44, 53)],
            "net_pnl": [(37, 37), (43, 43), (44, 44), (45, 45), (48, 48), (50, 50), (52, 52)],
        },
    },
    REFERENCE_IDS[1]: {
        "counts": (42, 42), "alignment": [("equal", 0, 42, 0, 42)],
        "unmatched_merlin": [], "exit_mismatches": [],
        "nonzero": {"entry_price": 0, "exit_price": 20, "quantity": 10, "net_pnl": 42},
        "max_at": {"entry_price": (1, 1), "exit_price": (8, 8), "quantity": (33, 33), "net_pnl": (35, 35)},
        "above": dict.fromkeys(("entry_price", "exit_price", "quantity", "net_pnl"), []),
    },
    REFERENCE_IDS[2]: {
        "counts": (68, 69),
        "alignment": [("equal", 0, 58, 0, 58), ("insert", 58, 58, 58, 59), ("equal", 58, 68, 59, 69)],
        "unmatched_merlin": [{"number": 59, "identity": ("long", "2025-11-03T16:00:00Z")}],
        "exit_mismatches": [
            (58, 58, "2025-11-03T15:30:00Z", "2025-11-03T15:00:00Z"),
            (68, 69, "2025-11-30T23:30:00Z", "2025-12-01T00:00:00Z"),
        ],
        "nonzero": {"entry_price": 0, "exit_price": 66, "quantity": 48, "net_pnl": 68},
        "max_at": {"entry_price": (1, 1), "exit_price": (68, 69), "quantity": (67, 68), "net_pnl": (68, 69)},
        "above": {
            "entry_price": [], "exit_price": [(58, 58), (68, 69)],
            "quantity": [(32, 32), (42, 42), (45, 45), (48, 48), (49, 49), (50, 50), (52, 52), (53, 53), (54, 54), (55, 55), (56, 56), (58, 58), (59, 60), (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 68), (68, 69)],
            "net_pnl": [(41, 41), (58, 58), (59, 60), (61, 62), (65, 66), (66, 67), (68, 69)],
        },
    },
    REFERENCE_IDS[3]: {
        "counts": (47, 47), "alignment": [("equal", 0, 47, 0, 47)],
        "unmatched_merlin": [], "exit_mismatches": [],
        "nonzero": {"entry_price": 0, "exit_price": 44, "quantity": 12, "net_pnl": 47},
        "max_at": {"entry_price": (1, 1), "exit_price": (34, 34), "quantity": (47, 47), "net_pnl": (40, 40)},
        "above": dict.fromkeys(("entry_price", "exit_price", "quantity", "net_pnl"), []),
    },
    REFERENCE_IDS[4]: {
        "counts": (51, 52),
        "alignment": [("equal", 0, 42, 0, 42), ("insert", 42, 42, 42, 43), ("equal", 42, 51, 43, 52)],
        "unmatched_merlin": [{"number": 43, "identity": ("long", "2025-11-03T16:00:00Z")}],
        "exit_mismatches": [
            (42, 42, "2025-11-03T15:30:00Z", "2025-11-03T15:00:00Z"),
            (51, 52, "2025-11-30T23:30:00Z", "2025-12-01T00:00:00Z"),
        ],
        "nonzero": {"entry_price": 0, "exit_price": 40, "quantity": 28, "net_pnl": 51},
        "max_at": {"entry_price": (1, 1), "exit_price": (51, 52), "quantity": (45, 46), "net_pnl": (51, 52)},
        "above": {
            "entry_price": [], "exit_price": [(42, 42), (51, 52)],
            "quantity": [(35, 35), (37, 37), (38, 38), (39, 39), (42, 42), (43, 44), (44, 45), (45, 46), (46, 47), (47, 48), (48, 49), (49, 50), (50, 51), (51, 52)],
            "net_pnl": [(42, 42), (43, 44), (49, 50), (51, 52)],
        },
    },
    REFERENCE_IDS[5]: {
        "counts": (34, 34), "alignment": [("equal", 0, 34, 0, 34)],
        "unmatched_merlin": [], "exit_mismatches": [],
        "nonzero": {"entry_price": 0, "exit_price": 18, "quantity": 3, "net_pnl": 34},
        "max_at": {"entry_price": (1, 1), "exit_price": (34, 34), "quantity": (32, 32), "net_pnl": (11, 11)},
        "above": dict.fromkeys(("entry_price", "exit_price", "quantity", "net_pnl"), []),
    },
}


@pytest.mark.parametrize("reference_id", REFERENCE_IDS)
def test_tradingview_differences_remain_exactly_within_the_accepted_cases(reference_id):
    expected = TV_EXPECTED[reference_id]
    facts = tradingview_comparison(reference_id)
    assert (facts["tradingview_count"], facts["merlin_count"]) == expected["counts"]
    assert facts["matched_count"] == facts["tradingview_count"]
    assert facts["unmatched_tradingview"] == []
    assert facts["unmatched_merlin"] == expected["unmatched_merlin"]
    assert entry_alignment(reference_id) == expected["alignment"]
    assert facts["exact"]["direction"] == {"count": 0, "mismatches": []}
    assert facts["exact"]["entry_time"] == {"count": 0, "mismatches": []}
    assert [
        (item["tradingview_number"], item["merlin_number"], item["expected"], item["actual"])
        for item in facts["exact"]["exit_time"]["mismatches"]
    ] == expected["exit_mismatches"]
    for field, numeric in facts["numeric"].items():
        assert numeric["nonzero"] == expected["nonzero"][field]
        assert numeric["max_at"] == expected["max_at"][field]
        assert numeric["above_csv_rounding"] == expected["above"][field]
    if reference_id in (REFERENCE_IDS[1], REFERENCE_IDS[3], REFERENCE_IDS[5]):
        assert facts["numeric"]["exit_price"]["max_abs"] < 0.0001
        assert facts["numeric"]["quantity"]["max_abs"] <= 0.01000000000001
        assert facts["numeric"]["net_pnl"]["max_abs"] < 0.007


@pytest.mark.parametrize("entry_mode", ["Reversal @ Triangle", "Trend @ Square"])
def test_complete_480_candidate_bracket_fast_outputs_and_mtm_match_old_s06(entry_mode):
    frame, trade_start_idx = prepared_reference_dataset()
    common = {"entryMode": entry_mode, "useTrailMA": False, "dateFilter": False}
    old_plan = build_grid_v2_plan(
        old_strategy.load_config(), GridV2Settings(enabled_variants=("bracket",)), common
    )
    new_plan = build_grid_v2_plan(
        CONFIG,
        GridV2Settings(enabled_variants=("bracket",)),
        {"entryMode": entry_mode, "trailMode": "Off (Bracket)", "dateFilter": False},
    )
    assert old_plan.deduped_candidate_count == new_plan.deduped_candidate_count == 480
    indices = tuple(range(480))
    old = execute_grid_v2_candidates(
        old_plan, frame, trade_start_idx, GridV2StrategyHooks.from_strategy(old_strategy),
        indices, compute_max_drawdown_mtm=True,
    )
    new = execute_grid_v2_candidates(
        new_plan, frame, trade_start_idx, GridV2StrategyHooks.from_strategy(new_strategy),
        indices, compute_max_drawdown_mtm=True,
    )
    numeric = (
        "net_profit_pct", "max_drawdown_pct", "romad", "profit_factor", "win_rate_pct",
        "gross_profit", "gross_loss", "final_balance", "sharpe_ratio", "sqn", "sharpe_daily",
        "max_drawdown_mtm_pct",
    )
    integers = (
        "total_trades", "winning_trades", "losing_trades", "max_consecutive_losses",
        "sharpe_daily_observations", "sharpe_daily_active_days",
    )
    for old_row, new_row in zip(old.rows, new.rows):
        for field in numeric:
            _assert_float(getattr(new_row, field), getattr(old_row, field))
        for field in integers:
            assert getattr(new_row, field) == getattr(old_row, field)
        assert new_row.guardrail_summary == old_row.guardrail_summary
