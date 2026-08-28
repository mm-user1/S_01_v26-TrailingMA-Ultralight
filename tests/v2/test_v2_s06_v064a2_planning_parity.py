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


@pytest.mark.parametrize("budget", list(SAMPLED_PINS))
def test_sampled_boundary_membership_and_order_are_pinned(budget):
    settings = GridV2Settings(planning_policy="sampled", requested_budget=budget, seed=42)
    first = build_grid_v2_plan(CONFIG, settings)
    repeated = build_grid_v2_plan(CONFIG, settings)
    assert (semantic_key_digest(first), first.plan_fingerprint) == SAMPLED_PINS[budget]
    assert first.candidate_table.semantic_keys_by_row == repeated.candidate_table.semantic_keys_by_row
    assert len(set(first.candidate_table.semantic_keys_by_row or ())) == budget


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
    REFERENCE_IDS[0]: (52, 52, [("equal", 0, 52, 0, 52)], 1),
    REFERENCE_IDS[1]: (42, 42, [("equal", 0, 42, 0, 42)], 0),
    REFERENCE_IDS[2]: (68, 69, [("equal", 0, 58, 0, 58), ("insert", 58, 58, 58, 59), ("equal", 58, 68, 59, 69)], 11),
    REFERENCE_IDS[3]: (47, 47, [("equal", 0, 47, 0, 47)], 0),
    REFERENCE_IDS[4]: (51, 52, [("equal", 0, 42, 0, 42), ("insert", 42, 42, 42, 43), ("equal", 42, 51, 43, 52)], 10),
    REFERENCE_IDS[5]: (34, 34, [("equal", 0, 34, 0, 34)], 0),
}


@pytest.mark.parametrize("reference_id", REFERENCE_IDS)
def test_tradingview_differences_remain_exactly_within_the_accepted_cases(reference_id):
    expected_count, merlin_count, alignment, zipped_exit_mismatches = TV_EXPECTED[reference_id]
    facts = tradingview_comparison(reference_id)
    assert (facts["tradingview_count"], facts["merlin_count"]) == (expected_count, merlin_count)
    assert entry_alignment(reference_id) == alignment
    assert facts["exact"]["exit_time"]["count"] == zipped_exit_mismatches
    if reference_id in (REFERENCE_IDS[1], REFERENCE_IDS[3], REFERENCE_IDS[5]):
        assert facts["exact"] == {
            "direction": {"count": 0, "first": None},
            "entry_time": {"count": 0, "first": None},
            "exit_time": {"count": 0, "first": None},
        }
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
