from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.strategy_lab.analysis.dataset import (
    AnalysisError,
    CandidateGeometry,
    ISView,
    ScopeLockedError,
    open_dataset,
)
from tools.strategy_lab.analysis.evaluate import (
    CustomRule,
    descriptive_bootstrap,
    evaluate_evidence,
    evaluate_scope,
    monthly_headline,
    outlier_robustness,
    profitable_share,
)
from tools.strategy_lab.analysis.output import OUTPUT_FILES, write_analysis
from tools.strategy_lab.analysis.rules import (
    RuleResult,
    evaluate_custom_rule,
    evaluate_rule,
    percentile_rank,
    percentile_scores,
    select_candidates,
    star_neighbours,
)
from tools.strategy_lab.config import canonical_json_bytes


METRICS = (
    "profit_factor",
    "net_profit_pct",
    "total_trades",
    "sharpe_daily",
    "max_drawdown_mtm_pct",
    "max_drawdown_pct",
    "gross_profit",
    "gross_loss",
    "win_rate_pct",
    "sqn",
    "rejected_fill_count",
    "zero_size_entry_count",
    "invalid_stop_distance_count",
    "flags",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract(window_ids=(1, 2)):
    selectable = [
        "primary_profit",
        "trade_gate15_profit",
        "trade_gate15_profit_factor",
        "trade_gate15_daily_sharpe",
        "trade_gate15_romad_mtm",
        "star_mean_profit",
        "star_worst_profit",
        "balanced_percentile_raw",
        "balanced_percentile_star",
    ]
    diagnostics = [
        "trade_gate15_romad_realized",
        "balanced_percentile_raw_realized",
        "balanced_percentile_star_realized",
        "pareto_plus_primary",
        "population_no_skill",
        "oos_oracle",
        "oos_anti_oracle",
    ]
    evidence = {
        "broad_population_edge": {
            "median_candidate_oos_net_profit_pct": {"operator": ">", "value": 0},
            "profitable_observation_share": {"operator": ">", "value": 0.5},
            "positive_monthly_population_medians": {"operator": ">=", "value": 1, "of": 2},
        },
        "selected_strategy_viability": {
            "six_month_headline_mean_oos_net_profit_pct": {"operator": ">", "value": 0},
            "median_selected_oos_net_profit_pct": {"operator": ">", "value": 0},
            "profitable_selected_observation_share": {"operator": ">", "value": 0.5},
            "positive_monthly_selected_means": {"operator": ">=", "value": 1, "of": 2},
        },
        "selection_lift": {
            "comparison": "rule_selected_minus_primary_profit_oos_net_profit_pct",
            "six_month_equal_weight_monthly_mean": {"operator": ">", "value": 0},
            "positive_monthly_top1_paired_means": {"operator": ">=", "value": 1, "of": 2},
            "top5_equal_weight_monthly_mean": {"operator": ">=", "value": 0},
            "robust_top1_headline_mean": {"operator": ">", "value": 0},
        },
        "nomination": {"maximum_rules": 3},
        "outlier_procedure": {
            "contribution": "ticker_mean_top1_paired_lift_over_six_windows",
            "removal_candidates": "strictly_positive_contributors_only",
            "quota": "ceil(0.10*total_ticker_count_in_cell)",
            "remove_count": "min(quota,positive_contributor_count)",
            "order": ["mean_contribution_desc", "canonical_symbol_asc"],
            "recompute": "monthly_means_then_equal_weight_six_month_headline",
            "criterion_scope": "recomputed_headline_only_monthly_signs_descriptive",
        },
        "primary_confirmation": {"cell": "holdout"},
        "temporal_windows_7_8": {"descriptive_only": True},
        "uncertainty": {
            "type": "descriptive_month_block_bootstrap",
            "blocks": 2,
            "resamples": 50,
            "sample_size_per_resample": 2,
            "sampling": "with_replacement",
            "seed": 17,
            "percentiles": [2.5, 97.5],
            "pass_fail_criterion": False,
        },
    }
    return {
        "observation_contract": {
            "candidate_observation": "candidate_ticker_window",
            "monthly_ticker_aggregation": "equal_weight_valid_tickers",
            "six_month_headline_aggregation": "equal_weight_six_monthly_means",
            "headline_median": "all_valid_ticker_window_observations",
            "missing_required_value": "fail_or_mark_rule_pair_unavailable_with_count",
            "natural_zero_trade_net_profit_pct": 0,
            "natural_zero_trade_is_profitable": False,
            "natural_zero_trade_population_membership": "included",
            "unavailable_non_finite_metric": "excluded_from_selection_not_zero_filled",
        },
        "rule_registry": {
            "version": "strategy_lab_rules_v1",
            "minimum_completed_trades": 15,
            "baseline_rule": "primary_profit",
            "selectable_rules": selectable,
            "nomination_eligible_rules": selectable[1:],
            "non_nominatable_diagnostics": diagnostics,
            "tie_break": [
                {"field": "rule_score", "direction": "descending"},
                {"field": "IS net_profit_pct", "direction": "descending"},
                {"field": "semantic_key", "direction": "ascending"},
                {"field": "candidate_id", "direction": "ascending"},
            ],
        },
        "analysis_scopes": [
            {"name": "development", "ticker_cell": "dev", "window_numbers": list(window_ids), "requires_unlock": False},
            {"name": "holdout", "ticker_cell": "holdout", "window_numbers": list(window_ids), "requires_unlock": True},
        ],
        "split": {"development_ticker_count": 2, "holdout_ticker_count": 1},
        "evidence_criteria": evidence,
        "evidence_criteria_version": "strategy_lab_evidence_v1",
        "maximum_nominated_rules": 3,
        "primary_comparison": "top1_oos_net_profit_vs_primary_profit",
    }


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _rewrite_runspec(root: Path, mutate) -> None:
    path = root / "normalized_runspec.json"
    run_spec = json.loads(path.read_text())
    mutate(run_spec)
    _write_json(path, run_spec)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["identity"]["pre_registration_sha256"] = hashlib.sha256(
        canonical_json_bytes(run_spec)
    ).hexdigest()
    manifest["artifacts"]["normalized_runspec.json"].update(
        sha256=_sha(path), size=path.stat().st_size
    )
    _write_json(manifest_path, manifest)


def _remove_metric(root: Path, metric: str) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    axis = manifest["identity"]["metric_axis"]
    index = axis.index(metric)
    axis.pop(index)
    for record in manifest["groups"]:
        path = root / record["path"]
        matrix = np.load(path)
        matrix = np.delete(matrix, index, axis=2)
        np.save(path, matrix)
        record.update(
            sha256=_sha(path),
            size=path.stat().st_size,
            shape=list(matrix.shape),
        )
    _write_json(manifest_path, manifest)


def _synthetic_dataset(tmp_path: Path, *, actual_windows=(1, 2), metric_axis=METRICS) -> Path:
    root = tmp_path / "dataset"
    root.mkdir(parents=True)
    prereg = _contract()
    run_spec = {"schema_version": "strategy_lab_runspec_v1", "run_name": "synthetic", "generation": {}, "preregistration": prereg}
    _write_json(root / "normalized_runspec.json", run_spec)
    candidates = {
        "schema_version": "strategy_lab_candidates_v1",
        "candidate_count": 4,
        "global_axis_names": ["x"],
        "plan_fingerprint": "plan",
        "semantic_key_digest": "semantic",
        "identity_versions": {},
        "candidates": [
            {
                "row_index": row,
                "candidate_id": 10 + row * 10,
                "semantic_key": key,
                "params": {"x": row},
                "axis_value_codes": [row],
                "active_axis_mask": [True],
                "variant_name": "v",
                "grid_mode_name": "b",
            }
            for row, key in enumerate(("z", "a", "m", "x"))
        ],
    }
    _write_json(root / "candidates.json", candidates)
    tickers = (("AAA", "dev"), ("BBB", "dev"), ("HHH", "holdout"))
    with (root / "data_quality.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("segment", "canonical_symbol", "cell"))
        writer.writeheader()
        for ticker, cell in tickers:
            writer.writerow({"segment": "source", "canonical_symbol": ticker, "cell": cell})
    windows = [
        {
            "window_id": number,
            "is_start": f"2026-0{number}-01T00:00:00Z",
            "is_end": f"2026-0{number}-28T23:30:00Z",
            "oos_start": f"2026-0{number + 2}-01T00:00:00Z",
            "oos_end": f"2026-0{number + 2}-28T23:30:00Z",
        }
        for number in actual_windows
    ]
    groups = []
    index = {name: pos for pos, name in enumerate(metric_axis)}
    for ticker_index, (ticker, _) in enumerate(tickers):
        folder = root / "groups" / ticker
        folder.mkdir(parents=True)
        for window in windows:
            matrix = np.zeros((4, 2, len(metric_axis)), dtype=np.float64)
            for segment in range(2):
                matrix[:, segment, index["net_profit_pct"]] = np.array([-1.0, 2.0, 0.0, 1.0]) + ticker_index + window["window_id"] + segment
                matrix[:, segment, index["total_trades"]] = [20.0, 20.0, 0.0, 20.0]
                matrix[:, segment, index["profit_factor"]] = [1.0, 2.0, np.nan, 1.5]
                matrix[:, segment, index["sharpe_daily"]] = [0.1, 0.2, np.nan, 0.3]
                matrix[:, segment, index["max_drawdown_mtm_pct"]] = [2.0, 2.0, 0.0, 1.0]
                matrix[:, segment, index["max_drawdown_pct"]] = [3.0, 3.0, 0.0, 1.5]
                matrix[:, segment, index["gross_profit"]] = [2.0, 3.0, 0.0, 2.0]
                matrix[:, segment, index["gross_loss"]] = [1.0, 1.0, 0.0, 1.0]
                matrix[:, segment, index["win_rate_pct"]] = [40.0, 50.0, 0.0, 45.0]
                matrix[:, segment, index["sqn"]] = np.nan
                matrix[:, segment, index["flags"]] = [2.0, 0.0, 1.0, 0.0]
            path = folder / f"window_{window['window_id']:02d}.npy"
            np.save(path, matrix)
            groups.append({
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha(path),
                "size": path.stat().st_size,
                "shape": list(matrix.shape),
                "dtype": "float64",
            })
    artifacts = {}
    for name in ("normalized_runspec.json", "candidates.json", "data_quality.csv"):
        path = root / name
        artifacts[name] = {"path": name, "sha256": _sha(path), "size": path.stat().st_size}
    manifest = {
        "schema_version": "strategy_lab_dataset_v2",
        "scope": "full" if len(actual_windows) == 2 else "smoke",
        "status": "complete",
        "artifacts": artifacts,
        "identity": {
            "dataset_schema": "strategy_lab_dataset_v2",
            "metric_axis": list(metric_axis),
            "segment_axis": ["is", "oos"],
            "dtype": "float64",
            "candidate_count": 4,
            "candidate_axis": [10, 20, 30, 40],
            "plan_fingerprint": "plan",
            "semantic_key_digest": "semantic",
            "pre_registration_sha256": hashlib.sha256(canonical_json_bytes(run_spec)).hexdigest(),
            "included_tickers": [
                {"canonical_symbol": ticker, "cell": cell} for ticker, cell in tickers
            ],
            "windows": windows,
        },
        "groups": groups,
    }
    _write_json(root / "manifest.json", manifest)
    return root


def _geometry(*, keys=("z", "a", "m", "x")) -> CandidateGeometry:
    return CandidateGeometry(
        candidate_ids=np.array([10, 20, 30, 40]),
        semantic_keys=tuple(keys),
        params=tuple({"x": index} for index in range(4)),
        global_axis_names=("x",),
        axis_codes=np.array([[0], [1], [2], [3]]),
        active_masks=np.ones((4, 1), dtype=bool),
        block_keys=(("v", "b"),) * 4,
    )


def _view(**overrides) -> ISView:
    metrics = {
        "net_profit_pct": np.array([1.0, 4.0, 3.0, 2.0]),
        "total_trades": np.array([20.0, 20.0, 10.0, 20.0]),
        "profit_factor": np.array([1.0, 2.0, 9.0, 3.0]),
        "sharpe_daily": np.array([0.1, 0.3, 8.0, 0.2]),
        "max_drawdown_mtm_pct": np.array([2.0, 2.0, 1.0, 4.0]),
        "max_drawdown_pct": np.array([4.0, 2.0, 1.0, 2.0]),
    }
    metrics.update(overrides)
    return ISView(metrics=metrics, ticker="AAA", window_id=1, is_start="is0", is_end="is1")


def test_reader_uses_declared_axes_nonzero_ids_and_structural_is_view(tmp_path):
    root = _synthetic_dataset(tmp_path, metric_axis=tuple(reversed(METRICS)))
    dataset = open_dataset(root)
    scope = dataset.resolve_scope()
    assert dataset.geometry.candidate_ids.tolist() == [10, 20, 30, 40]
    view = dataset.load_is_window(scope, scope.windows[0])["AAA"]
    assert view.metrics["net_profit_pct"].tolist() == [0.0, 3.0, 1.0, 2.0]
    assert not hasattr(view, "oos")
    assert not hasattr(view, "load_oos")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "strategy_lab_dataset_v99", "unsupported dataset schema"),
        ("status", "partial", "dataset status"),
    ],
)
def test_reader_rejects_schema_and_status(tmp_path, field, value, message):
    root = _synthetic_dataset(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest[field] = value
    if field == "schema_version":
        manifest["identity"]["dataset_schema"] = value
    _write_json(root / "manifest.json", manifest)
    with pytest.raises(AnalysisError, match=message):
        open_dataset(root)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda run_spec: run_spec["preregistration"]["rule_registry"].update(version="rules_v99"),
            "unsupported rule registry version",
        ),
        (
            lambda run_spec: run_spec["preregistration"].update(evidence_criteria_version="evidence_v99"),
            "unsupported evidence criteria version",
        ),
        (
            lambda run_spec: run_spec["preregistration"]["rule_registry"]["tie_break"][0].update(field="mystery"),
            "unknown rule tie-break contract",
        ),
        (
            lambda run_spec: run_spec["preregistration"]["evidence_criteria"]["broad_population_edge"]["median_candidate_oos_net_profit_pct"].update(operator="=="),
            "malformed evidence leaf",
        ),
    ],
)
def test_reader_rejects_unknown_contract_versions_operators_and_ties(tmp_path, mutate, message):
    root = _synthetic_dataset(tmp_path)
    _rewrite_runspec(root, mutate)
    with pytest.raises(AnalysisError, match=message):
        open_dataset(root)


def test_reader_rejects_quality_disagreement_and_escaping_group(tmp_path):
    root = _synthetic_dataset(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["groups"][0]["path"] = "../escape.npy"
    _write_json(root / "manifest.json", manifest)
    with pytest.raises(AnalysisError, match="escapes the dataset root"):
        open_dataset(root)
    root = _synthetic_dataset(tmp_path / "other")
    text = (root / "data_quality.csv").read_text().replace("AAA,dev", "AAA,holdout")
    (root / "data_quality.csv").write_text(text)
    manifest = json.loads((root / "manifest.json").read_text())
    artifact = root / "data_quality.csv"
    manifest["artifacts"]["data_quality.csv"].update(sha256=_sha(artifact), size=artifact.stat().st_size)
    _write_json(root / "manifest.json", manifest)
    with pytest.raises(AnalysisError, match="ownership disagrees"):
        open_dataset(root)


def test_partial_scope_and_synthetic_holdout_lock(tmp_path):
    root = _synthetic_dataset(tmp_path, actual_windows=(1,))
    dataset = open_dataset(root)
    with pytest.raises(AnalysisError, match="missing actual window.*2"):
        dataset.resolve_scope()
    partial = dataset.resolve_scope(allow_partial=True)
    assert partial.missing_window_ids == (2,)
    with pytest.raises(ScopeLockedError, match="requires --unlock-scope"):
        dataset.resolve_scope("holdout")
    policy = tmp_path / "policy.json"
    policy.write_text('{"freeze":true}')
    unlocked = dataset.resolve_scope("holdout", allow_partial=True, unlock=True, policy_path=policy)
    assert unlocked.unlock_evidence["policy_sha256"] == _sha(policy)


@pytest.mark.parametrize(
    ("rule", "expected_scores", "expected_eligible"),
    [
        ("primary_profit", [1.0, 4.0, 3.0, 2.0], [1, 1, 1, 1]),
        ("trade_gate15_profit", [1.0, 4.0, 3.0, 2.0], [1, 1, 0, 1]),
        ("trade_gate15_profit_factor", [1.0, 2.0, 9.0, 3.0], [1, 1, 0, 1]),
        ("trade_gate15_daily_sharpe", [0.1, 0.3, 8.0, 0.2], [1, 1, 0, 1]),
        ("trade_gate15_romad_mtm", [0.5, 2.0, None, 0.5], [1, 1, 0, 1]),
    ],
)
def test_direct_rule_formulas(rule, expected_scores, expected_eligible):
    result = evaluate_rule(rule, _view(), _geometry(), 15)
    actual = [None if not np.isfinite(value) else value for value in result.score]
    assert actual == expected_scores
    assert result.eligible.astype(int).tolist() == expected_eligible


def test_missing_metric_nan_and_natural_zero_remain_distinct():
    geometry = _geometry()
    missing = ISView({"net_profit_pct": np.array([0.0, np.nan, 1.0, 2.0])}, "A", 1, "a", "b")
    primary = evaluate_rule("primary_profit", missing, geometry, 15)
    assert primary.eligible.tolist() == [True, False, True, True]
    with pytest.raises(AnalysisError, match="unsupported for dataset; missing: total_trades"):
        evaluate_rule("trade_gate15_profit", missing, geometry, 15)


def test_missing_mtm_marks_only_dependent_rules_unsupported(tmp_path):
    root = _synthetic_dataset(tmp_path)
    _remove_metric(root, "max_drawdown_mtm_pct")
    dataset = open_dataset(root)
    result = evaluate_scope(dataset, dataset.resolve_scope())
    assert result.run_metadata["rule_states"]["trade_gate15_romad_mtm"] == {
        "status": "unsupported_for_dataset",
        "missing_metrics": ["max_drawdown_mtm_pct"],
        "kind": "selectable",
    }
    assert "trade_gate15_romad_mtm" not in result.summary["rules"]
    assert result.summary["rules"]["primary_profit"]["selected_pairs"] == 4


def test_star_geometry_respects_active_mask_variant_and_ordered_domains():
    geometry = CandidateGeometry(
        candidate_ids=np.arange(100, 106),
        semantic_keys=tuple(f"k{x}" for x in range(6)),
        params=tuple({} for _ in range(6)),
        global_axis_names=("x", "y"),
        axis_codes=np.array([[0, -1], [2, -1], [4, -1], [0, -1], [0, 0], [0, 2]]),
        active_masks=np.array([[1, 0], [1, 0], [1, 0], [1, 0], [1, 1], [1, 1]], dtype=bool),
        block_keys=(("A", "b"), ("A", "b"), ("A", "b"), ("B", "b"), ("A", "b"), ("A", "b")),
    )
    neighbours = star_neighbours(geometry)
    assert neighbours[0].tolist() == [0, 1]
    assert neighbours[1].tolist() == [0, 1, 2]
    assert 3 not in neighbours[0] and 4 not in neighbours[0]
    assert neighbours[4].tolist() == [4, 5]


def test_star_profit_and_balanced_star_use_exact_valid_set():
    geometry = _geometry()
    neighbours = star_neighbours(geometry)
    mean = evaluate_rule("star_mean_profit", _view(), geometry, 15, neighbours=neighbours)
    worst = evaluate_rule("star_worst_profit", _view(), geometry, 15, neighbours=neighbours)
    assert mean.score[[0, 1, 3]].tolist() == [2.5, 2.5, 2.0]
    assert worst.score[[0, 1, 3]].tolist() == [1.0, 1.0, 2.0]
    assert np.isnan(mean.score[2]) and np.isnan(worst.score[2])
    view = _view(profit_factor=np.array([1.0, np.nan, 4.0, 3.0]))
    balanced = evaluate_rule("balanced_percentile_star", view, geometry, 15, neighbours=neighbours)
    assert not balanced.eligible[1]
    assert balanced.support["star_support_profit_factor"].tolist() == [1, 0, 0, 1]


def test_percentile_formula_ties_and_small_n():
    values = np.array([1.0, 1.0, 3.0, np.nan])
    assert percentile_scores(values, np.array([1, 1, 1, 1], dtype=bool)).tolist()[:3] == [0.25, 0.25, 1.0]
    assert np.isnan(percentile_scores(values, np.array([1, 0, 0, 0], dtype=bool))).all()


def test_three_way_rank1_tie_uses_verbatim_semantic_key_and_records_tie():
    geometry = _geometry(keys=("β", "A", "a", "Z"))
    result = RuleResult("tie", np.array([5.0, 5.0, 5.0, 1.0]), np.ones(4, bool), (), {})
    selection = select_candidates(result, np.array([2.0, 2.0, 2.0, 9.0]), geometry)
    assert selection.candidate_ids[:3] == (20, 30, 10)
    assert selection.tie_depth_at_rank1 == 3
    assert selection.tie_break_level_used == "semantic_key"


def test_pareto_strict_epsilon_and_topk_percentile_semantics():
    view = _view(
        net_profit_pct=np.array([4.0, 3.0, 2.0, 1.0]),
        max_drawdown_mtm_pct=np.array([2.0, 2.0 - 5e-13, 1.0, 3.0]),
        total_trades=np.full(4, 20.0),
    )
    result = evaluate_rule("pareto_plus_primary", view, _geometry(), 15)
    assert result.eligible.tolist() == [True, False, True, False]
    values = np.array([0.0, 10.0, 20.0, 30.0])
    assert percentile_rank(values, 1) == pytest.approx(1 / 3)
    assert np.mean([percentile_rank(values, row) for row in (1, 3)]) == pytest.approx(2 / 3)


def test_statistic_specific_aggregation_and_all_cell_denominator():
    values = np.array([[100.0, 1.0], [np.nan, 1.0], [np.nan, 1.0]])
    monthly, headline = monthly_headline(values)
    assert monthly.tolist() == [100.0, 1.0]
    assert headline == 50.5
    assert np.nanmean(values) == pytest.approx(25.75)
    assert profitable_share(np.array([[1.0, np.nan], [-1.0, np.nan]]), denominator=4) == 0.25


def test_all_eleven_evidence_leaves_and_metadata_blocks_not_evaluated():
    evidence = _contract()["evidence_criteria"]
    population = {"median": 1.0, "profitable_share": 0.75, "positive_monthly_medians": 2}
    rule = {
        "top1_headline_mean": 1.0,
        "top1_pooled_median": 1.0,
        "profitable_selected_observation_share": 0.75,
        "positive_monthly_selected_means": 2,
        "top1_lift_headline": 0.5,
        "positive_monthly_top1_lift": 2,
        "top5_lift_headline": 0.0,
        "robustness": {"robust_lift_headline": 0.1},
    }
    rows = evaluate_evidence(evidence, population, rule, actual_block_count=2)
    assert len(rows) == 11 and all(row["status"] == "pass" for row in rows)
    assert not any(row["criterion"].startswith("nomination.") for row in rows)
    partial = evaluate_evidence(evidence, population, rule, actual_block_count=1)
    assert next(row for row in partial if row["criterion"].endswith("positive_monthly_population_medians"))["status"] == "unavailable"


def test_outlier_symbol_tie_and_robust_lift_not_absolute_profit():
    lift = np.array([[2.0, 2.0], [2.0, 2.0], [-1.0, -1.0]])
    selected = np.array([[100.0, 100.0], [-50.0, -50.0], [5.0, 5.0]])
    result = outlier_robustness(lift, selected, ("BBB", "AAA", "CCC"), _contract()["evidence_criteria"]["outlier_procedure"])
    assert result["removed_tickers"] == ["AAA"]
    assert result["robust_lift_headline"] == 0.5
    assert result["robust_absolute_selected_headline"] == 52.5


def test_bootstrap_is_fresh_order_independent_and_requires_all_blocks():
    uncertainty = _contract()["evidence_criteria"]["uncertainty"]
    first = descriptive_bootstrap([1.0, 3.0], uncertainty)
    descriptive_bootstrap([-100.0, 100.0], uncertainty)
    second = descriptive_bootstrap([1.0, 3.0], uncertainty)
    assert first == second
    assert descriptive_bootstrap([2.0, 2.0], uncertainty)["width"] == 0.0
    unavailable = descriptive_bootstrap([2.0, np.nan], uncertainty)
    assert unavailable["status"] == "unavailable" and "observed 1" in unavailable["reason"]


def test_custom_callable_validation_and_exploratory_label():
    result = evaluate_custom_rule("custom", "v1", lambda view, geometry, context: np.arange(4.0), _view(), _geometry(), {})
    assert result.exploratory and result.eligible.all()
    with pytest.raises(AnalysisError, match="align"):
        evaluate_custom_rule("bad", "v1", lambda *args: np.arange(3.0), _view(), _geometry(), {})
    with pytest.raises(AnalysisError, match="non-finite"):
        evaluate_custom_rule("bad", "v1", lambda *args: RuleResult("bad", np.array([1.0, np.nan, 2.0, 3.0]), np.ones(4, bool), (), {}), _view(), _geometry(), {})


def test_end_to_end_guardrails_outputs_determinism_and_protection(tmp_path):
    root = _synthetic_dataset(tmp_path)
    dataset = open_dataset(root)
    result = evaluate_scope(dataset, dataset.resolve_scope(), custom_rules=(CustomRule("custom", "v1", lambda view, geometry, context: view.metrics["net_profit_pct"]),))
    assert result.summary["diagnostics"]["unknown_flag_bits"] == 1
    assert result.summary["diagnostics"]["known_flag_bits"] == {
        "rejected_fill": 2,
        "invalid_stop_distance": 4,
        "zero_size_entry": 8,
    }
    assert result.summary["diagnostics"]["guardrail_fault_observation_counts"] == {"zero_size_entry_count": 0, "invalid_stop_distance_count": 0}
    custom_rows = [row for row in result.pair_decisions if row["rule"] == "custom"]
    assert custom_rows and all(row["result_label"] == "exploratory" for row in custom_rows)
    baseline = next(
        row
        for row in result.pair_decisions
        if row["rule"] == "primary_profit" and row["ticker"] == "AAA" and row["window_id"] == 1
    )
    assert baseline["selected_candidate_ids"] == [20, 40, 30, 10]
    assert baseline["top1_oos_net_profit_pct"] == 4.0
    assert baseline["top5_oos_net_profit_pct"] == 2.5
    output = tmp_path / "analysis"
    first = write_analysis(result, output, dataset_root=root)
    before = {name: (output / name).read_bytes() for name in OUTPUT_FILES}
    second = write_analysis(result, output, dataset_root=root)
    assert first["status"] == "published" and second["status"] == "verified_noop"
    assert before == {name: (output / name).read_bytes() for name in OUTPUT_FILES}
    assert set(path.name for path in output.iterdir()) == set(OUTPUT_FILES)
    with pytest.raises(AnalysisError, match="outside every input dataset root"):
        write_analysis(result, root / "analysis", dataset_root=root)
    (output / "summary.json").write_text("{}")
    with pytest.raises(AnalysisError, match="nonmatching"):
        write_analysis(result, output, dataset_root=root)
