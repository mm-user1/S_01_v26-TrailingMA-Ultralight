"""Explicit opt-in real-data certification for Phase 3L-A."""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .dataset import AnalysisError, open_dataset
from .evaluate import evaluate_scope


def _near(observed: Any, expected: float, label: str, tolerance: float = 5e-4) -> None:
    if observed is None or not np.isfinite(float(observed)) or abs(float(observed) - expected) > tolerance:
        raise AnalysisError(
            f"real certification mismatch for {label}: observed {observed!r}, "
            f"expected {expected} ± {tolerance}."
        )


def _series(observed: Sequence[Any], expected: Sequence[float], label: str) -> None:
    if len(observed) != len(expected):
        raise AnalysisError(f"real certification length mismatch for {label}.")
    for index, (actual, wanted) in enumerate(zip(observed, expected), start=1):
        _near(actual, wanted, f"{label}[{index}]")


def _audit_access(dataset, allowed_tickers: set[str], allowed_windows: set[int]) -> None:
    forbidden = [
        entry
        for entry in dataset.access_log
        if entry[0] not in allowed_tickers or entry[1] not in allowed_windows
    ]
    if forbidden:
        raise AnalysisError(f"certification loaded outside its development scope: {forbidden[:3]}.")


def certify(dataset_path: Path) -> Mapping[str, Any]:
    started = time.perf_counter()
    bounded_dataset = open_dataset(dataset_path)
    development = bounded_dataset.resolve_scope("development")
    bounded_scope = bounded_dataset.subset_scope(
        development,
        tickers=development.tickers[:4],
        window_ids=[window.window_id for window in development.windows[:2]],
    )
    bounded_started = time.perf_counter()
    bounded = evaluate_scope(bounded_dataset, bounded_scope)
    bounded_seconds = time.perf_counter() - bounded_started
    _audit_access(
        bounded_dataset, set(bounded_scope.tickers), {window.window_id for window in bounded_scope.windows}
    )
    if bounded.summary["dimensions"] != {
        "tickers": 4,
        "windows": 2,
        "candidates": 480,
        "segments": 2,
        "metrics": 21,
        "pairs": 8,
    }:
        raise AnalysisError("bounded certification dimensions changed.")

    dataset = open_dataset(dataset_path)
    scope = dataset.resolve_scope("development")
    tracemalloc.start()
    full_started = time.perf_counter()
    result = evaluate_scope(dataset, scope)
    full_seconds = time.perf_counter() - full_started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    _audit_access(dataset, set(scope.tickers), {1, 2, 3, 4, 5, 6})
    if any(entry[1] in (7, 8) for entry in dataset.access_log):
        raise AnalysisError("certification loaded temporal windows 7-8.")
    if any(dataset.ticker_cells[entry[0]] != "dev" for entry in dataset.access_log):
        raise AnalysisError("certification loaded a holdout ticker.")

    summary = result.summary
    population = summary["population"]
    diagnostics = summary["diagnostics"]
    rules = summary["rules"]
    if population["observation_count"] != 69120:
        raise AnalysisError("real certification population count changed.")
    _near(population["mean"], -0.8114, "population mean")
    _near(population["median"], -1.4109, "population median")
    _near(population["profitable_share"], 0.4070, "population profitable share")
    _near(population["zero_share"], 0.0432, "population zero share")
    _series(population["monthly_medians"], [1.880, -3.490, -0.949, -3.790, 4.111, -5.187], "monthly medians")
    _series(population["monthly_means"], [2.868, -3.736, -1.093, -3.098, 5.844, -5.654], "monthly means")
    _series(population["monthly_profitable_shares"], [0.564, 0.293, 0.391, 0.307, 0.655, 0.233], "monthly profitable shares")
    _near(diagnostics["gate_retained_candidate_share"], 0.78458, "gate retained share")
    if diagnostics["gate_pairs_with_no_candidate"] != 0:
        raise AnalysisError("real certification gate produced an unavailable pair.")
    denominator = population["observation_count"]
    finite = diagnostics["metric_finite_counts"]
    for segment, metric, expected in (
        ("is", "profit_factor", 0.9625),
        ("is", "sharpe_daily", 0.9684),
        ("is", "sqn", 0.3510),
        ("oos", "profit_factor", 0.9479),
        ("oos", "sharpe_daily", 0.9568),
        ("oos", "sqn", 0.0011),
    ):
        _near(finite[segment][metric] / denominator, expected, f"{segment} {metric} availability")
    _near(diagnostics["flags_nonzero_share"], 0.75605, "flags non-zero share")
    _near(rules["primary_profit"]["top1_headline_mean"], -1.0210, "primary top1")
    _near(rules["primary_profit"]["top1_pooled_median"], -0.5270, "primary median")
    _near(rules["population_no_skill"]["top1_headline_mean"], -0.8114, "population no-skill")
    _near(rules["oos_oracle"]["top1_headline_mean"], 19.0830, "OOS oracle")
    _near(rules["oos_anti_oracle"]["top1_headline_mean"], -16.9390, "OOS anti-oracle")
    _near(rules["balanced_percentile_star"]["top1_headline_mean"], -1.4060, "balanced star")
    daily = rules["trade_gate15_daily_sharpe"]
    _near(daily["top1_headline_mean"], -0.4806, "Daily Sharpe top1")
    _near(daily["top1_lift_headline"], 0.5404, "Daily Sharpe lift")
    if daily["positive_monthly_top1_lift"] != 4:
        raise AnalysisError("Daily Sharpe positive-lift month count changed.")
    _near(daily["robustness"]["robust_lift_headline"], 0.0410, "Daily Sharpe robust lift")
    _near(daily["top5_lift_headline"], -0.6102, "Daily Sharpe top5 lift")
    if daily["robustness"]["removed_tickers"] != ["QTUMUSDT", "ETHFIUSDT", "RSRUSDT"]:
        raise AnalysisError("Daily Sharpe outlier-removal order changed.")
    _near(diagnostics["oos_aggregate_profit_factor"], 0.9504, "OOS aggregate PF")
    _near(diagnostics["oos_mean_win_rate_pct"], 35.1350, "OOS mean win rate")
    _near(diagnostics["candidate_rank_split_half_tickers_spearman"], 0.1935, "ticker split rank", tolerance=1.5e-3)
    _near(diagnostics["candidate_rank_split_half_time_spearman"], -0.1105, "time split rank")
    _near(diagnostics["pooled_is_to_oos_spearman"], 0.0076, "pooled IS/OOS Spearman")
    _near(diagnostics["best_fixed_candidate_headline"], 0.6902, "best fixed candidate")
    return {
        "status": "passed",
        "dataset": str(dataset.root),
        "manifest_sha256": dataset.manifest_sha256,
        "bounded": {
            "tickers": list(bounded_scope.tickers),
            "windows": [window.window_id for window in bounded_scope.windows],
            "seconds": bounded_seconds,
            "outside_scope_loads": [],
        },
        "full_development": {
            "ticker_count": len(scope.tickers),
            "windows": [window.window_id for window in scope.windows],
            "seconds": full_seconds,
            "tracemalloc_peak_bytes": peak_bytes,
            "outside_scope_loads": [],
            "holdout_loaded": False,
            "temporal_windows_loaded": False,
        },
        "total_seconds": time.perf_counter() - started,
        "tolerance": "5e-4 for four-decimal point oracles; 1.5e-3 for preserved ticker split rank",
        "bootstrap_ci_pinned": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        evidence = certify(args.dataset)
    except AnalysisError as exc:
        print(f"Strategy Lab analysis certification error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(evidence, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
