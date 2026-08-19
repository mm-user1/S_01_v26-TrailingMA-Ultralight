"""Explicit opt-in real development certification for Phase 3L-B allocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .allocation import DatasetInput, evaluate_allocation
from .dataset import AnalysisError, open_dataset
from .json_utils import canonical_json_bytes
from .rules import evaluate_rule, select_candidates, star_neighbours


def _dataset_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise AnalysisError("--dataset must use the explicit label=path form.")
    label, raw = value.split("=", 1)
    if not label or not raw:
        raise AnalysisError("--dataset requires a non-empty label and path.")
    return label, Path(raw)


def _inputs(arguments: Sequence[tuple[str, Path]]) -> list[DatasetInput]:
    if len({label for label, _ in arguments}) != len(arguments):
        raise AnalysisError("dataset labels must be unique.")
    output = []
    for label, path in arguments:
        dataset = open_dataset(path)
        scope = dataset.resolve_scope("development")
        if scope.requires_unlock or scope.ticker_cell != "dev":
            raise AnalysisError("allocation certification is development-only.")
        if any(window.window_id in (7, 8) for window in scope.windows):
            raise AnalysisError("allocation certification cannot use temporal windows 7-8.")
        output.append(DatasetInput(label, dataset, scope))
    return output


def _bounded(inputs: Sequence[DatasetInput], ticker_count: int, block_count: int) -> list[DatasetInput]:
    common_tickers = set(inputs[0].scope.tickers)
    common_blocks = set(inputs[0].scope.block_keys)
    for item in inputs[1:]:
        common_tickers &= set(item.scope.tickers)
        common_blocks &= set(item.scope.block_keys)
    tickers = tuple(sorted(common_tickers))[:ticker_count]
    blocks = tuple(sorted(common_blocks))[:block_count]
    if not tickers or not blocks:
        raise AnalysisError("certification inputs have no bounded common development cell.")
    output = []
    for item in inputs:
        by_block = {window.block_key: window for window in item.scope.windows}
        scope = item.dataset.subset_scope(
            item.scope,
            tickers=tickers,
            window_ids=tuple(by_block[block].window_id for block in blocks),
        )
        output.append(DatasetInput(item.label, item.dataset, scope))
    return output


def _reference(
    item: DatasetInput, candidate_rule: str, primary_k: int, sensitivity_k: int
) -> Mapping[tuple[str, str], Mapping[str, Any]]:
    """Small direct oracle; intentionally does not call allocation helpers."""
    neighbours = star_neighbours(item.dataset.geometry)
    output = {}
    minimum = float(item.dataset.contract.rule_registry["minimum_completed_trades"])
    declared = int(item.dataset.contract.split["development_ticker_count"])
    for window in item.scope.windows:
        is_views = item.dataset.load_is_window(item.scope, window)
        rows = []
        for ticker in sorted(item.scope.tickers):
            view = is_views[ticker]
            rule = evaluate_rule(
                candidate_rule,
                view,
                item.dataset.geometry,
                minimum,
                neighbours=neighbours,
            )
            selected = select_candidates(
                rule, view.metrics["net_profit_pct"], item.dataset.geometry, maximum=1
            )
            if not selected.row_indices:
                continue
            row = selected.row_indices[0]
            score = float(view.metrics["net_profit_pct"][row])
            if not np.isfinite(score):
                continue
            trades_raw = float(view.metrics["total_trades"][row])
            trades = trades_raw if np.isfinite(trades_raw) else None
            rows.append(
                {
                    "ticker": ticker,
                    "row": row,
                    "candidate_id": selected.candidate_ids[0],
                    "score": score,
                    "trades": trades,
                }
            )
        rows.sort(
            key=lambda row: (
                -row["score"],
                row["trades"] is None,
                -(row["trades"] or 0.0),
                row["ticker"],
            )
        )
        bottom = sorted(
            rows,
            key=lambda row: (
                row["score"],
                row["trades"] is None,
                -(row["trades"] or 0.0),
                row["ticker"],
            ),
        )
        oos = item.dataset.load_oos_window(item.scope, window)
        returns = {
            row["ticker"]: float(oos[row["ticker"]][row["row"], item.dataset.metric_index["net_profit_pct"]])
            for row in rows
        }
        if not all(np.isfinite(value) for value in returns.values()):
            raise AnalysisError("independent certification oracle encountered non-finite OOS.")
        available = len(rows)
        matched = max(1, min(available, math.ceil(primary_k / declared * available)))
        variants = {}
        for name, k in (("primary", primary_k), ("sensitivity", sensitivity_k), ("matched_fraction", matched)):
            top_symbols = [row["ticker"] for row in rows[: min(k, available)]]
            bottom_symbols = [row["ticker"] for row in bottom[: min(k, available)]]
            oracle_symbols = sorted(returns, key=lambda ticker: (-returns[ticker], ticker))[: min(k, available)]
            anti_symbols = sorted(returns, key=lambda ticker: (returns[ticker], ticker))[: min(k, available)]
            top_return = sum(returns[ticker] for ticker in top_symbols) / k
            payload = {
                "schema": "strategy_lab_random_k_v1",
                "base_seed": item.dataset.contract.evidence["uncertainty"]["seed"],
                "dataset_manifest_sha256": item.dataset.manifest_sha256.lower(),
                "dataset_label": item.label,
                "candidate_rule": candidate_rule,
                "ticker_scorer": {
                    "name": "selected_is_net_profit",
                    "version": "strategy_lab_ticker_score_v1",
                    "configuration": {},
                },
                "allocation_kind": "random_k",
                "k": k,
                "oos_start_utc": window.oos_start,
                "oos_end_utc": window.oos_end,
            }
            digest = hashlib.sha256(canonical_json_bytes(payload)).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "big", signed=False))
            values = np.asarray([returns[row["ticker"]] for row in rows], dtype=np.float64)
            draws = np.empty(item.dataset.contract.evidence["uncertainty"]["resamples"])
            for index in range(draws.size):
                chosen = rng.choice(values.size, size=min(k, available), replace=False)
                draws[index] = np.sum(values[chosen]) / k
            percentile = float((np.sum(draws < top_return) + 0.5 * np.sum(draws == top_return)) / draws.size)
            variants[name] = {
                "k": k,
                "top": top_symbols,
                "top_return": top_return,
                "bottom_return": sum(returns[ticker] for ticker in bottom_symbols) / k,
                "oracle_return": sum(returns[ticker] for ticker in oracle_symbols) / k,
                "anti_return": sum(returns[ticker] for ticker in anti_symbols) / k,
                "all_available": float(np.mean(list(returns.values()))),
                "random_mean": float(np.mean(draws)),
                "random_median": float(np.median(draws)),
                "random_percentile": percentile,
                "seed_sha": digest.hex(),
                "candidate_ids": {
                    row["ticker"]: row["candidate_id"] for row in rows
                },
            }
        output[window.block_key] = {"variants": variants}
    return output


def _near(observed: Any, expected: Any, label: str) -> None:
    if isinstance(expected, list):
        if observed != expected:
            raise AnalysisError(f"allocation certification mismatch for {label}.")
    elif isinstance(expected, str):
        if observed != expected:
            raise AnalysisError(f"allocation certification mismatch for {label}.")
    elif observed is None or not np.isclose(float(observed), float(expected), rtol=0.0, atol=1e-12):
        raise AnalysisError(
            f"allocation certification mismatch for {label}: {observed!r} != {expected!r}."
        )


def _compare(result, inputs: Sequence[DatasetInput], candidate_rule: str) -> None:
    for item in inputs:
        alignment = result.run_metadata["alignment"]
        by_block = {window.block_key: window for window in item.scope.windows}
        reference_scope = item.dataset.subset_scope(
            item.scope,
            tickers=tuple(alignment["common_tickers"]),
            window_ids=tuple(
                by_block[tuple(block)].window_id
                for block in alignment["common_blocks"]
            ),
        )
        reference = _reference(
            DatasetInput(item.label, item.dataset, reference_scope),
            candidate_rule,
            6,
            8,
        )
        blocks = result.summary["datasets"][item.label]["blocks"]
        observed_by_block = {tuple(block["block_key"]): block for block in blocks}
        for block_key, expected_block in reference.items():
            observed_block = observed_by_block[block_key]
            pairs = {
                row["ticker"]: row
                for row in result.pair_decisions
                if row["dataset_label"] == item.label
                and (row["oos_start"], row["oos_end"]) == block_key
                and row["candidate_id"] is not None
            }
            for name, expected in expected_block["variants"].items():
                observed = observed_block["variants"][name]
                for key, observed_key in (
                    ("k", "k"),
                    ("top", "selected_tickers"),
                    ("top_return", "capacity_return_pct"),
                    ("bottom_return", ("bottom_k", "capacity_return_pct")),
                    ("oracle_return", ("oracle_k", "capacity_return_pct")),
                    ("anti_return", ("anti_oracle_k", "capacity_return_pct")),
                    ("all_available", "all_available_mean_pct"),
                    ("random_mean", ("random_k", "random_mean_pct")),
                    ("random_median", ("random_k", "random_median_pct")),
                    ("random_percentile", ("random_k", "top_k_random_percentile_fraction")),
                    ("seed_sha", ("random_k", "seed_payload_sha256")),
                ):
                    actual = observed
                    for part in (observed_key if isinstance(observed_key, tuple) else (observed_key,)):
                        actual = actual[part]
                    _near(actual, expected[key], f"{item.label}/{block_key}/{name}/{key}")
                observed_ids = {
                    ticker: row["candidate_id"] for ticker, row in pairs.items()
                }
                if observed_ids != expected["candidate_ids"]:
                    raise AnalysisError("selected candidate identities disagree with the oracle.")
        ordered_reference = [reference[key] for key in sorted(reference)]
        for name in ("primary", "sensitivity", "matched_fraction"):
            expected_monthly = [block["variants"][name]["top_return"] for block in ordered_reference]
            equity = 1.0
            curve = [equity]
            for value in expected_monthly:
                equity *= 1.0 + value / 100.0
                curve.append(equity)
            peaks = np.maximum.accumulate(np.asarray(curve, dtype=np.float64))
            expected_drawdown = float(
                np.max(1.0 - np.asarray(curve, dtype=np.float64) / peaks) * 100.0
            )
            summary = result.summary["datasets"][item.label]["variants"][name]
            _near(
                summary["portfolio"]["compounded_return_pct"],
                (equity - 1.0) * 100.0,
                f"{item.label}/{name}/compounding",
            )
            _near(
                summary["portfolio"]["monthly_series_max_drawdown_pct"],
                expected_drawdown,
                f"{item.label}/{name}/drawdown",
            )
            expected_turnover = []
            for index in range(1, len(ordered_reference)):
                prior = ordered_reference[index - 1]["variants"][name]
                current = ordered_reference[index]["variants"][name]
                if prior["k"] != current["k"]:
                    expected_turnover.append(None)
                    continue
                k = current["k"]
                prior_set, current_set = set(prior["top"]), set(current["top"])
                retained = len(prior_set & current_set) + min(
                    k - len(prior_set), k - len(current_set)
                )
                expected_turnover.append(1.0 - retained / k)
            observed_turnover = [
                row["value"] for row in summary["turnover"]["transitions"][1:]
            ]
            if len(observed_turnover) != len(expected_turnover):
                raise AnalysisError("allocation certification turnover length mismatch.")
            for index, (observed, expected) in enumerate(
                zip(observed_turnover, expected_turnover), start=1
            ):
                if expected is None:
                    if observed is not None:
                        raise AnalysisError(
                            "allocation certification varying-K turnover mismatch."
                        )
                else:
                    _near(
                        observed,
                        expected,
                        f"{item.label}/{name}/turnover[{index}]",
                    )


def _audit(inputs: Sequence[DatasetInput]) -> Mapping[str, Any]:
    facts = {}
    for item in inputs:
        allowed_tickers = set(item.scope.tickers)
        allowed_windows = {window.window_id for window in item.scope.windows}
        outside = [
            entry
            for entry in item.dataset.access_log
            if entry[0] not in allowed_tickers
            or entry[1] not in allowed_windows
            or item.dataset.ticker_cells[entry[0]] != "dev"
            or entry[1] in (7, 8)
        ]
        if outside:
            raise AnalysisError(f"certification loaded outside development scope: {outside[:3]}.")
        facts[item.label] = {
            "ticker_count": len(item.scope.tickers),
            "window_ids": sorted(allowed_windows),
            "group_segment_load_count": len(item.dataset.access_log),
            "outside_scope_loads": [],
            "holdout_loaded": False,
            "temporal_windows_loaded": False,
        }
    return facts


def certify(arguments: Sequence[tuple[str, Path]], candidate_rule: str) -> Mapping[str, Any]:
    started = time.perf_counter()
    bounded_inputs = _bounded(_inputs(arguments), 10, 2)
    bounded_started = time.perf_counter()
    bounded = evaluate_allocation(bounded_inputs, candidate_rule=candidate_rule)
    _compare(bounded, bounded_inputs, candidate_rule)
    bounded_seconds = time.perf_counter() - bounded_started
    bounded_audit = _audit(bounded_inputs)

    underfill_inputs = _bounded(_inputs(arguments), 4, 2)
    underfill = evaluate_allocation(underfill_inputs, candidate_rule=candidate_rule)
    _compare(underfill, underfill_inputs, candidate_rule)
    underfill_audit = _audit(underfill_inputs)

    full_inputs = _inputs(arguments)
    tracemalloc.start()
    full_started = time.perf_counter()
    full = evaluate_allocation(full_inputs, candidate_rule=candidate_rule)
    _compare(full, full_inputs, candidate_rule)
    full_seconds = time.perf_counter() - full_started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    full_audit = _audit(full_inputs)
    return {
        "status": "passed",
        "candidate_rule": candidate_rule,
        "dataset_count": len(arguments),
        "datasets": {
            item.label: {
                "path": str(item.dataset.root),
                "manifest_sha256": item.dataset.manifest_sha256,
                "schema_version": item.dataset.schema_version,
                "scope": item.dataset.scope_label,
                "status": item.dataset.status,
                "total_ticker_count": len(item.dataset.tickers),
            }
            for item in full_inputs
        },
        "bounded": {
            "seconds": bounded_seconds,
            "common_ticker_count": bounded.run_metadata["alignment"]["common_ticker_count"],
            "common_block_count": bounded.run_metadata["alignment"]["common_block_count"],
            "audit": bounded_audit,
            "independent_oracle": "passed",
        },
        "underfill": {"audit": underfill_audit, "independent_oracle": "passed"},
        "full_development": {
            "seconds": full_seconds,
            "tracemalloc_peak_bytes": peak_bytes,
            "common_ticker_count": full.run_metadata["alignment"]["common_ticker_count"],
            "common_block_count": full.run_metadata["alignment"]["common_block_count"],
            "audit": full_audit,
            "independent_oracle": "passed",
        },
        "total_seconds": time.perf_counter() - started,
        "interpretation": "software certification only; no policy or strategy-quality conclusion",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--rule", default="primary_profit")
    args = parser.parse_args(argv)
    try:
        evidence = certify(tuple(_dataset_argument(value) for value in args.dataset), args.rule)
    except AnalysisError as exc:
        print(f"Strategy Lab allocation certification error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(evidence, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
