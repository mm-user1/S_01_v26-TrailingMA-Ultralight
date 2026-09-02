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


def _reference_outcome(
    symbols: Sequence[str], returns: Mapping[str, float | None], k: int
) -> Mapping[str, Any]:
    selected = list(symbols[: min(k, len(symbols))])
    values = [returns.get(ticker) for ticker in selected]
    if not returns:
        return {"status": "unavailable", "reason": "zero available tickers", "return": None}
    if any(value is None or not math.isfinite(value) for value in values):
        return {
            "status": "unavailable",
            "reason": "selected OOS return is unavailable",
            "return": None,
        }
    return {"status": "available", "reason": None, "return": float(sum(values) / k)}


def _reference(
    inputs: Sequence[DatasetInput],
    candidate_rule: str,
    primary_k: int,
    sensitivity_k: int,
) -> Mapping[str, Mapping[tuple[str, str], Mapping[str, Any]]]:
    """Small direct oracle; intentionally does not call allocation helpers."""
    local: dict[str, dict[tuple[str, str], Mapping[str, Any]]] = {}
    for item in inputs:
        neighbours = star_neighbours(item.dataset.geometry)
        minimum = float(item.dataset.contract.rule_registry["minimum_completed_trades"])
        local[item.label] = {}
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
                row_index = selected.row_indices[0]
                score = float(view.metrics["net_profit_pct"][row_index])
                if not np.isfinite(score):
                    continue
                trades_raw = float(view.metrics["total_trades"][row_index])
                rows.append(
                    {
                        "ticker": ticker,
                        "row": row_index,
                        "candidate_id": selected.candidate_ids[0],
                        "score": score,
                        "trades": trades_raw if np.isfinite(trades_raw) else None,
                    }
                )
            local[item.label][window.block_key] = {"window": window, "rows": rows}

    output: dict[str, dict[tuple[str, str], Mapping[str, Any]]] = {
        item.label: {} for item in inputs
    }
    block_keys = set(local[inputs[0].label])
    for item in inputs[1:]:
        block_keys &= set(local[item.label])
    for block_key in sorted(block_keys):
        shared = {
            row["ticker"] for row in local[inputs[0].label][block_key]["rows"]
        }
        for item in inputs[1:]:
            shared &= {row["ticker"] for row in local[item.label][block_key]["rows"]}

        frozen: dict[str, Mapping[str, Any]] = {}
        for item in inputs:
            rows = [
                row for row in local[item.label][block_key]["rows"] if row["ticker"] in shared
            ]
            ranked = sorted(
                rows,
                key=lambda row: (
                    -row["score"],
                    row["trades"] is None,
                    -(row["trades"] or 0.0),
                    row["ticker"],
                ),
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
            available = len(ranked)
            declared = int(item.dataset.contract.split["development_ticker_count"])
            matched = (
                max(1, min(available, math.ceil(primary_k / declared * available)))
                if available
                else None
            )
            frozen[item.label] = {
                "window": local[item.label][block_key]["window"],
                "ranked": ranked,
                "bottom": bottom,
                "ks": {
                    "primary": primary_k,
                    "sensitivity": sensitivity_k,
                    "matched_fraction": matched,
                },
            }

        # Every rank and selected set above is frozen before this block reads any OOS.
        for item in inputs:
            state = frozen[item.label]
            window = state["window"]
            rows = state["ranked"]
            oos = item.dataset.load_oos_window(item.scope, window)
            returns: dict[str, float | None] = {}
            for row in rows:
                raw = oos[row["ticker"]][
                    row["row"], item.dataset.metric_index["net_profit_pct"]
                ]
                returns[row["ticker"]] = float(raw) if np.isfinite(raw) else None
            finite_returns = [value for value in returns.values() if value is not None]
            all_available = float(np.mean(finite_returns)) if finite_returns else None
            variants = {}
            for name, k in state["ks"].items():
                candidate_ids = {row["ticker"]: row["candidate_id"] for row in rows}
                if k is None:
                    variants[name] = {
                        "k": None,
                        "top": [],
                        "top_return": None,
                        "top_status": "unavailable",
                        "top_reason": "zero available tickers",
                        "bottom_return": None,
                        "bottom_status": "unavailable",
                        "bottom_reason": "zero available tickers",
                        "oracle_return": None,
                        "oracle_status": "unavailable",
                        "oracle_reason": "zero available tickers",
                        "anti_return": None,
                        "anti_status": "unavailable",
                        "anti_reason": "zero available tickers",
                        "all_available": None,
                        "random_mean": None,
                        "random_median": None,
                        "random_percentile": None,
                        "random_status": "unavailable",
                        "random_reason": "zero available tickers; matched K is unavailable",
                        "seed_sha": None,
                        "candidate_ids": candidate_ids,
                    }
                    continue
                top_symbols = [row["ticker"] for row in rows[: min(k, len(rows))]]
                bottom_symbols = [
                    row["ticker"] for row in state["bottom"][: min(k, len(rows))]
                ]
                finite_symbols = [ticker for ticker, value in returns.items() if value is not None]
                oracle_symbols = sorted(
                    finite_symbols, key=lambda ticker: (-returns[ticker], ticker)
                )[: min(k, len(rows))]
                anti_symbols = sorted(
                    finite_symbols, key=lambda ticker: (returns[ticker], ticker)
                )[: min(k, len(rows))]
                top = _reference_outcome(top_symbols, returns, k)
                bottom = _reference_outcome(bottom_symbols, returns, k)
                oracle = _reference_outcome(oracle_symbols, returns, k)
                anti = _reference_outcome(anti_symbols, returns, k)
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
                random_mean = random_median = random_percentile = None
                if not returns:
                    random_status, random_reason = "unavailable", "zero available tickers"
                elif any(value is None for value in returns.values()) or top["return"] is None:
                    random_status = "unavailable"
                    random_reason = "non-finite available or observed OOS return"
                else:
                    rng = np.random.default_rng(
                        int.from_bytes(digest[:8], "big", signed=False)
                    )
                    values = np.asarray(
                        [returns[row["ticker"]] for row in rows], dtype=np.float64
                    )
                    draws = np.empty(
                        item.dataset.contract.evidence["uncertainty"]["resamples"]
                    )
                    for index in range(draws.size):
                        chosen = rng.choice(
                            values.size, size=min(k, len(rows)), replace=False
                        )
                        draws[index] = np.sum(values[chosen]) / k
                    random_mean = float(np.mean(draws))
                    random_median = float(np.median(draws))
                    random_percentile = float(
                        (
                            np.sum(draws < top["return"])
                            + 0.5 * np.sum(draws == top["return"])
                        )
                        / draws.size
                    )
                    random_status, random_reason = "available", None
                variants[name] = {
                    "k": k,
                    "top": top_symbols,
                    "top_return": top["return"],
                    "top_status": top["status"],
                    "top_reason": top["reason"],
                    "bottom_return": bottom["return"],
                    "bottom_status": bottom["status"],
                    "bottom_reason": bottom["reason"],
                    "oracle_return": oracle["return"],
                    "oracle_status": oracle["status"],
                    "oracle_reason": oracle["reason"],
                    "anti_return": anti["return"],
                    "anti_status": anti["status"],
                    "anti_reason": anti["reason"],
                    "all_available": all_available,
                    "random_mean": random_mean,
                    "random_median": random_median,
                    "random_percentile": random_percentile,
                    "random_status": random_status,
                    "random_reason": random_reason,
                    "seed_sha": digest.hex(),
                    "candidate_ids": candidate_ids,
                }
            output[item.label][block_key] = {
                "available_tickers": len(rows),
                "variants": variants,
            }
    return output


def _near(observed: Any, expected: Any, label: str) -> None:
    if expected is None:
        if observed is not None:
            raise AnalysisError(
                f"allocation certification mismatch for {label}: {observed!r} != None."
            )
    elif isinstance(expected, list):
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
    reference_inputs = []
    alignment = result.run_metadata["alignment"]
    for item in inputs:
        by_block = {window.block_key: window for window in item.scope.windows}
        reference_scope = item.dataset.subset_scope(
            item.scope,
            tickers=tuple(alignment["common_tickers"]),
            window_ids=tuple(
                by_block[tuple(block)].window_id
                for block in alignment["common_blocks"]
            ),
        )
        reference_inputs.append(DatasetInput(item.label, item.dataset, reference_scope))
    references = _reference(reference_inputs, candidate_rule, 6, 8)
    for item in inputs:
        reference = references[item.label]
        blocks = result.summary["datasets"][item.label]["blocks"]
        observed_by_block = {tuple(block["block_key"]): block for block in blocks}
        for block_key, expected_block in reference.items():
            observed_block = observed_by_block[block_key]
            _near(
                observed_block["available_tickers"],
                expected_block["available_tickers"],
                f"{item.label}/{block_key}/available_tickers",
            )
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
                    ("top_status", "status"),
                    ("top_reason", "reason"),
                    ("bottom_status", ("bottom_k", "status")),
                    ("bottom_reason", ("bottom_k", "reason")),
                    ("oracle_status", ("oracle_k", "status")),
                    ("oracle_reason", ("oracle_k", "reason")),
                    ("anti_status", ("anti_oracle_k", "status")),
                    ("anti_reason", ("anti_oracle_k", "reason")),
                    ("random_status", ("random_k", "status")),
                    ("random_reason", ("random_k", "reason")),
                ):
                    actual = observed
                    for part in (observed_key if isinstance(observed_key, tuple) else (observed_key,)):
                        actual = actual[part]
                    _near(actual, expected[key], f"{item.label}/{block_key}/{name}/{key}")
                observed_ids = {
                    ticker: pairs[ticker]["candidate_id"]
                    for ticker in expected["candidate_ids"]
                    if ticker in pairs
                }
                if observed_ids != expected["candidate_ids"]:
                    raise AnalysisError("selected candidate identities disagree with the oracle.")
        ordered_reference = [reference[key] for key in sorted(reference)]
        for name in ("primary", "sensitivity", "matched_fraction"):
            expected_monthly = [block["variants"][name]["top_return"] for block in ordered_reference]
            summary = result.summary["datasets"][item.label]["variants"][name]
            if any(value is None for value in expected_monthly):
                _near(summary["portfolio"]["status"], "unavailable", f"{item.label}/{name}/status")
                _near(
                    summary["portfolio"]["reason"],
                    "one or more required calendar blocks are unavailable",
                    f"{item.label}/{name}/reason",
                )
                _near(summary["portfolio"]["compounded_return_pct"], None, f"{item.label}/{name}/compounding")
                _near(
                    summary["portfolio"]["monthly_series_max_drawdown_pct"],
                    None,
                    f"{item.label}/{name}/drawdown",
                )
            elif any(
                not math.isfinite(1.0 + value / 100.0)
                or 1.0 + value / 100.0 <= 0.0
                for value in expected_monthly
            ):
                _near(summary["portfolio"]["status"], "unavailable", f"{item.label}/{name}/status")
                _near(
                    summary["portfolio"]["reason"],
                    "monthly gross factor is non-finite or non-positive",
                    f"{item.label}/{name}/reason",
                )
                _near(summary["portfolio"]["compounded_return_pct"], None, f"{item.label}/{name}/compounding")
                _near(
                    summary["portfolio"]["monthly_series_max_drawdown_pct"],
                    None,
                    f"{item.label}/{name}/drawdown",
                )
            else:
                equity = 1.0
                curve = [equity]
                for value in expected_monthly:
                    equity *= 1.0 + value / 100.0
                    curve.append(equity)
                peaks = np.maximum.accumulate(np.asarray(curve, dtype=np.float64))
                expected_drawdown = float(
                    np.max(1.0 - np.asarray(curve, dtype=np.float64) / peaks) * 100.0
                )
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
                if (
                    prior["k"] is None
                    or current["k"] is None
                    or prior["k"] != current["k"]
                ):
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
