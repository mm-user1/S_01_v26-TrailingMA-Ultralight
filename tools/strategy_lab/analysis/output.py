"""Deterministic five-file publication for Strategy Lab analysis."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .dataset import AnalysisError
from .evaluate import AnalysisResult


OUTPUT_FILES = (
    "run_metadata.json",
    "summary.json",
    "pair_decisions.csv",
    "monthly_results.csv",
    "report.md",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    return value


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _csv_bytes(rows: tuple[Mapping[str, Any], ...], columns: tuple[str, ...]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        encoded = {}
        for column in columns:
            value = _jsonable(row.get(column))
            if isinstance(value, (list, dict)):
                encoded[column] = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            elif value is None:
                encoded[column] = ""
            else:
                encoded[column] = value
        writer.writerow(encoded)
    return stream.getvalue().encode("utf-8")


def _report(result: AnalysisResult) -> bytes:
    metadata, summary = result.run_metadata, result.summary
    dataset = metadata["dataset"]
    scope = metadata["analysis_scope"]
    population = summary["population"]
    def display(value: Any) -> str:
        return "unavailable" if value is None else f"{value:.6f}"

    lines = [
        "# Strategy Lab analysis report",
        "",
        f"- Dataset schema/status: `{dataset['schema_version']}` / `{dataset['status']}`",
        f"- Manifest SHA-256: `{dataset['manifest_sha256']}`",
        f"- Analysis scope: `{scope['name']}` ({scope['ticker_count']} tickers, {len(scope['actual_window_ids'])} UTC blocks)",
        f"- Partial scope: `{str(scope['is_partial']).lower()}`",
        f"- Population observations: {population['observation_count']}",
        f"- OOS Net Profit mean / median: {display(population['mean'])} / {display(population['median'])}",
        f"- OOS profitable share: {display(population['profitable_share'])}",
        "",
        "## Rule results",
        "",
        "| Rule | State | Selected pairs | Top-1 headline | Lift | Robust lift |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary["rules"].items():
        lines.append(
            f"| `{name}` | {row['rule_status']} | {row['selected_pairs']} / {row['total_pairs']} "
            f"| {display(row['top1_headline_mean'])} | {display(row['top1_lift_headline'])} "
            f"| {display(row['robustness']['robust_lift_headline'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This artifact applies frozen formulas and reports evidence mechanically. It does not nominate rules, allocate tickers, or conclude strategy quality. The descriptive month-block intervals are weak with five or six blocks.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def render_files(result: AnalysisResult) -> Mapping[str, bytes]:
    pair_columns = (
        "rule",
        "rule_kind",
        "result_label",
        "ticker",
        "window_id",
        "is_start",
        "is_end",
        "oos_start",
        "oos_end",
        "status",
        "reason",
        "selected_candidate_ids",
        "selected_semantic_keys",
        "ordered_scores",
        "ordered_is_net_profit_pct",
        "required_is_metrics",
        "matching_oos_metrics",
        "tie_depth_at_rank1",
        "tie_break_level_used",
        "top1_oos_net_profit_pct",
        "top5_oos_net_profit_pct",
        "top10_oos_net_profit_pct",
        "top1_oos_percentile",
        "top5_mean_individual_oos_percentile",
    )
    monthly_columns = (
        "row_kind",
        "rule",
        "window_id",
        "oos_start",
        "oos_end",
        "top1_mean",
        "top1_median",
        "top1_profitable_share",
        "top1_lift_mean",
        "top5_lift_mean",
        "population_mean",
        "population_median",
        "population_profitable_share",
    )
    return {
        "run_metadata.json": _json_bytes(result.run_metadata),
        "summary.json": _json_bytes(result.summary),
        "pair_decisions.csv": _csv_bytes(result.pair_decisions, pair_columns),
        "monthly_results.csv": _csv_bytes(result.monthly_results, monthly_columns),
        "report.md": _report(result),
    }


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def write_analysis(
    result: AnalysisResult,
    output_dir: str | Path,
    *,
    dataset_root: str | Path,
) -> Mapping[str, Any]:
    output = Path(output_dir).resolve()
    dataset = Path(dataset_root).resolve()
    if output == dataset or _inside(output, dataset):
        raise AnalysisError("analysis output must be outside every input dataset root.")
    files = render_files(result)
    if output.exists() and not output.is_dir():
        raise AnalysisError(f"analysis output is not a directory: {output}.")
    if output.exists():
        existing = {path.name for path in output.iterdir()}
        unknown = sorted(existing - set(OUTPUT_FILES))
        if unknown:
            raise AnalysisError(f"analysis output contains unexpected files: {unknown}.")
        if existing:
            if existing == set(OUTPUT_FILES) and all(
                (output / name).read_bytes() == content for name, content in files.items()
            ):
                return {"status": "verified_noop", "output": str(output), "files": list(OUTPUT_FILES)}
            raise AnalysisError(
                "analysis output already contains nonmatching results; choose a new directory."
            )
    output.mkdir(parents=True, exist_ok=True)
    published: list[Path] = []
    try:
        for name in OUTPUT_FILES:
            target = output / name
            temporary = output / f".{name}.{uuid.uuid4().hex}.tmp"
            temporary.write_bytes(files[name])
            os.replace(temporary, target)
            published.append(target)
    except OSError as exc:
        for target in published:
            try:
                target.unlink()
            except OSError:
                pass
        raise AnalysisError(f"cannot publish analysis output: {exc}") from exc
    return {"status": "published", "output": str(output), "files": list(OUTPUT_FILES)}
