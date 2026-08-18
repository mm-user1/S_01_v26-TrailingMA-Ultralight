"""Frozen Strategy Lab rule formulas and deterministic candidate selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from .dataset import AnalysisError, CandidateGeometry, ISView


SELECTABLE_RULES = (
    "primary_profit",
    "trade_gate15_profit",
    "trade_gate15_profit_factor",
    "trade_gate15_daily_sharpe",
    "trade_gate15_romad_mtm",
    "star_mean_profit",
    "star_worst_profit",
    "balanced_percentile_raw",
    "balanced_percentile_star",
)
FORMULA_DIAGNOSTICS = (
    "trade_gate15_romad_realized",
    "balanced_percentile_raw_realized",
    "balanced_percentile_star_realized",
    "pareto_plus_primary",
)
OOS_DIAGNOSTICS = (
    "population_no_skill",
    "oos_oracle",
    "oos_anti_oracle",
)
RULE_REQUIREMENTS: Mapping[str, tuple[str, ...]] = {
    "primary_profit": ("net_profit_pct",),
    "trade_gate15_profit": ("total_trades", "net_profit_pct"),
    "trade_gate15_profit_factor": ("total_trades", "profit_factor"),
    "trade_gate15_daily_sharpe": ("total_trades", "sharpe_daily"),
    "trade_gate15_romad_mtm": (
        "total_trades",
        "net_profit_pct",
        "max_drawdown_mtm_pct",
    ),
    "star_mean_profit": ("total_trades", "net_profit_pct"),
    "star_worst_profit": ("total_trades", "net_profit_pct"),
    "balanced_percentile_raw": (
        "total_trades",
        "net_profit_pct",
        "profit_factor",
        "max_drawdown_mtm_pct",
    ),
    "balanced_percentile_star": (
        "total_trades",
        "net_profit_pct",
        "profit_factor",
        "max_drawdown_mtm_pct",
    ),
    "trade_gate15_romad_realized": (
        "total_trades",
        "net_profit_pct",
        "max_drawdown_pct",
    ),
    "balanced_percentile_raw_realized": (
        "total_trades",
        "net_profit_pct",
        "profit_factor",
        "max_drawdown_pct",
    ),
    "balanced_percentile_star_realized": (
        "total_trades",
        "net_profit_pct",
        "profit_factor",
        "max_drawdown_pct",
    ),
    "pareto_plus_primary": (
        "total_trades",
        "net_profit_pct",
        "max_drawdown_mtm_pct",
    ),
}


@dataclass(frozen=True)
class RuleResult:
    name: str
    score: np.ndarray
    eligible: np.ndarray
    required_metrics: tuple[str, ...]
    support: Mapping[str, np.ndarray]
    exploratory: bool = False


@dataclass(frozen=True)
class Selection:
    row_indices: tuple[int, ...]
    candidate_ids: tuple[int, ...]
    semantic_keys: tuple[str, ...]
    scores: tuple[float, ...]
    is_net_profit_pct: tuple[float | None, ...]
    tie_depth_at_rank1: int
    tie_break_level_used: str


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise AnalysisError("average_ranks requires one finite vector.")
    order = np.argsort(values, kind="stable")
    ranked = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranked[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranked


def percentile_scores(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if values.ndim != 1 or valid.shape != values.shape:
        raise AnalysisError("percentile inputs must be aligned vectors.")
    output = np.full(values.shape, np.nan, dtype=np.float64)
    selected = valid & np.isfinite(values)
    count = int(selected.sum())
    if count < 2:
        return output
    output[selected] = (average_ranks(values[selected]) - 1.0) / (count - 1.0)
    return output


def star_neighbours(geometry: CandidateGeometry) -> tuple[np.ndarray, ...]:
    """Return self plus same-block/same-mask one-step ordered-axis neighbours."""
    count, width = geometry.axis_codes.shape
    domains: list[tuple[int, ...]] = []
    for axis in range(width):
        domains.append(
            tuple(
                sorted(
                    {
                        int(geometry.axis_codes[row, axis])
                        for row in range(count)
                        if geometry.active_masks[row, axis]
                    }
                )
            )
        )
    lookup: dict[tuple[Any, ...], int] = {}
    for row in range(count):
        key = (
            geometry.block_keys[row],
            tuple(bool(value) for value in geometry.active_masks[row]),
            tuple(int(value) for value in geometry.axis_codes[row]),
        )
        if key in lookup:
            raise AnalysisError("candidate geometry has duplicate block/mask/code identity.")
        lookup[key] = row
    output: list[np.ndarray] = []
    for row in range(count):
        neighbours = [row]
        mask = geometry.active_masks[row]
        codes = geometry.axis_codes[row]
        for axis in range(width):
            if not mask[axis]:
                continue
            domain = domains[axis]
            try:
                position = domain.index(int(codes[axis]))
            except ValueError as exc:
                raise AnalysisError("active candidate code is absent from its domain.") from exc
            for adjacent in (position - 1, position + 1):
                if adjacent < 0 or adjacent >= len(domain):
                    continue
                moved = codes.copy()
                moved[axis] = domain[adjacent]
                key = (
                    geometry.block_keys[row],
                    tuple(bool(value) for value in mask),
                    tuple(int(value) for value in moved),
                )
                candidate = lookup.get(key)
                if candidate is not None:
                    neighbours.append(candidate)
        output.append(np.asarray(sorted(set(neighbours)), dtype=np.int64))
    return tuple(output)


def _star_values(
    values: np.ndarray,
    member_valid: np.ndarray,
    center_valid: np.ndarray,
    neighbours: tuple[np.ndarray, ...],
    *,
    reducer: str,
) -> tuple[np.ndarray, np.ndarray]:
    output = np.full(values.shape, np.nan, dtype=np.float64)
    support = np.zeros(values.shape, dtype=np.int64)
    for center, rows in enumerate(neighbours):
        if not center_valid[center]:
            continue
        members = rows[member_valid[rows] & np.isfinite(values[rows])]
        if members.size == 0:
            continue
        support[center] = members.size
        if reducer == "mean":
            output[center] = float(np.mean(values[members]))
        elif reducer == "min":
            output[center] = float(np.min(values[members]))
        else:
            raise AnalysisError(f"unknown star reducer: {reducer!r}.")
    return output, support


def _aligned(view: ISView, name: str, count: int) -> np.ndarray:
    values = np.asarray(view.metrics[name], dtype=np.float64)
    if values.shape != (count,):
        raise AnalysisError(f"metric {name!r} is not candidate-aligned.")
    return values


def evaluate_rule(
    name: str,
    view: ISView,
    geometry: CandidateGeometry,
    gate: float,
    *,
    neighbours: tuple[np.ndarray, ...] | None = None,
) -> RuleResult:
    if name not in RULE_REQUIREMENTS:
        raise AnalysisError(f"requested rule has no strategy_lab_rules_v1 formula: {name!r}.")
    required = RULE_REQUIREMENTS[name]
    absent = tuple(metric for metric in required if metric not in view.metrics)
    if absent:
        raise AnalysisError(
            f"rule {name!r} is unsupported for dataset; missing: {', '.join(absent)}."
        )
    count = geometry.count
    metrics = {metric: _aligned(view, metric, count) for metric in required}
    net = metrics.get("net_profit_pct")
    trades = metrics.get("total_trades")
    trade_ok = (
        np.isfinite(trades) & (trades >= gate)
        if trades is not None
        else np.ones(count, dtype=bool)
    )
    support: dict[str, np.ndarray] = {}
    if name == "primary_profit":
        score = net.copy()
        eligible = np.isfinite(net)
    elif name == "trade_gate15_profit":
        score = net.copy()
        eligible = trade_ok & np.isfinite(net)
    elif name == "trade_gate15_profit_factor":
        score = metrics["profit_factor"].copy()
        eligible = trade_ok & np.isfinite(score)
    elif name == "trade_gate15_daily_sharpe":
        score = metrics["sharpe_daily"].copy()
        eligible = trade_ok & np.isfinite(score)
    elif name in ("trade_gate15_romad_mtm", "trade_gate15_romad_realized"):
        denominator_name = (
            "max_drawdown_mtm_pct"
            if name.endswith("mtm")
            else "max_drawdown_pct"
        )
        denominator = metrics[denominator_name]
        valid = trade_ok & np.isfinite(net) & np.isfinite(denominator) & (denominator > 0)
        score = np.full(count, np.nan)
        score[valid] = net[valid] / denominator[valid]
        eligible = valid
    elif name in ("star_mean_profit", "star_worst_profit"):
        if neighbours is None:
            neighbours = star_neighbours(geometry)
        member_valid = trade_ok & np.isfinite(net)
        score, count_support = _star_values(
            net,
            member_valid,
            trade_ok,
            neighbours,
            reducer="mean" if name == "star_mean_profit" else "min",
        )
        support["star_support_count"] = count_support
        eligible = trade_ok & np.isfinite(score)
    elif name.startswith("balanced_percentile_"):
        if neighbours is None:
            neighbours = star_neighbours(geometry)
        denominator_name = (
            "max_drawdown_pct" if name.endswith("_realized") else "max_drawdown_mtm_pct"
        )
        pf = metrics["profit_factor"]
        denominator = metrics[denominator_name]
        valid = (
            trade_ok
            & np.isfinite(net)
            & np.isfinite(pf)
            & np.isfinite(denominator)
        )
        if "_star" in name:
            star_net, net_support = _star_values(
                net, valid, valid, neighbours, reducer="mean"
            )
            star_pf, pf_support = _star_values(
                pf, valid, valid, neighbours, reducer="mean"
            )
            star_dd, dd_support = _star_values(
                denominator, valid, valid, neighbours, reducer="mean"
            )
            common = valid & np.isfinite(star_net) & np.isfinite(star_pf) & np.isfinite(star_dd)
            components = (
                percentile_scores(star_net, common),
                percentile_scores(star_pf, common),
                percentile_scores(-star_dd, common),
            )
            support.update(
                star_support_net=net_support,
                star_support_profit_factor=pf_support,
                star_support_drawdown=dd_support,
            )
        else:
            common = valid
            components = (
                percentile_scores(net, common),
                percentile_scores(pf, common),
                percentile_scores(-denominator, common),
            )
        score = np.mean(np.vstack(components), axis=0)
        eligible = common & np.isfinite(score)
    elif name == "pareto_plus_primary":
        denominator = metrics["max_drawdown_mtm_pct"]
        valid = trade_ok & np.isfinite(net) & np.isfinite(denominator)
        front = np.zeros(count, dtype=bool)
        rows = np.flatnonzero(valid)
        ordered = sorted(rows.tolist(), key=lambda row: (-net[row], denominator[row]))
        running_min = np.inf
        for row in ordered:
            if denominator[row] < running_min - 1e-12:
                front[row] = True
                running_min = denominator[row]
        score = net.copy()
        eligible = front
    else:  # defensive against registry/formula drift
        raise AnalysisError(f"unimplemented rule formula: {name!r}.")
    return RuleResult(name, score, eligible, required, support)


def evaluate_custom_rule(
    name: str,
    version: str,
    function: Callable[[ISView, CandidateGeometry, Mapping[str, Any]], Any],
    view: ISView,
    geometry: CandidateGeometry,
    context: Mapping[str, Any],
) -> RuleResult:
    if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
        raise AnalysisError("custom rule name and version must be non-empty strings.")
    raw = function(view, geometry, context)
    if isinstance(raw, RuleResult):
        if raw.name != name:
            raise AnalysisError(
                f"custom rule result name {raw.name!r} does not match {name!r}."
            )
        score = np.asarray(raw.score, dtype=np.float64)
        eligible = np.asarray(raw.eligible, dtype=bool)
        required = raw.required_metrics
        support = raw.support
    else:
        score = np.asarray(raw, dtype=np.float64)
        eligible = np.isfinite(score)
        required = ()
        support = {}
    if score.shape != (geometry.count,) or eligible.shape != score.shape:
        raise AnalysisError("custom rule score and eligibility must align to candidates.")
    if np.any(eligible & ~np.isfinite(score)):
        raise AnalysisError("custom rule marked a non-finite score eligible.")
    return RuleResult(name, score, eligible, tuple(required), support, exploratory=True)


def select_candidates(
    result: RuleResult,
    is_net_profit_pct: np.ndarray,
    geometry: CandidateGeometry,
    *,
    maximum: int = 10,
) -> Selection:
    score = np.asarray(result.score, dtype=np.float64)
    eligible = np.asarray(result.eligible, dtype=bool)
    is_net = np.asarray(is_net_profit_pct, dtype=np.float64)
    expected = (geometry.count,)
    if score.shape != expected or eligible.shape != expected or is_net.shape != expected:
        raise AnalysisError("selection vectors are not candidate-aligned.")
    rows = np.flatnonzero(eligible & np.isfinite(score)).tolist()

    def sort_key(row: int) -> tuple[Any, ...]:
        net = is_net[row]
        return (
            -score[row],
            0 if np.isfinite(net) else 1,
            -net if np.isfinite(net) else 0.0,
            geometry.semantic_keys[row],
            int(geometry.candidate_ids[row]),
        )

    ordered = sorted(rows, key=sort_key)
    if not ordered:
        return Selection((), (), (), (), (), 0, "unavailable")
    winner_score = score[ordered[0]]
    tied = [row for row in ordered if score[row] == winner_score]
    if len(tied) == 1:
        tie_level = "rule_score"
    else:
        finite_nets = [is_net[row] for row in tied if np.isfinite(is_net[row])]
        best_net = max(finite_nets) if finite_nets else None
        net_tied = [
            row
            for row in tied
            if (best_net is None and not np.isfinite(is_net[row]))
            or (best_net is not None and np.isfinite(is_net[row]) and is_net[row] == best_net)
        ]
        if len(net_tied) == 1:
            tie_level = "IS net_profit_pct"
        else:
            best_key = min(geometry.semantic_keys[row] for row in net_tied)
            key_tied = [row for row in net_tied if geometry.semantic_keys[row] == best_key]
            tie_level = "semantic_key" if len(key_tied) == 1 else "candidate_id"
    chosen = tuple(ordered[:maximum])
    return Selection(
        row_indices=chosen,
        candidate_ids=tuple(int(geometry.candidate_ids[row]) for row in chosen),
        semantic_keys=tuple(geometry.semantic_keys[row] for row in chosen),
        scores=tuple(float(score[row]) for row in chosen),
        is_net_profit_pct=tuple(
            float(is_net[row]) if np.isfinite(is_net[row]) else None for row in chosen
        ),
        tie_depth_at_rank1=len(tied),
        tie_break_level_used=tie_level,
    )


def percentile_rank(values: np.ndarray, row: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if row < 0 or row >= values.size or not finite[row] or finite.sum() < 2:
        return float("nan")
    ranks = average_ranks(values[finite])
    finite_position = int(np.cumsum(finite)[row] - 1)
    return float((ranks[finite_position] - 1.0) / (finite.sum() - 1.0))
