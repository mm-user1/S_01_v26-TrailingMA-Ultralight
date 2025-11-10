"""Walk-Forward Analysis engine for strategy optimisation."""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest_engine import StrategyParams, run_strategy
from optuna_engine import OptunaConfig, run_optuna_optimization
from optimizer_engine import OptimizationConfig, OptimizationResult


class WFMode(Enum):
    """Walk-Forward window mode."""

    ROLLING = "rolling"
    ANCHORED = "anchored"


class CVMode(Enum):
    """Cross-validation mode."""

    DISABLED = "disabled"
    ENABLED = "enabled"
    AUTO = "auto"


@dataclass
class WalkForwardConfig:
    """Configuration values for Walk-Forward Analysis."""

    mode: WFMode = WFMode.ROLLING
    wf_zone_pct: float = 80.0
    forward_reserve_pct: float = 20.0
    is_pct: float = 70.0
    oos_pct: float = 30.0
    gap_bars: int = 2
    step_pct: float = 100.0
    cv_mode: CVMode = CVMode.AUTO
    cv_folds: int = 5
    cv_gap_bars: int = 0
    topk_per_window: int = 20
    min_oos_win_rate: float = 0.70
    max_degradation: float = 0.40
    min_trades_oos: int = 10
    min_forward_profit: float = 0.0
    warmup_multiplier: float = 1.5
    min_warmup_bars: int = 1000

    def ensure_valid(self) -> None:
        """Normalise numeric values to safe ranges."""

        self.wf_zone_pct = float(np.clip(self.wf_zone_pct, 40.0, 95.0))
        self.forward_reserve_pct = float(np.clip(self.forward_reserve_pct, 5.0, 60.0))
        total_allocation = self.wf_zone_pct + self.forward_reserve_pct
        if total_allocation <= 0:
            self.wf_zone_pct, self.forward_reserve_pct = 80.0, 20.0
        elif not math.isclose(total_allocation, 100.0, rel_tol=1e-3, abs_tol=1e-3):
            scale = 100.0 / total_allocation
            self.wf_zone_pct *= scale
            self.forward_reserve_pct *= scale

        self.is_pct = max(1.0, float(self.is_pct))
        self.oos_pct = max(1.0, float(self.oos_pct))
        total_window = self.is_pct + self.oos_pct
        if total_window <= 0:
            self.is_pct, self.oos_pct = 70.0, 30.0
        else:
            scale = 100.0 / total_window
            self.is_pct *= scale
            self.oos_pct *= scale

        self.step_pct = float(np.clip(self.step_pct, 10.0, 200.0))
        self.gap_bars = max(0, int(self.gap_bars))
        self.topk_per_window = max(1, int(self.topk_per_window))
        self.min_trades_oos = max(0, int(self.min_trades_oos))
        self.cv_folds = max(2, int(self.cv_folds))
        self.cv_gap_bars = max(0, int(self.cv_gap_bars))
        self.min_forward_profit = float(self.min_forward_profit)
        self.min_oos_win_rate = float(np.clip(self.min_oos_win_rate, 0.0, 1.0))
        self.max_degradation = float(np.clip(self.max_degradation, 0.0, 1.0))
        self.warmup_multiplier = float(max(1.0, self.warmup_multiplier))
        self.min_warmup_bars = max(100, int(self.min_warmup_bars))


@dataclass
class WindowSplit:
    """Definition of a single optimisation/validation window."""

    window_id: int
    warmup_start_idx: int
    warmup_end_idx: int
    is_start_idx: int
    is_end_idx: int
    gap_start_idx: int
    gap_end_idx: int
    oos_start_idx: int
    oos_end_idx: int

    def get_is_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.iloc[self.warmup_start_idx : self.is_end_idx]

    def get_is_only_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.iloc[self.is_start_idx : self.is_end_idx]

    def get_oos_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.iloc[self.warmup_start_idx : self.oos_end_idx]

    def get_oos_only_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.iloc[self.oos_start_idx : self.oos_end_idx]


@dataclass
class WindowResult:
    """Collection of optimisation and validation results per window."""

    window_id: int
    window_split: WindowSplit
    top_params: List[StrategyParams]
    is_results: List[Dict[str, float]]
    oos_results: List[Dict[str, float]]
    passed_params: List[StrategyParams]
    passed_oos_results: List[Dict[str, float]]
    passed_is_results: List[Dict[str, float]]
    cv_used: bool = False


@dataclass
class AggregatedParamResult:
    """Aggregated metrics for parameter sets across all windows."""

    param_hash: str
    params: StrategyParams
    window_count: int
    oos_profits: List[float]
    oos_drawdowns: List[float]
    oos_trades: List[int]
    oos_sharpes: List[float]
    avg_oos_profit: float
    median_oos_profit: float
    std_oos_profit: float
    oos_win_rate: float
    avg_is_profit: float
    avg_degradation: float
    consistency_score: float
    aggregate_score: float


@dataclass
class ForwardTestResult:
    """Performance of a parameter set on the forward reserve."""

    param_hash: str
    params: StrategyParams
    forward_profit: float
    forward_drawdown: float
    forward_sharpe: float
    forward_trades: int
    status: str
    passed: bool


@dataclass
class WalkForwardResult:
    """Full Walk-Forward Analysis output."""

    config: WalkForwardConfig
    windows: List[WindowSplit]
    forward_start_idx: int
    forward_end_idx: int
    window_results: List[WindowResult]
    aggregated_results: List[AggregatedParamResult]
    forward_results: List[ForwardTestResult]
    final_ranking: List[Tuple[str, float]]


def _strategy_hash(params: StrategyParams) -> str:
    """Create a deterministic hash of strategy parameters."""

    import hashlib

    payload = params.to_dict()
    normalized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


class WalkForwardEngine:
    """Main engine orchestrating Walk-Forward Analysis."""

    def __init__(self, config: WalkForwardConfig) -> None:
        self.config = config
        self.config.ensure_valid()

    # ------------------------------------------------------------------
    # Warmup utilities
    # ------------------------------------------------------------------
    def calculate_required_warmup(self, param_ranges: Dict[str, Tuple[float, float, float]]) -> int:
        """Estimate the warmup period based on the longest indicator length."""

        def _range_max(key: str, default: int) -> int:
            values = param_ranges.get(key)
            if not values:
                return default
            try:
                return int(values[1])
            except (TypeError, ValueError, IndexError):
                return default

        max_ma = _range_max("maLength", 200)
        max_trail_long = _range_max("trailLongLength", 200)
        max_trail_short = _range_max("trailShortLength", 200)
        atr_period = _range_max("atrPeriod", 14)

        max_period = max(max_ma, max_trail_long, max_trail_short, atr_period)
        warmup = int(math.ceil(max_period * self.config.warmup_multiplier))
        return max(warmup, self.config.min_warmup_bars)

    def validate_data_sufficiency(
        self, df: pd.DataFrame, param_ranges: Dict[str, Tuple[float, float, float]]
    ) -> Dict[str, Any]:
        """Check whether enough candles are available for WFA."""

        total_bars = len(df)
        required_warmup = self.calculate_required_warmup(param_ranges)
        available_for_wf = max(0, total_bars - required_warmup)
        min_window_bars = max(
            1000,
            int((self.config.is_pct + self.config.oos_pct) / 100.0 * 1000),
        ) + self.config.gap_bars
        min_required_total = required_warmup + max(min_window_bars * 2, 2000)
        return {
            "sufficient": total_bars >= min_required_total,
            "total_bars": total_bars,
            "required_warmup": required_warmup,
            "available_for_wf": available_for_wf,
            "recommended_warmup": int(required_warmup * 1.2),
            "min_required_total": min_required_total,
        }

    # ------------------------------------------------------------------
    # Window splitting
    # ------------------------------------------------------------------
    def split_data(
        self, df: pd.DataFrame, param_ranges: Dict[str, Tuple[float, float, float]]
    ) -> Tuple[List[WindowSplit], int, int]:
        """Split the dataset into walk-forward windows and forward reserve."""

        total_bars = len(df)
        warmup_bars = self.calculate_required_warmup(param_ranges)
        min_window_bars = max(
            1000,
            int((self.config.is_pct + self.config.oos_pct) / 100.0 * 1000),
        ) + self.config.gap_bars
        wf_zone_bars = int(total_bars * (self.config.wf_zone_pct / 100.0))
        wf_zone_bars = max(
            warmup_bars + min_window_bars,
            min(total_bars - 1, wf_zone_bars),
        )
        forward_start_idx = wf_zone_bars
        forward_end_idx = total_bars

        wf_start = warmup_bars
        wf_end = wf_zone_bars
        available = max(0, wf_end - wf_start)
        if available < min_window_bars:
            raise ValueError("Not enough data for walk-forward analysis after warmup trimming.")

        if self.config.mode == WFMode.ROLLING:
            windows = self._split_rolling(wf_start, wf_end, available, warmup_bars)
        else:
            windows = self._split_anchored(wf_start, wf_end, available, warmup_bars)

        if not windows:
            raise ValueError("Failed to construct walk-forward windows with provided settings.")

        return windows, forward_start_idx, forward_end_idx

    def _derive_window_lengths(self, available: int) -> Tuple[int, int]:
        """Derive IS and OOS lengths that honour gap and minimum sizes."""

        gap = max(0, self.config.gap_bars)
        usable = available - gap
        if usable <= 0:
            return 0, 0

        min_is = 200
        min_oos = 100
        min_total = min_is + min_oos
        if usable < min_total:
            return 0, 0

        ratio = max(1.0, self.config.is_pct + self.config.oos_pct)
        is_share = self.config.is_pct / ratio
        oos_share = self.config.oos_pct / ratio

        is_bars = max(min_is, int(round(usable * is_share)))
        oos_bars = max(min_oos, int(round(usable * oos_share)))

        total = is_bars + oos_bars
        if total > usable:
            overflow = total - usable
            reduce_oos = min(overflow, max(0, oos_bars - min_oos))
            oos_bars -= reduce_oos
            overflow -= reduce_oos

            if overflow > 0:
                reduce_is = min(overflow, max(0, is_bars - min_is))
                is_bars -= reduce_is
                overflow -= reduce_is

            if overflow > 0:
                if oos_bars - overflow >= min_oos:
                    oos_bars -= overflow
                    overflow = 0
                elif is_bars - overflow >= min_is:
                    is_bars -= overflow
                    overflow = 0

            if overflow > 0:
                return 0, 0

        return is_bars, oos_bars

    def _split_rolling(
        self, wf_start: int, wf_end: int, available: int, warmup: int
    ) -> List[WindowSplit]:
        windows: List[WindowSplit] = []
        is_bars, oos_bars = self._derive_window_lengths(available)
        if is_bars == 0 or oos_bars == 0:
            return windows
        step_bars = max(1, int(oos_bars * (self.config.step_pct / 100.0)))

        window_id = 1
        is_start = wf_start
        while True:
            is_end = is_start + is_bars
            gap_start = is_end
            gap_end = gap_start + self.config.gap_bars
            oos_start = gap_end
            oos_end = oos_start + oos_bars
            if oos_end > wf_end:
                break

            warmup_start = max(0, is_start - warmup)
            warmup_end = is_start

            windows.append(
                WindowSplit(
                    window_id=window_id,
                    warmup_start_idx=warmup_start,
                    warmup_end_idx=warmup_end,
                    is_start_idx=is_start,
                    is_end_idx=is_end,
                    gap_start_idx=gap_start,
                    gap_end_idx=gap_end,
                    oos_start_idx=oos_start,
                    oos_end_idx=oos_end,
                )
            )

            window_id += 1
            is_start += step_bars

        return windows

    def _split_anchored(
        self, wf_start: int, wf_end: int, available: int, warmup: int
    ) -> List[WindowSplit]:
        windows: List[WindowSplit] = []
        base_is, oos_bars = self._derive_window_lengths(available)
        if base_is == 0 or oos_bars == 0:
            return windows
        step_bars = max(1, int(oos_bars * (self.config.step_pct / 100.0)))

        window_id = 1
        is_end = wf_start + base_is
        while True:
            gap_start = is_end
            gap_end = gap_start + self.config.gap_bars
            oos_start = gap_end
            oos_end = oos_start + oos_bars
            if oos_end > wf_end:
                break

            warmup_start = max(0, wf_start - warmup)
            warmup_end = wf_start

            windows.append(
                WindowSplit(
                    window_id=window_id,
                    warmup_start_idx=warmup_start,
                    warmup_end_idx=warmup_end,
                    is_start_idx=wf_start,
                    is_end_idx=is_end,
                    gap_start_idx=gap_start,
                    gap_end_idx=gap_end,
                    oos_start_idx=oos_start,
                    oos_end_idx=oos_end,
                )
            )

            window_id += 1
            is_end += step_bars

        return windows

    # ------------------------------------------------------------------
    # Optimisation helpers
    # ------------------------------------------------------------------
    def should_use_cv(self, optuna_config: OptunaConfig, window: WindowSplit) -> bool:
        if self.config.cv_mode == CVMode.DISABLED:
            return False
        if self.config.cv_mode == CVMode.ENABLED:
            return True
        trials = getattr(optuna_config, "n_trials", 0)
        is_length = window.is_end_idx - window.is_start_idx
        min_length = max(2000, self.config.cv_folds * 500)
        return trials >= 200 and is_length >= min_length

    def _clone_base_config(
        self, base_config: OptimizationConfig, window_df: pd.DataFrame
    ) -> OptimizationConfig:
        buffer = io.StringIO()
        window_df.to_csv(buffer)
        buffer.seek(0)
        return replace(base_config, csv_file=buffer)

    def _results_to_params(
        self, results: Iterable[OptimizationResult], base_config: OptimizationConfig
    ) -> List[StrategyParams]:
        params: List[StrategyParams] = []
        for row in results:
            payload = {
                "backtester": True,
                "dateFilter": False,
                "maType": row.ma_type,
                "maLength": row.ma_length,
                "closeCountLong": row.close_count_long,
                "closeCountShort": row.close_count_short,
                "stopLongX": row.stop_long_atr,
                "stopLongRR": row.stop_long_rr,
                "stopLongLP": row.stop_long_lp,
                "stopShortX": row.stop_short_atr,
                "stopShortRR": row.stop_short_rr,
                "stopShortLP": row.stop_short_lp,
                "stopLongMaxPct": row.stop_long_max_pct,
                "stopShortMaxPct": row.stop_short_max_pct,
                "stopLongMaxDays": row.stop_long_max_days,
                "stopShortMaxDays": row.stop_short_max_days,
                "trailRRLong": row.trail_rr_long,
                "trailRRShort": row.trail_rr_short,
                "trailLongType": row.trail_ma_long_type,
                "trailLongLength": row.trail_ma_long_length,
                "trailLongOffset": row.trail_ma_long_offset,
                "trailShortType": row.trail_ma_short_type,
                "trailShortLength": row.trail_ma_short_length,
                "trailShortOffset": row.trail_ma_short_offset,
                "riskPerTrade": base_config.risk_per_trade_pct,
                "contractSize": base_config.contract_size,
                "commissionRate": base_config.commission_rate,
                "atrPeriod": base_config.atr_period,
            }
            params.append(StrategyParams.from_dict(payload))
        return params

    def _optimize_standard(
        self,
        base_config: OptimizationConfig,
        optuna_config: OptunaConfig,
        is_df: pd.DataFrame,
    ) -> List[StrategyParams]:
        cloned = self._clone_base_config(base_config, is_df)
        results = run_optuna_optimization(cloned, optuna_config, dataframe=is_df)
        top = results[: self.config.topk_per_window]
        return self._results_to_params(top, base_config)

    def _create_cv_folds(self, window: WindowSplit) -> List[Tuple[int, int]]:
        """Create sequential CV folds inside the IS region."""

        total_is = window.is_end_idx - window.is_start_idx
        min_fold_bars = max(200, self.config.cv_gap_bars + 1)
        max_possible_folds = total_is // min_fold_bars if min_fold_bars else 0
        if max_possible_folds < 2:
            return []

        n_folds = min(self.config.cv_folds, max_possible_folds)
        boundaries = np.linspace(window.is_start_idx, window.is_end_idx, n_folds + 1, dtype=int)
        folds: List[Tuple[int, int]] = []
        for i in range(n_folds):
            start = int(boundaries[i])
            end = int(boundaries[i + 1])
            if end - start <= 0:
                continue
            if self.config.cv_gap_bars:
                start = min(end, start + self.config.cv_gap_bars)
            if end - start <= 0:
                continue
            folds.append((start, end))
        return folds

    def _apply_cv_reranking(
        self,
        df: pd.DataFrame,
        window: WindowSplit,
        params: List[StrategyParams],
    ) -> List[StrategyParams]:
        """Re-rank parameter sets using median performance across CV folds."""

        if not params:
            return []

        folds = self._create_cv_folds(window)
        if not folds:
            return params

        scored: List[Tuple[float, StrategyParams]] = []
        for sp in params:
            fold_profits: List[float] = []
            for test_start, test_end in folds:
                metrics = self._calculate_metrics(
                    df,
                    [sp],
                    window.warmup_start_idx,
                    test_start,
                    test_end,
                )
                if metrics:
                    fold_profits.append(metrics[0]["net_profit_pct"])
            if not fold_profits:
                score = float("-inf")
            else:
                score = float(np.median(fold_profits))
            scored.append((score, sp))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored]

    def _calculate_metrics(
        self,
        df: pd.DataFrame,
        params_list: Iterable[StrategyParams],
        warmup_start_idx: int,
        period_start_idx: int,
        period_end_idx: int,
    ) -> List[Dict[str, float]]:
        """Evaluate parameter sets on a specific period of data."""

        period_end_idx = min(period_end_idx, len(df))
        if period_end_idx <= period_start_idx or period_end_idx <= 0:
            return []

        warmup_start_idx = max(0, min(warmup_start_idx, period_start_idx))
        data_slice = df.iloc[warmup_start_idx:period_end_idx]
        if data_slice.empty:
            return []

        index = df.index
        period_start_idx = max(0, min(period_start_idx, len(index) - 1))
        period_end_idx = max(period_start_idx + 1, period_end_idx)
        start_ts = index[period_start_idx]
        end_ts = index[period_end_idx - 1]

        metrics: List[Dict[str, float]] = []
        for params in params_list:
            filtered_params = replace(
                params,
                use_date_filter=True,
                start=start_ts,
                end=end_ts,
            )
            result = run_strategy(data_slice, filtered_params)
            metrics.append(
                {
                    "net_profit_pct": float(result.net_profit_pct),
                    "max_drawdown_pct": float(result.max_drawdown_pct),
                    "total_trades": int(result.total_trades),
                    "sharpe_ratio": float(getattr(result, "sharpe_ratio", 0.0)),
                    "profit_factor": float(getattr(result, "profit_factor", 0.0)),
                }
            )
        return metrics

    def _filter_params(
        self,
        params: List[StrategyParams],
        is_metrics: List[Dict[str, float]],
        oos_metrics: List[Dict[str, float]],
    ) -> Tuple[List[StrategyParams], List[Dict[str, float]], List[Dict[str, float]]]:
        filtered_params: List[StrategyParams] = []
        filtered_metrics: List[Dict[str, float]] = []
        filtered_is: List[Dict[str, float]] = []
        for sp, is_res, oos_res in zip(params, is_metrics, oos_metrics):
            if oos_res["net_profit_pct"] <= 0:
                continue
            if oos_res["total_trades"] < self.config.min_trades_oos:
                continue
            if is_res["net_profit_pct"] <= 0:
                continue
            degradation = 1.0 - (oos_res["net_profit_pct"] / is_res["net_profit_pct"])
            if degradation > self.config.max_degradation:
                continue
            filtered_params.append(sp)
            filtered_metrics.append(oos_res)
            filtered_is.append(is_res)
        return filtered_params, filtered_metrics, filtered_is

    # ------------------------------------------------------------------
    # Aggregation and ranking
    # ------------------------------------------------------------------
    def _aggregate_results(self, windows: List[WindowResult]) -> List[AggregatedParamResult]:
        grouped: Dict[str, Dict[str, Any]] = {}
        for window in windows:
            for params, is_res, oos_res in zip(
                window.passed_params, window.passed_is_results, window.passed_oos_results
            ):
                key = _strategy_hash(params)
                entry = grouped.setdefault(
                    key,
                    {
                        "params": params,
                        "oos_profits": [],
                        "oos_drawdowns": [],
                        "oos_trades": [],
                        "oos_sharpes": [],
                        "is_profits": [],
                    },
                )
                entry["oos_profits"].append(oos_res["net_profit_pct"])
                entry["oos_drawdowns"].append(oos_res["max_drawdown_pct"])
                entry["oos_trades"].append(oos_res["total_trades"])
                entry["oos_sharpes"].append(oos_res.get("sharpe_ratio", 0.0))
                entry["is_profits"].append(is_res["net_profit_pct"])

        aggregated: List[AggregatedParamResult] = []
        for key, entry in grouped.items():
            oos_profits = entry["oos_profits"]
            is_profits = entry["is_profits"]
            avg_is = float(np.mean(is_profits)) if is_profits else 0.0
            avg_oos = float(np.mean(oos_profits)) if oos_profits else 0.0
            median_oos = float(np.median(oos_profits)) if oos_profits else 0.0
            std_oos = float(np.std(oos_profits)) if len(oos_profits) > 1 else 0.0
            win_rate = float(sum(p > 0 for p in oos_profits)) / len(oos_profits) if oos_profits else 0.0
            degradation = 1.0 - (avg_oos / avg_is) if avg_is > 0 else 0.0
            consistency = 1.0 / (1.0 + std_oos)
            aggregate_score = (
                0.4 * avg_oos
                + 0.3 * win_rate * 100
                + 0.2 * (1.0 - degradation) * 100
                + 0.1 * consistency * 100
            )

            aggregated.append(
                AggregatedParamResult(
                    param_hash=key,
                    params=entry["params"],
                    window_count=len(oos_profits),
                    oos_profits=oos_profits,
                    oos_drawdowns=entry["oos_drawdowns"],
                    oos_trades=entry["oos_trades"],
                    oos_sharpes=entry["oos_sharpes"],
                    avg_oos_profit=avg_oos,
                    median_oos_profit=median_oos,
                    std_oos_profit=std_oos,
                    oos_win_rate=win_rate,
                    avg_is_profit=avg_is,
                    avg_degradation=degradation,
                    consistency_score=consistency,
                    aggregate_score=aggregate_score,
                )
            )

        filtered = [
            item
            for item in aggregated
            if item.window_count > 0 and item.oos_win_rate >= self.config.min_oos_win_rate
        ]
        filtered.sort(key=lambda item: item.aggregate_score, reverse=True)
        return filtered

    def _run_forward_test(
        self,
        df: pd.DataFrame,
        aggregated: List[AggregatedParamResult],
        warmup_start_idx: int,
        forward_start_idx: int,
        forward_end_idx: int,
    ) -> List[ForwardTestResult]:
        if forward_end_idx <= forward_start_idx:
            return []

        top_params = [item.params for item in aggregated[:50]]
        metrics = self._calculate_metrics(
            df,
            top_params,
            warmup_start_idx,
            forward_start_idx,
            forward_end_idx,
        )

        results: List[ForwardTestResult] = []
        for agg, metric in zip(aggregated[:50], metrics):
            profit = float(metric.get("net_profit_pct", 0.0))
            drawdown = float(metric.get("max_drawdown_pct", 0.0))
            trades = int(metric.get("total_trades", 0))
            sharpe = float(metric.get("sharpe_ratio", 0.0))

            if profit < self.config.min_forward_profit:
                status = "FAILED"
                passed = False
            elif profit < agg.avg_oos_profit * 0.5:
                status = "WEAK"
                passed = True
            else:
                status = "PASSED"
                passed = True

            results.append(
                ForwardTestResult(
                    param_hash=agg.param_hash,
                    params=agg.params,
                    forward_profit=profit,
                    forward_drawdown=drawdown,
                    forward_sharpe=sharpe,
                    forward_trades=trades,
                    status=status,
                    passed=passed,
                )
            )
        return results

    def _rank_by_forward(
        self, aggregated: List[AggregatedParamResult], forward_results: List[ForwardTestResult]
    ) -> List[Tuple[str, float]]:
        forward_map = {item.param_hash: item for item in forward_results}
        ranking: List[Tuple[str, float]] = []
        for agg in aggregated:
            forward = forward_map.get(agg.param_hash)
            if not forward:
                continue
            forward_norm = max(0.0, min(100.0, forward.forward_profit))
            oos_norm = max(0.0, min(100.0, agg.avg_oos_profit))
            final_score = (
                0.50 * forward_norm
                + 0.20 * oos_norm
                + 0.15 * (1.0 - agg.avg_degradation) * 100
                + 0.10 * agg.oos_win_rate * 100
                + 0.05 * agg.consistency_score * 100
            )
            ranking.append((agg.param_hash, final_score))
        ranking.sort(key=lambda item: item[1], reverse=True)
        return ranking

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_optimization(
        self,
        df: pd.DataFrame,
        base_config: OptimizationConfig,
        optuna_config: OptunaConfig,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> WalkForwardResult:
        validation = self.validate_data_sufficiency(df, base_config.param_ranges)
        if not validation["sufficient"]:
            raise ValueError(
                "Insufficient data for walk-forward analysis. "
                f"Required: {validation['min_required_total']}, available: {validation['total_bars']}"
            )

        warmup_bars = self.calculate_required_warmup(base_config.param_ranges)
        windows, forward_start, forward_end = self.split_data(df, base_config.param_ranges)
        if progress_callback:
            progress_callback(
                {
                    "stage": "split",
                    "total_windows": len(windows),
                    "forward_start": forward_start,
                    "forward_end": forward_end,
                }
            )

        window_results: List[WindowResult] = []
        for index, window in enumerate(windows, start=1):
            if progress_callback:
                progress_callback(
                    {
                        "stage": "window",
                        "window": window.window_id,
                        "index": index,
                        "total": len(windows),
                    }
                )

            is_df = window.get_is_df(df)

            params = self._optimize_standard(base_config, optuna_config, is_df)
            if use_cv:
                params = self._apply_cv_reranking(df, window, params)
            if len(params) > self.config.topk_per_window:
                params = params[: self.config.topk_per_window]
            if not params:
                continue

            is_metrics = self._calculate_metrics(
                df,
                params,
                window.warmup_start_idx,
                window.is_start_idx,
                window.is_end_idx,
            )
            oos_metrics = self._calculate_metrics(
                df,
                params,
                window.warmup_start_idx,
                window.oos_start_idx,
                window.oos_end_idx,
            )
            passed_params, passed_metrics, passed_is = self._filter_params(
                params, is_metrics, oos_metrics
            )

            window_results.append(
                WindowResult(
                    window_id=window.window_id,
                    window_split=window,
                    top_params=params,
                    is_results=is_metrics,
                    oos_results=oos_metrics,
                    passed_params=passed_params,
                    passed_oos_results=passed_metrics,
                    passed_is_results=passed_is,
                    cv_used=use_cv,
                )
            )

        if progress_callback:
            progress_callback({"stage": "aggregate"})
        aggregated = self._aggregate_results(window_results)

        if progress_callback:
            progress_callback({"stage": "forward_test"})
        forward_results = self._run_forward_test(
            df,
            aggregated,
            max(0, forward_start - warmup_bars),
            forward_start,
            forward_end,
        )

        if progress_callback:
            progress_callback({"stage": "rank"})
        final_ranking = self._rank_by_forward(aggregated, forward_results)

        return WalkForwardResult(
            config=self.config,
            windows=windows,
            forward_start_idx=forward_start,
            forward_end_idx=forward_end,
            window_results=window_results,
            aggregated_results=aggregated,
            forward_results=forward_results,
            final_ranking=final_ranking,
        )


def export_wf_results_to_csv(result: WalkForwardResult, output: Any) -> None:
    """Write walk-forward results to CSV. ``output`` may be a path or file-like object."""

    def _open_target(target: Any) -> Tuple[io.TextIOBase, bool]:
        if isinstance(target, (str, bytes, bytearray)):
            handle = open(target, "w", newline="")
            return handle, True
        if isinstance(target, io.TextIOBase):
            return target, False
        if hasattr(target, "write"):
            return target, False
        raise TypeError("Unsupported output type for CSV export.")

    handle, should_close = _open_target(output)
    writer = csv.writer(handle)

    try:
        writer.writerow(["=== WALK-FORWARD ANALYSIS SUMMARY ==="])
        writer.writerow(["Mode", result.config.mode.value])
        writer.writerow(["WF Zone", f"{result.config.wf_zone_pct:.1f}%"])
        writer.writerow(["Forward Reserve", f"{result.config.forward_reserve_pct:.1f}%"])
        writer.writerow(["Windows", len(result.windows)])
        writer.writerow([])

        writer.writerow(["=== FINAL RANKING (Top 20) ==="])
        writer.writerow(
            [
                "Rank",
                "Param Hash",
                "Forward Profit %",
                "Forward Max DD %",
                "Forward Sharpe",
                "Avg OOS Profit %",
                "OOS Win Rate %",
                "Score",
                "Status",
            ]
        )
        forward_map = {item.param_hash: item for item in result.forward_results}
        agg_map = {item.param_hash: item for item in result.aggregated_results}
        for rank, (param_hash, score) in enumerate(result.final_ranking[:20], start=1):
            forward = forward_map.get(param_hash)
            agg = agg_map.get(param_hash)
            if not forward or not agg:
                continue
            writer.writerow(
                [
                    rank,
                    param_hash[:12],
                    f"{forward.forward_profit:.2f}",
                    f"{forward.forward_drawdown:.2f}",
                    f"{forward.forward_sharpe:.2f}",
                    f"{agg.avg_oos_profit:.2f}",
                    f"{agg.oos_win_rate * 100:.1f}",
                    f"{score:.1f}",
                    forward.status,
                ]
            )

        writer.writerow([])
        writer.writerow(["=== AGGREGATED PARAMETER RESULTS (Top 20) ==="])
        writer.writerow(
            [
                "Rank",
                "Param Hash",
                "Windows",
                "Avg OOS %",
                "Median OOS %",
                "OOS Win Rate %",
                "Avg IS %",
                "Degradation %",
            ]
        )
        for rank, agg in enumerate(result.aggregated_results[:20], start=1):
            writer.writerow(
                [
                    rank,
                    agg.param_hash[:12],
                    agg.window_count,
                    f"{agg.avg_oos_profit:.2f}",
                    f"{agg.median_oos_profit:.2f}",
                    f"{agg.oos_win_rate * 100:.1f}",
                    f"{agg.avg_is_profit:.2f}",
                    f"{agg.avg_degradation * 100:.2f}",
                ]
            )

        writer.writerow([])
        writer.writerow(["=== WINDOW RESULTS ==="])
        writer.writerow(
            [
                "Window",
                "Type",
                "Net Profit %",
                "Max DD %",
                "Trades",
                "Param Hash",
            ]
        )
        for window in result.window_results:
            for params, is_metrics, oos_metrics in zip(
                window.top_params, window.is_results, window.oos_results
            ):
                param_hash = _strategy_hash(params)
                writer.writerow(
                    [
                        window.window_id,
                        "IS",
                        f"{is_metrics['net_profit_pct']:.2f}",
                        f"{is_metrics['max_drawdown_pct']:.2f}",
                        is_metrics["total_trades"],
                        param_hash[:12],
                    ]
                )
                writer.writerow(
                    [
                        window.window_id,
                        "OOS",
                        f"{oos_metrics['net_profit_pct']:.2f}",
                        f"{oos_metrics['max_drawdown_pct']:.2f}",
                        oos_metrics["total_trades"],
                        "",
                    ]
                )
    finally:
        if should_close:
            handle.close()
