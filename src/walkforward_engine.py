"""
Walk-Forward Analysis Engine - Stage 1 MVP
Simple implementation focusing on core functionality.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from copy import deepcopy
import hashlib
import io
import json

import numpy as np
import pandas as pd

from backtest_engine import StrategyParams, run_strategy
from optimizer_engine import OptimizationConfig
from optuna_engine import OptunaConfig, run_optuna_optimization


@dataclass
class WFConfig:
    """Simple configuration for Walk-Forward"""

    num_windows: int = 5
    gap_bars: int = 100
    topk_per_window: int = 20

    # Fixed percentages (not configurable in Stage 1)
    wf_zone_pct: float = 80.0
    forward_pct: float = 20.0
    is_pct: float = 70.0
    oos_pct: float = 30.0
    warmup_bars: int = 1000


@dataclass
class WindowSplit:
    """One IS/OOS window"""

    window_id: int
    is_start: int
    is_end: int
    gap_start: int
    gap_end: int
    oos_start: int
    oos_end: int


@dataclass
class WindowResult:
    """Results from one window"""

    window_id: int
    top_params: List[Dict[str, Any]]  # Top-K from IS optimization
    oos_profits: List[float]  # OOS profit for each param
    oos_drawdowns: List[float]
    oos_trades: List[int]


@dataclass
class AggregatedResult:
    """Aggregated results for one param set"""

    param_id: str  # "EMA 45_abc123"
    params: Dict[str, Any]
    appearances: int  # How many windows
    avg_oos_profit: float
    oos_win_rate: float  # % windows profitable
    oos_profits: List[float]  # All OOS profits


@dataclass
class WFResult:
    """Complete Walk-Forward results"""

    config: WFConfig
    windows: List[WindowSplit]
    window_results: List[WindowResult]
    aggregated: List[AggregatedResult]
    forward_profits: List[float]  # Forward test results for Top-10
    forward_params: List[Dict[str, Any]]
    wf_zone_start: int  # Start bar of WF zone (after warmup)
    wf_zone_end: int  # End bar of WF zone
    forward_start: int  # Start bar of Forward Reserve
    forward_end: int  # End bar of Forward Reserve


class WalkForwardEngine:
    """Main engine for Walk-Forward Analysis"""

    def __init__(self, config: WFConfig, base_config_template: Dict[str, Any], optuna_settings: Dict[str, Any]):
        self.config = config
        self.base_config_template = deepcopy(base_config_template)
        self.optuna_settings = deepcopy(optuna_settings)

    def split_data(self, df: pd.DataFrame) -> Tuple[List[WindowSplit], int, int]:
        """
        Split data into WF windows + Forward Reserve

        Returns:
            windows: List of WindowSplit objects
            forward_start: Start index of forward period
            forward_end: End index of forward period
        """
        total_bars = len(df)
        warmup = self.config.warmup_bars

        # Minimum dataset size check (need space for warmup + at least one window)
        min_required_bars = warmup + 100  # Warmup + minimal window
        if total_bars < min_required_bars:
            raise ValueError(f"Dataset is too small for walk-forward analysis. "
                           f"Need at least {min_required_bars} bars, have {total_bars}.")

        # Calculate zones
        wf_zone_end = int(total_bars * (self.config.wf_zone_pct / 100))
        forward_start = wf_zone_end
        forward_end = total_bars

        # Available bars for WF windows (NO warmup reservation here)
        # Warmup will be added when preparing datasets for each window
        wf_available_start = 0
        wf_available_end = wf_zone_end
        wf_available_bars = wf_available_end - wf_available_start

        if wf_available_bars <= 0:
            raise ValueError("No data available for walk-forward windows.")

        total_gap = self.config.gap_bars * self.config.num_windows
        effective_bars = wf_available_bars - total_gap
        if effective_bars <= 0:
            raise ValueError("Gap configuration leaves no data for windows. Reduce gap or number of windows.")

        # Calculate window size
        window_total_bars = effective_bars // self.config.num_windows
        if window_total_bars <= 0:
            raise ValueError("Dataset is too small for the requested number of windows.")

        is_bars = int(window_total_bars * (self.config.is_pct / 100))
        oos_bars = int(window_total_bars * (self.config.oos_pct / 100))
        if is_bars <= 0 or oos_bars <= 0:
            raise ValueError("Window configuration produced zero-length IS or OOS segments.")

        windows: List[WindowSplit] = []
        stride = window_total_bars + self.config.gap_bars
        for i in range(self.config.num_windows):
            is_start = wf_available_start + i * stride
            is_end = is_start + is_bars
            gap_start = is_end
            gap_end = gap_start + self.config.gap_bars
            oos_start = gap_end
            oos_end = oos_start + oos_bars

            if oos_end > wf_available_end:
                break

            windows.append(
                WindowSplit(
                    window_id=i + 1,
                    is_start=is_start,
                    is_end=is_end,
                    gap_start=gap_start,
                    gap_end=gap_end,
                    oos_start=oos_start,
                    oos_end=oos_end,
                )
            )

        if not windows:
            raise ValueError("Failed to create any walk-forward windows. Provide more data or adjust settings.")

        return windows, forward_start, forward_end

    def run_wf_optimization(self, df: pd.DataFrame) -> WFResult:
        """
        Main function - runs complete Walk-Forward Analysis

        Steps:
        1. Split data
        2. For each window: optimize IS, test OOS
        3. Aggregate results
        4. Forward test
        5. Return results
        """
        print("Starting Walk-Forward Analysis...")

        # Step 1: Split data
        windows, fwd_start, fwd_end = self.split_data(df)
        print(f"Created {len(windows)} windows")
        print(f"Forward Reserve: bars {fwd_start} to {fwd_end}")

        # Step 2: Process each window
        window_results: List[WindowResult] = []
        for window in windows:
            print(f"\n--- Window {window.window_id}/{len(windows)} ---")

            # Prepare IS dataset with warmup using timestamps
            is_start_time = df.index[window.is_start]
            is_end_time = df.index[window.is_end - 1]

            print(f"IS optimization: bars {window.is_start} to {window.is_end} "
                  f"(dates {is_start_time.date()} to {is_end_time.date()})")

            # Run Optuna optimization on IS window
            # The optimization engine will handle warmup preparation internally
            optimization_results = self._run_optuna_on_window(
                df, is_start_time, is_end_time
            )

            topk = min(self.config.topk_per_window, len(optimization_results))
            top_results = optimization_results[:topk]
            top_params = [self._result_to_params(result) for result in top_results]

            print(f"Got {len(top_params)} top parameter sets")

            # Prepare OOS dataset with accumulated history (IS + gap for warmup)
            oos_start_time = df.index[window.oos_start]
            oos_end_time = df.index[window.oos_end - 1]

            print(f"OOS validation: bars {window.oos_start} to {window.oos_end} "
                  f"(dates {oos_start_time.date()} to {oos_end_time.date()})")

            oos_profits: List[float] = []
            oos_drawdowns: List[float] = []
            oos_trades: List[int] = []

            for params in top_params:
                from backtest_engine import prepare_dataset_with_warmup

                # Create params object for warmup calculation
                strategy_params = StrategyParams.from_dict(params)
                strategy_params.use_date_filter = True
                strategy_params.start = oos_start_time
                strategy_params.end = oos_end_time

                # Prepare dataset with proper warmup
                oos_df_prepared, trade_start_idx = prepare_dataset_with_warmup(
                    df, oos_start_time, oos_end_time, strategy_params
                )

                # Run strategy with trade_start_idx
                result = run_strategy(oos_df_prepared, strategy_params, trade_start_idx)

                oos_period_trades = [
                    trade
                    for trade in result.trades
                    if oos_start_time <= trade.entry_time <= oos_end_time
                ]

                if oos_period_trades:
                    oos_pnl = sum(trade.net_pnl for trade in oos_period_trades)
                    initial_equity = 10000
                    oos_profit_pct = (oos_pnl / initial_equity) * 100

                    equity_curve = [initial_equity]
                    running_equity = initial_equity
                    for trade in oos_period_trades:
                        running_equity += trade.net_pnl
                        equity_curve.append(running_equity)

                    peak = equity_curve[0]
                    max_dd = 0.0
                    for equity in equity_curve:
                        if equity > peak:
                            peak = equity
                        drawdown = ((equity - peak) / peak) * 100
                        if drawdown < max_dd:
                            max_dd = drawdown

                    oos_profits.append(oos_profit_pct)
                    oos_drawdowns.append(max_dd)
                    oos_trades.append(len(oos_period_trades))
                else:
                    oos_profits.append(0.0)
                    oos_drawdowns.append(0.0)
                    oos_trades.append(0)

            window_results.append(
                WindowResult(
                    window_id=window.window_id,
                    top_params=top_params,
                    oos_profits=oos_profits,
                    oos_drawdowns=oos_drawdowns,
                    oos_trades=oos_trades,
                )
            )

            print(
                f"OOS results: {len([profit for profit in oos_profits if profit > 0])}/{len(oos_profits)} profitable"
            )

        # Step 3: Aggregate results
        print("\n--- Aggregating Results ---")
        aggregated = self._aggregate_results(window_results)
        print(f"Found {len(aggregated)} unique parameter sets")

        # Step 4: Forward Test
        print("\n--- Forward Test ---")
        from backtest_engine import prepare_dataset_with_warmup

        forward_start_time = df.index[fwd_start]
        forward_end_time = df.index[fwd_end - 1]
        print(f"Forward test period: {forward_start_time.date()} to {forward_end_time.date()}")

        top10 = aggregated[:10]
        forward_profits: List[float] = []
        forward_params: List[Dict[str, Any]] = []

        for agg in top10:
            # Create params object for warmup calculation
            strategy_params = StrategyParams.from_dict(agg.params)
            strategy_params.use_date_filter = True
            strategy_params.start = forward_start_time
            strategy_params.end = forward_end_time

            # CRITICAL: Prepare dataset with warmup for forward test
            forward_df_prepared, trade_start_idx = prepare_dataset_with_warmup(
                df, forward_start_time, forward_end_time, strategy_params
            )

            if not forward_df_prepared.empty:
                result = run_strategy(forward_df_prepared, strategy_params, trade_start_idx)
                forward_profits.append(result.net_profit_pct)
            else:
                forward_profits.append(0.0)
            forward_params.append(agg.params)

        print(
            f"Forward Test complete: {len([profit for profit in forward_profits if profit > 0])}/{len(forward_profits)} profitable"
        )

        # Calculate WF zone boundaries
        wf_zone_start = windows[0].is_start if windows else self.config.warmup_bars
        wf_zone_end = fwd_start

        wf_result = WFResult(
            config=self.config,
            windows=windows,
            window_results=window_results,
            aggregated=aggregated,
            forward_profits=forward_profits,
            forward_params=forward_params,
            wf_zone_start=wf_zone_start,
            wf_zone_end=wf_zone_end,
            forward_start=fwd_start,
            forward_end=fwd_end,
        )

        return wf_result

    def _run_optuna_on_window(self, df: pd.DataFrame, start_time: pd.Timestamp, end_time: pd.Timestamp):
        # Create CSV buffer from full dataframe
        csv_buffer = self._dataframe_to_csv_buffer(df)

        # Update fixed params with date filter and start/end dates
        fixed_params = deepcopy(self.base_config_template["fixed_params"])
        fixed_params["dateFilter"] = True
        fixed_params["start"] = start_time.isoformat()
        fixed_params["end"] = end_time.isoformat()

        base_config = OptimizationConfig(
            csv_file=csv_buffer,
            enabled_params=deepcopy(self.base_config_template["enabled_params"]),
            param_ranges=deepcopy(self.base_config_template["param_ranges"]),
            fixed_params=fixed_params,
            ma_types_trend=list(self.base_config_template["ma_types_trend"]),
            ma_types_trail_long=list(self.base_config_template["ma_types_trail_long"]),
            ma_types_trail_short=list(self.base_config_template["ma_types_trail_short"]),
            lock_trail_types=bool(self.base_config_template["lock_trail_types"]),
            risk_per_trade_pct=float(self.base_config_template["risk_per_trade_pct"]),
            contract_size=float(self.base_config_template["contract_size"]),
            commission_rate=float(self.base_config_template["commission_rate"]),
            atr_period=int(self.base_config_template["atr_period"]),
            worker_processes=int(self.base_config_template["worker_processes"]),
            filter_min_profit=bool(self.base_config_template["filter_min_profit"]),
            min_profit_threshold=float(self.base_config_template["min_profit_threshold"]),
            score_config=deepcopy(self.base_config_template["score_config"]),
            optimization_mode="optuna",
        )

        optuna_cfg = OptunaConfig(
            target=self.optuna_settings["target"],
            budget_mode=self.optuna_settings["budget_mode"],
            n_trials=self.optuna_settings["n_trials"],
            time_limit=self.optuna_settings["time_limit"],
            convergence_patience=self.optuna_settings["convergence_patience"],
            enable_pruning=self.optuna_settings["enable_pruning"],
            sampler=self.optuna_settings["sampler"],
            pruner=self.optuna_settings["pruner"],
            warmup_trials=self.optuna_settings["warmup_trials"],
            save_study=self.optuna_settings["save_study"],
            study_name=None,
        )

        return run_optuna_optimization(base_config, optuna_cfg)

    def _dataframe_to_csv_buffer(self, df_window: pd.DataFrame) -> io.StringIO:
        buffer = io.StringIO()
        working_df = df_window.copy()
        working_df["time"] = working_df.index.view("int64") // 10**9
        ordered_cols = ["time", "Open", "High", "Low", "Close", "Volume"]
        working_df = working_df[ordered_cols]
        working_df.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer

    def _result_to_params(self, result) -> Dict[str, Any]:
        params = {
            "backtester": True,
            "dateFilter": False,
            "start": None,
            "end": None,
            "maType": result.ma_type,
            "maLength": int(result.ma_length),
            "closeCountLong": int(result.close_count_long),
            "closeCountShort": int(result.close_count_short),
            "stopLongX": float(result.stop_long_atr),
            "stopLongRR": float(result.stop_long_rr),
            "stopLongLP": int(result.stop_long_lp),
            "stopShortX": float(result.stop_short_atr),
            "stopShortRR": float(result.stop_short_rr),
            "stopShortLP": int(result.stop_short_lp),
            "stopLongMaxPct": float(result.stop_long_max_pct),
            "stopShortMaxPct": float(result.stop_short_max_pct),
            "stopLongMaxDays": int(result.stop_long_max_days),
            "stopShortMaxDays": int(result.stop_short_max_days),
            "trailRRLong": float(result.trail_rr_long),
            "trailRRShort": float(result.trail_rr_short),
            "trailLongType": result.trail_ma_long_type,
            "trailLongLength": int(result.trail_ma_long_length),
            "trailLongOffset": float(result.trail_ma_long_offset),
            "trailShortType": result.trail_ma_short_type,
            "trailShortLength": int(result.trail_ma_short_length),
            "trailShortOffset": float(result.trail_ma_short_offset),
            "riskPerTrade": float(self.base_config_template["risk_per_trade_pct"]),
            "contractSize": float(self.base_config_template["contract_size"]),
            "commissionRate": float(self.base_config_template["commission_rate"]),
            "atrPeriod": int(self.base_config_template["atr_period"]),
        }
        return params

    def _aggregate_results(self, window_results: List[WindowResult]) -> List[AggregatedResult]:
        """
        Aggregate results across windows

        Simple approach:
        - Group params that appear in multiple windows
        - Calculate average OOS profit
        - Calculate win rate
        - Sort by average OOS profit
        """
        param_map: Dict[str, Dict[str, Any]] = {}

        for window_result in window_results:
            for index, params in enumerate(window_result.top_params):
                if index >= len(window_result.oos_profits):
                    continue
                oos_profit = window_result.oos_profits[index]
                if oos_profit <= 0:
                    continue

                param_id = self._create_param_id(params)

                if param_id not in param_map:
                    param_map[param_id] = {
                        "params": params,
                        "appearances": 0,
                        "oos_profits": [],
                    }

                param_map[param_id]["appearances"] += 1
                param_map[param_id]["oos_profits"].append(oos_profit)

        aggregated: List[AggregatedResult] = []
        for param_id, data in param_map.items():
            oos_profits = data["oos_profits"]
            avg_profit = float(np.mean(oos_profits)) if oos_profits else 0.0
            win_rate = (
                len([profit for profit in oos_profits if profit > 0]) / len(oos_profits)
                if oos_profits
                else 0.0
            )

            aggregated.append(
                AggregatedResult(
                    param_id=param_id,
                    params=data["params"],
                    appearances=data["appearances"],
                    avg_oos_profit=avg_profit,
                    oos_win_rate=win_rate,
                    oos_profits=oos_profits,
                )
            )

        aggregated.sort(key=lambda item: item.avg_oos_profit, reverse=True)
        return aggregated

    def _create_param_id(self, params: Dict[str, Any]) -> str:
        """
        Create unique ID for param set
        Format: "MA_TYPE MA_LENGTH_hash"
        Example: "EMA 45_6d4ad0df"
        """
        ma_type = params.get("maType", "UNKNOWN")
        ma_length = params.get("maLength", 0)

        param_str = json.dumps(params, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]

        return f"{ma_type} {ma_length}_{param_hash}"


def export_wf_results_csv(result: WFResult, df: Optional[pd.DataFrame] = None) -> str:
    """Export Walk-Forward results to CSV string

    Args:
        result: WFResult object containing walk-forward analysis results
        df: Optional DataFrame with datetime index for converting bar numbers to dates
    """
    import csv
    from io import StringIO

    output = StringIO()
    writer = csv.writer(output)

    def bar_to_date(bar_idx: int) -> str:
        """Convert bar index to date string (YYYY-MM-DD)"""
        if df is not None and 0 <= bar_idx < len(df):
            timestamp = df.index[bar_idx]
            return timestamp.strftime('%Y-%m-%d')
        return str(bar_idx)

    writer.writerow(["=== WALK-FORWARD ANALYSIS - RESULTS ==="])
    writer.writerow([])

    writer.writerow(["=== SUMMARY ===", "", "", ""])
    writer.writerow(["Total Windows", len(result.windows), "Start", "End"])
    writer.writerow([
        "WF Zone",
        f"{result.config.wf_zone_pct}%",
        bar_to_date(result.wf_zone_start),
        bar_to_date(result.wf_zone_end - 1)
    ])
    writer.writerow([
        "Forward Reserve",
        f"{result.config.forward_pct}%",
        bar_to_date(result.forward_start),
        bar_to_date(result.forward_end - 1)
    ])
    writer.writerow(["Gap Between IS/OOS", f"{result.config.gap_bars} bars", "", ""])
    writer.writerow(["Top-K Per Window", result.config.topk_per_window])
    writer.writerow([])

    writer.writerow(["=== TOP 10 PARAMETER SETS (by Avg OOS Profit) ==="])
    writer.writerow(
        [
            "Rank",
            "Param ID",
            "Appearances",
            "Avg OOS Profit %",
            "OOS Win Rate",
            "Forward Profit %",
        ]
    )

    for rank, agg in enumerate(result.aggregated[:10], 1):
        forward_profit = (
            result.forward_profits[rank - 1]
            if rank <= len(result.forward_profits)
            else "N/A"
        )

        writer.writerow(
            [
                rank,
                agg.param_id,
                f"{agg.appearances}/{len(result.windows)}",
                f"{agg.avg_oos_profit:.2f}%",
                f"{agg.oos_win_rate * 100:.1f}%",
                f"{forward_profit:.2f}%"
                if isinstance(forward_profit, float)
                else forward_profit,
            ]
        )

    writer.writerow([])

    writer.writerow(["=== WINDOW DETAILS ==="])
    writer.writerow(
        [
            "Window",
            "IS Start",
            "IS End",
            "Gap Start",
            "Gap End",
            "OOS Start",
            "OOS End",
            "Top Param ID",
            "OOS Profit %",
        ]
    )

    helper_engine = WalkForwardEngine(result.config, {}, {})

    for window_result in result.window_results:
        window = result.windows[window_result.window_id - 1]

        if window_result.oos_profits:
            best_index = int(np.argmax(window_result.oos_profits))
            best_param = window_result.top_params[best_index]
            best_param_id = helper_engine._create_param_id(best_param)
            best_oos_profit = window_result.oos_profits[best_index]
        else:
            best_param_id = "N/A"
            best_oos_profit = 0.0

        writer.writerow(
            [
                window.window_id,
                bar_to_date(window.is_start),
                bar_to_date(window.is_end),
                bar_to_date(window.gap_start),
                bar_to_date(window.gap_end),
                bar_to_date(window.oos_start),
                bar_to_date(window.oos_end),
                best_param_id,
                f"{best_oos_profit:.2f}%",
            ]
        )

    writer.writerow([])

    writer.writerow(["=== FORWARD TEST RESULTS ==="])
    writer.writerow(["Rank", "Param ID", "Forward Profit %"])

    for rank, agg in enumerate(result.aggregated[:10], 1):
        if rank <= len(result.forward_profits):
            forward_profit = result.forward_profits[rank - 1]
            writer.writerow([rank, agg.param_id, f"{forward_profit:.2f}%"])

    writer.writerow([])

    writer.writerow(["=== DETAILED PARAMETERS FOR TOP 10 ==="])
    writer.writerow([])

    for rank, agg in enumerate(result.aggregated[:10], 1):
        writer.writerow([f"--- Rank #{rank}: {agg.param_id} ---"])
        params = agg.params
        writer.writerow(["Parameter", "Value"])
        writer.writerow(["MA Type", params.get("maType", "N/A")])
        writer.writerow(["MA Length", params.get("maLength", "N/A")])
        writer.writerow(["Close Count Long", params.get("closeCountLong", "N/A")])
        writer.writerow(["Close Count Short", params.get("closeCountShort", "N/A")])
        writer.writerow(["Stop Long ATR", params.get("stopLongX", "N/A")])
        writer.writerow(["Stop Long RR", params.get("stopLongRR", "N/A")])
        writer.writerow(["Stop Long LP", params.get("stopLongLP", "N/A")])
        writer.writerow(["Stop Short ATR", params.get("stopShortX", "N/A")])
        writer.writerow(["Stop Short RR", params.get("stopShortRR", "N/A")])
        writer.writerow(["Stop Short LP", params.get("stopShortLP", "N/A")])
        writer.writerow(["Stop Long Max %", params.get("stopLongMaxPct", "N/A")])
        writer.writerow(["Stop Short Max %", params.get("stopShortMaxPct", "N/A")])
        writer.writerow(["Stop Long Max Days", params.get("stopLongMaxDays", "N/A")])
        writer.writerow(["Stop Short Max Days", params.get("stopShortMaxDays", "N/A")])
        writer.writerow(["Trail RR Long", params.get("trailRRLong", "N/A")])
        writer.writerow(["Trail RR Short", params.get("trailRRShort", "N/A")])
        writer.writerow(["Trail MA Long Type", params.get("trailLongType", "N/A")])
        writer.writerow(["Trail MA Long Length", params.get("trailLongLength", "N/A")])
        writer.writerow(["Trail MA Long Offset", params.get("trailLongOffset", "N/A")])
        writer.writerow(["Trail MA Short Type", params.get("trailShortType", "N/A")])
        writer.writerow(["Trail MA Short Length", params.get("trailShortLength", "N/A")])
        writer.writerow(["Trail MA Short Offset", params.get("trailShortOffset", "N/A")])

        writer.writerow([])
        writer.writerow(["Performance Metrics", ""])
        writer.writerow(["Appearances", f"{agg.appearances}/{len(result.windows)}"])
        writer.writerow(["Avg OOS Profit %", f"{agg.avg_oos_profit:.2f}%"])
        writer.writerow(["OOS Win Rate", f"{agg.oos_win_rate * 100:.1f}%"])
        writer.writerow(
            [
                "OOS Profits by Window",
                ", ".join([f"{profit:.2f}%" for profit in agg.oos_profits]),
            ]
        )
        rank_index = rank - 1
        if rank_index < len(result.forward_profits):
            writer.writerow([
                "Forward Test Profit %",
                f"{result.forward_profits[rank_index]:.2f}%",
            ])

        writer.writerow([])
        writer.writerow([])

    return output.getvalue()
