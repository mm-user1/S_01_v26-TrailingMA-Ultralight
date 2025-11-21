"""Optimization engine for multi-strategy grid and Optuna search."""
from __future__ import annotations

import bisect
import itertools
import itertools
import logging
import math
import multiprocessing as mp
from dataclasses import dataclass, field
from decimal import Decimal
from typing import IO, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from backtest_engine import compute_max_drawdown, load_data
from indicators import DEFAULT_ATR_PERIOD, atr, get_ma
from strategy_registry import StrategyRegistry

# Constants
CHUNK_SIZE = 2000

logger = logging.getLogger(__name__)

SCORE_METRIC_ATTRS: Dict[str, str] = {
    "romad": "romad",
    "sharpe": "sharpe_ratio",
    "pf": "profit_factor",
    "ulcer": "ulcer_index",
    "recovery": "recovery_factor",
    "consistency": "consistency_score",
}

DEFAULT_SCORE_CONFIG: Dict[str, Any] = {
    "weights": {},
    "enabled_metrics": {},
    "invert_metrics": {},
    "normalization_method": "percentile",
    "filter_enabled": False,
    "min_score_threshold": 0.0,
}


@dataclass
class OptimizationConfig:
    """Configuration received from the optimizer form."""

    csv_file: IO[Any]
    enabled_params: Dict[str, bool]
    param_ranges: Dict[str, Tuple[float, float, float]]
    fixed_params: Dict[str, Any]
    strategy_id: str = "s01_trailing_ma"
    lock_trail_types: bool = False
    risk_per_trade_pct: float = 2.0
    contract_size: float = 0.01
    commission_rate: float = 0.0004
    atr_period: int = DEFAULT_ATR_PERIOD
    worker_processes: int = 6
    filter_min_profit: bool = False
    min_profit_threshold: float = 0.0
    score_config: Optional[Dict[str, Any]] = None
    optimization_mode: str = "grid"
    export_trades: bool = False


@dataclass
class OptimizationResult:
    """Represents a single optimization result row."""

    params: Dict[str, Any] = field(default_factory=dict)
    net_profit_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    total_trades: int = 0
    winning_trades: Optional[int] = None
    losing_trades: Optional[int] = None
    sharpe_ratio: Optional[float] = None
    profit_factor: Optional[float] = None
    romad: Optional[float] = None
    recovery_factor: Optional[float] = None
    ulcer_index: Optional[float] = None
    consistency_score: Optional[float] = None
    score: float = 0.0

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - defensive
        # Avoid recursion during unpickling when __dict__ may be incomplete.
        try:
            params = object.__getattribute__(self, "params")
        except AttributeError:
            raise AttributeError(item)

        if item in params:
            return params[item]
        raise AttributeError(item)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "net_profit_pct": self.net_profit_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "sharpe_ratio": self.sharpe_ratio,
            "profit_factor": self.profit_factor,
            "romad": self.romad,
            "recovery_factor": self.recovery_factor,
            "ulcer_index": self.ulcer_index,
            "consistency_score": self.consistency_score,
            "score": self.score,
        }
        data.update(self.params)
        return data


# Globals populated inside worker processes
_df: Optional[pd.DataFrame] = None
_strategy_class: Optional[type] = None
_cached_data: Dict[str, Any] = {}
_trade_start_idx: int = 0
_config: Optional[OptimizationConfig] = None


def _generate_numeric_sequence(
    start: float, stop: float, step: float, is_int: bool
) -> List[Any]:
    if step == 0:
        raise ValueError("Step must be non-zero for optimization ranges.")
    delta = abs(step)
    step_value = delta if start <= stop else -delta
    decimals = max(0, -Decimal(str(step)).normalize().as_tuple().exponent)
    epsilon = delta * 1e-9

    values: List[Any] = []
    index = 0

    while True:
        raw_value = start + index * step_value
        if step_value > 0:
            if raw_value > stop + epsilon:
                break
        else:
            if raw_value < stop - epsilon:
                break

        if is_int:
            values.append(int(round(raw_value)))
        else:
            rounded_value = round(raw_value, decimals)
            if rounded_value == 0:
                rounded_value = 0.0
            values.append(float(rounded_value))

        index += 1

    if not values:
        if is_int:
            values.append(int(round(start)))
        else:
            rounded_start = round(start, decimals)
            values.append(float(0.0 if rounded_start == 0 else rounded_start))
    return values


def generate_parameter_grid(
    config: OptimizationConfig, param_definitions: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Generate the cartesian product of all parameter combinations."""

    param_values: Dict[str, List[Any]] = {}

    for param_name, param_def in param_definitions.items():
        enabled = bool(config.enabled_params.get(param_name, False))

        if enabled:
            if param_name not in config.param_ranges:
                raise ValueError(
                    f"Parameter '{param_name}' is enabled but has no range specified"
                )

            start, stop, step = config.param_ranges[param_name]
            param_type = param_def.get("type")

            if param_type == "int":
                values = _generate_numeric_sequence(start, stop, step, is_int=True)
                values = [int(round(v)) for v in values]
            elif param_type == "float":
                values = _generate_numeric_sequence(start, stop, step, is_int=False)
                values = [float(v) for v in values]
            elif param_type == "categorical":
                if isinstance(config.param_ranges[param_name], (list, tuple)):
                    values = list(config.param_ranges[param_name])
                else:
                    values = param_def.get("choices", [param_def.get("default")])
            elif param_type == "bool":
                values = [True, False]
            else:
                raise ValueError(f"Unsupported parameter type: {param_type}")

            param_values[param_name] = values
        else:
            value = config.fixed_params.get(param_name, param_def.get("default"))
            param_values[param_name] = [value]

    param_names = list(param_values.keys())
    param_lists = [param_values[name] for name in param_names]

    combinations: List[Dict[str, Any]] = []
    for values in itertools.product(*param_lists):
        combo = dict(zip(param_names, values))
        combinations.append(combo)

    return combinations


def _parse_timestamp(value: Any) -> Optional[pd.Timestamp]:
    if value in (None, ""):
        return None
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):  # pragma: no cover - defensive
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _determine_warmup_bars(cache_requirements: Dict[str, Any]) -> int:
    max_length = 0
    for _, length in cache_requirements.get("ma_types_and_lengths", []):
        try:
            max_length = max(max_length, int(length))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            continue
    if max_length <= 0:
        return 0
    return max(500, int(max_length * 1.5))


def _prepare_dataset_for_optimization(
    df: pd.DataFrame, config: OptimizationConfig, cache_requirements: Dict[str, Any]
) -> Tuple[pd.DataFrame, int]:
    use_date_filter = bool(config.fixed_params.get("dateFilter", True))
    start_value = config.fixed_params.get("start") or config.fixed_params.get("startDate")
    end_value = config.fixed_params.get("end") or config.fixed_params.get("endDate")

    start = _parse_timestamp(start_value)
    end = _parse_timestamp(end_value)

    if not use_date_filter or (start is None and end is None):
        return df.copy(), 0

    required_warmup = _determine_warmup_bars(cache_requirements)
    times = df.index

    if start is not None:
        start_mask = times >= start
        if not start_mask.any():
            return df.iloc[0:0].copy(), 0
        start_idx = int(start_mask.argmax())
    else:
        start_idx = 0

    if end is not None:
        end_mask = times <= end
        if not end_mask.any():
            return df.iloc[0:0].copy(), 0
        end_idx = len(df)
    else:
        end_idx = len(df)

    warmup_start = max(0, start_idx - required_warmup)
    trimmed_df = df.iloc[warmup_start:end_idx].copy()
    trade_start_idx = start_idx - warmup_start

    return trimmed_df, trade_start_idx


def _init_worker(
    df: pd.DataFrame,
    cache_requirements: Dict[str, Any],
    strategy_class: type,
    config: OptimizationConfig,
    trade_start_idx: int,
) -> None:
    """Initialise worker globals."""

    global _df, _cached_data, _strategy_class, _config, _trade_start_idx

    _df = df
    _cached_data = {}
    _strategy_class = strategy_class
    _config = config
    _trade_start_idx = int(trade_start_idx)

    logger.info(f"Initializing worker for {strategy_class.STRATEGY_ID}")

    if not cache_requirements:
        return

    close_series = df["Close"].astype(float)
    high_series = df["High"].astype(float)
    low_series = df["Low"].astype(float)

    if "ma_types_and_lengths" in cache_requirements:
        _cached_data["ma_cache"] = {}
        for ma_type, length in cache_requirements.get("ma_types_and_lengths", []):
            ma_values = get_ma(
                close_series,
                ma_type,
                int(length),
                df["Volume"],
                high_series,
                low_series,
            ).to_numpy()
            _cached_data["ma_cache"][(ma_type, int(length))] = ma_values

    if cache_requirements.get("needs_atr") or cache_requirements.get("atr_periods"):
        _cached_data["atr"] = {}
        for period in cache_requirements.get("atr_periods", [config.atr_period]):
            atr_values = atr(high_series, low_series, close_series, int(period)).to_numpy()
            _cached_data["atr"][int(period)] = atr_values

    if "long_lp_values" in cache_requirements:
        _cached_data["lowest"] = {}
        for lp in cache_requirements.get("long_lp_values", []):
            if lp > 0:
                lowest_values = (
                    low_series.rolling(window=int(lp), min_periods=1).min().to_numpy()
                )
                _cached_data["lowest"][int(lp)] = lowest_values

    if "short_lp_values" in cache_requirements:
        _cached_data["highest"] = {}
        for lp in cache_requirements.get("short_lp_values", []):
            if lp > 0:
                highest_values = (
                    high_series.rolling(window=int(lp), min_periods=1).max().to_numpy()
                )
                _cached_data["highest"][int(lp)] = highest_values


def _simulate_combination(params_dict: Dict[str, Any]) -> OptimizationResult:
    """Run a single simulation using pre-computed caches."""

    if _strategy_class is None or _df is None:
        raise RuntimeError("Worker not initialized")

    try:
        strategy = _strategy_class(params_dict)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Failed to instantiate strategy: {exc}")
        return OptimizationResult(
            params=params_dict,
            net_profit_pct=0.0,
            max_drawdown_pct=100.0,
            total_trades=0,
        )

    if hasattr(strategy, "trade_start_idx"):
        strategy.trade_start_idx = _trade_start_idx

    try:
        result = strategy.simulate(_df, cached_data=_cached_data)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Simulation error: {exc}")
        return OptimizationResult(
            params=params_dict,
            net_profit_pct=0.0,
            max_drawdown_pct=100.0,
            total_trades=0,
        )

    net_profit_pct = float(result.get("net_profit_pct", 0.0))
    max_drawdown_pct = float(result.get("max_drawdown_pct", 0.0))
    total_trades = int(result.get("total_trades", 0))
    winning_trades = result.get("winning_trades")
    losing_trades = result.get("losing_trades")
    sharpe_ratio = result.get("sharpe_ratio")
    profit_factor = result.get("profit_factor")
    ulcer_index = result.get("ulcer_index")
    consistency_score = result.get("consistency_score") or result.get("consistency")

    if not math.isnan(max_drawdown_pct) and max_drawdown_pct != 0:
        romad = net_profit_pct / max_drawdown_pct
        recovery_factor = net_profit_pct / max_drawdown_pct
    else:
        romad = 0.0
        recovery_factor = 0.0

    return OptimizationResult(
        params=params_dict,
        net_profit_pct=net_profit_pct,
        max_drawdown_pct=max_drawdown_pct,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        sharpe_ratio=sharpe_ratio,
        profit_factor=profit_factor,
        romad=romad,
        recovery_factor=recovery_factor,
        ulcer_index=ulcer_index,
        consistency_score=consistency_score,
    )


def calculate_score(
    results: List[OptimizationResult],
    config: Optional[Dict[str, Any]],
) -> List[OptimizationResult]:
    """Calculate composite score for optimization results."""

    if not results:
        return results

    if config is None:
        config = {}

    normalized_config = DEFAULT_SCORE_CONFIG.copy()
    normalized_config.update({k: v for k, v in (config or {}).items() if v is not None})

    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y", "on"}:
                return True
            if lowered in {"false", "0", "no", "n", "off"}:
                return False
        return False

    weights = normalized_config.get("weights") or {}
    enabled_metrics = normalized_config.get("enabled_metrics") or {}
    invert_metrics = normalized_config.get("invert_metrics") or {}
    filter_enabled = _as_bool(normalized_config.get("filter_enabled", False))
    try:
        min_score_threshold = float(normalized_config.get("min_score_threshold", 0.0))
    except (TypeError, ValueError):
        min_score_threshold = 0.0
    min_score_threshold = max(0.0, min(100.0, min_score_threshold))

    normalization_method_raw = normalized_config.get("normalization_method", "percentile")
    normalization_method = (
        str(normalization_method_raw).strip().lower() if normalization_method_raw is not None else "percentile"
    )
    if normalization_method not in {"", "percentile"}:
        normalization_method = "percentile"

    metrics_to_normalize: List[str] = []
    for metric in SCORE_METRIC_ATTRS:
        if _as_bool(enabled_metrics.get(metric, False)):
            metrics_to_normalize.append(metric)

    normalized_values: Dict[str, Dict[int, float]] = {}
    for metric_name in metrics_to_normalize:
        attr_name = SCORE_METRIC_ATTRS[metric_name]
        metric_values = [
            getattr(item, attr_name)
            for item in results
            if getattr(item, attr_name) is not None
        ]
        if not metric_values:
            normalized_values[metric_name] = {id(item): 50.0 for item in results}
            continue
        sorted_vals = sorted(float(value) for value in metric_values)
        total = len(sorted_vals)
        normalized_values[metric_name] = {}
        invert = _as_bool(invert_metrics.get(metric_name, False))
        for item in results:
            value = getattr(item, attr_name)
            if value is None:
                rank = 50.0
            else:
                idx = bisect.bisect_left(sorted_vals, float(value))
                rank = (idx / total) * 100.0
                if invert:
                    rank = 100.0 - rank
            normalized_values[metric_name][id(item)] = rank

    for item in results:
        item.score = 0.0
        score_total = 0.0
        weight_total = 0.0
        for metric_name in metrics_to_normalize:
            weight_raw = weights.get(metric_name, 0.0)
            try:
                weight = float(weight_raw)
            except (TypeError, ValueError):
                weight = 0.0
            weight = max(0.0, min(1.0, weight))
            if weight <= 0:
                continue
            score_total += normalized_values[metric_name][id(item)] * weight
            weight_total += weight
        if weight_total > 0:
            item.score = score_total / weight_total

    if filter_enabled:
        results = [item for item in results if item.score >= min_score_threshold]

    return results


def run_grid_optimization(config: OptimizationConfig) -> List[OptimizationResult]:
    """Execute the grid search optimization."""
    strategy_class = StrategyRegistry.get_strategy_class(config.strategy_id)
    param_definitions = strategy_class.get_param_definitions()

    for name, definition in param_definitions.items():
        if not config.enabled_params.get(name, False) and name not in config.fixed_params:
            config.fixed_params[name] = definition.get("default")

    df = load_data(config.csv_file)
    combinations = generate_parameter_grid(config, param_definitions)
    total = len(combinations)
    if total == 0:
        raise ValueError("No parameter combinations generated for optimization.")

    cache_requirements = strategy_class.get_cache_requirements(combinations)
    df_prepared, trade_start_idx = _prepare_dataset_for_optimization(
        df, config, cache_requirements
    )

    results: List[OptimizationResult] = []
    pool_args = (
        df_prepared,
        cache_requirements,
        strategy_class,
        config,
        trade_start_idx,
    )
    processes = min(32, max(1, int(config.worker_processes)))
    with mp.Pool(processes=processes, initializer=_init_worker, initargs=pool_args) as pool:
        progress_iter = tqdm(
            range(0, total, CHUNK_SIZE),
            desc="Optimizing",
            total=total,
            unit="combo",
        )
        for start_idx in progress_iter:
            batch = combinations[start_idx : start_idx + CHUNK_SIZE]
            batch_results = pool.map(_simulate_combination, batch)
            results.extend(batch_results)
            progress_iter.update(len(batch) - 1)

    score_config = DEFAULT_SCORE_CONFIG if config.score_config is None else config.score_config
    results = calculate_score(results, score_config)

    if config.filter_min_profit:
        threshold = float(config.min_profit_threshold)
        results = [
            item for item in results if float(item.net_profit_pct) >= threshold
        ]

    results.sort(key=lambda item: item.net_profit_pct, reverse=True)
    return results


def run_optimization(config: OptimizationConfig) -> List[OptimizationResult]:
    """Router that delegates to grid or Optuna optimization engines."""

    StrategyRegistry.get_strategy_class  # noqa: B018 (trigger import for type checkers)
    logger.info(f"Optimizing strategy: {config.strategy_id}")

    strategy_class = StrategyRegistry.get_strategy_class(config.strategy_id)
    mode = getattr(config, "optimization_mode", "grid")
    if mode == "optuna":
        from optuna_engine import OptunaConfig, run_optuna_optimization

        optuna_config = OptunaConfig(
            target=getattr(config, "optuna_target", "score"),
            budget_mode=getattr(config, "optuna_budget_mode", "trials"),
            n_trials=int(getattr(config, "optuna_n_trials", 500) or 500),
            time_limit=int(getattr(config, "optuna_time_limit", 3600) or 3600),
            convergence_patience=int(
                getattr(config, "optuna_convergence", 50) or 50
            ),
            enable_pruning=bool(getattr(config, "optuna_enable_pruning", True)),
            sampler=getattr(config, "optuna_sampler", "tpe"),
            pruner=getattr(config, "optuna_pruner", "median"),
            warmup_trials=int(
                getattr(config, "optuna_warmup_trials", 20) or 20
            ),
            save_study=bool(getattr(config, "optuna_save_study", False)),
            study_name=getattr(config, "optuna_study_name", None),
        )

        return run_optuna_optimization(config, optuna_config, strategy_class)

    return run_grid_optimization(config)


def _format_csv_value(value: Any, formatter: Optional[str]) -> str:
    if value is None:
        return ""
    if formatter == "percent":
        return f"{float(value):.2f}%"
    if formatter == "float":
        return f"{float(value):.2f}"
    if formatter == "float1":
        return f"{float(value):.1f}"
    if formatter == "optional_float":
        if value is None:
            return ""
        return f"{float(value):.2f}"
    return str(value)


def _format_fixed_param_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def export_to_csv(
    results: List[OptimizationResult],
    config: OptimizationConfig,
    strategy_class: type,
    *,
    filter_min_profit: bool = False,
    min_profit_threshold: float = 0.0,
    optimization_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Export results to CSV format string with fixed parameter metadata."""

    import io

    param_definitions = strategy_class.get_param_definitions()

    fixed_items = []
    for name, definition in param_definitions.items():
        if bool(config.enabled_params.get(name, False)):
            continue
        value = config.fixed_params.get(name, definition.get("default"))
        fixed_items.append((name, value))

    output = io.StringIO()

    if optimization_metadata:
        output.write("Optuna Metadata\n")
        output.write(f"Method,{optimization_metadata.get('method', 'Grid Search')}\n")
        if optimization_metadata.get("method") == "Optuna":
            output.write(
                f"Target,{optimization_metadata.get('target', 'Composite Score')}\n"
            )
            output.write(
                f"Total Trials,{optimization_metadata.get('total_trials', 0)}\n"
            )
            output.write(
                f"Completed Trials,{optimization_metadata.get('completed_trials', 0)}\n"
            )
            output.write(
                f"Pruned Trials,{optimization_metadata.get('pruned_trials', 0)}\n"
            )
            output.write(
                f"Best Trial Number,{optimization_metadata.get('best_trial_number', 0)}\n"
            )
            output.write(
                f"Best Value,{optimization_metadata.get('best_value', 0)}\n"
            )
            output.write(
                f"Optimization Time,{optimization_metadata.get('optimization_time', '-')}\n"
            )
        else:
            output.write(
                f"Total Combinations,{optimization_metadata.get('total_combinations', 0)}\n"
            )
            output.write(
                f"Optimization Time,{optimization_metadata.get('optimization_time', '-')}\n"
            )
        output.write("\n")

    output.write("Fixed Parameters\n")
    output.write("Parameter Name,Value\n")
    for name, value in fixed_items:
        formatted_value = _format_fixed_param_value(value)
        output.write(f"{name},{formatted_value}\n")
    output.write("\n")

    metric_columns: List[Tuple[str, str]] = [
        ("Net Profit %", "net_profit_pct"),
        ("Max DD %", "max_drawdown_pct"),
        ("Trades", "total_trades"),
        ("Score", "score"),
        ("RoMaD", "romad"),
        ("Sharpe", "sharpe_ratio"),
        ("PF", "profit_factor"),
        ("Ulcer", "ulcer_index"),
        ("Recover", "recovery_factor"),
        ("Consist", "consistency_score"),
    ]

    param_columns: List[Tuple[str, Optional[str]]] = []
    for name, definition in param_definitions.items():
        if any(name == fixed_name for fixed_name, _ in fixed_items):
            continue
        formatter = None
        if definition.get("type") == "float":
            formatter = "float1"
        param_columns.append((name, formatter))

    header_line = ",".join([name for name, _ in param_columns] + [label for label, _ in metric_columns])
    output.write(header_line + "\n")

    if filter_min_profit:
        threshold = float(min_profit_threshold)
        filtered_results = [
            item for item in results if float(item.net_profit_pct) >= threshold
        ]
    else:
        filtered_results = results

    for item in filtered_results:
        row_values = []
        for param_name, formatter in param_columns:
            value = item.params.get(param_name)
            row_values.append(_format_csv_value(value, formatter))
        for _, attr_name in metric_columns:
            value = getattr(item, attr_name, "")
            formatter = None
            if attr_name in {"net_profit_pct", "max_drawdown_pct"}:
                formatter = "percent"
            elif attr_name in {"score", "romad", "sharpe_ratio", "profit_factor", "ulcer_index", "recovery_factor", "consistency_score"}:
                formatter = "float"
            row_values.append(_format_csv_value(value, formatter))
        output.write(",".join(row_values) + "\n")

    return output.getvalue()
