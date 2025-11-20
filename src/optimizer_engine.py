"""Optimization engine for S_01 TrailingMA grid search."""
from __future__ import annotations

import bisect
import itertools
import math
import multiprocessing as mp
from dataclasses import dataclass
from decimal import Decimal
from typing import IO, Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from backtest_engine import (
    DEFAULT_ATR_PERIOD,
    atr,
    compute_max_drawdown,
    get_ma,
    load_data,
)

# Constants
CHUNK_SIZE = 2000

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
    ma_types_trend: List[str]
    ma_types_trail_long: List[str]
    ma_types_trail_short: List[str]
    lock_trail_types: bool = False
    risk_per_trade_pct: float = 2.0
    contract_size: float = 0.01
    commission_rate: float = 0.0005
    atr_period: int = DEFAULT_ATR_PERIOD
    worker_processes: int = 6
    filter_min_profit: bool = False
    min_profit_threshold: float = 0.0
    score_config: Optional[Dict[str, Any]] = None
    optimization_mode: str = "grid"


@dataclass
class OptimizationResult:
    """Represents a single optimization result row."""

    ma_type: str
    ma_length: int
    close_count_long: int
    close_count_short: int
    stop_long_atr: float
    stop_long_rr: float
    stop_long_lp: int
    stop_short_atr: float
    stop_short_rr: float
    stop_short_lp: int
    stop_long_max_pct: float
    stop_short_max_pct: float
    stop_long_max_days: int
    stop_short_max_days: int
    trail_rr_long: float
    trail_rr_short: float
    trail_ma_long_type: str
    trail_ma_long_length: int
    trail_ma_long_offset: float
    trail_ma_short_type: str
    trail_ma_short_length: int
    trail_ma_short_offset: float
    net_profit_pct: float
    max_drawdown_pct: float
    total_trades: int
    romad: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    profit_factor: Optional[float] = None
    ulcer_index: Optional[float] = None
    recovery_factor: Optional[float] = None
    consistency_score: Optional[float] = None
    score: float = 0.0


# Globals populated inside worker processes
_data_close: np.ndarray
_data_high: np.ndarray
_data_low: np.ndarray
_times: np.ndarray
_time_index: pd.DatetimeIndex
_ma_cache: Dict[Tuple[str, int], np.ndarray]
_lowest_cache: Dict[int, np.ndarray]
_highest_cache: Dict[int, np.ndarray]
_atr_values: np.ndarray
_time_in_range: np.ndarray
_month_arr: np.ndarray
_risk_per_trade_pct: float
_contract_size: float
_commission_rate: float


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


PARAMETER_MAP: Dict[str, Tuple[str, bool]] = {
    "maLength": ("ma_length", True),
    "closeCountLong": ("close_count_long", True),
    "closeCountShort": ("close_count_short", True),
    "stopLongX": ("stop_long_atr", False),
    "stopLongRR": ("stop_long_rr", False),
    "stopLongLP": ("stop_long_lp", True),
    "stopShortX": ("stop_short_atr", False),
    "stopShortRR": ("stop_short_rr", False),
    "stopShortLP": ("stop_short_lp", True),
    "stopLongMaxPct": ("stop_long_max_pct", False),
    "stopShortMaxPct": ("stop_short_max_pct", False),
    "stopLongMaxDays": ("stop_long_max_days", True),
    "stopShortMaxDays": ("stop_short_max_days", True),
    "trailRRLong": ("trail_rr_long", False),
    "trailRRShort": ("trail_rr_short", False),
    "trailLongLength": ("trail_ma_long_length", True),
    "trailLongOffset": ("trail_ma_long_offset", False),
    "trailShortLength": ("trail_ma_short_length", True),
    "trailShortOffset": ("trail_ma_short_offset", False),
}


def generate_parameter_grid(config: OptimizationConfig) -> List[Dict[str, Any]]:
    """Generate the cartesian product of all parameter combinations."""

    if not config.ma_types_trend or not config.ma_types_trail_long or not config.ma_types_trail_short:
        raise ValueError("At least one MA type must be selected in each group.")

    param_values: Dict[str, List[Any]] = {}
    for frontend_name, (internal_name, is_int) in PARAMETER_MAP.items():
        enabled = bool(config.enabled_params.get(frontend_name, False))
        if enabled:
            if frontend_name not in config.param_ranges:
                raise ValueError(f"Missing range for parameter '{frontend_name}'.")
            start, stop, step = config.param_ranges[frontend_name]
            values = _generate_numeric_sequence(start, stop, step, is_int)
        else:
            if frontend_name not in config.fixed_params:
                raise ValueError(f"Missing fixed value for parameter '{frontend_name}'.")
            value = config.fixed_params[frontend_name]
            values = [int(value) if is_int else float(value)]
        param_values[internal_name] = values

    trend_types = [ma.upper() for ma in config.ma_types_trend]
    trail_long_types = [ma.upper() for ma in config.ma_types_trail_long]
    trail_short_types = [ma.upper() for ma in config.ma_types_trail_short]

    param_names = list(param_values.keys())
    param_lists = [param_values[name] for name in param_names]

    combinations: List[Dict[str, Any]] = []

    if config.lock_trail_types:
        trail_short_set = set(trail_short_types)
        paired_trail_types = [trail for trail in trail_long_types if trail in trail_short_set]
        if not paired_trail_types:
            raise ValueError(
                "No overlapping trail MA types available when lock_trail_types is enabled."
            )

        for ma_type in trend_types:
            for paired_type in paired_trail_types:
                for values in itertools.product(*param_lists):
                    combo = dict(zip(param_names, values))
                    combo.update(
                        {
                            "ma_type": ma_type,
                            "trail_ma_long_type": paired_type,
                            "trail_ma_short_type": paired_type,
                        }
                    )
                    combinations.append(combo)
    else:
        for ma_type, trail_long_type, trail_short_type in itertools.product(
            trend_types, trail_long_types, trail_short_types
        ):
            for values in itertools.product(*param_lists):
                combo = dict(zip(param_names, values))
                combo.update(
                    {
                        "ma_type": ma_type,
                        "trail_ma_long_type": trail_long_type,
                        "trail_ma_short_type": trail_short_type,
                    }
                )
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


def _init_worker(
    df: pd.DataFrame,
    risk_per_trade_pct: float,
    contract_size: float,
    commission_rate: float,
    atr_period: int,
    ma_specs: Iterable[Tuple[str, int]],
    long_lp_values: Iterable[int],
    short_lp_values: Iterable[int],
    use_date_filter: bool,
    trade_start_idx: int,
) -> None:
    """Initialise worker globals."""

    global _data_close, _data_high, _data_low, _times, _time_index
    global _ma_cache, _lowest_cache, _highest_cache, _atr_values, _time_in_range
    global _month_arr
    global _risk_per_trade_pct, _contract_size, _commission_rate

    close_series = df["Close"].astype(float)
    high_series = df["High"].astype(float)
    low_series = df["Low"].astype(float)

    _data_close = close_series.to_numpy()
    _data_high = high_series.to_numpy()
    _data_low = low_series.to_numpy()
    _time_index = df.index
    _times = df.index.to_numpy()
    try:
        _month_arr = _time_index.month.to_numpy(dtype=np.int16)
    except AttributeError:  # pragma: no cover - defensive
        _month_arr = pd.to_datetime(_time_index).month.to_numpy(dtype=np.int16)

    _ma_cache = {}
    for ma_type, length in ma_specs:
        if length <= 0:
            _ma_cache[(ma_type, length)] = np.full_like(_data_close, np.nan)
            continue
        series = get_ma(close_series, ma_type, length, df["Volume"], high_series, low_series)
        _ma_cache[(ma_type, length)] = series.to_numpy(dtype=float)

    _atr_values = atr(high_series, low_series, close_series, atr_period).to_numpy(dtype=float)

    lp_values_long = set(long_lp_values) or {1}
    _lowest_cache = {}
    for length in lp_values_long:
        if length <= 0:
            _lowest_cache[length] = np.full_like(_data_low, np.nan)
        else:
            rolled = low_series.rolling(length, min_periods=1).min().to_numpy(dtype=float)
            _lowest_cache[length] = rolled

    lp_values_short = set(short_lp_values) or {1}
    _highest_cache = {}
    for length in lp_values_short:
        if length <= 0:
            _highest_cache[length] = np.full_like(_data_high, np.nan)
        else:
            rolled = high_series.rolling(length, min_periods=1).max().to_numpy(dtype=float)
            _highest_cache[length] = rolled

    # Use trade_start_idx to define trading zone
    if use_date_filter:
        _time_in_range = np.zeros(len(df), dtype=bool)
        _time_in_range[trade_start_idx:] = True
    else:
        _time_in_range = np.ones(len(df), dtype=bool)

    _risk_per_trade_pct = float(risk_per_trade_pct)
    _contract_size = float(contract_size)
    _commission_rate = float(commission_rate)


def _simulate_combination(params_dict: Dict[str, Any]) -> OptimizationResult:
    """Run a single simulation using pre-computed caches."""

    ma_type = params_dict["ma_type"]
    ma_length = int(params_dict["ma_length"])
    close_count_long = int(params_dict["close_count_long"])
    close_count_short = int(params_dict["close_count_short"])
    stop_long_atr = float(params_dict["stop_long_atr"])
    stop_long_rr = float(params_dict["stop_long_rr"])
    stop_long_lp = max(1, int(params_dict["stop_long_lp"]))
    stop_short_atr = float(params_dict["stop_short_atr"])
    stop_short_rr = float(params_dict["stop_short_rr"])
    stop_short_lp = max(1, int(params_dict["stop_short_lp"]))
    stop_long_max_pct = float(params_dict["stop_long_max_pct"])
    stop_short_max_pct = float(params_dict["stop_short_max_pct"])
    stop_long_max_days = int(params_dict["stop_long_max_days"])
    stop_short_max_days = int(params_dict["stop_short_max_days"])
    trail_rr_long = float(params_dict["trail_rr_long"])
    trail_rr_short = float(params_dict["trail_rr_short"])
    trail_ma_long_type = params_dict["trail_ma_long_type"]
    trail_ma_long_length = int(params_dict["trail_ma_long_length"])
    trail_ma_long_offset = float(params_dict["trail_ma_long_offset"])
    trail_ma_short_type = params_dict["trail_ma_short_type"]
    trail_ma_short_length = int(params_dict["trail_ma_short_length"])
    trail_ma_short_offset = float(params_dict["trail_ma_short_offset"])

    ma_series = _ma_cache.get((ma_type, ma_length))
    if ma_series is None:
        ma_series = np.full_like(_data_close, np.nan)
    trail_ma_long = _ma_cache.get((trail_ma_long_type, trail_ma_long_length))
    if trail_ma_long is None:
        trail_ma_long = np.full_like(_data_close, np.nan)
    trail_ma_short = _ma_cache.get((trail_ma_short_type, trail_ma_short_length))
    if trail_ma_short is None:
        trail_ma_short = np.full_like(_data_close, np.nan)

    lowest_long = _lowest_cache.get(stop_long_lp)
    if lowest_long is None:
        lowest_long = _lowest_cache[next(iter(_lowest_cache))]
    highest_short = _highest_cache.get(stop_short_lp)
    if highest_short is None:
        highest_short = _highest_cache[next(iter(_highest_cache))]

    equity = 100.0
    realized_equity = equity
    position = 0
    prev_position = 0
    position_size = 0.0
    entry_price = math.nan
    stop_price = math.nan
    target_price = math.nan
    trail_price_long = math.nan
    trail_price_short = math.nan
    trail_activated_long = False
    trail_activated_short = False
    entry_time_long: Optional[pd.Timestamp] = None
    entry_time_short: Optional[pd.Timestamp] = None
    entry_commission = 0.0

    counter_close_trend_long = 0
    counter_close_trend_short = 0
    counter_trade_long = 0
    counter_trade_short = 0

    trades_count = 0
    realized_curve: List[float] = []
    equity_curve: List[float] = []
    monthly_returns: List[float] = []
    month_start_equity: Optional[float] = None
    current_month: Optional[int] = None
    last_equity = realized_equity
    in_range_previous = False
    gross_profit = 0.0
    gross_loss = 0.0
    has_month_data = len(_month_arr) == len(_data_close)
    has_time_filter = len(_time_in_range) == len(_data_close)

    for i in range(len(_data_close)):
        c = float(_data_close[i])
        h = float(_data_high[i])
        l = float(_data_low[i])
        ma_value = float(ma_series[i]) if i < len(ma_series) else math.nan
        atr_value = float(_atr_values[i]) if i < len(_atr_values) else math.nan
        lowest_value = float(lowest_long[i]) if i < len(lowest_long) else math.nan
        highest_value = float(highest_short[i]) if i < len(highest_short) else math.nan
        trail_long_value = float(trail_ma_long[i]) if i < len(trail_ma_long) else math.nan
        trail_short_value = float(trail_ma_short[i]) if i < len(trail_ma_short) else math.nan

        if trail_ma_long_length > 0 and not math.isnan(trail_long_value):
            trail_long_value *= 1 + trail_ma_long_offset / 100.0
        if trail_ma_short_length > 0 and not math.isnan(trail_short_value):
            trail_short_value *= 1 + trail_ma_short_offset / 100.0

        if not math.isnan(ma_value):
            if c > ma_value:
                counter_close_trend_long += 1
                counter_close_trend_short = 0
            elif c < ma_value:
                counter_close_trend_short += 1
                counter_close_trend_long = 0
            else:
                counter_close_trend_long = 0
                counter_close_trend_short = 0

        if position > 0:
            counter_trade_long = 1
            counter_trade_short = 0
        elif position < 0:
            counter_trade_long = 0
            counter_trade_short = 1

        exit_price: Optional[float] = None
        current_time = _time_index[i]

        in_range = bool(_time_in_range[i]) if has_time_filter else True

        if has_month_data:
            month_value = int(_month_arr[i])
            if in_range:
                if not in_range_previous:
                    month_start_equity = last_equity
                    current_month = month_value
                elif current_month is not None and month_value != current_month:
                    if month_start_equity is not None and month_start_equity > 0:
                        monthly_returns.append((last_equity / month_start_equity - 1.0) * 100.0)
                    month_start_equity = last_equity
                    current_month = month_value
            elif in_range_previous and month_start_equity is not None and month_start_equity > 0:
                monthly_returns.append((last_equity / month_start_equity - 1.0) * 100.0)
                month_start_equity = None
                current_month = None

        if position > 0:
            if (
                not trail_activated_long
                and not math.isnan(entry_price)
                and not math.isnan(stop_price)
            ):
                activation_price = entry_price + (entry_price - stop_price) * trail_rr_long
                if h >= activation_price:
                    trail_activated_long = True
                    if math.isnan(trail_price_long):
                        trail_price_long = stop_price
            if not math.isnan(trail_price_long) and not math.isnan(trail_long_value):
                if math.isnan(trail_price_long) or trail_long_value > trail_price_long:
                    trail_price_long = trail_long_value
            if trail_activated_long:
                if not math.isnan(trail_price_long) and l <= trail_price_long:
                    exit_price = h if trail_price_long > h else trail_price_long
            else:
                if l <= stop_price:
                    exit_price = stop_price
                elif h >= target_price:
                    exit_price = target_price
            if exit_price is None and entry_time_long is not None and stop_long_max_days > 0:
                delta_days = int(((current_time - entry_time_long).total_seconds()) // 86400)
                if delta_days >= stop_long_max_days:
                    exit_price = c
            if exit_price is not None:
                gross_pnl = (exit_price - entry_price) * position_size
                exit_commission = exit_price * position_size * _commission_rate
                trade_pnl = gross_pnl - entry_commission - exit_commission
                if trade_pnl > 0:
                    gross_profit += trade_pnl
                elif trade_pnl < 0:
                    gross_loss += abs(trade_pnl)
                realized_equity += gross_pnl - exit_commission
                entry_commission = 0.0
                position = 0
                position_size = 0.0
                entry_price = math.nan
                stop_price = math.nan
                target_price = math.nan
                trail_price_long = math.nan
                trail_activated_long = False
                entry_time_long = None
                trades_count += 1

        elif position < 0:
            if (
                not trail_activated_short
                and not math.isnan(entry_price)
                and not math.isnan(stop_price)
            ):
                activation_price = entry_price - (stop_price - entry_price) * trail_rr_short
                if l <= activation_price:
                    trail_activated_short = True
                    if math.isnan(trail_price_short):
                        trail_price_short = stop_price
            if not math.isnan(trail_price_short) and not math.isnan(trail_short_value):
                if math.isnan(trail_price_short) or trail_short_value < trail_price_short:
                    trail_price_short = trail_short_value
            if trail_activated_short:
                if not math.isnan(trail_price_short) and h >= trail_price_short:
                    exit_price = l if trail_price_short < l else trail_price_short
            else:
                if h >= stop_price:
                    exit_price = stop_price
                elif l <= target_price:
                    exit_price = target_price
            if exit_price is None and entry_time_short is not None and stop_short_max_days > 0:
                delta_days = int(((current_time - entry_time_short).total_seconds()) // 86400)
                if delta_days >= stop_short_max_days:
                    exit_price = c
            if exit_price is not None:
                gross_pnl = (entry_price - exit_price) * position_size
                exit_commission = exit_price * position_size * _commission_rate
                trade_pnl = gross_pnl - entry_commission - exit_commission
                if trade_pnl > 0:
                    gross_profit += trade_pnl
                elif trade_pnl < 0:
                    gross_loss += abs(trade_pnl)
                realized_equity += gross_pnl - exit_commission
                entry_commission = 0.0
                position = 0
                position_size = 0.0
                entry_price = math.nan
                stop_price = math.nan
                target_price = math.nan
                trail_price_short = math.nan
                trail_activated_short = False
                entry_time_short = None
                trades_count += 1

        up_trend = counter_close_trend_long >= close_count_long and counter_trade_long == 0
        down_trend = counter_close_trend_short >= close_count_short and counter_trade_short == 0

        can_open_long = (
            up_trend
            and position == 0
            and prev_position == 0
            and _time_in_range[i]
            and not math.isnan(atr_value)
            and not math.isnan(lowest_value)
        )
        can_open_short = (
            down_trend
            and position == 0
            and prev_position == 0
            and _time_in_range[i]
            and not math.isnan(atr_value)
            and not math.isnan(highest_value)
        )

        if can_open_long:
            stop_size = atr_value * stop_long_atr
            long_stop_price = lowest_value - stop_size
            long_stop_distance = c - long_stop_price
            if long_stop_distance > 0:
                long_stop_pct = (long_stop_distance / c) * 100
                if long_stop_pct <= stop_long_max_pct or stop_long_max_pct <= 0:
                    risk_cash = realized_equity * (_risk_per_trade_pct / 100)
                    qty = risk_cash / long_stop_distance if long_stop_distance != 0 else 0.0
                    if _contract_size > 0:
                        qty = math.floor(qty / _contract_size) * _contract_size
                    if qty > 0:
                        position = 1
                        position_size = qty
                        entry_price = c
                        stop_price = long_stop_price
                        target_price = c + long_stop_distance * stop_long_rr
                        trail_price_long = long_stop_price
                        trail_activated_long = False
                        entry_time_long = current_time
                        entry_commission = entry_price * position_size * _commission_rate
                        realized_equity -= entry_commission

        if can_open_short and position == 0:
            stop_size = atr_value * stop_short_atr
            short_stop_price = highest_value + stop_size
            short_stop_distance = short_stop_price - c
            if short_stop_distance > 0:
                short_stop_pct = (short_stop_distance / c) * 100
                if short_stop_pct <= stop_short_max_pct or stop_short_max_pct <= 0:
                    risk_cash = realized_equity * (_risk_per_trade_pct / 100)
                    qty = risk_cash / short_stop_distance if short_stop_distance != 0 else 0.0
                    if _contract_size > 0:
                        qty = math.floor(qty / _contract_size) * _contract_size
                    if qty > 0:
                        position = -1
                        position_size = qty
                        entry_price = c
                        stop_price = short_stop_price
                        target_price = c - short_stop_distance * stop_short_rr
                        trail_price_short = short_stop_price
                        trail_activated_short = False
                        entry_time_short = current_time
                        entry_commission = entry_price * position_size * _commission_rate
                        realized_equity -= entry_commission

        realized_curve.append(realized_equity)
        current_equity = realized_equity
        if position > 0 and not math.isnan(entry_price):
            current_equity += (c - entry_price) * position_size
        elif position < 0 and not math.isnan(entry_price):
            current_equity += (entry_price - c) * position_size
        equity_curve.append(current_equity)
        last_equity = current_equity
        prev_position = position
        in_range_previous = in_range

    equity_series = pd.Series(realized_curve, index=_time_index[: len(realized_curve)])
    net_profit_pct = ((realized_equity - equity) / equity) * 100
    max_drawdown_pct = compute_max_drawdown(equity_series)

    if (
        has_month_data
        and month_start_equity is not None
        and month_start_equity > 0
        and in_range_previous
    ):
        monthly_returns.append((last_equity / month_start_equity - 1.0) * 100.0)

    profit_factor: Optional[float]
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else None
    else:
        if gross_profit > 0:
            profit_factor = 999.0
        else:
            profit_factor = 1.0 if trades_count > 0 else None

    if profit_factor is None:
        profit_factor = 1.0

    romad: Optional[float]
    recovery_factor: Optional[float]
    if net_profit_pct < 0:
        romad = 0.0
        recovery_factor = 0.0
    else:
        if abs(max_drawdown_pct) < 1e-9:
            romad = net_profit_pct * 100.0
            recovery_factor = net_profit_pct * 100.0
        elif max_drawdown_pct != 0:
            romad = net_profit_pct / abs(max_drawdown_pct)
            recovery_factor = net_profit_pct / abs(max_drawdown_pct)
        else:
            romad = 0.0
            recovery_factor = 0.0

    sharpe_ratio: Optional[float] = None
    if len(monthly_returns) >= 2:
        monthly_array = np.array(monthly_returns, dtype=float)
        if monthly_array.size >= 2:
            avg_return = float(np.mean(monthly_array))
            sd_return = float(np.std(monthly_array, ddof=0))
            if sd_return != 0:
                rfr = (0.02 * 100.0) / 12.0
                sharpe_ratio = (avg_return - rfr) / sd_return

    ulcer_index: Optional[float] = None
    if equity_curve:
        equity_array = np.asarray(equity_curve, dtype=float)
        if equity_array.size:
            running_max = np.maximum.accumulate(equity_array)
            with np.errstate(divide="ignore", invalid="ignore"):
                drawdowns = np.where(running_max > 0, equity_array / running_max - 1.0, 0.0)
            drawdown_squared_sum = float(np.square(drawdowns).sum())
            ulcer_index = math.sqrt(drawdown_squared_sum / equity_array.size) * 100.0

    consistency_score: Optional[float] = None
    if len(monthly_returns) >= 3:
        total_months = len(monthly_returns)
        profitable_months = sum(1 for value in monthly_returns if value is not None and value > 0)
        consistency_score = (profitable_months / total_months) * 100.0 if total_months > 0 else None

    return OptimizationResult(
        ma_type=ma_type,
        ma_length=ma_length,
        close_count_long=close_count_long,
        close_count_short=close_count_short,
        stop_long_atr=stop_long_atr,
        stop_long_rr=stop_long_rr,
        stop_long_lp=stop_long_lp,
        stop_short_atr=stop_short_atr,
        stop_short_rr=stop_short_rr,
        stop_short_lp=stop_short_lp,
        stop_long_max_pct=stop_long_max_pct,
        stop_short_max_pct=stop_short_max_pct,
        stop_long_max_days=stop_long_max_days,
        stop_short_max_days=stop_short_max_days,
        trail_rr_long=trail_rr_long,
        trail_rr_short=trail_rr_short,
        trail_ma_long_type=trail_ma_long_type,
        trail_ma_long_length=trail_ma_long_length,
        trail_ma_long_offset=trail_ma_long_offset,
        trail_ma_short_type=trail_ma_short_type,
        trail_ma_short_length=trail_ma_short_length,
        trail_ma_short_offset=trail_ma_short_offset,
        net_profit_pct=net_profit_pct,
        max_drawdown_pct=max_drawdown_pct,
        total_trades=trades_count,
        romad=romad,
        sharpe_ratio=sharpe_ratio,
        profit_factor=profit_factor,
        ulcer_index=ulcer_index,
        recovery_factor=recovery_factor,
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

    from backtest_engine import prepare_dataset_with_warmup, StrategyParams

    df = load_data(config.csv_file)
    combinations = generate_parameter_grid(config)
    total = len(combinations)
    if total == 0:
        raise ValueError("No parameter combinations generated for optimization.")

    ma_specs = set()
    long_lp_values = set()
    short_lp_values = set()
    for combo in combinations:
        ma_specs.add((combo["ma_type"], int(combo["ma_length"])))
        ma_specs.add((combo["trail_ma_long_type"], int(combo["trail_ma_long_length"])))
        ma_specs.add((combo["trail_ma_short_type"], int(combo["trail_ma_short_length"])))
        long_lp_values.add(max(1, int(combo["stop_long_lp"])))
        short_lp_values.add(max(1, int(combo["stop_short_lp"])))

    use_date_filter = bool(config.fixed_params.get("dateFilter", False))
    start = _parse_timestamp(config.fixed_params.get("start"))
    end = _parse_timestamp(config.fixed_params.get("end"))

    # Prepare dataset with warmup if date filtering is enabled
    trade_start_idx = 0
    if use_date_filter and (start is not None or end is not None):
        # Find the maximum MA length across all combinations to calculate warmup
        max_ma_length = 0
        for combo in combinations:
            max_ma_length = max(
                max_ma_length,
                int(combo["ma_length"]),
                int(combo["trail_ma_long_length"]),
                int(combo["trail_ma_short_length"])
            )

        # Create a dummy StrategyParams with max MA lengths for warmup calculation
        dummy_params = StrategyParams(
            use_backtester=True,
            use_date_filter=use_date_filter,
            start=start,
            end=end,
            ma_type="SMA",
            ma_length=max_ma_length,
            trail_ma_long_type="SMA",
            trail_ma_long_length=max_ma_length,
            trail_ma_short_type="SMA",
            trail_ma_short_length=max_ma_length,
            close_count_long=1,
            close_count_short=1,
            stop_long_atr=1.0,
            stop_long_rr=1.0,
            stop_long_lp=1,
            stop_short_atr=1.0,
            stop_short_rr=1.0,
            stop_short_lp=1,
            stop_long_max_pct=0.0,
            stop_short_max_pct=0.0,
            stop_long_max_days=0,
            stop_short_max_days=0,
            trail_rr_long=1.0,
            trail_rr_short=1.0,
            trail_ma_long_offset=0.0,
            trail_ma_short_offset=0.0,
            risk_per_trade_pct=config.risk_per_trade_pct,
            contract_size=config.contract_size,
            commission_rate=config.commission_rate,
            atr_period=config.atr_period
        )

        df, trade_start_idx = prepare_dataset_with_warmup(df, start, end, dummy_params)

    results: List[OptimizationResult] = []
    pool_args = (
        df,
        config.risk_per_trade_pct,
        config.contract_size,
        config.commission_rate,
        int(config.atr_period),
        list(ma_specs),
        list(long_lp_values),
        list(short_lp_values),
        use_date_filter,
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

    if config.score_config is None:
        score_config = DEFAULT_SCORE_CONFIG
    else:
        score_config = config.score_config
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

        return run_optuna_optimization(config, optuna_config)

    return run_grid_optimization(config)


CSV_COLUMN_SPECS: List[Tuple[str, Optional[str], str, Optional[str]]] = [
    ("MA Type", "maType", "ma_type", None),
    ("MA Length", "maLength", "ma_length", None),
    ("CC L", "closeCountLong", "close_count_long", None),
    ("CC S", "closeCountShort", "close_count_short", None),
    ("St L X", "stopLongX", "stop_long_atr", "float1"),
    ("Stop Long RR", "stopLongRR", "stop_long_rr", "float1"),
    ("St L LP", "stopLongLP", "stop_long_lp", None),
    ("St S X", "stopShortX", "stop_short_atr", "float1"),
    ("Stop Short RR", "stopShortRR", "stop_short_rr", "float1"),
    ("St S LP", "stopShortLP", "stop_short_lp", None),
    ("St L Max %", "stopLongMaxPct", "stop_long_max_pct", "float1"),
    ("St S Max %", "stopShortMaxPct", "stop_short_max_pct", "float1"),
    ("St L Max D", "stopLongMaxDays", "stop_long_max_days", None),
    ("St S Max D", "stopShortMaxDays", "stop_short_max_days", None),
    ("Trail RR Long", "trailRRLong", "trail_rr_long", "float1"),
    ("Trail RR Short", "trailRRShort", "trail_rr_short", "float1"),
    ("Tr L Type", "trailLongType", "trail_ma_long_type", None),
    ("Tr L Len", "trailLongLength", "trail_ma_long_length", None),
    ("Tr L Off", "trailLongOffset", "trail_ma_long_offset", "float1"),
    ("Tr S Type", "trailShortType", "trail_ma_short_type", None),
    ("Tr S Len", "trailShortLength", "trail_ma_short_length", None),
    ("Tr S Off", "trailShortOffset", "trail_ma_short_offset", "float1"),
    ("Net Profit%", None, "net_profit_pct", "percent"),
    ("Max DD%", None, "max_drawdown_pct", "percent"),
    ("Trades", None, "total_trades", None),
    ("Score", None, "score", "float"),
    ("RoMaD", None, "romad", "optional_float"),
    ("Sharpe", None, "sharpe_ratio", "optional_float"),
    ("PF", None, "profit_factor", "optional_float"),
    ("Ulcer", None, "ulcer_index", "optional_float"),
    ("Recover", None, "recovery_factor", "optional_float"),
    ("Consist", None, "consistency_score", "optional_float"),
]


def _format_csv_value(value: Any, formatter: Optional[str]) -> str:
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
    fixed_params: Union[Mapping[str, Any], Iterable[Tuple[str, Any]]],
    *,
    filter_min_profit: bool = False,
    min_profit_threshold: float = 0.0,
    optimization_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Export results to CSV format string with fixed parameter metadata.

    When ``filter_min_profit`` is enabled, rows whose ``net_profit_pct`` is
    strictly below ``min_profit_threshold`` are omitted from the export. The
    optimisation itself remains unaffected.
    """

    import io

    output = io.StringIO()

    if isinstance(fixed_params, Mapping):
        fixed_items = list(fixed_params.items())
    else:
        fixed_items = list(fixed_params)
    fixed_lookup = {name: value for name, value in fixed_items}

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

    filtered_columns = [
        spec for spec in CSV_COLUMN_SPECS if spec[1] is None or spec[1] not in fixed_lookup
    ]

    header_line = ",".join(column[0] for column in filtered_columns)
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
        for _, frontend_name, attr_name, formatter in filtered_columns:
            value = getattr(item, attr_name)
            row_values.append(_format_csv_value(value, formatter))
        output.write(",".join(row_values) + "\n")

    return output.getvalue()
