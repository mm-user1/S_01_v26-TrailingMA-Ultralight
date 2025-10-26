"""Optimization engine for S_01 TrailingMA grid search."""
from __future__ import annotations

import itertools
import math
import multiprocessing as mp
from dataclasses import dataclass
from typing import IO, Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest_engine import (
    DEFAULT_ATR_PERIOD,
    atr,
    compute_max_drawdown,
    get_ma,
    load_data,
)

# Constants
CHUNK_SIZE = 2000


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
    risk_per_trade_pct: float = 2.0
    contract_size: float = 0.01
    commission_rate: float = 0.0005
    atr_period: int = DEFAULT_ATR_PERIOD
    worker_processes: int = 6


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
    values: List[Any] = []
    current = start

    def should_continue(val: float) -> bool:
        if step_value > 0:
            return val <= stop + delta * 1e-9
        return val >= stop - delta * 1e-9

    while should_continue(current):
        values.append(int(round(current)) if is_int else float(current))
        current += step_value

    if not values:
        values.append(int(round(start)) if is_int else float(start))
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

    ma_type_combos = list(
        itertools.product(
            [ma.upper() for ma in config.ma_types_trend],
            [ma.upper() for ma in config.ma_types_trail_long],
            [ma.upper() for ma in config.ma_types_trail_short],
        )
    )

    param_names = list(param_values.keys())
    param_lists = [param_values[name] for name in param_names]

    combinations: List[Dict[str, Any]] = []
    for ma_type, trail_long_type, trail_short_type in ma_type_combos:
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
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> None:
    """Initialise worker globals."""

    global _data_close, _data_high, _data_low, _times, _time_index
    global _ma_cache, _lowest_cache, _highest_cache, _atr_values, _time_in_range
    global _risk_per_trade_pct, _contract_size, _commission_rate

    close_series = df["Close"].astype(float)
    high_series = df["High"].astype(float)
    low_series = df["Low"].astype(float)

    _data_close = close_series.to_numpy()
    _data_high = high_series.to_numpy()
    _data_low = low_series.to_numpy()
    _time_index = df.index
    _times = df.index.to_numpy()

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

    if use_date_filter and (start is not None or end is not None):
        mask = np.ones(len(df), dtype=bool)
        if start is not None:
            mask &= _time_index >= start
        if end is not None:
            mask &= _time_index <= end
        _time_in_range = mask
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
                    exit_price = trail_price_long
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
                    exit_price = trail_price_short
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
        prev_position = position

    equity_series = pd.Series(realized_curve, index=_time_index[: len(realized_curve)])
    net_profit_pct = ((realized_equity - equity) / equity) * 100
    max_drawdown_pct = compute_max_drawdown(equity_series)

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
    )


def run_optimization(
    config: OptimizationConfig, progress_callback: Optional[Any] = None
) -> List[OptimizationResult]:
    """Execute the grid search optimization."""

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

    if progress_callback:
        progress_callback(0, total)

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
        start,
        end,
    )
    processes = min(32, max(1, int(config.worker_processes)))
    with mp.Pool(processes=processes, initializer=_init_worker, initargs=pool_args) as pool:
        for start_idx in range(0, total, CHUNK_SIZE):
            batch = combinations[start_idx : start_idx + CHUNK_SIZE]
            batch_results = pool.map(_simulate_combination, batch)
            results.extend(batch_results)
            if progress_callback:
                progress_callback(min(start_idx + len(batch), total), total)

    results.sort(key=lambda item: item.net_profit_pct, reverse=True)
    return results


CSV_COLUMN_SPECS: List[Tuple[str, Optional[str], str, Optional[str]]] = [
    ("MA Type", None, "ma_type", None),
    ("MA Length", "maLength", "ma_length", None),
    ("Close Count Long", "closeCountLong", "close_count_long", None),
    ("Close Count Short", "closeCountShort", "close_count_short", None),
    ("Stop Long X", "stopLongX", "stop_long_atr", None),
    ("Stop Long RR", "stopLongRR", "stop_long_rr", None),
    ("Stop Long LP", "stopLongLP", "stop_long_lp", None),
    ("Stop Short X", "stopShortX", "stop_short_atr", None),
    ("Stop Short RR", "stopShortRR", "stop_short_rr", None),
    ("Stop Short LP", "stopShortLP", "stop_short_lp", None),
    ("Stop Long Max %", "stopLongMaxPct", "stop_long_max_pct", None),
    ("Stop Short Max %", "stopShortMaxPct", "stop_short_max_pct", None),
    ("Stop Long Max Days", "stopLongMaxDays", "stop_long_max_days", None),
    ("Stop Short Max Days", "stopShortMaxDays", "stop_short_max_days", None),
    ("Trail RR Long", "trailRRLong", "trail_rr_long", None),
    ("Trail RR Short", "trailRRShort", "trail_rr_short", None),
    ("Trail MA Long Type", None, "trail_ma_long_type", None),
    ("Trail MA Long Length", "trailLongLength", "trail_ma_long_length", None),
    ("Trail MA Long Offset", "trailLongOffset", "trail_ma_long_offset", None),
    ("Trail MA Short Type", None, "trail_ma_short_type", None),
    ("Trail MA Short Length", "trailShortLength", "trail_ma_short_length", None),
    ("Trail MA Short Offset", "trailShortOffset", "trail_ma_short_offset", None),
    ("Net Profit%", None, "net_profit_pct", "percent"),
    ("Max Drawdown%", None, "max_drawdown_pct", "percent"),
    ("Total Trades", None, "total_trades", None),
]


def _format_csv_value(value: Any, formatter: Optional[str]) -> str:
    if formatter == "percent":
        return f"{float(value):.2f}%"
    return str(value)


def _format_fixed_param_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def export_to_csv(
    results: List[OptimizationResult], fixed_params: Dict[str, Any]
) -> str:
    """Export results to CSV format string with fixed parameter metadata."""

    import io

    output = io.StringIO()

    fixed_items = [
        f"{name}={_format_fixed_param_value(fixed_params[name])}"
        for name in fixed_params
    ]
    output.write("Fixed Parameters:")
    if fixed_items:
        output.write("," + ",".join(fixed_items))
    output.write("\n")

    filtered_columns = [
        spec for spec in CSV_COLUMN_SPECS if spec[1] is None or spec[1] not in fixed_params
    ]

    header_line = ",".join(column[0] for column in filtered_columns)
    output.write(header_line + "\n")

    for item in results:
        row_values = []
        for _, frontend_name, attr_name, formatter in filtered_columns:
            value = getattr(item, attr_name)
            row_values.append(_format_csv_value(value, formatter))
        output.write(",".join(row_values) + "\n")

    return output.getvalue()
