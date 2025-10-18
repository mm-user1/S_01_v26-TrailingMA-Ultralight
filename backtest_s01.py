import argparse
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from backtesting import _stats

FACTOR_T3 = 0.7
FAST_KAMA = 2
SLOW_KAMA = 30
START_DATE = pd.Timestamp('2025-04-01', tz='UTC')
END_DATE = pd.Timestamp('2025-09-01', tz='UTC')
DEFAULT_CONTRACT_SIZE = 0.01
COMMISSION_RATE = 0.0005
MA_TYPE_OPTIONS = [
    "EMA",
    "SMA",
    "HMA",
    "WMA",
    "ALMA",
    "KAMA",
    "TMA",
    "T3",
    "DEMA",
    "VWMA",
    "VWAP",
]


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()


def wma(series: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length + 1, dtype=float)
    return series.rolling(length, min_periods=length).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def hma(series: pd.Series, length: int) -> pd.Series:
    if length <= 0:
        return pd.Series(np.nan, index=series.index)
    half_length = max(1, length // 2)
    sqrt_length = max(1, int(math.sqrt(length)))
    return wma(2 * wma(series, half_length) - wma(series, length), sqrt_length)


def vwma(series: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    weighted = (series * volume).rolling(length, min_periods=length).sum()
    vol_sum = volume.rolling(length, min_periods=length).sum()
    return weighted / vol_sum


def alma(series: pd.Series, length: int, offset: float = 0.85, sigma: float = 6) -> pd.Series:
    if length <= 0:
        return pd.Series(np.nan, index=series.index)
    m = offset * (length - 1)
    s = length / sigma

    def _alma(values: np.ndarray) -> float:
        weights = np.exp(-((np.arange(len(values)) - m) ** 2) / (2 * s * s))
        weights /= weights.sum()
        return np.dot(weights, values)

    return series.rolling(length, min_periods=length).apply(_alma, raw=True)


def dema(series: pd.Series, length: int) -> pd.Series:
    e1 = ema(series, length)
    e2 = ema(e1, length)
    return 2 * e1 - e2


def kama(series: pd.Series, length: int) -> pd.Series:
    if length <= 0:
        return pd.Series(np.nan, index=series.index)
    mom = series.diff(length).abs()
    volatility = series.diff().abs().rolling(length, min_periods=length).sum()
    er = pd.Series(np.where(volatility != 0, mom / volatility, 0), index=series.index)
    fast_alpha = 2 / (FAST_KAMA + 1)
    slow_alpha = 2 / (SLOW_KAMA + 1)
    alpha = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2
    kama_values = np.empty(len(series))
    kama_values[:] = np.nan
    for i in range(len(series)):
        price = series.iat[i]
        if np.isnan(price):
            continue
        a = alpha.iat[i]
        if np.isnan(a):
            kama_values[i] = price if i == 0 else kama_values[i - 1]
            continue
        prev = kama_values[i - 1] if i > 0 and not np.isnan(kama_values[i - 1]) else (series.iat[i - 1] if i > 0 else price)
        kama_values[i] = a * price + (1 - a) * prev
    return pd.Series(kama_values, index=series.index)


def tma(series: pd.Series, length: int) -> pd.Series:
    if length <= 0:
        return pd.Series(np.nan, index=series.index)
    first = sma(series, math.ceil(length / 2))
    return sma(first, math.floor(length / 2) + 1)


def gd(series: pd.Series, length: int) -> pd.Series:
    ema1 = ema(series, length)
    ema2 = ema(ema1, length)
    return ema1 * (1 + FACTOR_T3) - ema2 * FACTOR_T3


def t3(series: pd.Series, length: int) -> pd.Series:
    if length <= 0:
        return pd.Series(np.nan, index=series.index)
    return gd(gd(gd(series, length), length), length)


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    typical = (high + low + close) / 3
    tp_vol = typical * volume
    cumulative = tp_vol.cumsum()
    cumulative_vol = volume.cumsum()
    return cumulative / cumulative_vol


def get_ma(series: pd.Series,
           ma_type: str,
           length: int,
           volume: Optional[pd.Series] = None,
           high: Optional[pd.Series] = None,
           low: Optional[pd.Series] = None) -> pd.Series:
    if length == 0 and ma_type.upper() != "VWAP":
        return pd.Series(np.nan, index=series.index)
    ma_type = ma_type.upper()
    if ma_type == "SMA":
        return sma(series, length)
    if ma_type == "EMA":
        return ema(series, length)
    if ma_type == "HMA":
        return hma(series, length)
    if ma_type == "WMA":
        return wma(series, length)
    if ma_type == "VWMA":
        if volume is None:
            raise ValueError("Volume data required for VWMA")
        return vwma(series, volume, length)
    if ma_type == "VWAP":
        if any(v is None for v in (high, low, volume)):
            raise ValueError("High, Low, Volume required for VWAP")
        return vwap(high, low, series, volume)
    if ma_type == "ALMA":
        return alma(series, length)
    if ma_type == "DEMA":
        return dema(series, length)
    if ma_type == "KAMA":
        return kama(series, length)
    if ma_type == "TMA":
        return tma(series, length)
    if ma_type == "T3":
        return t3(series, length)
    return ema(series, length)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


@dataclass
class TradeRecord:
    direction: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    size: float
    net_pnl: float


@dataclass
class StrategyParams:
    date_filter: bool = True
    start_date: pd.Timestamp = START_DATE
    end_date: pd.Timestamp = END_DATE
    t_ma_type: str = "EMA"
    ma_length3: int = 45
    close_count_trend_long: int = 7
    close_count_trend_short: int = 5
    stop_multiplier_long: float = 2.0
    rr_long: float = 3.0
    lp_long: int = 2
    stop_multiplier_short: float = 2.0
    rr_short: float = 3.0
    lp_short: int = 2
    long_stop_pct_filter_size: float = 3.0
    short_stop_pct_filter_size: float = 3.0
    long_stop_days_filter_size: int = 2
    short_stop_days_filter_size: int = 4
    trail_rr_long: float = 1.0
    trail_rr_short: float = 1.0
    trail_ma_type_long: str = "SMA"
    trail_ma_length_long: int = 160
    trail_ma_offset_long: float = -1.0
    trail_ma_type_short: str = "SMA"
    trail_ma_length_short: int = 160
    trail_ma_offset_short: float = 1.0
    risk_per_trade: float = 2.0
    contract_size: float = DEFAULT_CONTRACT_SIZE


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.set_index('time').sort_index()
    return df[['open', 'high', 'low', 'close', 'Volume']].rename(columns=str.capitalize)


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    equity_curve = equity_curve.ffill()
    drawdown = 1 - equity_curve / equity_curve.cummax()
    _, peak_dd = _stats.compute_drawdown_duration_peaks(drawdown)
    if peak_dd.isna().all():
        return 0.0
    return peak_dd.max() * 100


def parse_date_value(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    else:
        ts = ts.tz_convert('UTC')
    return ts


def run_strategy(
    df: pd.DataFrame,
    params: StrategyParams,
    ma_type_override: Optional[str] = None,
) -> Tuple[float, float, int]:
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    ma_type = (ma_type_override or params.t_ma_type).upper()
    ma3 = get_ma(
        close,
        ma_type,
        params.ma_length3,
        volume=volume,
        high=high,
        low=low,
    )
    atr14 = atr(high, low, close, 14)
    lp_long = max(1, params.lp_long)
    lp_short = max(1, params.lp_short)
    lowest_long = low.rolling(lp_long, min_periods=1).min()
    highest_short = high.rolling(lp_short, min_periods=1).max()

    trail_ma_long_base = get_ma(
        close,
        params.trail_ma_type_long,
        params.trail_ma_length_long,
        volume=volume,
        high=high,
        low=low,
    )
    if params.trail_ma_offset_long != 0:
        trail_ma_long = trail_ma_long_base * (1 + params.trail_ma_offset_long / 100.0)
    else:
        trail_ma_long = trail_ma_long_base

    trail_ma_short_base = get_ma(
        close,
        params.trail_ma_type_short,
        params.trail_ma_length_short,
        volume=volume,
        high=high,
        low=low,
    )
    if params.trail_ma_offset_short != 0:
        trail_ma_short = trail_ma_short_base * (1 + params.trail_ma_offset_short / 100.0)
    else:
        trail_ma_short = trail_ma_short_base

    times = df.index
    if params.date_filter:
        time_in_range = (times >= params.start_date) & (times <= params.end_date)
    else:
        time_in_range = np.ones(len(times), dtype=bool)

    equity = 100.0
    realized_equity = equity
    position = 0  # 0 flat, 1 long, -1 short
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

    trades: List[TradeRecord] = []
    equity_curve: List[float] = []
    realized_curve: List[float] = []

    for i in range(len(df)):
        time = times[i]
        c = close.iat[i]
        h = high.iat[i]
        l = low.iat[i]
        ma_value = ma3.iat[i]
        atr_value = atr14.iat[i]
        lowest_value = lowest_long.iat[i]
        highest_value = highest_short.iat[i]
        trail_long_value = trail_ma_long.iat[i]
        trail_short_value = trail_ma_short.iat[i]

        if not np.isnan(ma_value):
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

        exit_price = None
        if position > 0:
            if not trail_activated_long and not math.isnan(entry_price) and not math.isnan(stop_price):
                activation_price = entry_price + (entry_price - stop_price) * params.trail_rr_long
                if h >= activation_price:
                    trail_activated_long = True
                    if math.isnan(trail_price_long):
                        trail_price_long = stop_price
            if not math.isnan(trail_price_long) and not np.isnan(trail_long_value):
                if np.isnan(trail_price_long) or trail_long_value > trail_price_long:
                    trail_price_long = trail_long_value
            if trail_activated_long:
                if not math.isnan(trail_price_long) and l <= trail_price_long:
                    exit_price = trail_price_long
            else:
                if l <= stop_price:
                    exit_price = stop_price
                elif h >= target_price:
                    exit_price = target_price
            if exit_price is None and entry_time_long is not None:
                days_in_trade = int(math.floor((time - entry_time_long).total_seconds() / 86400))
                if days_in_trade >= params.long_stop_days_filter_size:
                    exit_price = c
            if exit_price is not None:
                gross_pnl = (exit_price - entry_price) * position_size
                exit_commission = exit_price * position_size * COMMISSION_RATE
                realized_equity += gross_pnl - exit_commission
                trades.append(TradeRecord(
                    direction='long',
                    entry_time=entry_time_long,
                    exit_time=time,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    size=position_size,
                    net_pnl=gross_pnl - exit_commission - entry_commission,
                ))
                position = 0
                position_size = 0.0
                entry_price = math.nan
                stop_price = math.nan
                target_price = math.nan
                trail_price_long = math.nan
                trail_activated_long = False
                entry_time_long = None
                entry_commission = 0.0

        elif position < 0:
            if not trail_activated_short and not math.isnan(entry_price) and not math.isnan(stop_price):
                activation_price = entry_price - (stop_price - entry_price) * params.trail_rr_short
                if l <= activation_price:
                    trail_activated_short = True
                    if math.isnan(trail_price_short):
                        trail_price_short = stop_price
            if not math.isnan(trail_price_short) and not np.isnan(trail_short_value):
                if np.isnan(trail_price_short) or trail_short_value < trail_price_short:
                    trail_price_short = trail_short_value
            if trail_activated_short:
                if not math.isnan(trail_price_short) and h >= trail_price_short:
                    exit_price = trail_price_short
            else:
                if h >= stop_price:
                    exit_price = stop_price
                elif l <= target_price:
                    exit_price = target_price
            if exit_price is None and entry_time_short is not None:
                days_in_trade = int(math.floor((time - entry_time_short).total_seconds() / 86400))
                if days_in_trade >= params.short_stop_days_filter_size:
                    exit_price = c
            if exit_price is not None:
                gross_pnl = (entry_price - exit_price) * position_size
                exit_commission = exit_price * position_size * COMMISSION_RATE
                realized_equity += gross_pnl - exit_commission
                trades.append(TradeRecord(
                    direction='short',
                    entry_time=entry_time_short,
                    exit_time=time,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    size=position_size,
                    net_pnl=gross_pnl - exit_commission - entry_commission,
                ))
                position = 0
                position_size = 0.0
                entry_price = math.nan
                stop_price = math.nan
                target_price = math.nan
                trail_price_short = math.nan
                trail_activated_short = False
                entry_time_short = None
                entry_commission = 0.0

        up_trend = (
            counter_close_trend_long >= params.close_count_trend_long
            and counter_trade_long == 0
        )
        down_trend = (
            counter_close_trend_short >= params.close_count_trend_short
            and counter_trade_short == 0
        )

        can_open_long = (
            up_trend
            and position == 0
            and prev_position == 0
            and time_in_range[i]
            and not np.isnan(atr_value)
            and not np.isnan(lowest_value)
        )
        can_open_short = (
            down_trend
            and position == 0
            and prev_position == 0
            and time_in_range[i]
            and not np.isnan(atr_value)
            and not np.isnan(highest_value)
        )

        if can_open_long:
            stop_size = atr_value * params.stop_multiplier_long
            long_stop_price = lowest_value - stop_size
            long_stop_distance = c - long_stop_price
            if long_stop_distance > 0 and c != 0:
                long_stop_pct = (long_stop_distance / c) * 100
                if long_stop_pct <= params.long_stop_pct_filter_size:
                    risk_cash = realized_equity * (params.risk_per_trade / 100.0)
                    if params.contract_size <= 0:
                        qty = 0
                    else:
                        qty = risk_cash / long_stop_distance
                        qty = math.floor((qty / params.contract_size)) * params.contract_size
                    if qty > 0:
                        position = 1
                        position_size = qty
                        entry_price = c
                        stop_price = long_stop_price
                        target_price = c + long_stop_distance * params.rr_long
                        trail_price_long = long_stop_price
                        trail_activated_long = False
                        entry_time_long = time
                        entry_commission = entry_price * position_size * COMMISSION_RATE
                        realized_equity -= entry_commission

        if can_open_short and position == 0:
            stop_size = atr_value * params.stop_multiplier_short
            short_stop_price = highest_value + stop_size
            short_stop_distance = short_stop_price - c
            if short_stop_distance > 0 and c != 0:
                short_stop_pct = (short_stop_distance / c) * 100
                if short_stop_pct <= params.short_stop_pct_filter_size:
                    risk_cash = realized_equity * (params.risk_per_trade / 100.0)
                    if params.contract_size <= 0:
                        qty = 0
                    else:
                        qty = risk_cash / short_stop_distance
                        qty = math.floor((qty / params.contract_size)) * params.contract_size
                    if qty > 0:
                        position = -1
                        position_size = qty
                        entry_price = c
                        stop_price = short_stop_price
                        target_price = c - short_stop_distance * params.rr_short
                        trail_price_short = short_stop_price
                        trail_activated_short = False
                        entry_time_short = time
                        entry_commission = entry_price * position_size * COMMISSION_RATE
                        realized_equity -= entry_commission

        mark_to_market = realized_equity
        if position > 0 and not math.isnan(entry_price):
            mark_to_market += (c - entry_price) * position_size
        elif position < 0 and not math.isnan(entry_price):
            mark_to_market += (entry_price - c) * position_size
        realized_curve.append(realized_equity)
        equity_curve.append(mark_to_market)
        prev_position = position

    equity_series = pd.Series(realized_curve, index=df.index)
    net_profit_pct = ((realized_equity - 100.0) / 100.0) * 100
    max_drawdown_pct = compute_max_drawdown(equity_series)
    total_trades = len(trades)
    return net_profit_pct, max_drawdown_pct, total_trades


def launch_gui(df: pd.DataFrame, data_path: str) -> None:
    from dearpygui import dearpygui as dpg

    dpg.create_context()

    with dpg.theme() as light_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (255, 255, 255, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (248, 248, 248, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Border, (208, 208, 208, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (51, 51, 51, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (255, 255, 255, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (240, 240, 240, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (255, 255, 255, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Header, (245, 245, 245, 255))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (238, 238, 238, 255))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (230, 230, 230, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 120, 212, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (16, 110, 190, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (0, 90, 158, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Tab, (245, 245, 245, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, (232, 232, 232, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, (255, 255, 255, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocused, (245, 245, 245, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocusedActive, (245, 245, 245, 255))
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (0, 120, 212, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, (240, 240, 240, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, (192, 192, 192, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, (160, 160, 160, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive, (128, 128, 128, 255))
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 3)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 4)

    with dpg.theme() as secondary_button_theme:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (225, 225, 225, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (208, 208, 208, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (192, 192, 192, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (51, 51, 51, 255))

    with dpg.theme() as section_header_theme:
        with dpg.theme_component(dpg.mvText):
            dpg.add_theme_color(dpg.mvThemeCol_Text, (85, 85, 85, 255))

    date_filter_tag = "input_date_filter"
    start_date_tag = "input_start_date"
    start_time_tag = "input_start_time"
    end_date_tag = "input_end_date"
    end_time_tag = "input_end_time"
    ma_type_tag = "input_t_ma_type"
    ma_length_tag = "input_ma_length"
    close_long_tag = "input_close_count_long"
    close_short_tag = "input_close_count_short"
    stop_long_tag = "input_stop_multiplier_long"
    rr_long_tag = "input_rr_long"
    lp_long_tag = "input_lp_long"
    stop_short_tag = "input_stop_multiplier_short"
    rr_short_tag = "input_rr_short"
    lp_short_tag = "input_lp_short"
    long_stop_pct_tag = "input_long_stop_pct"
    short_stop_pct_tag = "input_short_stop_pct"
    long_stop_days_tag = "input_long_stop_days"
    short_stop_days_tag = "input_short_stop_days"
    trail_rr_long_tag = "input_trail_rr_long"
    trail_rr_short_tag = "input_trail_rr_short"
    trail_ma_long_type_tag = "input_trail_ma_long_type"
    trail_ma_long_length_tag = "input_trail_ma_long_length"
    trail_ma_long_offset_tag = "input_trail_ma_long_offset"
    trail_ma_short_type_tag = "input_trail_ma_short_type"
    trail_ma_short_length_tag = "input_trail_ma_short_length"
    trail_ma_short_offset_tag = "input_trail_ma_short_offset"
    risk_per_trade_tag = "input_risk_per_trade"
    contract_size_tag = "input_contract_size"
    results_text_tag = "results_text"

    default_start_date = START_DATE.tz_convert("UTC") if START_DATE.tzinfo else START_DATE
    default_end_date = END_DATE.tz_convert("UTC") if END_DATE.tzinfo else END_DATE

    def combine_date_and_time(date_tag: str, time_tag: str) -> pd.Timestamp:
        date_value = dpg.get_value(date_tag)
        time_value = dpg.get_value(time_tag)
        if not isinstance(date_value, str) or not isinstance(time_value, str):
            raise ValueError("Date and time inputs must be strings")
        return parse_date_value(f"{date_value.strip()} {time_value.strip()}")

    def gather_params() -> Optional[StrategyParams]:
        try:
            start_date = combine_date_and_time(start_date_tag, start_time_tag)
            end_date = combine_date_and_time(end_date_tag, end_time_tag)
        except Exception as exc:  # noqa: BLE001 - console feedback is acceptable here
            message = f"Date parsing error: {exc}"
            print(message)
            dpg.set_value(results_text_tag, message)
            return None

        try:
            contract_size = float(dpg.get_value(contract_size_tag))
        except (TypeError, ValueError) as exc:
            message = f"Contract Size parsing error: {exc}"
            print(message)
            dpg.set_value(results_text_tag, message)
            return None

        return StrategyParams(
            date_filter=bool(dpg.get_value(date_filter_tag)),
            start_date=start_date,
            end_date=end_date,
            t_ma_type=str(dpg.get_value(ma_type_tag)),
            ma_length3=int(dpg.get_value(ma_length_tag)),
            close_count_trend_long=int(dpg.get_value(close_long_tag)),
            close_count_trend_short=int(dpg.get_value(close_short_tag)),
            stop_multiplier_long=float(dpg.get_value(stop_long_tag)),
            rr_long=float(dpg.get_value(rr_long_tag)),
            lp_long=int(dpg.get_value(lp_long_tag)),
            stop_multiplier_short=float(dpg.get_value(stop_short_tag)),
            rr_short=float(dpg.get_value(rr_short_tag)),
            lp_short=int(dpg.get_value(lp_short_tag)),
            long_stop_pct_filter_size=float(dpg.get_value(long_stop_pct_tag)),
            short_stop_pct_filter_size=float(dpg.get_value(short_stop_pct_tag)),
            long_stop_days_filter_size=int(dpg.get_value(long_stop_days_tag)),
            short_stop_days_filter_size=int(dpg.get_value(short_stop_days_tag)),
            trail_rr_long=float(dpg.get_value(trail_rr_long_tag)),
            trail_rr_short=float(dpg.get_value(trail_rr_short_tag)),
            trail_ma_type_long=str(dpg.get_value(trail_ma_long_type_tag)),
            trail_ma_length_long=int(dpg.get_value(trail_ma_long_length_tag)),
            trail_ma_offset_long=float(dpg.get_value(trail_ma_long_offset_tag)),
            trail_ma_type_short=str(dpg.get_value(trail_ma_short_type_tag)),
            trail_ma_length_short=int(dpg.get_value(trail_ma_short_length_tag)),
            trail_ma_offset_short=float(dpg.get_value(trail_ma_short_offset_tag)),
            risk_per_trade=float(dpg.get_value(risk_per_trade_tag)),
            contract_size=contract_size,
        )

    def run_backtest() -> None:
        dpg.set_value(
            results_text_tag,
            "Backtesting in progress...\n\nPlease wait...",
        )
        params = gather_params()
        if params is None:
            return

        net_profit, max_drawdown, trades = run_strategy(
            df, params, ma_type_override=params.t_ma_type
        )

        result_lines = [
            "=" * 60,
            f"Data: {data_path}",
            f"Trend MA №3: {params.t_ma_type} (Length {params.ma_length3})",
            (
                f"Trail Long: {params.trail_ma_type_long} (Len {params.trail_ma_length_long}, "
                f"Offset {params.trail_ma_offset_long:.2f})"
            ),
            (
                f"Trail Short: {params.trail_ma_type_short} (Len {params.trail_ma_length_short}, "
                f"Offset {params.trail_ma_offset_short:.2f})"
            ),
            f"Net Profit %: {net_profit:.2f}",
            f"Max Portfolio Drawdown %: {max_drawdown:.2f}",
            f"Total Trades: {trades}",
            "=" * 60,
        ]

        results_text = "\n".join(result_lines)
        print(results_text)
        dpg.set_value(results_text_tag, results_text)

    def reset_defaults() -> None:
        dpg.set_value(date_filter_tag, True)
        dpg.set_value(start_date_tag, default_start_date.strftime("%Y-%m-%d"))
        dpg.set_value(start_time_tag, default_start_date.strftime("%H:%M"))
        dpg.set_value(end_date_tag, default_end_date.strftime("%Y-%m-%d"))
        dpg.set_value(end_time_tag, default_end_date.strftime("%H:%M"))
        dpg.set_value(ma_type_tag, "EMA")
        dpg.set_value(ma_length_tag, 45)
        dpg.set_value(close_long_tag, 7)
        dpg.set_value(close_short_tag, 5)
        dpg.set_value(stop_long_tag, 2.0)
        dpg.set_value(rr_long_tag, 3.0)
        dpg.set_value(lp_long_tag, 2)
        dpg.set_value(stop_short_tag, 2.0)
        dpg.set_value(rr_short_tag, 3.0)
        dpg.set_value(lp_short_tag, 2)
        dpg.set_value(long_stop_pct_tag, 3.0)
        dpg.set_value(short_stop_pct_tag, 3.0)
        dpg.set_value(long_stop_days_tag, 2)
        dpg.set_value(short_stop_days_tag, 4)
        dpg.set_value(trail_rr_long_tag, 1.0)
        dpg.set_value(trail_rr_short_tag, 1.0)
        dpg.set_value(trail_ma_long_type_tag, "SMA")
        dpg.set_value(trail_ma_long_length_tag, 160)
        dpg.set_value(trail_ma_long_offset_tag, -1.0)
        dpg.set_value(trail_ma_short_type_tag, "SMA")
        dpg.set_value(trail_ma_short_length_tag, 160)
        dpg.set_value(trail_ma_short_offset_tag, 1.0)
        dpg.set_value(risk_per_trade_tag, 2.0)
        dpg.set_value(contract_size_tag, DEFAULT_CONTRACT_SIZE)
        dpg.set_value(
            results_text_tag,
            "Press 'Run' to launch the backtest...",
        )

    def cancel_window() -> None:
        dpg.configure_item("main_window", show=False)
        dpg.stop_dearpygui()

    with dpg.window(
        label="S_01 TrailingMA Backtester",
        tag="main_window",
        width=850,
        height=750,
        pos=(100, 100),
        no_close=True,
    ):
        with dpg.tab_bar():
            with dpg.tab(label="Inputs"):
                dpg.add_spacing(count=2)

                with dpg.group(horizontal=True):
                    dpg.add_checkbox(
                        label="Date Filter",
                        default_value=True,
                        tag=date_filter_tag,
                    )
                    dpg.add_spacing(count=5)
                    dpg.add_checkbox(label="Backtester", default_value=True, enabled=False)

                dpg.add_spacing(count=3)

                with dpg.group(horizontal=True):
                    label = dpg.add_text("Start Date")
                    dpg.bind_item_theme(label, section_header_theme)
                    dpg.add_input_text(
                        default_value=default_start_date.strftime("%Y-%m-%d"),
                        width=120,
                        tag=start_date_tag,
                    )
                    dpg.add_button(label="📅", width=35, enabled=False)
                    dpg.add_input_text(
                        default_value=default_start_date.strftime("%H:%M"),
                        width=70,
                        tag=start_time_tag,
                    )

                with dpg.group(horizontal=True):
                    label = dpg.add_text("End Date  ")
                    dpg.bind_item_theme(label, section_header_theme)
                    dpg.add_input_text(
                        default_value=default_end_date.strftime("%Y-%m-%d"),
                        width=120,
                        tag=end_date_tag,
                    )
                    dpg.add_button(label="📅", width=35, enabled=False)
                    dpg.add_input_text(
                        default_value=default_end_date.strftime("%H:%M"),
                        width=70,
                        tag=end_time_tag,
                    )

                dpg.add_separator()
                dpg.add_spacing(count=2)

                with dpg.group(horizontal=True):
                    label = dpg.add_text("T MA Type")
                    dpg.bind_item_theme(label, section_header_theme)
                    dpg.add_combo(
                        items=MA_TYPE_OPTIONS,
                        default_value="EMA",
                        width=110,
                        tag=ma_type_tag,
                    )
                    dpg.add_spacing(count=5)
                    dpg.add_text("Length")
                    dpg.add_input_int(
                        default_value=45,
                        width=80,
                        min_value=1,
                        tag=ma_length_tag,
                    )

                dpg.add_spacing(count=2)

                with dpg.group(horizontal=True):
                    dpg.add_text("Close Count Long")
                    dpg.add_input_int(default_value=7, width=80, tag=close_long_tag, min_value=1)
                    dpg.add_spacing(count=5)
                    dpg.add_text("Close Count Short")
                    dpg.add_input_int(default_value=5, width=80, tag=close_short_tag, min_value=1)

                dpg.add_separator()
                dpg.add_spacing(count=2)

                with dpg.collapsing_header(label="STOPS AND FILTERS", default_open=True):
                    dpg.add_spacing(count=2)

                    with dpg.child_window(height=160, border=True):
                        dpg.add_spacing(count=2)

                        with dpg.group(horizontal=True):
                            dpg.add_text("Stop Long X ")
                            dpg.add_input_float(default_value=2.0, width=70, tag=stop_long_tag, step=0.1)
                            dpg.add_spacing(count=3)
                            dpg.add_text("RR")
                            dpg.add_input_float(default_value=3.0, width=70, tag=rr_long_tag, step=0.1)
                            dpg.add_spacing(count=3)
                            dpg.add_text("LP")
                            dpg.add_input_int(default_value=2, width=70, tag=lp_long_tag, min_value=1)

                        with dpg.group(horizontal=True):
                            dpg.add_text("Stop Short X")
                            dpg.add_input_float(default_value=2.0, width=70, tag=stop_short_tag, step=0.1)
                            dpg.add_spacing(count=3)
                            dpg.add_text("RR")
                            dpg.add_input_float(default_value=3.0, width=70, tag=rr_short_tag, step=0.1)
                            dpg.add_spacing(count=3)
                            dpg.add_text("LP")
                            dpg.add_input_int(default_value=2, width=70, tag=lp_short_tag, min_value=1)

                        dpg.add_spacing(count=2)

                        with dpg.group(horizontal=True):
                            dpg.add_text("L Stop Max %")
                            dpg.add_input_float(default_value=3.0, width=70, tag=long_stop_pct_tag, step=0.1)
                            dpg.add_spacing(count=3)
                            dpg.add_text("S Stop Max %")
                            dpg.add_input_float(default_value=3.0, width=70, tag=short_stop_pct_tag, step=0.1)

                        with dpg.group(horizontal=True):
                            dpg.add_text("L Stop Max D")
                            dpg.add_input_int(default_value=2, width=70, tag=long_stop_days_tag, min_value=1)
                            dpg.add_spacing(count=3)
                            dpg.add_text("S Stop Max D")
                            dpg.add_input_int(default_value=4, width=70, tag=short_stop_days_tag, min_value=1)

                dpg.add_spacing(count=2)

                with dpg.collapsing_header(label="TRAILING STOPS", default_open=True):
                    dpg.add_spacing(count=2)

                    with dpg.child_window(height=140, border=True):
                        dpg.add_spacing(count=2)

                        with dpg.group(horizontal=True):
                            dpg.add_text("Trail RR Long ")
                            dpg.add_input_float(default_value=1.0, width=70, tag=trail_rr_long_tag, step=0.1)
                            dpg.add_spacing(count=3)
                            dpg.add_text("Trail RR Short")
                            dpg.add_input_float(default_value=1.0, width=70, tag=trail_rr_short_tag, step=0.1)

                        dpg.add_spacing(count=2)

                        with dpg.group(horizontal=True):
                            dpg.add_text("Trail MA Long ")
                            dpg.add_combo(
                                items=MA_TYPE_OPTIONS,
                                default_value="SMA",
                                width=100,
                                tag=trail_ma_long_type_tag,
                            )
                            dpg.add_text("Length")
                            dpg.add_input_int(
                                default_value=160,
                                width=70,
                                tag=trail_ma_long_length_tag,
                                min_value=0,
                            )
                            dpg.add_text("Offset")
                            dpg.add_input_float(
                                default_value=-1.0,
                                width=70,
                                tag=trail_ma_long_offset_tag,
                                step=0.1,
                            )

                        with dpg.group(horizontal=True):
                            dpg.add_text("Trail MA Short")
                            dpg.add_combo(
                                items=MA_TYPE_OPTIONS,
                                default_value="SMA",
                                width=100,
                                tag=trail_ma_short_type_tag,
                            )
                            dpg.add_text("Length")
                            dpg.add_input_int(
                                default_value=160,
                                width=70,
                                tag=trail_ma_short_length_tag,
                                min_value=0,
                            )
                            dpg.add_text("Offset")
                            dpg.add_input_float(
                                default_value=1.0,
                                width=70,
                                tag=trail_ma_short_offset_tag,
                                step=0.1,
                            )

                dpg.add_separator()
                dpg.add_spacing(count=2)

                with dpg.group(horizontal=True):
                    dpg.add_text("Risk Per Trade")
                    dpg.add_input_float(
                        default_value=2.0,
                        width=80,
                        format="%.2f",
                        tag=risk_per_trade_tag,
                    )
                    dpg.add_spacing(count=5)
                    dpg.add_text("Contract Size")
                    dpg.add_input_float(
                        default_value=DEFAULT_CONTRACT_SIZE,
                        width=80,
                        format="%.4f",
                        tag=contract_size_tag,
                    )

                dpg.add_separator()
                dpg.add_spacing(count=2)

                with dpg.child_window(height=180, border=True, tag="results_window"):
                    dpg.add_spacing(count=2)
                    dpg.add_text(
                        "Нажмите 'Run' для запуска бэктеста...",
                        tag=results_text_tag,
                        color=(136, 136, 136),
                    )

                dpg.add_spacing(count=3)

                with dpg.group(horizontal=True):
                    defaults_btn = dpg.add_button(
                        label="Defaults",
                        width=100,
                        callback=reset_defaults,
                    )
                    dpg.bind_item_theme(defaults_btn, secondary_button_theme)
                    dpg.add_spacer(width=450)
                    cancel_btn = dpg.add_button(
                        label="Cancel",
                        width=100,
                        callback=cancel_window,
                    )
                    dpg.bind_item_theme(cancel_btn, secondary_button_theme)
                    dpg.add_spacing(count=2)
                    dpg.add_button(label="Run", width=100, callback=run_backtest)

            with dpg.tab(label="Properties"):
                dpg.add_spacing(count=5)
                dpg.add_text("Properties tab - coming soon...", color=(136, 136, 136))

            with dpg.tab(label="Style"):
                dpg.add_spacing(count=5)
                dpg.add_text("Style tab - coming soon...", color=(136, 136, 136))

            with dpg.tab(label="Visibility"):
                dpg.add_spacing(count=5)
                dpg.add_text("Visibility tab - coming soon...", color=(136, 136, 136))

    reset_defaults()
    dpg.bind_theme(light_theme)
    dpg.create_viewport(title="TrailingMA Backtester", width=900, height=800)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()

def main() -> None:
    parser = argparse.ArgumentParser(description="S_01 TrailingMA Ultralight backtester")
    parser.add_argument(
        "--data",
        default="OKX_LINKUSDT.P, 15 2025.02.01-2025.09.09.csv",
        help="Path to CSV data file",
    )
    parser.add_argument(
        "--ma-type",
        default="EMA",
        choices=MA_TYPE_OPTIONS,
        help="Trend moving average type to use when running without GUI",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run a single backtest with current parameters without launching the GUI",
    )
    args = parser.parse_args()

    df = load_data(args.data)

    if args.no_gui:
        params = StrategyParams(t_ma_type=args.ma_type)
        net_profit, max_drawdown, trades = run_strategy(
            df, params, ma_type_override=args.ma_type
        )
        print(f"Net Profit %: {net_profit:.2f}")
        print(f"Max Portfolio Drawdown %: {max_drawdown:.2f}")
        print(f"Total Trades: {trades}")
    else:
        launch_gui(df, args.data)


if __name__ == "__main__":
    main()
