import argparse
import math
from dataclasses import dataclass, replace
from pathlib import Path
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

    font_candidates = [
        Path("segoeui.ttf"),
        Path.cwd() / "fonts" / "segoeui.ttf",
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("~/Library/Fonts/Segoe UI.ttf").expanduser(),
        Path("/usr/share/fonts/truetype/segoeui/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/SegoeUI.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/segoeui.ttf"),
    ]

    for candidate in font_candidates:
        if candidate.is_file():
            with dpg.font_registry():
                default_font = dpg.add_font(str(candidate), 16)
            dpg.bind_font(default_font)
            break
    else:
        print("Segoe UI font not found; using Dear PyGui default font.")

    ma_checkbox_tags = {opt: f"ma_select_{opt.lower()}" for opt in MA_TYPE_OPTIONS}
    all_tag = "ma_select_all"

    trail_long_checkbox_tags = {
        opt: f"trail_long_select_{opt.lower()}" for opt in MA_TYPE_OPTIONS
    }
    trail_short_checkbox_tags = {
        opt: f"trail_short_select_{opt.lower()}" for opt in MA_TYPE_OPTIONS
    }
    trail_long_all_tag = "trail_long_select_all"
    trail_short_all_tag = "trail_short_select_all"

    def make_checkbox_handlers(option_tags: dict[str, str], all_checkbox_tag: str):
        def on_all_toggle(sender: int, app_data: bool) -> None:
            state = bool(app_data)
            for tag in option_tags.values():
                dpg.set_value(tag, state)

        def on_type_toggle(sender: int, app_data: bool, user_data: str) -> None:
            if not app_data:
                dpg.set_value(all_checkbox_tag, False)
            elif all(dpg.get_value(tag) for tag in option_tags.values()):
                dpg.set_value(all_checkbox_tag, True)

        return on_all_toggle, on_type_toggle

    ma_all_toggle, ma_type_toggle = make_checkbox_handlers(ma_checkbox_tags, all_tag)
    trail_long_all_toggle, trail_long_type_toggle = make_checkbox_handlers(
        trail_long_checkbox_tags, trail_long_all_tag
    )
    trail_short_all_toggle, trail_short_type_toggle = make_checkbox_handlers(
        trail_short_checkbox_tags, trail_short_all_tag
    )

    with dpg.theme() as light_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 0, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (245, 245, 245, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (245, 245, 245, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (245, 245, 245, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgCollapsed, (245, 245, 245, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (230, 230, 230, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (210, 210, 210, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (190, 190, 190, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Button, (225, 225, 225, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (200, 200, 200, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (180, 180, 180, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Header, (220, 220, 220, 255))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (200, 200, 200, 255))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (180, 180, 180, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (245, 245, 245, 255))
            dpg.add_theme_color(dpg.mvThemeCol_MenubarBg, (245, 245, 245, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Border, (200, 200, 200, 255))
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, (245, 245, 245, 255))

    def gather_params() -> Optional[StrategyParams]:
        start_raw = dpg.get_value("start_date")
        end_raw = dpg.get_value("end_date")
        try:
            start_date = parse_date_value(start_raw)
            end_date = parse_date_value(end_raw)
        except Exception as exc:  # noqa: BLE001 - reporting to console is acceptable here
            print(f"Date parsing error: {exc}")
            return None
        try:
            contract_size = float(dpg.get_value("contract_size"))
        except (TypeError, ValueError) as exc:
            print(f"Contract Size parsing error: {exc}")
            return None
        return StrategyParams(
            date_filter=dpg.get_value("date_filter"),
            start_date=start_date,
            end_date=end_date,
            ma_length3=dpg.get_value("ma_length3"),
            close_count_trend_long=dpg.get_value("close_count_long"),
            close_count_trend_short=dpg.get_value("close_count_short"),
            stop_multiplier_long=dpg.get_value("stop_multiplier_long"),
            rr_long=dpg.get_value("rr_long"),
            lp_long=dpg.get_value("lp_long"),
            stop_multiplier_short=dpg.get_value("stop_multiplier_short"),
            rr_short=dpg.get_value("rr_short"),
            lp_short=dpg.get_value("lp_short"),
            long_stop_pct_filter_size=dpg.get_value("long_stop_pct_filter_size"),
            short_stop_pct_filter_size=dpg.get_value("short_stop_pct_filter_size"),
            long_stop_days_filter_size=dpg.get_value("long_stop_days_filter_size"),
            short_stop_days_filter_size=dpg.get_value("short_stop_days_filter_size"),
            trail_rr_long=dpg.get_value("trail_rr_long"),
            trail_rr_short=dpg.get_value("trail_rr_short"),
            trail_ma_length_long=dpg.get_value("trail_ma_length_long"),
            trail_ma_offset_long=dpg.get_value("trail_ma_offset_long"),
            trail_ma_length_short=dpg.get_value("trail_ma_length_short"),
            trail_ma_offset_short=dpg.get_value("trail_ma_offset_short"),
            risk_per_trade=dpg.get_value("risk_per_trade"),
            contract_size=contract_size,
        )

    def on_run(sender: int, app_data: int) -> None:
        params = gather_params()
        if params is None:
            return
        if dpg.get_value(all_tag):
            selected_types = list(MA_TYPE_OPTIONS)
        else:
            selected_types = [
                opt for opt in MA_TYPE_OPTIONS if dpg.get_value(ma_checkbox_tags[opt])
            ]
        if not selected_types:
            print("Select at least one moving average type to run.")
            dpg.set_value(
                "results_output",
                "Please select at least one moving average type for MA №3.",
            )
            return

        if dpg.get_value(trail_long_all_tag):
            selected_trail_long = list(MA_TYPE_OPTIONS)
        else:
            selected_trail_long = [
                opt
                for opt in MA_TYPE_OPTIONS
                if dpg.get_value(trail_long_checkbox_tags[opt])
            ]
        if not selected_trail_long:
            message = "Select at least one trailing MA type for long positions."
            print(message)
            dpg.set_value("results_output", message)
            return

        if dpg.get_value(trail_short_all_tag):
            selected_trail_short = list(MA_TYPE_OPTIONS)
        else:
            selected_trail_short = [
                opt
                for opt in MA_TYPE_OPTIONS
                if dpg.get_value(trail_short_checkbox_tags[opt])
            ]
        if not selected_trail_short:
            message = "Select at least one trailing MA type for short positions."
            print(message)
            dpg.set_value("results_output", message)
            return

        output_lines = [
            "=" * 60,
            (
                f"Running backtests for {len(selected_types)} MA type(s) with "
                f"{len(selected_trail_long)} long trailing option(s) and "
                f"{len(selected_trail_short)} short trailing option(s) using data: {data_path}"
            ),
        ]

        for ma_name in selected_types:
            for trail_long_name in selected_trail_long:
                for trail_short_name in selected_trail_short:
                    current_params = replace(
                        params,
                        t_ma_type=ma_name,
                        trail_ma_type_long=trail_long_name,
                        trail_ma_type_short=trail_short_name,
                    )
                    net_profit, max_drawdown, trades = run_strategy(
                        df, current_params, ma_type_override=ma_name
                    )
                    output_lines.append(
                        (
                            f"[{ma_name}] Trail L:{trail_long_name} S:{trail_short_name} "
                            f"Net Profit %: {net_profit:.2f} | "
                            f"Max DD %: {max_drawdown:.2f} | Trades: {trades}"
                        )
                    )

        output_lines.append("=" * 60)
        results_text = "\n".join(output_lines)
        print(results_text)
        dpg.set_value("results_output", results_text)

    with dpg.window(
        label="S_01 TrailingMA Backtester",
        width=820,
        height=980,
        no_resize=True,
    ):
        dpg.add_text("Backtest Settings")
        dpg.add_text(f"Data: {data_path}")
        dpg.add_separator()

        with dpg.group(horizontal=True):
            dpg.add_checkbox(label="Date Filter", default_value=True, tag="date_filter")
            dpg.add_checkbox(label="Backtester", default_value=True, tag="backtester")
        dpg.add_input_text(label="Start Date", default_value="2025-04-01", tag="start_date")
        dpg.add_input_text(label="End Date", default_value="2025-09-01", tag="end_date")

        dpg.add_separator()
        dpg.add_text("MA №3")
        with dpg.group(horizontal=True):
            dpg.add_checkbox(label="ALL", tag=all_tag, callback=ma_all_toggle)
            for opt in MA_TYPE_OPTIONS[:5]:
                dpg.add_checkbox(
                    label=opt,
                    tag=ma_checkbox_tags[opt],
                    callback=ma_type_toggle,
                    user_data=opt,
                    default_value=(opt == "EMA"),
                )
        with dpg.group(horizontal=True):
            dpg.add_spacer(width=42)
            for opt in MA_TYPE_OPTIONS[5:]:
                dpg.add_checkbox(
                    label=opt,
                    tag=ma_checkbox_tags[opt],
                    callback=ma_type_toggle,
                    user_data=opt,
                )
        dpg.add_input_int(label="Length", default_value=45, min_value=0, tag="ma_length3")

        dpg.add_separator()
        dpg.add_text("Trend BB")
        dpg.add_input_int(
            label="Close Count Long",
            default_value=7,
            min_value=1,
            tag="close_count_long",
        )
        dpg.add_input_int(
            label="Close Count Short",
            default_value=5,
            min_value=1,
            tag="close_count_short",
        )

        dpg.add_separator()
        dpg.add_text("Stops and Filters")
        with dpg.group(horizontal=True):
            dpg.add_input_float(
                label="Stop Long X",
                default_value=2.0,
                step=0.1,
                tag="stop_multiplier_long",
            )
            dpg.add_input_float(label="RR", default_value=3.0, step=0.1, tag="rr_long")
            dpg.add_input_int(label="LP", default_value=2, min_value=1, tag="lp_long")
        with dpg.group(horizontal=True):
            dpg.add_input_float(
                label="Stop Short X",
                default_value=2.0,
                step=0.1,
                tag="stop_multiplier_short",
            )
            dpg.add_input_float(label="RR", default_value=3.0, step=0.1, tag="rr_short")
            dpg.add_input_int(label="LP", default_value=2, min_value=1, tag="lp_short")
        with dpg.group(horizontal=True):
            dpg.add_input_int(
                label="L Stop Max %",
                default_value=3,
                min_value=1,
                tag="long_stop_pct_filter_size",
            )
            dpg.add_input_int(
                label="S Stop Max %",
                default_value=3,
                min_value=1,
                tag="short_stop_pct_filter_size",
            )
        with dpg.group(horizontal=True):
            dpg.add_input_int(
                label="L Stop Max D",
                default_value=2,
                min_value=1,
                tag="long_stop_days_filter_size",
            )
            dpg.add_input_int(
                label="S Stop Max D",
                default_value=4,
                min_value=1,
                tag="short_stop_days_filter_size",
            )

        dpg.add_separator()
        dpg.add_text("Trailing Stops")
        with dpg.group(horizontal=True):
            dpg.add_input_float(
                label="Trail RR Long",
                default_value=1.0,
                step=0.1,
                tag="trail_rr_long",
            )
            dpg.add_input_float(
                label="Trail RR Short",
                default_value=1.0,
                step=0.1,
                tag="trail_rr_short",
            )
        dpg.add_text("Trail MA Long")
        with dpg.group(horizontal=True):
            dpg.add_checkbox(
                label="ALL",
                tag=trail_long_all_tag,
                callback=trail_long_all_toggle,
            )
            for opt in MA_TYPE_OPTIONS[:5]:
                dpg.add_checkbox(
                    label=opt,
                    tag=trail_long_checkbox_tags[opt],
                    callback=trail_long_type_toggle,
                    user_data=opt,
                    default_value=(opt == "SMA"),
                )
        with dpg.group(horizontal=True):
            dpg.add_spacer(width=42)
            for opt in MA_TYPE_OPTIONS[5:]:
                dpg.add_checkbox(
                    label=opt,
                    tag=trail_long_checkbox_tags[opt],
                    callback=trail_long_type_toggle,
                    user_data=opt,
                )
        with dpg.group(horizontal=True):
            dpg.add_input_int(
                label="Length",
                default_value=160,
                step=5,
                min_value=0,
                tag="trail_ma_length_long",
            )
            dpg.add_input_float(
                label="Offset",
                default_value=-1.0,
                step=0.5,
                tag="trail_ma_offset_long",
            )
        dpg.add_text("Trail MA Short")
        with dpg.group(horizontal=True):
            dpg.add_checkbox(
                label="ALL",
                tag=trail_short_all_tag,
                callback=trail_short_all_toggle,
            )
            for opt in MA_TYPE_OPTIONS[:5]:
                dpg.add_checkbox(
                    label=opt,
                    tag=trail_short_checkbox_tags[opt],
                    callback=trail_short_type_toggle,
                    user_data=opt,
                    default_value=(opt == "SMA"),
                )
        with dpg.group(horizontal=True):
            dpg.add_spacer(width=42)
            for opt in MA_TYPE_OPTIONS[5:]:
                dpg.add_checkbox(
                    label=opt,
                    tag=trail_short_checkbox_tags[opt],
                    callback=trail_short_type_toggle,
                    user_data=opt,
                )
        with dpg.group(horizontal=True):
            dpg.add_input_int(
                label="Length",
                default_value=160,
                step=5,
                min_value=0,
                tag="trail_ma_length_short",
            )
            dpg.add_input_float(
                label="Offset",
                default_value=1.0,
                step=0.5,
                tag="trail_ma_offset_short",
            )

        dpg.add_separator()
        dpg.add_text("Risk Per Trade")
        with dpg.group(horizontal=True):
            dpg.add_input_float(
                label="Risk Per Trade",
                default_value=2.0,
                step=0.5,
                tag="risk_per_trade",
            )
            dpg.add_combo(
                label="Contract Size",
                items=[
                    "0.0001",
                    "0.001",
                    "0.01",
                    "0.1",
                    "1",
                    "10",
                    "100",
                    "1000",
                    "10000",
                    "100000",
                    "1000000",
                ],
                default_value="0.01",
                tag="contract_size",
            )

        dpg.add_separator()
        dpg.add_spacer(height=10)
        with dpg.group(horizontal=True):
            dpg.add_spacer(width=360)
            dpg.add_button(label="Run", width=120, callback=on_run)

        dpg.add_separator()
        dpg.add_text("Results")
        dpg.add_input_text(
            tag="results_output",
            multiline=True,
            readonly=True,
            width=760,
            height=320,
        )

    dpg.bind_theme(light_theme)
    dpg.create_viewport(title="S_01 TrailingMA Backtester", width=920, height=1080)
    dpg.set_viewport_clear_color((245, 245, 245, 255))
    dpg.setup_dearpygui()
    dpg.show_viewport()
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
