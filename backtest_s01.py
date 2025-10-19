import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from backtesting import _stats
from dearpygui import dearpygui as dpg

FACTOR_T3 = 0.7
FAST_KAMA = 2
SLOW_KAMA = 30
START_DATE = pd.Timestamp('2025-04-01 08:00', tz='UTC')
END_DATE = pd.Timestamp('2025-09-01 08:00', tz='UTC')
DEFAULT_MA_LENGTH = 45
DEFAULT_RISK_PER_TRADE = 2.0
DEFAULT_CONTRACT_SIZE = 0.01
COMMISSION_RATE = 0.0005

MA_TYPES_ROW_1 = ["ALL", "EMA", "SMA", "HMA", "WMA", "ALMA"]
MA_TYPES_ROW_2 = ["KAMA", "TMA", "T3", "DEMA", "VWMA", "VWAP"]
ALL_MA_TYPES = MA_TYPES_ROW_1[1:] + MA_TYPES_ROW_2


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
        prev = (
            kama_values[i - 1]
            if i > 0 and not np.isnan(kama_values[i - 1])
            else (series.iat[i - 1] if i > 0 else price)
        )
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


def run_strategy(df: pd.DataFrame,
                 ma_type: str,
                 ma_length: int = DEFAULT_MA_LENGTH,
                 start_date: pd.Timestamp = START_DATE,
                 end_date: pd.Timestamp = END_DATE,
                 risk_per_trade_pct: float = DEFAULT_RISK_PER_TRADE,
                 contract_size: float = DEFAULT_CONTRACT_SIZE) -> Tuple[float, float, int]:
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    ma3 = get_ma(close,
                 ma_type,
                 ma_length,
                 volume=volume,
                 high=high,
                 low=low)
    atr14 = atr(high, low, close, 14)
    lowest_long = low.rolling(2, min_periods=1).min()
    highest_short = high.rolling(2, min_periods=1).max()

    trail_ma_long = get_ma(close, 'SMA', 160)
    trail_ma_long = trail_ma_long * (1 + (-1.0) / 100)
    trail_ma_short = get_ma(close, 'SMA', 160)
    trail_ma_short = trail_ma_short * (1 + (1.0) / 100)

    times = df.index
    time_in_range = (times >= start_date) & (times <= end_date)

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
                activation_price = entry_price + (entry_price - stop_price) * 1.0
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
                if days_in_trade >= 2:
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
                activation_price = entry_price - (stop_price - entry_price) * 1.0
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
                if days_in_trade >= 4:
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

        up_trend = counter_close_trend_long >= 7 and counter_trade_long == 0
        down_trend = counter_close_trend_short >= 5 and counter_trade_short == 0

        can_open_long = (
            up_trend and position == 0 and prev_position == 0 and time_in_range[i] and
            not np.isnan(atr_value) and not np.isnan(lowest_value)
        )
        can_open_short = (
            down_trend and position == 0 and prev_position == 0 and time_in_range[i] and
            not np.isnan(atr_value) and not np.isnan(highest_value)
        )

        if can_open_long:
            stop_size = atr_value * 2.0
            long_stop_price = lowest_value - stop_size
            long_stop_distance = c - long_stop_price
            if long_stop_distance > 0:
                long_stop_pct = (long_stop_distance / c) * 100
                if long_stop_pct <= 3:
                    risk_cash = realized_equity * (risk_per_trade_pct / 100)
                    qty = risk_cash / long_stop_distance
                    qty = math.floor((qty / contract_size)) * contract_size
                    if qty > 0:
                        position = 1
                        position_size = qty
                        entry_price = c
                        stop_price = long_stop_price
                        target_price = c + long_stop_distance * 3.0
                        trail_price_long = long_stop_price
                        trail_activated_long = False
                        entry_time_long = time
                        entry_commission = entry_price * position_size * COMMISSION_RATE
                        realized_equity -= entry_commission

        if can_open_short and position == 0:
            stop_size = atr_value * 2.0
            short_stop_price = highest_value + stop_size
            short_stop_distance = short_stop_price - c
            if short_stop_distance > 0:
                short_stop_pct = (short_stop_distance / c) * 100
                if short_stop_pct <= 3:
                    risk_cash = realized_equity * (risk_per_trade_pct / 100)
                    qty = risk_cash / short_stop_distance
                    qty = math.floor((qty / contract_size)) * contract_size
                    if qty > 0:
                        position = -1
                        position_size = qty
                        entry_price = c
                        stop_price = short_stop_price
                        target_price = c - short_stop_distance * 3.0
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


def format_results(ma_type: str,
                   net_profit: float,
                   max_drawdown: float,
                   trades: int) -> str:
    return (
        f"MA: {ma_type}\n"
        f"  Net Profit %: {net_profit:.2f}\n"
        f"  Max Portfolio Drawdown %: {max_drawdown:.2f}\n"
        f"  Total Trades: {trades}\n"
    )


def add_section_title(text: str) -> None:
    dpg.add_text(text.upper(), color=(58, 58, 58, 255))
    dpg.add_separator()
    dpg.add_spacing(count=2)


def add_label(text: str, width: int = 120) -> None:
    """Render a label followed by spacer to emulate a fixed width."""
    dpg.add_text(text)
    approx_remaining = max(width - len(text) * 7, 0)
    if approx_remaining:
        dpg.add_spacer(width=approx_remaining)


def add_datetime_input(label: str,
                       default_date: str,
                       default_time: str,
                       date_tag: str,
                       time_tag: str) -> None:
    with dpg.group(horizontal=True):
        add_label(f"{label}:")
        dpg.add_input_text(width=150, default_value=default_date, tag=date_tag)
        dpg.add_spacer(width=10)
        dpg.add_input_text(width=80, default_value=default_time, tag=time_tag)


def create_monochrome_theme() -> None:
    with dpg.theme() as monochrome_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (245, 245, 245, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (245, 245, 245, 255))

            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (74, 74, 74, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (74, 74, 74, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgCollapsed, (74, 74, 74, 255))

            dpg.add_theme_color(dpg.mvThemeCol_Border, (153, 153, 153, 255))
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
            dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 1)

            dpg.add_theme_color(dpg.mvThemeCol_Text, (42, 42, 42, 255))

            dpg.add_theme_color(dpg.mvThemeCol_Button, (74, 74, 74, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (58, 58, 58, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (90, 90, 90, 255))

            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (255, 255, 255, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (248, 248, 248, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (255, 255, 255, 255))

            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (42, 42, 42, 255))

            dpg.add_theme_color(dpg.mvThemeCol_Header, (232, 232, 232, 255))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (221, 221, 221, 255))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (204, 204, 204, 255))

            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 20, 20)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 6, 6)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 10, 10)
            dpg.add_theme_style(dpg.mvStyleVar_ItemInnerSpacing, 8, 8)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 3)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 3)

    dpg.bind_theme(monochrome_theme)


def update_all_checkbox_state() -> None:
    all_checked = all(dpg.get_value(f"ma_checkbox_{ma}") for ma in ALL_MA_TYPES)
    dpg.set_value("ma_checkbox_ALL", all_checked)


def on_all_checkbox(sender: int, app_data: bool, user_data: Sequence[str]) -> None:
    for ma in user_data:
        dpg.set_value(f"ma_checkbox_{ma}", app_data)


def on_individual_checkbox(sender: int, app_data: bool, user_data: None) -> None:
    update_all_checkbox_state()


def add_ma_selector(label: str, tag_prefix: str) -> None:
    dpg.add_text(label)
    dpg.add_spacing(count=1)
    with dpg.child_window(height=80, border=True, tag=f"{tag_prefix}_container"):
        with dpg.group(horizontal=True):
            for ma in MA_TYPES_ROW_1:
                tag = f"ma_checkbox_{ma}"
                if ma == "ALL":
                    dpg.add_checkbox(label=ma,
                                     tag=tag,
                                     callback=on_all_checkbox,
                                     user_data=ALL_MA_TYPES)
                else:
                    dpg.add_checkbox(label=ma,
                                     tag=tag,
                                     callback=on_individual_checkbox,
                                     default_value=(ma == "EMA"))
        with dpg.group(horizontal=True):
            for ma in MA_TYPES_ROW_2:
                dpg.add_checkbox(label=ma,
                                 tag=f"ma_checkbox_{ma}",
                                 callback=on_individual_checkbox)
    dpg.add_spacing(count=2)


def add_results_area() -> None:
    with dpg.child_window(height=200, border=True, tag="results_window"):
        dpg.add_text(
            "Нажмите 'Run' для запуска бэктеста...",
            color=(119, 119, 119, 255),
            tag="results_text",
        )


def parse_datetime(date_str: str, time_str: str) -> pd.Timestamp:
    combined = f"{date_str.strip()} {time_str.strip()}"
    return pd.Timestamp(combined, tz='UTC')


def run_backtests(df: pd.DataFrame,
                  selected_ma_types: Sequence[str],
                  params: Dict[str, float]) -> str:
    results = []
    for ma in selected_ma_types:
        net_profit, max_drawdown, trades = run_strategy(
            df,
            ma,
            ma_length=int(params["ma_length"]),
            start_date=params["start_date"],
            end_date=params["end_date"],
            risk_per_trade_pct=float(params["risk_pct"]),
            contract_size=float(params["contract_size"]),
        )
        results.append(format_results(ma, net_profit, max_drawdown, trades))
    return "\n".join(results)


def on_run_clicked(sender: int, app_data: None, user_data: pd.DataFrame) -> None:
    selected = [
        ma for ma in ALL_MA_TYPES
        if dpg.get_value(f"ma_checkbox_{ma}")
    ]
    if not selected:
        dpg.set_value("results_text", "Выберите хотя бы один тип мувинга.")
        return

    try:
        start_date = parse_datetime(dpg.get_value("start_date"),
                                    dpg.get_value("start_time"))
        end_date = parse_datetime(dpg.get_value("end_date"),
                                  dpg.get_value("end_time"))
    except Exception:
        dpg.set_value("results_text", "Неверный формат даты или времени.")
        return

    if end_date <= start_date:
        dpg.set_value("results_text", "Дата окончания должна быть позже даты начала.")
        return

    params = {
        "ma_length": max(1, dpg.get_value("ma_length")),
        "risk_pct": max(0.0, dpg.get_value("risk_per_trade")),
        "contract_size": max(0.0001, dpg.get_value("contract_size")),
        "start_date": start_date,
        "end_date": end_date,
    }

    results_text = run_backtests(user_data, selected, params)
    dpg.set_value("results_text", results_text)


def on_defaults_clicked(sender: int, app_data: None, user_data: None) -> None:
    dpg.set_value("start_date", START_DATE.strftime("%Y-%m-%d"))
    dpg.set_value("start_time", START_DATE.strftime("%H:%M"))
    dpg.set_value("end_date", END_DATE.strftime("%Y-%m-%d"))
    dpg.set_value("end_time", END_DATE.strftime("%H:%M"))
    dpg.set_value("ma_length", DEFAULT_MA_LENGTH)
    dpg.set_value("risk_per_trade", DEFAULT_RISK_PER_TRADE)
    dpg.set_value("contract_size", DEFAULT_CONTRACT_SIZE)
    for ma in ALL_MA_TYPES:
        dpg.set_value(f"ma_checkbox_{ma}", ma == "EMA")
    update_all_checkbox_state()
    dpg.set_value("results_text", "Параметры сброшены. Нажмите 'Run'.")


def build_gui(df: pd.DataFrame) -> None:
    dpg.create_context()
    dpg.create_viewport(title="S_01 TrailingMA Backtester", width=840, height=900)
    dpg.setup_dearpygui()
    create_monochrome_theme()

    with dpg.window(label="S_01 TrailingMA Backtester",
                    tag="main_window",
                    width=800,
                    height=850,
                    pos=[20, 20],
                    no_resize=True,
                    no_move=False,
                    no_close=False):

        dpg.add_spacing(count=2)
        with dpg.group():
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="Date Filter", default_value=True)
                dpg.add_spacing(count=3)
                dpg.add_checkbox(label="Backtester", default_value=True)

            dpg.add_spacing(count=2)
            add_datetime_input("Start Date",
                               START_DATE.strftime("%Y-%m-%d"),
                               START_DATE.strftime("%H:%M"),
                               "start_date",
                               "start_time")
            add_datetime_input("End Date",
                               END_DATE.strftime("%Y-%m-%d"),
                               END_DATE.strftime("%H:%M"),
                               "end_date",
                               "end_time")
            dpg.add_spacing(count=3)

        with dpg.group():
            add_ma_selector("T MA Type", "t_ma")
            dpg.add_spacing(count=2)

            with dpg.group(horizontal=True):
                add_label("Length:")
                dpg.add_input_int(width=100,
                                  default_value=DEFAULT_MA_LENGTH,
                                  tag="ma_length",
                                  min_value=1,
                                  min_clamped=True)

            with dpg.group(horizontal=True):
                add_label("Close Count Long:")
                dpg.add_input_int(width=100, default_value=7, enabled=False)
                add_label("Close Count Short:")
                dpg.add_input_int(width=100, default_value=5, enabled=False)
            dpg.add_spacing(count=3)

        with dpg.collapsing_header(label="STOPS AND FILTERS", default_open=True):
            with dpg.group(horizontal=True):
                add_label("ATR Period:")
                dpg.add_input_int(width=100, default_value=14, enabled=False)
                add_label("Stop RR:")
                dpg.add_input_float(width=100, default_value=2.0, enabled=False)

        with dpg.collapsing_header(label="TRAILING STOPS", default_open=True):
            with dpg.group(horizontal=True):
                add_label("Trail RR:")
                dpg.add_input_float(width=100, default_value=1.0, enabled=False)
                add_label("Trail Offset:")
                dpg.add_input_float(width=100, default_value=1.0, enabled=False)

        with dpg.group():
            dpg.add_spacing(count=2)
            with dpg.group(horizontal=True):
                add_label("Risk Per Trade:")
                dpg.add_input_float(width=100,
                                    default_value=DEFAULT_RISK_PER_TRADE,
                                    step=0.01,
                                    tag="risk_per_trade")
                add_label("Contract Size:")
                dpg.add_input_float(width=100,
                                    default_value=DEFAULT_CONTRACT_SIZE,
                                    step=0.01,
                                    tag="contract_size")
            dpg.add_spacing(count=3)

        add_results_area()
        dpg.add_spacing(count=3)

        with dpg.group(horizontal=True):
            dpg.add_button(label="Defaults", width=100, height=30, callback=on_defaults_clicked)
            dpg.add_spacer(width=400)
            dpg.add_button(label="Cancel", width=80, height=30, callback=lambda: dpg.stop_dearpygui())
            dpg.add_button(label="Run", width=80, height=30, callback=on_run_clicked, user_data=df)

    update_all_checkbox_state()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()


def main() -> None:
    df = load_data("OKX_LINKUSDT.P, 15 2025.02.01-2025.09.09.csv")
    build_gui(df)


if __name__ == "__main__":
    main()
