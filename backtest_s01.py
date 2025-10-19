import math
import sys
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from backtesting import _stats

try:
    import tkinter as tk
    from tkinter import messagebox
except ImportError:  # pragma: no cover - tkinter is part of stdlib
    tk = None
    messagebox = None

FACTOR_T3 = 0.7
FAST_KAMA = 2
SLOW_KAMA = 30
START_DATE = pd.Timestamp('2025-04-01', tz='UTC')
END_DATE = pd.Timestamp('2025-09-01', tz='UTC')
CONTRACT_SIZE = 0.01
COMMISSION_RATE = 0.0005


@dataclass
class StrategyParameters:
    ma_type: str = "EMA"
    ma_length: int = 45
    close_count_long: int = 7
    close_count_short: int = 5
    stop_long_atr_multiplier: float = 2.0
    stop_long_rr: float = 3.0
    stop_long_lookback: int = 2
    stop_short_atr_multiplier: float = 2.0
    stop_short_rr: float = 3.0
    stop_short_lookback: int = 2
    long_stop_max_pct: float = 3.0
    short_stop_max_pct: float = 3.0
    long_stop_max_days: int = 2
    short_stop_max_days: int = 4
    trail_rr_long: float = 1.0
    trail_rr_short: float = 1.0
    trail_ma_long_type: str = "SMA"
    trail_ma_long_length: int = 160
    trail_ma_long_offset: float = -1.0
    trail_ma_short_type: str = "SMA"
    trail_ma_short_length: int = 160
    trail_ma_short_offset: float = 1.0
    risk_per_trade: float = 2.0
    contract_size: float = CONTRACT_SIZE
    date_filter_enabled: bool = True
    backtester_enabled: bool = True
    start_date: pd.Timestamp = START_DATE
    end_date: pd.Timestamp = END_DATE


def default_parameters() -> StrategyParameters:
    return StrategyParameters()


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


def run_strategy(df: pd.DataFrame, params: StrategyParameters) -> Tuple[float, float, int]:
    if not params.backtester_enabled:
        return 0.0, 0.0, 0

    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    ma3 = get_ma(close, params.ma_type, params.ma_length, volume, high, low)
    atr14 = atr(high, low, close, 14)
    lowest_long = low.rolling(max(1, params.stop_long_lookback), min_periods=1).min()
    highest_short = high.rolling(max(1, params.stop_short_lookback), min_periods=1).max()

    trail_ma_long = get_ma(
        close,
        params.trail_ma_long_type,
        params.trail_ma_long_length,
        volume,
        high,
        low,
    )
    trail_ma_long = trail_ma_long * (1 + (params.trail_ma_long_offset or 0.0) / 100)
    trail_ma_short = get_ma(
        close,
        params.trail_ma_short_type,
        params.trail_ma_short_length,
        volume,
        high,
        low,
    )
    trail_ma_short = trail_ma_short * (1 + (params.trail_ma_short_offset or 0.0) / 100)

    times = df.index
    if params.date_filter_enabled:
        start = params.start_date if params.start_date is not None else times.min()
        end = params.end_date if params.end_date is not None else times.max()
        time_in_range = ((times >= start) & (times <= end)).astype(bool)
    else:
        time_in_range = np.ones(len(df), dtype=bool)

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
                if days_in_trade >= params.long_stop_max_days:
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
                if days_in_trade >= params.short_stop_max_days:
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

        up_trend = counter_close_trend_long >= params.close_count_long and counter_trade_long == 0
        down_trend = counter_close_trend_short >= params.close_count_short and counter_trade_short == 0

        can_open_long = (
            up_trend and position == 0 and prev_position == 0 and time_in_range[i] and
            not np.isnan(atr_value) and not np.isnan(lowest_value)
        )
        can_open_short = (
            down_trend and position == 0 and prev_position == 0 and time_in_range[i] and
            not np.isnan(atr_value) and not np.isnan(highest_value)
        )

        if can_open_long:
            stop_size = atr_value * params.stop_long_atr_multiplier
            long_stop_price = lowest_value - stop_size
            long_stop_distance = c - long_stop_price
            if long_stop_distance > 0:
                long_stop_pct = (long_stop_distance / c) * 100
                if long_stop_pct <= params.long_stop_max_pct:
                    risk_cash = realized_equity * (params.risk_per_trade / 100)
                    if params.contract_size <= 0:
                        qty = 0.0
                    else:
                        qty = risk_cash / long_stop_distance
                        qty = math.floor((qty / params.contract_size)) * params.contract_size
                    if qty > 0:
                        position = 1
                        position_size = qty
                        entry_price = c
                        stop_price = long_stop_price
                        target_price = c + long_stop_distance * params.stop_long_rr
                        trail_price_long = long_stop_price
                        trail_activated_long = False
                        entry_time_long = time
                        entry_commission = entry_price * position_size * COMMISSION_RATE
                        realized_equity -= entry_commission

        if can_open_short and position == 0:
            stop_size = atr_value * params.stop_short_atr_multiplier
            short_stop_price = highest_value + stop_size
            short_stop_distance = short_stop_price - c
            if short_stop_distance > 0:
                short_stop_pct = (short_stop_distance / c) * 100
                if short_stop_pct <= params.short_stop_max_pct:
                    risk_cash = realized_equity * (params.risk_per_trade / 100)
                    if params.contract_size <= 0:
                        qty = 0.0
                    else:
                        qty = risk_cash / short_stop_distance
                        qty = math.floor((qty / params.contract_size)) * params.contract_size
                    if qty > 0:
                        position = -1
                        position_size = qty
                        entry_price = c
                        stop_price = short_stop_price
                        target_price = c - short_stop_distance * params.stop_short_rr
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


MA_TYPES: Sequence[str] = (
    "SMA",
    "EMA",
    "WMA",
    "HMA",
    "VWMA",
    "VWAP",
    "ALMA",
    "DEMA",
    "KAMA",
    "TMA",
    "T3",
)


class BacktesterGUI:
    WINDOW_BG = "#f5f5f5"
    VIEWPORT_BG = "#e8e8e8"
    TITLE_BG = "#4a4a4a"
    TITLE_FG = "#ffffff"
    TEXT_PRIMARY = "#2a2a2a"
    TEXT_SECONDARY = "#3a3a3a"
    BORDER_PRIMARY = "#999999"
    BORDER_SECONDARY = "#bbbbbb"
    BORDER_TERTIARY = "#cccccc"
    SCALE_FACTOR = 0.92

    def _scale(self, value: float) -> int:
        if value == 0:
            return 0
        return max(1, int(round(value * self.SCALE_FACTOR)))

    def _font(self, size: int, *, weight: str = "normal", slant: Optional[str] = None) -> Tuple:
        scaled_size = max(1, int(round(size * self.SCALE_FACTOR)))
        styles: List[str] = []
        if weight != "normal":
            styles.append(weight)
        if slant:
            styles.append(slant)
        if styles:
            return ("Segoe UI", scaled_size, " ".join(styles))
        return ("Segoe UI", scaled_size)

    def _pad(self, value):
        if isinstance(value, tuple):
            return tuple(self._scale(v) for v in value)
        return self._scale(value)

    def __init__(self) -> None:
        if tk is None:
            raise RuntimeError("Tkinter is required to run the GUI")

        self.root = tk.Tk()
        self.root.title("TrailingMA Backtester")
        self.root.configure(bg=self.VIEWPORT_BG)
        self.root.tk.call("tk", "scaling", self.SCALE_FACTOR)
        self.root.minsize(self._scale(820), self._scale(880))

        self.params = default_parameters()
        self.data = load_data("OKX_LINKUSDT.P, 15 2025.02.01-2025.09.09.csv")

        self.entries: Dict[str, tk.Entry] = {}
        self.trail_ma_long_vars: Dict[str, tk.BooleanVar] = {}
        self.trail_ma_short_vars: Dict[str, tk.BooleanVar] = {}
        self.trail_ma_long_all_var = tk.BooleanVar(value=False)
        self.trail_ma_short_all_var = tk.BooleanVar(value=False)

        self.date_filter_var = tk.BooleanVar(value=self.params.date_filter_enabled)
        self.backtester_var = tk.BooleanVar(value=self.params.backtester_enabled)
        self.ma_vars: Dict[str, tk.BooleanVar] = {}
        self.all_ma_var = tk.BooleanVar(value=False)
        self.start_date_var = tk.StringVar(value=self.params.start_date.strftime("%Y-%m-%d"))
        self.start_time_var = tk.StringVar(value=self.params.start_date.strftime("%H:%M"))
        self.end_date_var = tk.StringVar(value=self.params.end_date.strftime("%Y-%m-%d"))
        self.end_time_var = tk.StringVar(value=self.params.end_date.strftime("%H:%M"))
        self.results_placeholder = "Нажмите 'Run' для запуска бэктеста..."

        self._build_window()
        self.root.update_idletasks()
        required_width = self.root.winfo_reqwidth()
        required_height = self.root.winfo_reqheight()
        self.root.geometry(f"{required_width}x{required_height}")

    def _build_window(self) -> None:
        window = tk.Frame(
            self.root,
            bg=self.WINDOW_BG,
            highlightbackground=self.BORDER_PRIMARY,
            highlightcolor=self.BORDER_PRIMARY,
            highlightthickness=1,
            bd=0,
        )
        window.pack(fill="both", expand=True, padx=0, pady=0)

        title_bar = tk.Frame(window, bg=self.TITLE_BG, height=self._scale(38))
        title_bar.pack(fill="x")
        title_label = tk.Label(
            title_bar,
            text="TrailingMA Backtester",
            bg=self.TITLE_BG,
            fg=self.TITLE_FG,
            font=self._font(15),
        )
        title_label.pack(side="left", padx=self._pad((12, 0)), pady=self._scale(6))

        controls = tk.Frame(title_bar, bg=self.TITLE_BG)
        controls.pack(side="right", padx=self._pad(12), pady=self._scale(6))
        self._add_title_button(controls, "–", lambda: self.root.iconify())
        self._add_title_button(controls, "□", self._maximize_window)
        self._add_title_button(controls, "✕", self.root.destroy)

        content_wrapper = tk.Frame(window, bg=self.WINDOW_BG)
        content_wrapper.pack(fill="both", expand=True)

        canvas = tk.Canvas(
            content_wrapper,
            bg=self.WINDOW_BG,
            highlightthickness=0,
            bd=0,
        )
        scrollbar = tk.Scrollbar(content_wrapper, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self.content = tk.Frame(canvas, bg=self.WINDOW_BG)
        canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.content.bind(
            "<Configure>",
            lambda event: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        container = tk.Frame(self.content, bg=self.WINDOW_BG)
        container.pack(fill="both", expand=True, padx=self._pad(20), pady=self._pad(16))

        self._build_date_section(container)
        self._build_ma_section(container)
        self._build_stops_section(container)
        self._build_trailing_section(container)
        self._build_risk_section(container)
        self._build_results_section(container)
        self._build_action_bar(container)

    def _add_title_button(self, parent: tk.Widget, text: str, command) -> None:
        button = tk.Button(
            parent,
            text=text,
            command=command,
            bg=self.TITLE_BG,
            fg=self.TITLE_FG,
            activebackground="#3a3a3a",
            activeforeground=self.TITLE_FG,
            bd=0,
            font=self._font(12),
            padx=self._scale(8),
        )
        button.pack(side="left", padx=self._scale(4))

    def _maximize_window(self) -> None:
        try:
            self.root.state("zoomed")
        except tk.TclError:
            try:
                self.root.attributes("-zoomed", True)
            except tk.TclError:
                pass

    def _build_section(self, parent: tk.Widget, title: str) -> tk.Frame:
        section = tk.Frame(parent, bg=self.WINDOW_BG)
        section.pack(fill="x", pady=self._pad((0, 16)))
        label = tk.Label(
            section,
            text=title.upper(),
            bg=self.WINDOW_BG,
            fg=self.TEXT_SECONDARY,
            font=self._font(12, weight="bold"),
        )
        label.pack(fill="x", pady=self._pad((0, 8)))
        underline = tk.Frame(section, bg=self.BORDER_SECONDARY, height=1)
        underline.pack(fill="x", pady=self._pad((0, 10)))
        return section

    def _build_date_section(self, parent: tk.Widget) -> None:
        section = self._build_section(parent, "Date Filter")

        checkbox_row = tk.Frame(section, bg=self.WINDOW_BG)
        checkbox_row.pack(fill="x", pady=self._pad((0, 10)))

        self._add_checkbox(checkbox_row, "Date Filter", self.date_filter_var)
        self._add_checkbox(checkbox_row, "Backtester", self.backtester_var)

        self._build_date_inputs(section, "Start Date", self.start_date_var, self.start_time_var)
        self._build_date_inputs(section, "End Date", self.end_date_var, self.end_time_var)

    def _build_date_inputs(self, section: tk.Frame, label_text: str, date_var: tk.StringVar, time_var: tk.StringVar) -> None:
        group = tk.Frame(section, bg=self.WINDOW_BG)
        group.pack(fill="x", pady=self._pad((0, 10)))
        label = tk.Label(
            group,
            text=label_text,
            bg=self.WINDOW_BG,
            fg=self.TEXT_PRIMARY,
            font=self._font(14),
            width=14,
            anchor="w",
        )
        label.pack(side="left")

        date_entry = self._create_entry(group, width=14)
        date_entry.configure(textvariable=date_var)
        date_entry.pack(side="left", padx=self._pad((8, 6)))

        calendar_button = tk.Button(
            group,
            text="📅",
            bg="#cccccc",
            fg=self.TEXT_PRIMARY,
            activebackground="#bbbbbb",
            activeforeground=self.TEXT_PRIMARY,
            bd=0,
            padx=self._scale(8),
            pady=self._scale(4),
            font=self._font(12),
        )
        calendar_button.pack(side="left")

        time_entry = self._create_entry(group, width=8)
        time_entry.configure(textvariable=time_var)
        time_entry.pack(side="left", padx=self._pad((8, 0)))

    def _build_ma_section(self, parent: tk.Widget) -> None:
        section = self._build_section(parent, "MA Settings")

        label = tk.Label(
            section,
            text="T MA Type",
            bg=self.WINDOW_BG,
            fg=self.TEXT_PRIMARY,
            font=self._font(14),
        )
        label.pack(anchor="w", pady=self._pad((0, 8)))

        ma_container = tk.Frame(section, bg=self.WINDOW_BG)
        ma_container.pack(fill="x", pady=self._pad((0, 10)))

        self._add_checkbox(ma_container, "ALL", self.all_ma_var, command=self._toggle_all_ma)

        grid = tk.Frame(section, bg=self.WINDOW_BG)
        grid.pack(fill="x", pady=self._pad((4, 10)))

        for idx, ma_type in enumerate(MA_TYPES):
            var = tk.BooleanVar(value=(ma_type == self.params.ma_type))
            self.ma_vars[ma_type] = var
            checkbox = tk.Checkbutton(
                grid,
                text=ma_type,
                variable=var,
                command=self._ma_selection_changed,
                bg=self.WINDOW_BG,
                fg=self.TEXT_PRIMARY,
                selectcolor=self.WINDOW_BG,
                activebackground=self.WINDOW_BG,
                activeforeground=self.TEXT_PRIMARY,
                highlightthickness=0,
                font=self._font(13),
                padx=self._scale(6),
            )
            row = idx // 4
            col = idx % 4
            checkbox.grid(row=row, column=col, padx=self._scale(12), pady=self._scale(3), sticky="w")

        self._create_labeled_entry(section, "Length", "ma_length", str(self.params.ma_length))
        self._create_labeled_entry(section, "Close Count Long", "close_count_long", str(self.params.close_count_long))
        self._create_labeled_entry(section, "Close Count Short", "close_count_short", str(self.params.close_count_short))

    def _build_stops_section(self, parent: tk.Widget) -> None:
        container = self._build_collapsible(parent, "Stops and Filters")
        grid = tk.Frame(container, bg=self.WINDOW_BG)
        grid.pack(fill="x")

        group_configs = [
            (
                ("Stop Long X", "stop_long_atr_multiplier", str(self.params.stop_long_atr_multiplier)),
                ("RR", "stop_long_rr", str(self.params.stop_long_rr)),
                ("LP", "stop_long_lookback", str(self.params.stop_long_lookback)),
            ),
            (
                ("Stop Short X", "stop_short_atr_multiplier", str(self.params.stop_short_atr_multiplier)),
                ("RR", "stop_short_rr", str(self.params.stop_short_rr)),
                ("LP", "stop_short_lookback", str(self.params.stop_short_lookback)),
            ),
            (
                ("L Stop Max %", "long_stop_max_pct", str(self.params.long_stop_max_pct)),
                ("S Stop Max %", "short_stop_max_pct", str(self.params.short_stop_max_pct)),
            ),
            (
                ("L Stop Max D", "long_stop_max_days", str(self.params.long_stop_max_days)),
                ("S Stop Max D", "short_stop_max_days", str(self.params.short_stop_max_days)),
            ),
        ]

        for idx, params in enumerate(group_configs):
            frame = self._create_param_group(grid, params, use_pack=False)
            row = idx // 2
            column = idx % 2
            pad_tuple = (0, 12) if column == 0 else (12, 0)
            frame.grid(
                row=row,
                column=column,
                padx=self._pad(pad_tuple),
                pady=self._scale(5),
                sticky="nsew",
            )

        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

    def _build_trailing_section(self, parent: tk.Widget) -> None:
        container = self._build_collapsible(parent, "Trailing Stops")

        self._create_param_group(
            container,
            (
                ("Trail RR Long", "trail_rr_long", str(self.params.trail_rr_long)),
                ("Trail RR Short", "trail_rr_short", str(self.params.trail_rr_short)),
            ),
        ).pack_configure(pady=self._scale(3))

        selectors = tk.Frame(container, bg=self.WINDOW_BG)
        selectors.pack(fill="x", pady=self._pad((4, 0)))

        long_column = tk.Frame(selectors, bg=self.WINDOW_BG)
        long_column.pack(side="left", fill="both", expand=True, padx=self._pad((0, 8)))
        short_column = tk.Frame(selectors, bg=self.WINDOW_BG)
        short_column.pack(side="left", fill="both", expand=True, padx=self._pad((8, 0)))

        self._build_trailing_selector(
            long_column,
            title="Trail MA Long",
            vars_map=self.trail_ma_long_vars,
            default=self.params.trail_ma_long_type,
            all_var=self.trail_ma_long_all_var,
        )
        long_group = self._create_param_group(
            long_column,
            (
                ("Length", "trail_ma_long_length", str(self.params.trail_ma_long_length)),
                ("Offset", "trail_ma_long_offset", str(self.params.trail_ma_long_offset)),
            ),
        )
        long_group.pack_configure(pady=self._scale(3))

        self._build_trailing_selector(
            short_column,
            title="Trail MA Short",
            vars_map=self.trail_ma_short_vars,
            default=self.params.trail_ma_short_type,
            all_var=self.trail_ma_short_all_var,
        )
        short_group = self._create_param_group(
            short_column,
            (
                ("Length", "trail_ma_short_length", str(self.params.trail_ma_short_length)),
                ("Offset", "trail_ma_short_offset", str(self.params.trail_ma_short_offset)),
            ),
        )
        short_group.pack_configure(pady=self._scale(3))

    def _build_trailing_selector(
        self,
        parent: tk.Widget,
        title: str,
        vars_map: Dict[str, tk.BooleanVar],
        default: str,
        *,
        all_var: tk.BooleanVar,
    ) -> None:
        label = tk.Label(
            parent,
            text=title,
            bg=self.WINDOW_BG,
            fg=self.TEXT_PRIMARY,
            font=self._font(14),
        )
        label.pack(anchor="w", pady=self._pad((6, 4)))

        controls = tk.Frame(parent, bg=self.WINDOW_BG)
        controls.pack(fill="x", pady=self._pad((0, 4)))

        all_checkbox = tk.Checkbutton(
            controls,
            text="ALL",
            variable=all_var,
            command=lambda mapping=vars_map, flag=all_var: self._toggle_trailing_all(mapping, flag),
            bg=self.WINDOW_BG,
            fg=self.TEXT_PRIMARY,
            selectcolor=self.WINDOW_BG,
            activebackground=self.WINDOW_BG,
            activeforeground=self.TEXT_PRIMARY,
            highlightthickness=0,
            font=self._font(13),
            padx=self._scale(6),
        )
        all_checkbox.pack(anchor="w")

        grid = tk.Frame(parent, bg=self.WINDOW_BG)
        grid.pack(fill="x", pady=self._pad((0, 8)))
        for idx, ma_type in enumerate(MA_TYPES):
            var = tk.BooleanVar(value=(ma_type == default))
            vars_map[ma_type] = var
            checkbox = tk.Checkbutton(
                grid,
                text=ma_type,
                variable=var,
                command=lambda mapping=vars_map, flag=all_var: self._trailing_selection_changed(mapping, flag),
                bg=self.WINDOW_BG,
                fg=self.TEXT_PRIMARY,
                selectcolor=self.WINDOW_BG,
                activebackground=self.WINDOW_BG,
                activeforeground=self.TEXT_PRIMARY,
                highlightthickness=0,
                font=self._font(13),
                padx=self._scale(6),
            )
            row = idx // 4
            col = idx % 4
            checkbox.grid(row=row, column=col, padx=self._scale(12), pady=self._scale(3), sticky="w")

    def _toggle_trailing_all(
        self,
        mapping: Dict[str, tk.BooleanVar],
        all_var: tk.BooleanVar,
    ) -> None:
        state = all_var.get()
        for var in mapping.values():
            var.set(state)

    def _trailing_selection_changed(
        self,
        mapping: Dict[str, tk.BooleanVar],
        all_var: tk.BooleanVar,
    ) -> None:
        if all(var.get() for var in mapping.values()):
            all_var.set(True)
        else:
            all_var.set(False)

    def _build_risk_section(self, parent: tk.Widget) -> None:
        section = self._build_section(parent, "Risk Settings")
        self._create_labeled_entry(section, "Risk Per Trade", "risk_per_trade", str(self.params.risk_per_trade), width=10)
        self._create_labeled_entry(section, "Contract Size", "contract_size", str(self.params.contract_size), width=10)

    def _build_results_section(self, parent: tk.Widget) -> None:
        section = self._build_section(parent, "Results")
        results_frame = tk.Frame(section, bg=self.WINDOW_BG)
        results_frame.pack(fill="both", expand=True)
        self.results_text = tk.Text(
            results_frame,
            bg=self.VIEWPORT_BG,
            fg="#777777",
            font=self._font(14, slant="italic"),
            height=7,
            width=60,
            relief="solid",
            bd=1,
            wrap="word",
            highlightbackground=self.BORDER_PRIMARY,
            highlightcolor=self.BORDER_PRIMARY,
        )
        self.results_text.insert("1.0", self.results_placeholder)
        self.results_text.configure(state="disabled")
        self.results_text.pack(fill="both", expand=True, pady=self._pad((0, 0)))

    def _build_action_bar(self, parent: tk.Widget) -> None:
        separator = tk.Frame(parent, bg=self.BORDER_SECONDARY, height=1)
        separator.pack(fill="x", pady=self._pad((16, 0)))
        bar = tk.Frame(parent, bg=self.WINDOW_BG)
        bar.pack(fill="x")
        left = tk.Frame(bar, bg=self.WINDOW_BG)
        left.pack(side="left", padx=self._pad(10), pady=self._pad(12))
        right = tk.Frame(bar, bg=self.WINDOW_BG)
        right.pack(side="right", padx=self._pad(10), pady=self._pad(12))

        self._add_button(left, "Defaults", self._reset_defaults, secondary=True)
        self._add_button(right, "Cancel", self.root.destroy, secondary=True)
        self._add_button(right, "Run", self._on_run)

    def _add_button(self, parent: tk.Widget, text: str, command, secondary: bool = False) -> None:
        bg = "#cccccc" if secondary else self.TITLE_BG
        fg = self.TEXT_PRIMARY if secondary else self.TITLE_FG
        hover = "#bbbbbb" if secondary else "#3a3a3a"
        button = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=hover,
            activeforeground=fg,
            bd=0,
            padx=self._scale(16),
            pady=self._scale(8),
            font=self._font(14),
        )
        button.pack(side="left", padx=self._scale(6))

    def _add_checkbox(self, parent: tk.Widget, text: str, variable: tk.BooleanVar, command=None) -> None:
        checkbox = tk.Checkbutton(
            parent,
            text=text,
            variable=variable,
            command=command,
            bg=self.WINDOW_BG,
            fg=self.TEXT_PRIMARY,
            selectcolor=self.WINDOW_BG,
            activebackground=self.WINDOW_BG,
            activeforeground=self.TEXT_PRIMARY,
            font=self._font(14),
            highlightthickness=0,
            padx=self._scale(6),
        )
        checkbox.pack(side="left", padx=self._pad((0, 20)))

    def _create_entry(self, parent: tk.Widget, width: int = 10) -> tk.Entry:
        entry = tk.Entry(
            parent,
            bg="#ffffff",
            fg=self.TEXT_PRIMARY,
            relief="solid",
            bd=1,
            highlightthickness=0,
            font=self._font(14),
            width=width,
        )
        return entry

    def _create_labeled_entry(
        self,
        parent: tk.Widget,
        label_text: str,
        key: str,
        default: str,
        width: int = 10,
    ) -> None:
        group = tk.Frame(parent, bg=self.WINDOW_BG)
        group.pack(fill="x", pady=self._pad((0, 10)))
        label = tk.Label(
            group,
            text=label_text,
            bg=self.WINDOW_BG,
            fg=self.TEXT_PRIMARY,
            font=self._font(14),
            width=18,
            anchor="w",
        )
        label.pack(side="left")
        entry = self._create_entry(group, width)
        entry.insert(0, default)
        entry.pack(side="left", padx=self._pad((8, 0)))
        self.entries[key] = entry

    def _create_param_group(
        self,
        parent: tk.Widget,
        params: Sequence[Tuple[str, str, str]],
        *,
        use_pack: bool = True,
    ) -> tk.Frame:
        frame = tk.Frame(
            parent,
            bg=self.VIEWPORT_BG,
            highlightbackground=self.BORDER_TERTIARY,
            highlightthickness=1,
            bd=0,
            padx=self._scale(10),
            pady=self._scale(6),
        )
        if use_pack:
            frame.pack(fill="x", pady=self._scale(6))

        for label_text, key, default in params:
            label = tk.Label(
                frame,
                text=label_text,
                bg=self.VIEWPORT_BG,
                fg=self.TEXT_PRIMARY,
                font=self._font(13),
            )
            label.pack(side="left", padx=self._pad((0, 6)))
            entry = self._create_entry(frame, width=8)
            entry.insert(0, default)
            entry.pack(side="left", padx=self._pad((0, 10)))
            self.entries[key] = entry

        return frame

    def _build_collapsible(self, parent: tk.Widget, title: str) -> tk.Frame:
        section = tk.Frame(parent, bg=self.WINDOW_BG)
        section.pack(fill="x", pady=self._pad((0, 16)))

        header = tk.Frame(section, bg=self.WINDOW_BG)
        header.pack(fill="x")
        label = tk.Label(
            header,
            text=title.upper(),
            bg=self.WINDOW_BG,
            fg=self.TEXT_SECONDARY,
            font=self._font(12, weight="bold"),
        )
        label.pack(side="left")

        toggle_var = tk.BooleanVar(value=True)

        def toggle() -> None:
            state = toggle_var.get()
            if state:
                content.pack(fill="x", pady=self._pad((10, 0)))
                button.configure(text="–")
            else:
                content.forget()
                button.configure(text="+")

        button = tk.Button(
            header,
            text="–",
            command=lambda: self._toggle_collapsible(toggle_var, toggle),
            bg="#cccccc",
            fg=self.TEXT_PRIMARY,
            activebackground="#bbbbbb",
            activeforeground=self.TEXT_PRIMARY,
            bd=0,
            width=3,
            font=self._font(12, weight="bold"),
        )
        button.pack(side="right")

        content = tk.Frame(section, bg=self.WINDOW_BG)
        content.pack(fill="x", pady=self._pad((10, 0)))
        section.toggle_var = toggle_var  # type: ignore[attr-defined]
        section.toggle_action = toggle  # type: ignore[attr-defined]
        section.toggle_button = button  # type: ignore[attr-defined]
        section.content_frame = content  # type: ignore[attr-defined]
        return content

    def _toggle_collapsible(self, toggle_var: tk.BooleanVar, action) -> None:
        toggle_var.set(not toggle_var.get())
        action()

    def _toggle_all_ma(self) -> None:
        state = self.all_ma_var.get()
        for var in self.ma_vars.values():
            var.set(state)

    def _ma_selection_changed(self) -> None:
        if all(var.get() for var in self.ma_vars.values()):
            self.all_ma_var.set(True)
        else:
            self.all_ma_var.set(False)

    def _reset_defaults(self) -> None:
        self.params = default_parameters()
        for key, entry in self.entries.items():
            entry.delete(0, tk.END)
        defaults = {
            "ma_length": str(self.params.ma_length),
            "close_count_long": str(self.params.close_count_long),
            "close_count_short": str(self.params.close_count_short),
            "stop_long_atr_multiplier": str(self.params.stop_long_atr_multiplier),
            "stop_long_rr": str(self.params.stop_long_rr),
            "stop_long_lookback": str(self.params.stop_long_lookback),
            "stop_short_atr_multiplier": str(self.params.stop_short_atr_multiplier),
            "stop_short_rr": str(self.params.stop_short_rr),
            "stop_short_lookback": str(self.params.stop_short_lookback),
            "long_stop_max_pct": str(self.params.long_stop_max_pct),
            "short_stop_max_pct": str(self.params.short_stop_max_pct),
            "long_stop_max_days": str(self.params.long_stop_max_days),
            "short_stop_max_days": str(self.params.short_stop_max_days),
            "trail_rr_long": str(self.params.trail_rr_long),
            "trail_rr_short": str(self.params.trail_rr_short),
            "trail_ma_long_length": str(self.params.trail_ma_long_length),
            "trail_ma_long_offset": str(self.params.trail_ma_long_offset),
            "trail_ma_short_length": str(self.params.trail_ma_short_length),
            "trail_ma_short_offset": str(self.params.trail_ma_short_offset),
            "risk_per_trade": str(self.params.risk_per_trade),
            "contract_size": str(self.params.contract_size),
        }
        for key, value in defaults.items():
            if key in self.entries:
                self.entries[key].insert(0, value)

        for ma_type, var in self.ma_vars.items():
            var.set(ma_type == self.params.ma_type)
        self.all_ma_var.set(False)

        self.trail_ma_long_all_var.set(False)
        self.trail_ma_short_all_var.set(False)
        for mapping, default in (
            (self.trail_ma_long_vars, self.params.trail_ma_long_type),
            (self.trail_ma_short_vars, self.params.trail_ma_short_type),
        ):
            for ma_type, var in mapping.items():
                var.set(ma_type == default)

        self.date_filter_var.set(self.params.date_filter_enabled)
        self.backtester_var.set(self.params.backtester_enabled)
        self.start_date_var.set(self.params.start_date.strftime("%Y-%m-%d"))
        self.start_time_var.set(self.params.start_date.strftime("%H:%M"))
        self.end_date_var.set(self.params.end_date.strftime("%Y-%m-%d"))
        self.end_time_var.set(self.params.end_date.strftime("%H:%M"))
        self._set_results(self.results_placeholder, placeholder=True)

    def _set_results(self, text: str, placeholder: bool = False) -> None:
        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert("1.0", text)
        if placeholder:
            self.results_text.configure(fg="#777777", font=self._font(14, slant="italic"))
        else:
            self.results_text.configure(fg=self.TEXT_PRIMARY, font=self._font(14))
        self.results_text.configure(state="disabled")

    def _on_run(self) -> None:
        if not self.backtester_var.get():
            self._set_results("Backtester отключен.")
            return

        selected_ma = self._selected_ma_types(
            self.ma_vars,
            self.all_ma_var,
            require_selection=True,
        )

        if not selected_ma:
            if messagebox is not None:
                messagebox.showerror("Ошибка", "Выберите хотя бы один тип MA для тестирования")
            return

        trailing_long_types = self._selected_ma_types(
            self.trail_ma_long_vars,
            self.trail_ma_long_all_var,
            default=self.params.trail_ma_long_type,
        )
        trailing_short_types = self._selected_ma_types(
            self.trail_ma_short_vars,
            self.trail_ma_short_all_var,
            default=self.params.trail_ma_short_type,
        )

        try:
            params_list = [
                self._collect_parameters(ma_type, trail_long, trail_short)
                for ma_type in selected_ma
                for trail_long in trailing_long_types
                for trail_short in trailing_short_types
            ]
        except ValueError as exc:  # invalid input
            if messagebox is not None:
                messagebox.showerror("Ошибка", str(exc))
            return

        self._set_results("Выполняется тестирование...")
        threading.Thread(
            target=self._execute_backtests,
            args=(params_list,),
            daemon=True,
        ).start()

    def _collect_parameters(
        self,
        ma_type: str,
        trail_long_type: str,
        trail_short_type: str,
    ) -> StrategyParameters:
        params = StrategyParameters()
        defaults = StrategyParameters()
        params.ma_type = ma_type
        params.ma_length = self._get_int("ma_length", minimum=1)
        params.close_count_long = self._get_int("close_count_long", minimum=0)
        params.close_count_short = self._get_int("close_count_short", minimum=0)
        params.stop_long_atr_multiplier = self._get_float("stop_long_atr_multiplier", minimum=0)
        params.stop_long_rr = self._get_float("stop_long_rr", minimum=0)
        params.stop_long_lookback = self._get_int("stop_long_lookback", minimum=1)
        params.stop_short_atr_multiplier = self._get_float("stop_short_atr_multiplier", minimum=0)
        params.stop_short_rr = self._get_float("stop_short_rr", minimum=0)
        params.stop_short_lookback = self._get_int("stop_short_lookback", minimum=1)
        params.long_stop_max_pct = self._get_float("long_stop_max_pct", minimum=0)
        params.short_stop_max_pct = self._get_float("short_stop_max_pct", minimum=0)
        params.long_stop_max_days = self._get_int("long_stop_max_days", minimum=0)
        params.short_stop_max_days = self._get_int("short_stop_max_days", minimum=0)
        params.trail_rr_long = self._get_float("trail_rr_long", minimum=0)
        params.trail_rr_short = self._get_float("trail_rr_short", minimum=0)
        params.trail_ma_long_length = self._get_int("trail_ma_long_length", minimum=1)
        params.trail_ma_long_offset = self._get_float("trail_ma_long_offset")
        params.trail_ma_short_length = self._get_int("trail_ma_short_length", minimum=1)
        params.trail_ma_short_offset = self._get_float("trail_ma_short_offset")
        params.risk_per_trade = self._get_float("risk_per_trade", minimum=0)
        params.contract_size = self._get_float("contract_size", minimum=0)
        params.date_filter_enabled = self.date_filter_var.get()
        params.backtester_enabled = self.backtester_var.get()
        params.start_date = self._parse_datetime(self.start_date_var.get(), self.start_time_var.get(), START_DATE)
        params.end_date = self._parse_datetime(self.end_date_var.get(), self.end_time_var.get(), END_DATE)
        if params.date_filter_enabled and params.start_date > params.end_date:
            raise ValueError("Дата начала должна быть меньше или равна дате окончания")
        params.trail_ma_long_type = trail_long_type or defaults.trail_ma_long_type
        params.trail_ma_short_type = trail_short_type or defaults.trail_ma_short_type
        return params

    def _selected_ma_types(
        self,
        mapping: Dict[str, tk.BooleanVar],
        all_var: tk.BooleanVar,
        *,
        default: Optional[str] = None,
        require_selection: bool = False,
    ) -> List[str]:
        if all_var.get():
            return list(MA_TYPES)
        selected = [ma_type for ma_type, var in mapping.items() if var.get()]
        if selected:
            return selected
        if default is not None:
            return [default]
        if require_selection:
            return []
        return []

    def _get_int(self, key: str, minimum: Optional[int] = None) -> int:
        value = self.entries[key].get().strip()
        if value == "":
            raise ValueError(f"Поле '{key}' не может быть пустым")
        try:
            result = int(float(value))
        except ValueError as exc:
            raise ValueError(f"Некорректное значение для '{key}'") from exc
        if minimum is not None and result < minimum:
            raise ValueError(f"Значение '{key}' должно быть ≥ {minimum}")
        return result

    def _get_float(self, key: str, minimum: Optional[float] = None) -> float:
        value = self.entries[key].get().strip()
        if value == "":
            raise ValueError(f"Поле '{key}' не может быть пустым")
        try:
            result = float(value)
        except ValueError as exc:
            raise ValueError(f"Некорректное значение для '{key}'") from exc
        if minimum is not None and result < minimum:
            raise ValueError(f"Значение '{key}' должно быть ≥ {minimum}")
        return result

    def _parse_datetime(self, date_str: str, time_str: str, default: pd.Timestamp) -> pd.Timestamp:
        if not date_str:
            return default
        time_component = time_str if time_str else "00:00"
        candidate = f"{date_str.strip()} {time_component.strip()}"
        try:
            ts = pd.Timestamp(candidate, tz="UTC")
        except Exception as exc:
            raise ValueError("Некорректная дата или время") from exc
        return ts

    def _execute_backtests(self, params_list: Sequence[StrategyParameters]) -> None:
        try:
            results: List[str] = []
            for params in params_list:
                net_profit, max_drawdown, trades = run_strategy(self.data, params)
                result_text = (
                    f"MA: {params.ma_type}\n"
                    f"Net Profit %: {net_profit:.2f}\n"
                    f"Max Portfolio Drawdown %: {max_drawdown:.2f}\n"
                    f"Total Trades: {trades}"
                )
                results.append(result_text)
            formatted = "\n\n".join(results)
            self.root.after(0, lambda: self._set_results(formatted))
        except Exception as exc:  # pragma: no cover - GUI feedback path
            self.root.after(0, lambda: self._set_results(f"Ошибка при выполнении: {exc}"))

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        df = load_data("OKX_LINKUSDT.P, 15 2025.02.01-2025.09.09.csv")
        params = default_parameters()
        net_profit, max_drawdown, trades = run_strategy(df, params)
        print(f"Net Profit %: {net_profit:.2f}")
        print(f"Max Portfolio Drawdown %: {max_drawdown:.2f}")
        print(f"Total Trades: {trades}")
        return

    if tk is None:
        raise RuntimeError(
            "Tkinter is required for GUI mode. Запустите со специальным флагом --cli для режима командной строки."
        )

    app = BacktesterGUI()
    app.run()


if __name__ == "__main__":
    main()
