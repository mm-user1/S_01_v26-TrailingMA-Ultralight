import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from backtesting import _stats
import dearpygui.dearpygui as dpg

# Strategy constants
FACTOR_T3 = 0.7
FAST_KAMA = 2
SLOW_KAMA = 30

# Moving average functions
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

def run_strategy(df: pd.DataFrame, ma_type: str, start_date: pd.Timestamp, end_date: pd.Timestamp,
                 t_ma_length: int, close_count_long: int, close_count_short: int,
                 stop_long_x: float, rr_long: float, lp_long: int,
                 stop_short_x: float, rr_short: float, lp_short: int,
                 l_stop_max_pct: float, s_stop_max_pct: float,
                 l_stop_max_d: int, s_stop_max_d: int,
                 trail_rr_long: float, trail_rr_short: float,
                 trail_ma_long_type: str, trail_ma_long_length: int, trail_ma_long_offset: float,
                 trail_ma_short_type: str, trail_ma_short_length: int, trail_ma_short_offset: float,
                 risk_per_trade: float, contract_size: float, commission_rate: float = 0.0005) -> Tuple[float, float, int]:
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    ma3 = get_ma(close, ma_type, t_ma_length)
    atr14 = atr(high, low, close, 14)
    lowest_long = low.rolling(lp_long, min_periods=1).min()
    highest_short = high.rolling(lp_short, min_periods=1).max()

    trail_ma_long = get_ma(close, trail_ma_long_type, trail_ma_long_length)
    trail_ma_long = trail_ma_long * (1 + trail_ma_long_offset / 100)
    trail_ma_short = get_ma(close, trail_ma_short_type, trail_ma_short_length)
    trail_ma_short = trail_ma_short * (1 + trail_ma_short_offset / 100)

    times = df.index
    time_in_range = (times >= start_date) & (times <= end_date)

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

    trades: List[TradeRecord] = []
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
                activation_price = entry_price + (entry_price - stop_price) * trail_rr_long
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
                if days_in_trade >= l_stop_max_d:
                    exit_price = c
            if exit_price is not None:
                gross_pnl = (exit_price - entry_price) * position_size
                exit_commission = exit_price * position_size * commission_rate
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
                activation_price = entry_price - (stop_price - entry_price) * trail_rr_short
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
                if days_in_trade >= s_stop_max_d:
                    exit_price = c
            if exit_price is not None:
                gross_pnl = (entry_price - exit_price) * position_size
                exit_commission = exit_price * position_size * commission_rate
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

        up_trend = counter_close_trend_long >= close_count_long and counter_trade_long == 0
        down_trend = counter_close_trend_short >= close_count_short and counter_trade_short == 0

        can_open_long = (
            up_trend and position == 0 and prev_position == 0 and time_in_range[i] and
            not np.isnan(atr_value) and not np.isnan(lowest_value)
        )
        can_open_short = (
            down_trend and position == 0 and prev_position == 0 and time_in_range[i] and
            not np.isnan(atr_value) and not np.isnan(highest_value)
        )

        if can_open_long:
            stop_size = atr_value * stop_long_x
            long_stop_price = lowest_value - stop_size
            long_stop_distance = c - long_stop_price
            if long_stop_distance > 0:
                long_stop_pct = (long_stop_distance / c) * 100
                if long_stop_pct <= l_stop_max_pct:
                    risk_cash = realized_equity * (risk_per_trade / 100)
                    qty = risk_cash / long_stop_distance
                    qty = math.floor((qty / contract_size)) * contract_size
                    if qty > 0:
                        position = 1
                        position_size = qty
                        entry_price = c
                        stop_price = long_stop_price
                        target_price = c + long_stop_distance * rr_long
                        trail_price_long = long_stop_price
                        trail_activated_long = False
                        entry_time_long = time
                        entry_commission = entry_price * position_size * commission_rate
                        realized_equity -= entry_commission

        if can_open_short and position == 0:
            stop_size = atr_value * stop_short_x
            short_stop_price = highest_value + stop_size
            short_stop_distance = short_stop_price - c
            if short_stop_distance > 0:
                short_stop_pct = (short_stop_distance / c) * 100
                if short_stop_pct <= s_stop_max_pct:
                    risk_cash = realized_equity * (risk_per_trade / 100)
                    qty = risk_cash / short_stop_distance
                    qty = math.floor((qty / contract_size)) * contract_size
                    if qty > 0:
                        position = -1
                        position_size = qty
                        entry_price = c
                        stop_price = short_stop_price
                        target_price = c - short_stop_distance * rr_short
                        trail_price_short = short_stop_price
                        trail_activated_short = False
                        entry_time_short = time
                        entry_commission = entry_price * position_size * commission_rate
                        realized_equity -= entry_commission

        realized_curve.append(realized_equity)
        prev_position = position

    equity_series = pd.Series(realized_curve, index=df.index)
    net_profit_pct = ((realized_equity - 100.0) / 100.0) * 100
    max_drawdown_pct = compute_max_drawdown(equity_series)
    total_trades = len(trades)
    return net_profit_pct, max_drawdown_pct, total_trades

# GUI Implementation
def create_gui():
    dpg.create_context()
    
    # Create monochrome theme
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
    
    def run_backtest():
        results_text = ""
        
        # Get selected MA types
        ma_types = []
        ma_checkboxes = ['ema', 'sma', 'hma', 'wma', 'alma', 'kama', 'tma', 't3', 'dema', 'vwma', 'vwap']
        
        if dpg.get_value("t_ma_all"):
            ma_types = [ma.upper() for ma in ma_checkboxes]
        else:
            for ma in ma_checkboxes:
                if dpg.get_value(f"t_ma_{ma}"):
                    ma_types.append(ma.upper())
        
        if not ma_types:
            dpg.set_value("results_text", "Error: No MA types selected!")
            return
        
        # Get parameters
        try:
            start_date = pd.Timestamp(dpg.get_value("start_date"), tz='UTC')
            end_date = pd.Timestamp(dpg.get_value("end_date"), tz='UTC')
            t_ma_length = dpg.get_value("t_ma_length")
            close_count_long = dpg.get_value("close_count_long")
            close_count_short = dpg.get_value("close_count_short")
            stop_long_x = dpg.get_value("stop_long_x")
            rr_long = dpg.get_value("rr_long")
            lp_long = dpg.get_value("lp_long")
            stop_short_x = dpg.get_value("stop_short_x")
            rr_short = dpg.get_value("rr_short")
            lp_short = dpg.get_value("lp_short")
            l_stop_max_pct = dpg.get_value("l_stop_max_pct")
            s_stop_max_pct = dpg.get_value("s_stop_max_pct")
            l_stop_max_d = dpg.get_value("l_stop_max_d")
            s_stop_max_d = dpg.get_value("s_stop_max_d")
            trail_rr_long = dpg.get_value("trail_rr_long")
            trail_rr_short = dpg.get_value("trail_rr_short")
            trail_ma_long_length = dpg.get_value("trail_ma_long_length")
            trail_ma_long_offset = dpg.get_value("trail_ma_long_offset")
            trail_ma_short_length = dpg.get_value("trail_ma_short_length")
            trail_ma_short_offset = dpg.get_value("trail_ma_short_offset")
            risk_per_trade = dpg.get_value("risk_per_trade")
            contract_size = dpg.get_value("contract_size")
            
            # Get trail MA types
            trail_ma_long_types = []
            trail_ma_short_types = []
            
            if dpg.get_value("trail_ma_long_all"):
                trail_ma_long_types = [ma.upper() for ma in ma_checkboxes]
            else:
                for ma in ma_checkboxes:
                    if dpg.get_value(f"trail_ma_long_{ma}"):
                        trail_ma_long_types.append(ma.upper())
            
            if dpg.get_value("trail_ma_short_all"):
                trail_ma_short_types = [ma.upper() for ma in ma_checkboxes]
            else:
                for ma in ma_checkboxes:
                    if dpg.get_value(f"trail_ma_short_{ma}"):
                        trail_ma_short_types.append(ma.upper())
            
            if not trail_ma_long_types:
                trail_ma_long_types = ['SMA']
            if not trail_ma_short_types:
                trail_ma_short_types = ['SMA']
            
            # Load data
            df = load_data("OKX_LINKUSDT.P, 15 2025.02.01-2025.09.09.csv")
            
            # Run backtests
            results_text = "=" * 60 + "\n"
            results_text += "BACKTEST RESULTS\n"
            results_text += "=" * 60 + "\n\n"
            
            for ma_type in ma_types:
                for trail_long in trail_ma_long_types:
                    for trail_short in trail_ma_short_types:
                        net_profit, max_dd, total_trades = run_strategy(
                            df, ma_type, start_date, end_date,
                            t_ma_length, close_count_long, close_count_short,
                            stop_long_x, rr_long, lp_long,
                            stop_short_x, rr_short, lp_short,
                            l_stop_max_pct, s_stop_max_pct,
                            l_stop_max_d, s_stop_max_d,
                            trail_rr_long, trail_rr_short,
                            trail_long, trail_ma_long_length, trail_ma_long_offset,
                            trail_short, trail_ma_short_length, trail_ma_short_offset,
                            risk_per_trade, contract_size
                        )
                        
                        results_text += f"T MA: {ma_type} | Trail Long: {trail_long} | Trail Short: {trail_short}\n"
                        results_text += f"  Net Profit %: {net_profit:.2f}\n"
                        results_text += f"  Max Drawdown %: {max_dd:.2f}\n"
                        results_text += f"  Total Trades: {total_trades}\n"
                        results_text += "-" * 60 + "\n"
            
        except Exception as e:
            results_text = f"Error: {str(e)}"
        
        dpg.set_value("results_text", results_text)
    
    def toggle_all_t_ma(sender, app_data):
        state = dpg.get_value("t_ma_all")
        ma_types = ['ema', 'sma', 'hma', 'wma', 'alma', 'kama', 'tma', 't3', 'dema', 'vwma', 'vwap']
        for ma in ma_types:
            dpg.set_value(f"t_ma_{ma}", state)
    
    def toggle_all_trail_long(sender, app_data):
        state = dpg.get_value("trail_ma_long_all")
        ma_types = ['ema', 'sma', 'hma', 'wma', 'alma', 'kama', 'tma', 't3', 'dema', 'vwma', 'vwap']
        for ma in ma_types:
            dpg.set_value(f"trail_ma_long_{ma}", state)
    
    def toggle_all_trail_short(sender, app_data):
        state = dpg.get_value("trail_ma_short_all")
        ma_types = ['ema', 'sma', 'hma', 'wma', 'alma', 'kama', 'tma', 't3', 'dema', 'vwma', 'vwap']
        for ma in ma_types:
            dpg.set_value(f"trail_ma_short_{ma}", state)
    
    # Main window
    with dpg.window(label="S_01 TrailingMA Backtester", tag="main_window", 
                    width=800, height=850, pos=[20, 20],
                    no_resize=True, no_move=False, no_close=False):
        
        # Date Filter Section
        with dpg.group():
            dpg.add_text("")
            dpg.add_text("")
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="Date Filter", default_value=True, tag="date_filter")
                dpg.add_text("   ")
                dpg.add_checkbox(label="Backtester", default_value=True, tag="backtester")
            
            dpg.add_spacing(count=2)
            with dpg.group(horizontal=True):
                dpg.add_text("Start Date")
                dpg.add_spacer(width=10)
                dpg.add_input_text(default_value="2025-04-01", width=150, tag="start_date")
                dpg.add_input_text(default_value="08:00", width=80, tag="start_time")

            with dpg.group(horizontal=True):
                dpg.add_text("End Date")
                dpg.add_spacer(width=10)
                dpg.add_input_text(default_value="2025-09-01", width=150, tag="end_date")
                dpg.add_input_text(default_value="08:00", width=80, tag="end_time")
            dpg.add_spacing(count=3)
        
        # MA Settings
        with dpg.group():
            dpg.add_text("T MA Type")
            dpg.add_spacing(count=1)
            
            with dpg.child_window(height=80, border=True, tag="t_ma_container"):
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(label="ALL", default_value=True, tag="t_ma_all", callback=toggle_all_t_ma)
                    dpg.add_checkbox(label="EMA", default_value=True, tag="t_ma_ema")
                    dpg.add_checkbox(label="SMA", default_value=True, tag="t_ma_sma")
                    dpg.add_checkbox(label="HMA", default_value=True, tag="t_ma_hma")
                    dpg.add_checkbox(label="WMA", default_value=True, tag="t_ma_wma")
                    dpg.add_checkbox(label="ALMA", default_value=True, tag="t_ma_alma")
                
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(label="KAMA", default_value=True, tag="t_ma_kama")
                    dpg.add_checkbox(label="TMA", default_value=True, tag="t_ma_tma")
                    dpg.add_checkbox(label="T3", default_value=True, tag="t_ma_t3")
                    dpg.add_checkbox(label="DEMA", default_value=True, tag="t_ma_dema")
                    dpg.add_checkbox(label="VWMA", default_value=True, tag="t_ma_vwma")
                    dpg.add_checkbox(label="VWAP", default_value=True, tag="t_ma_vwap")
            
            dpg.add_spacing(count=2)
            
            with dpg.group(horizontal=True):
                dpg.add_text("Length:")
                dpg.add_spacer(width=10)
                dpg.add_input_int(width=100, default_value=45, tag="t_ma_length")

            with dpg.group(horizontal=True):
                dpg.add_text("Close Count Long:")
                dpg.add_spacer(width=10)
                dpg.add_input_int(width=100, default_value=7, tag="close_count_long")
                dpg.add_text("Close Count Short:")
                dpg.add_spacer(width=10)
                dpg.add_input_int(width=100, default_value=5, tag="close_count_short")
            dpg.add_spacing(count=3)
        
        # Stops and Filters
        with dpg.collapsing_header(label="STOPS AND FILTERS", default_open=True):
            with dpg.child_window(height=40, border=True):
                with dpg.group(horizontal=True):
                    dpg.add_text("Stop Long X:")
                    dpg.add_input_float(width=70, default_value=2.0, tag="stop_long_x", step=0.1)
                    dpg.add_text("RR:")
                    dpg.add_input_float(width=70, default_value=3.0, tag="rr_long", step=0.1)
                    dpg.add_text("LP:")
                    dpg.add_input_int(width=70, default_value=2, tag="lp_long")
            
            with dpg.child_window(height=40, border=True):
                with dpg.group(horizontal=True):
                    dpg.add_text("Stop Short X:")
                    dpg.add_input_float(width=70, default_value=2.0, tag="stop_short_x", step=0.1)
                    dpg.add_text("RR:")
                    dpg.add_input_float(width=70, default_value=3.0, tag="rr_short", step=0.1)
                    dpg.add_text("LP:")
                    dpg.add_input_int(width=70, default_value=2, tag="lp_short")
            
            with dpg.child_window(height=40, border=True):
                with dpg.group(horizontal=True):
                    dpg.add_text("L Stop Max %:")
                    dpg.add_input_float(width=70, default_value=3.0, tag="l_stop_max_pct", step=0.1)
                    dpg.add_text("S Stop Max %:")
                    dpg.add_input_float(width=70, default_value=3.0, tag="s_stop_max_pct", step=0.1)
            
            with dpg.child_window(height=40, border=True):
                with dpg.group(horizontal=True):
                    dpg.add_text("L Stop Max D:")
                    dpg.add_input_int(width=70, default_value=2, tag="l_stop_max_d")
                    dpg.add_text("S Stop Max D:")
                    dpg.add_input_int(width=70, default_value=4, tag="s_stop_max_d")
        
        dpg.add_spacing(count=2)
        
        # Trailing Stops
        with dpg.collapsing_header(label="TRAILING STOPS", default_open=True):
            with dpg.child_window(height=40, border=True):
                with dpg.group(horizontal=True):
                    dpg.add_text("Trail RR Long:")
                    dpg.add_input_float(width=70, default_value=1.0, tag="trail_rr_long", step=0.1)
                    dpg.add_text("Trail RR Short:")
                    dpg.add_input_float(width=70, default_value=1.0, tag="trail_rr_short", step=0.1)
            
            dpg.add_spacing(count=2)
            dpg.add_text("Trail MA Long")
            dpg.add_spacing(count=1)
            
            with dpg.child_window(height=120, border=True):
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(label="ALL", default_value=True, tag="trail_ma_long_all", callback=toggle_all_trail_long)
                    dpg.add_checkbox(label="EMA", default_value=True, tag="trail_ma_long_ema")
                    dpg.add_checkbox(label="SMA", default_value=True, tag="trail_ma_long_sma")
                    dpg.add_checkbox(label="HMA", default_value=True, tag="trail_ma_long_hma")
                    dpg.add_checkbox(label="WMA", default_value=True, tag="trail_ma_long_wma")
                    dpg.add_checkbox(label="ALMA", default_value=True, tag="trail_ma_long_alma")
                
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(label="KAMA", default_value=True, tag="trail_ma_long_kama")
                    dpg.add_checkbox(label="TMA", default_value=True, tag="trail_ma_long_tma")
                    dpg.add_checkbox(label="T3", default_value=True, tag="trail_ma_long_t3")
                    dpg.add_checkbox(label="DEMA", default_value=True, tag="trail_ma_long_dema")
                    dpg.add_checkbox(label="VWMA", default_value=True, tag="trail_ma_long_vwma")
                    dpg.add_checkbox(label="VWAP", default_value=True, tag="trail_ma_long_vwap")
                
                dpg.add_spacing(count=1)
                with dpg.group(horizontal=True):
                    dpg.add_text("Length:")
                    dpg.add_input_int(width=80, default_value=160, tag="trail_ma_long_length")
                    dpg.add_text("Offset:")
                    dpg.add_input_float(width=80, default_value=-1.0, tag="trail_ma_long_offset", step=0.1)
            
            dpg.add_spacing(count=2)
            dpg.add_text("Trail MA Short")
            dpg.add_spacing(count=1)
            
            with dpg.child_window(height=120, border=True):
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(label="ALL", default_value=True, tag="trail_ma_short_all", callback=toggle_all_trail_short)
                    dpg.add_checkbox(label="EMA", default_value=True, tag="trail_ma_short_ema")
                    dpg.add_checkbox(label="SMA", default_value=True, tag="trail_ma_short_sma")
                    dpg.add_checkbox(label="HMA", default_value=True, tag="trail_ma_short_hma")
                    dpg.add_checkbox(label="WMA", default_value=True, tag="trail_ma_short_wma")
                    dpg.add_checkbox(label="ALMA", default_value=True, tag="trail_ma_short_alma")
                
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(label="KAMA", default_value=True, tag="trail_ma_short_kama")
                    dpg.add_checkbox(label="TMA", default_value=True, tag="trail_ma_short_tma")
                    dpg.add_checkbox(label="T3", default_value=True, tag="trail_ma_short_t3")
                    dpg.add_checkbox(label="DEMA", default_value=True, tag="trail_ma_short_dema")
                    dpg.add_checkbox(label="VWMA", default_value=True, tag="trail_ma_short_vwma")
                    dpg.add_checkbox(label="VWAP", default_value=True, tag="trail_ma_short_vwap")
                
                dpg.add_spacing(count=1)
                with dpg.group(horizontal=True):
                    dpg.add_text("Length:")
                    dpg.add_input_int(width=80, default_value=160, tag="trail_ma_short_length")
                    dpg.add_text("Offset:")
                    dpg.add_input_float(width=80, default_value=1.0, tag="trail_ma_short_offset", step=0.1)
        
        dpg.add_spacing(count=3)
        
        # Risk Settings
        with dpg.group():
            with dpg.group(horizontal=True):
                dpg.add_text("Risk Per Trade:")
                dpg.add_spacer(width=10)
                dpg.add_input_float(width=100, default_value=2.0, step=0.01, tag="risk_per_trade")
                dpg.add_text("Contract Size:")
                dpg.add_spacer(width=10)
                dpg.add_input_float(width=100, default_value=0.01, step=0.01, tag="contract_size")
            dpg.add_spacing(count=3)
        
        # Results Area
        dpg.add_text("RESULTS", color=(58, 58, 58, 255))
        dpg.add_separator()
        dpg.add_spacing(count=2)
        
        with dpg.child_window(height=200, border=True, tag="results_window"):
            dpg.add_text("Нажмите 'Run' для запуска бэктеста...", 
                        color=(119, 119, 119, 255), 
                        tag="results_text")
        
        dpg.add_spacing(count=3)
        
        # Action Buttons
        with dpg.group(horizontal=True):
            dpg.add_button(label="Defaults", width=100, height=30)
            dpg.add_text("", tag="spacer_text")
            dpg.add_button(label="Cancel", width=80, height=30)
            dpg.add_button(label="Run", width=80, height=30, callback=run_backtest)
    
    dpg.create_viewport(title="S_01 TrailingMA Backtester", width=840, height=900)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    create_gui()
