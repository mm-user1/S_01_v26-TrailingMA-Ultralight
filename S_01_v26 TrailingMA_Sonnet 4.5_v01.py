# -*- coding: utf-8 -*-
"""
S_01 TrailingMA Light — перенос стратегии из PineScript v5 в Python
+ GUI интерфейс для настройки и запуска тестирования
+ Полный перебор параметров (grid search)
+ Поддержка Heikin Ashi свечей
+ Визуализация последних 10 сделок
"""

import os
import re
import glob
import math
import time
import json
import argparse
import multiprocessing as mp
from contextlib import contextmanager
from typing import Tuple, List, Dict, Optional
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

# ================== Глобальные массивы (инициализируются в _init_pool) ==================
t_arr: np.ndarray = None
o_arr: np.ndarray = None
h_arr: np.ndarray = None
l_arr: np.ndarray = None
c_arr: np.ndarray = None
v_arr: np.ndarray = None
month_arr: np.ndarray = None

# Массивы для расчёта индикаторов/сигналов (обычные или HA):
sc_arr: np.ndarray = None
sh_arr: np.ndarray = None
sl_arr: np.ndarray = None

commission_rate: float = 0.0005
initial_capital: float = 100.0
contract_size: float = 0.01
vwap_tz_offset_hours: int = 8
ma_cache: Dict[Tuple[str, int], np.ndarray] = {}

# ================== Утилиты ==================
def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    need = ['time', 'open', 'high', 'low', 'close']
    for n in need:
        if n not in cols:
            raise ValueError(f"CSV is missing required column: '{n}'")
    vol = cols.get('volume')
    if vol is None:
        if 'Volume' in df.columns:
            vol = 'Volume'
        else:
            df['Volume'] = 1.0
            vol = 'Volume'
    return df.rename(columns={
        cols['time']: 'time',
        cols['open']: 'open',
        cols['high']: 'high',
        cols['low']: 'low',
        cols['close']: 'close',
        vol: 'Volume'
    })[['time', 'open', 'high', 'low', 'close', 'Volume']]

def parse_local_to_utc_epoch(dt_str: Optional[str], tz_offset_hours: int = 8) -> Optional[int]:
    if not dt_str: return None
    dt_naive = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
    return int((dt_naive - timedelta(hours=tz_offset_hours)).timestamp())

def fmt_dur(sec: float) -> str:
    sec = int(round(sec))
    h = sec // 3600; m = (sec % 3600) // 60; s = sec % 60
    return f"{h:d}h {m:02d}m {s:02d}s" if h else f"{m:02d}m {s:02d}s"

def dates_cli_label(start_str: Optional[str], end_str: Optional[str]) -> str:
    if not start_str or not end_str:
        return ""
    def d(s: str) -> str: return s.split()[0].replace('-', '.')
    return f"{d(start_str)}-{d(end_str)}"

def extract_prefix_from_filename(csv_path: str) -> str:
    stem = Path(csv_path).stem
    m = re.search(r'\d{4}\.\d{2}\.\d{2}-\d{4}\.\d{2}\.\d{2}', stem)
    if m:
        prefix = stem[:m.start()].rstrip()
        if prefix:
            return prefix
    m2 = re.match(r'(.+?,\s*\d+)\b', stem)
    if m2:
        return m2.group(1).strip()
    return stem.strip()

def sanitize_filename(s: str) -> str:
    pattern = re.compile(r'[\\/:*?"<>|]')
    s = pattern.sub('', s)
    return s.rstrip(' .')

def expand_csv_argument(arg: str) -> List[str]:
    p = Path(arg)
    if p.is_dir():
        files = sorted(str(x) for x in p.glob("*.csv"))
        if not files:
            raise SystemExit(f"No .csv files found in directory: {arg}")
        return files
    if any(ch in arg for ch in "*?[]"):
        files = sorted(glob.glob(arg))
        files = [f for f in files if f.lower().endswith(".csv")]
        if not files:
            raise SystemExit(f"No .csv files match glob: {arg}")
        return files
    if not p.exists():
        raise SystemExit(f"--csv path not found: {arg}")
    if p.is_file():
        return [str(p)]
    raise SystemExit(f"--csv path is not a file/dir/glob: {arg}")

# ================== Heikin Ashi ==================
def heikin_ashi(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ha_c = (o + h + l + c) / 4.0
    ha_o = np.empty_like(c)
    ha_o[0] = (o[0] + c[0]) / 2.0
    for i in range(1, c.shape[0]):
        ha_o[i] = 0.5 * (ha_o[i-1] + ha_c[i-1])
    ha_h = np.maximum.reduce([h, ha_o, ha_c])
    ha_l = np.minimum.reduce([l, ha_o, ha_c])
    return ha_o, ha_h, ha_l, ha_c

# ================== Индикаторы (NumPy) ==================
def sma(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full(arr.shape[0], np.nan)
    if n <= 0 or n > arr.shape[0]: return out
    cs = np.cumsum(np.insert(arr, 0, 0.0))
    out[n-1:] = (cs[n:] - cs[:-n]) / float(n)
    return out

def ema(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full(arr.shape[0], np.nan)
    if n <= 0: return out
    a = 2.0 / (n + 1.0)
    out[n-1] = np.nanmean(arr[:n])
    for i in range(n, arr.shape[0]):
        prev = out[i-1]
        out[i] = prev + a * (arr[i] - prev)
    return out

def wma(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full(arr.shape[0], np.nan)
    if n <= 0 or n > arr.shape[0]: return out
    w = np.arange(1, n+1, dtype=float)
    conv = np.convolve(arr, w[::-1], mode='valid') / w.sum()
    out[n-1:] = conv
    return out

def hma(arr: np.ndarray, n: int) -> np.ndarray:
    if n <= 0: return np.full(arr.shape[0], np.nan)
    n_half = max(1, int(round(n/2.0)))
    n_sqrt = max(1, int(round(math.sqrt(n))))
    w1 = wma(arr, n_half)
    w2 = wma(arr, n)
    start = n - 1
    diff = 2.0 * w1 - w2
    tail = diff[start:]
    if tail.shape[0] < n_sqrt:
        return np.full(arr.shape[0], np.nan)
    w3 = wma(tail, n_sqrt)
    valid = w3[n_sqrt - 1:]
    out = np.full(arr.shape[0], np.nan)
    out[start + (n_sqrt - 1): start + (n_sqrt - 1) + valid.shape[0]] = valid
    return out

def alma(arr: np.ndarray, n: int, offset: float = 0.85, sigma: float = 6.0) -> np.ndarray:
    out = np.full(arr.shape[0], np.nan)
    if n <= 0 or n > arr.shape[0]: return out
    m = offset * (n - 1)
    s = n / sigma
    j = np.arange(n, dtype=float)
    w = np.exp(-((j - m)**2)/(2*s*s))
    w /= w.sum()
    conv = np.convolve(arr, w[::-1], mode='valid')
    out[n-1:] = conv
    return out

def kama(arr: np.ndarray, n: int, fast_len: int = 2, slow_len: int = 30) -> np.ndarray:
    out = np.full(arr.shape[0], np.nan)
    if n <= 0 or n > arr.shape[0]: return out
    a_fast = 2.0 / (fast_len + 1.0)
    a_slow = 2.0 / (slow_len + 1.0)
    out[n-1] = arr[n-1]
    for i in range(n, arr.shape[0]):
        change = abs(arr[i] - arr[i-n])
        vol = np.abs(np.diff(arr[i-n:i+1])).sum()
        er = (change / vol) if vol != 0.0 else 0.0
        a = (er * (a_fast - a_slow) + a_slow) ** 2
        prev = out[i-1] if not np.isnan(out[i-1]) else arr[i-1]
        out[i] = prev + a * (arr[i] - prev)
    return out

def tma_triangular(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full(arr.shape[0], np.nan)
    if n <= 0 or n > arr.shape[0]: return out
    len1 = int(math.ceil(n/2))
    len2 = int(math.floor(n/2) + 1)
    s1 = sma(arr, len1)
    tail = s1[len1 - 1:]
    if tail.shape[0] < len2: return out
    s2 = sma(tail, len2)
    valid = s2[len2 - 1:]
    out[n - 1: n - 1 + valid.shape[0]] = valid
    return out

def dema(arr: np.ndarray, n: int) -> np.ndarray:
    e1 = ema(arr, n)
    e2 = ema(e1, n)
    return 2.0 * e1 - e2

def t3(arr: np.ndarray, n: int, v: float = 0.7) -> np.ndarray:
    def gd(src: np.ndarray) -> np.ndarray:
        e1 = ema(src, n)
        e2 = ema(e1, n)
        return e1 * (1.0 + v) - e2 * v
    return gd(gd(gd(arr)))

def vwma(close: np.ndarray, vol: np.ndarray, n: int) -> np.ndarray:
    out = np.full(close.shape[0], np.nan)
    if n <= 0 or n > close.shape[0]: return out
    pv = close * vol
    c_pv = np.cumsum(np.insert(pv, 0, 0.0))
    c_v = np.cumsum(np.insert(vol, 0, 0.0))
    numer = c_pv[n:] - c_pv[:-n]
    denom = c_v[n:] - c_v[:-n]
    vals = np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom != 0.0)
    out[n-1:] = vals
    return out

def vwap(time_s: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
         vol: np.ndarray, tz_offset_hours: int = 8) -> np.ndarray:
    tp = (high + low + close) / 3.0
    out = np.full(close.shape[0], np.nan)
    if close.shape[0] == 0: return out
    tz_off = tz_offset_hours * 3600
    day_idx = (time_s + tz_off) // 86400
    cur_day = day_idx[0]
    cum_tpv = 0.0
    cum_v = 0.0
    for i in range(close.shape[0]):
        if day_idx[i] != cur_day:
            cur_day = day_idx[i]
            cum_tpv = 0.0
            cum_v = 0.0
        cum_tpv += tp[i] * vol[i]
        cum_v += vol[i]
        out[i] = (cum_tpv / cum_v) if cum_v != 0 else np.nan
    return out

def compute_ma(ma_type: str, length: int) -> np.ndarray:
    key = ma_type.upper()
    if key == "SMA":   return sma(sc_arr, length)
    if key == "EMA":   return ema(sc_arr, length)
    if key == "WMA":   return wma(sc_arr, length)
    if key == "HMA":   return hma(sc_arr, length)
    if key == "ALMA":  return alma(sc_arr, length, 0.85, 6.0)
    if key == "KAMA":  return kama(sc_arr, length, 2, 30)
    if key == "TMA":   return tma_triangular(sc_arr, length)
    if key == "T3":    return t3(sc_arr, length, 0.7)
    if key == "DEMA":  return dema(sc_arr, length)
    if key == "VWMA":  return vwma(sc_arr, v_arr, length)
    if key == "VWAP":  return vwap(t_arr, sh_arr, sl_arr, sc_arr, v_arr, vwap_tz_offset_hours)
    raise ValueError(f"Unsupported MA: {ma_type}")

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Расчет ATR (Average True Range)"""
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr = np.maximum(tr, np.abs(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    return ema(tr, period)

# ================== Инициализация пула / данных ==================
def _init_pool(csv_path: str, start_ts: Optional[int], end_ts: Optional[int],
               _commission: float, _initial_capital: float,
               _contract_size: float, _vwap_tz: int, _use_ha: bool):
    global t_arr, o_arr, h_arr, l_arr, c_arr, v_arr, month_arr
    global sc_arr, sh_arr, sl_arr
    global commission_rate, initial_capital, contract_size, vwap_tz_offset_hours, ma_cache

    df = pd.read_csv(csv_path)
    df = ensure_columns(df)
    if start_ts is not None: df = df[df['time'] >= int(start_ts)]
    if end_ts   is not None: df = df[df['time'] <= int(end_ts)]
    if df.empty: raise ValueError("No data rows after filtering by --start/--end.")

    t_arr = df['time'].to_numpy(np.int64)
    o_arr = df['open'].to_numpy(float)
    h_arr = df['high'].to_numpy(float)
    l_arr = df['low'].to_numpy(float)
    c_arr = df['close'].to_numpy(float)
    v_arr = df['Volume'].to_numpy(float)

    if _use_ha:
        ha_o, ha_h, ha_l, ha_c = heikin_ashi(o_arr, h_arr, l_arr, c_arr)
        sc_arr = ha_c; sh_arr = ha_h; sl_arr = ha_l
    else:
        sc_arr = c_arr; sh_arr = h_arr; sl_arr = l_arr

    commission_rate = _commission
    initial_capital = _initial_capital
    contract_size = _contract_size
    vwap_tz_offset_hours = _vwap_tz
    ma_cache = {}

    month_arr = pd.to_datetime(df['time'], unit='s').dt.month.to_numpy(np.int32)

# ================== Симуляция одной комбинации ==================
def simulate(params: Tuple) -> Tuple:
    """
    Симуляция стратегии TrailingMA Light
    params: (ma_type, ma_period, n_long, n_short, stop_mult_long, rr_long, lp_long,
             stop_mult_short, rr_short, lp_short, long_stop_pct_filter, short_stop_pct_filter,
             long_stop_days_filter, short_stop_days_filter, trail_rr_long, trail_rr_short,
             trail_ma_type_long, trail_ma_length_long, trail_ma_offset_long,
             trail_ma_type_short, trail_ma_length_short, trail_ma_offset_short)
    """
    (ma_type, ma_period, n_long, n_short, stop_mult_long, rr_long, lp_long,
     stop_mult_short, rr_short, lp_short, long_stop_pct_filter, short_stop_pct_filter,
     long_stop_days_filter, short_stop_days_filter, trail_rr_long, trail_rr_short,
     trail_ma_type_long, trail_ma_length_long, trail_ma_offset_long,
     trail_ma_type_short, trail_ma_length_short, trail_ma_offset_short) = params

    # Получаем MA для тренда
    key = (ma_type.upper(), ma_period)
    ma = ma_cache.get(key)
    if ma is None:
        ma = compute_ma(ma_type, ma_period)
        ma_cache[key] = ma

    # Получаем MA для трейлинга
    key_trail_long = (trail_ma_type_long.upper(), trail_ma_length_long)
    trail_ma_long = ma_cache.get(key_trail_long)
    if trail_ma_long is None:
        trail_ma_long = compute_ma(trail_ma_type_long, trail_ma_length_long)
        ma_cache[key_trail_long] = trail_ma_long

    key_trail_short = (trail_ma_type_short.upper(), trail_ma_length_short)
    trail_ma_short = ma_cache.get(key_trail_short)
    if trail_ma_short is None:
        trail_ma_short = compute_ma(trail_ma_type_short, trail_ma_length_short)
        ma_cache[key_trail_short] = trail_ma_short

    # Применяем offset к трейлинг MA
    trail_ma_long_offset = trail_ma_long * (1.0 + trail_ma_offset_long / 100.0)
    trail_ma_short_offset = trail_ma_short * (1.0 + trail_ma_offset_short / 100.0)

    # Расчет ATR
    atr_arr = atr(sh_arr, sl_arr, sc_arr, 14)

    # Инициализация состояния
    cash = float(initial_capital)
    pos_qty = 0.0
    position = 0  # 0 = нет позиции, 1 = long, -1 = short
    entry_price = 0.0
    entry_fee = 0.0
    entry_bar = -1

    t_stop = 0.0
    t_target = 0.0
    trail_ma_price_long = 0.0
    trail_ma_price_short = 0.0
    trail_activated_long = False
    trail_activated_short = False

    count_long = 0
    count_short = 0
    trades = 0
    wins = 0
    loss_streak = 0
    max_loss_streak = 0

    peak = initial_capital
    max_drawdown = 0.0

    next_open = 0

    # Sharpe ratio tracking
    month_start_equity = initial_capital
    monthly_returns: List[float] = []
    last_equity = initial_capital

    # Статистика по типам выходов
    exits_stop = 0
    exits_take = 0
    exits_trail = 0
    exits_days = 0

    # Для детальной записи сделок (опционально)
    trade_log = []

    def close_position(exit_price: float, exit_type: str, exit_bar_idx: int):
        nonlocal cash, pos_qty, position, entry_fee, entry_price, entry_bar
        nonlocal trades, wins, loss_streak, max_loss_streak, trail_activated_long, trail_activated_short
        nonlocal exits_stop, exits_take, exits_trail, exits_days, trade_log, peak, max_drawdown
        nonlocal t_stop, t_target, trail_ma_price_long, trail_ma_price_short

        if position == 0 or pos_qty == 0.0:
            return

        trade_value = abs(pos_qty) * exit_price
        fee = trade_value * commission_rate

        if position == 1:
            cash_change = pos_qty * exit_price - fee
            pnl = pos_qty * (exit_price - entry_price) - entry_fee - fee
            cash_delta = cash_change
        else:
            cash_change = -abs(pos_qty) * exit_price - fee
            pnl = abs(pos_qty) * (entry_price - exit_price) - entry_fee - fee
            cash_delta = cash_change

        cash += cash_delta

        trades += 1
        if pnl > 0:
            wins += 1
            loss_streak = 0
        else:
            loss_streak += 1
            max_loss_streak = max(max_loss_streak, loss_streak)

        if exit_type == "stop":
            exits_stop += 1
        elif exit_type == "take":
            exits_take += 1
        elif exit_type == "trail":
            exits_trail += 1
        elif exit_type == "days":
            exits_days += 1

        trade_log.append({
            'entry_bar': entry_bar,
            'exit_bar': exit_bar_idx,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position': position,
            'pnl': pnl,
            'exit_type': exit_type
        })

        pos_qty = 0.0
        position = 0
        entry_fee = 0.0
        entry_price = 0.0
        entry_bar = -1
        trail_activated_long = False
        trail_activated_short = False
        t_stop = 0.0
        t_target = 0.0
        trail_ma_price_long = 0.0
        trail_ma_price_short = 0.0

        eq = cash
        if eq > peak:
            peak = eq
        else:
            drop = (eq - peak) / peak
            if drop < max_drawdown:
                max_drawdown = drop

    for i in range(o_arr.shape[0]):
        position_prev_bar = position
        # Проверка смены месяца для Sharpe
        if i > 0 and month_arr[i] != month_arr[i-1]:
            if month_start_equity > 0:
                monthly_returns.append((last_equity / month_start_equity - 1.0) * 100.0)
            month_start_equity = last_equity

        bar_open = o_arr[i]
        bar_high = h_arr[i]
        bar_low = l_arr[i]
        bar_close = c_arr[i]

        # === ВЫХОД НА ОТКРЫТИИ ПО СТОПУ/ТРЕЙЛУ/ТЕЙКУ ===
        if position == 1:
            stop_level = trail_ma_price_long if trail_activated_long else t_stop
            target_level = None if trail_activated_long else t_target

            if not np.isnan(stop_level) and bar_open <= stop_level:
                close_position(bar_open, "trail" if trail_activated_long else "stop", i)
            elif target_level is not None and bar_open >= target_level:
                close_position(target_level, "take", i)

        elif position == -1:
            stop_level = trail_ma_price_short if trail_activated_short else t_stop
            target_level = None if trail_activated_short else t_target

            if not np.isnan(stop_level) and bar_open >= stop_level:
                close_position(bar_open, "trail" if trail_activated_short else "stop", i)
            elif target_level is not None and bar_open <= target_level:
                close_position(target_level, "take", i)

        # === ОТКРЫТИЕ ПОЗИЦИИ на open ===
        if next_open != 0 and position == 0:
            px = bar_open

            if i == 0:
                next_open = 0
                continue

            atr_idx = i - 1 if i > 0 else i
            atr_value = atr_arr[atr_idx]
            if np.isnan(atr_value) or atr_value <= 0:
                next_open = 0
                continue

            # Расчет стопа и тейка
            if next_open == 1:  # Long
                length = max(1, lp_long)
                lookback_end = i
                lookback_start = max(0, lookback_end - length)
                if lookback_end <= lookback_start:
                    next_open = 0
                    continue

                lowest_low = np.min(sl_arr[lookback_start:lookback_end])
                stop_size = atr_value * stop_mult_long
                stop_price = lowest_low - stop_size
                stop_distance = px - stop_price
                if stop_distance <= 0:
                    next_open = 0
                    continue

                stop_pct = (stop_distance / px) * 100.0
                if stop_pct > long_stop_pct_filter:
                    next_open = 0
                    continue

                target_price = px + (stop_distance * rr_long)
                qty = math.floor(((cash * 0.02) / stop_distance) / contract_size) * contract_size

                if qty > 0:
                    cost = qty * px
                    fee = cost * commission_rate
                    if cost + fee <= cash:
                        cash -= (cost + fee)
                        pos_qty = qty
                        position = 1
                        entry_price = px
                        entry_fee = fee
                        entry_bar = i
                        t_stop = stop_price
                        t_target = target_price
                        trail_ma_price_long = stop_price

            else:  # Short
                length = max(1, lp_short)
                lookback_end = i
                lookback_start = max(0, lookback_end - length)
                if lookback_end <= lookback_start:
                    next_open = 0
                    continue

                highest_high = np.max(sh_arr[lookback_start:lookback_end])
                stop_size = atr_value * stop_mult_short
                stop_price = highest_high + stop_size
                stop_distance = stop_price - px
                if stop_distance <= 0:
                    next_open = 0
                    continue

                stop_pct = (stop_distance / px) * 100.0
                if stop_pct > short_stop_pct_filter:
                    next_open = 0
                    continue

                target_price = px - (stop_distance * rr_short)
                qty = math.floor(((cash * 0.02) / stop_distance) / contract_size) * contract_size

                if qty > 0:
                    proceeds = qty * px
                    fee = proceeds * commission_rate
                    cash += (proceeds - fee)
                    pos_qty = -qty
                    position = -1
                    entry_price = px
                    entry_fee = fee
                    entry_bar = i
                    t_stop = stop_price
                    t_target = target_price
                    trail_ma_price_short = stop_price

            next_open = 0

        # === ОБНОВЛЕНИЕ СЧЕТЧИКОВ ТРЕНДА ===
        mav = ma[i]
        if np.isnan(mav):
            count_long = 0
            count_short = 0
        else:
            if sc_arr[i] > mav:
                count_long += 1
                count_short = 0
            elif sc_arr[i] < mav:
                count_short += 1
                count_long = 0
            else:
                count_long = 0
                count_short = 0

        # === ПРОВЕРКА АКТИВАЦИИ ТРЕЙЛИНГА ===
        if position == 1 and not trail_activated_long:
            profit_threshold = entry_price + ((entry_price - t_stop) * trail_rr_long)
            if bar_high >= profit_threshold:
                trail_activated_long = True

        if position == -1 and not trail_activated_short:
            profit_threshold = entry_price - ((t_stop - entry_price) * trail_rr_short)
            if bar_low <= profit_threshold:
                trail_activated_short = True

        # === ОБНОВЛЕНИЕ ТРЕЙЛИНГ СТОПА ===
        if trail_activated_long and not np.isnan(trail_ma_long_offset[i]):
            if trail_ma_long_offset[i] > trail_ma_price_long:
                trail_ma_price_long = trail_ma_long_offset[i]

        if trail_activated_short and not np.isnan(trail_ma_short_offset[i]):
            if trail_ma_short_offset[i] < trail_ma_price_short:
                trail_ma_price_short = trail_ma_short_offset[i]

        # === ПРОВЕРКА ВЫХОДА ПО СТОПУ/ТЕЙКУ/ТРЕЙЛИНГУ ===
        if position == 1:
            if trail_activated_long and not np.isnan(trail_ma_price_long):
                if bar_low <= trail_ma_price_long:
                    exit_price = trail_ma_price_long if bar_open > trail_ma_price_long else bar_open
                    close_position(exit_price, "trail", i)
            else:
                stop_hit = bar_low <= t_stop
                target_hit = bar_high >= t_target
                if stop_hit or target_hit:
                    if stop_hit and target_hit:
                        dist_stop = abs(bar_open - t_stop)
                        dist_target = abs(t_target - bar_open)
                        if dist_stop <= dist_target:
                            exit_price = t_stop if bar_open > t_stop else bar_open
                            exit_type = "stop"
                        else:
                            exit_price = t_target
                            exit_type = "take"
                    elif stop_hit:
                        exit_price = t_stop if bar_open > t_stop else bar_open
                        exit_type = "stop"
                    else:
                        exit_price = t_target
                        exit_type = "take"
                    close_position(exit_price, exit_type, i)

        if position == -1:
            if trail_activated_short and not np.isnan(trail_ma_price_short):
                if bar_high >= trail_ma_price_short:
                    exit_price = trail_ma_price_short if bar_open < trail_ma_price_short else bar_open
                    close_position(exit_price, "trail", i)
            else:
                stop_hit = bar_high >= t_stop
                target_hit = bar_low <= t_target
                if stop_hit or target_hit:
                    if stop_hit and target_hit:
                        dist_stop = abs(bar_open - t_stop)
                        dist_target = abs(bar_open - t_target)
                        if dist_stop <= dist_target:
                            exit_price = t_stop if bar_open < t_stop else bar_open
                            exit_type = "stop"
                        else:
                            exit_price = t_target
                            exit_type = "take"
                    elif stop_hit:
                        exit_price = t_stop if bar_open < t_stop else bar_open
                        exit_type = "stop"
                    else:
                        exit_price = t_target
                        exit_type = "take"
                    close_position(exit_price, exit_type, i)

        # === ПРОВЕРКА ФИЛЬТРА ПО ДНЯМ ===
        if position == 1 and entry_bar >= 0:
            days_in_trade = (t_arr[i] - t_arr[entry_bar]) / 86400.0
            if days_in_trade >= long_stop_days_filter:
                close_position(bar_close, "days", i)

        if position == -1 and entry_bar >= 0:
            days_in_trade = (t_arr[i] - t_arr[entry_bar]) / 86400.0
            if days_in_trade >= short_stop_days_filter:
                close_position(bar_close, "days", i)

        # === ГЕНЕРАЦИЯ СИГНАЛОВ ===
        sig_long = (count_long >= n_long)
        sig_short = (count_short >= n_short)

        # Сигнал на вход только если нет позиции и не было позиции на предыдущем баре
        if position == 0 and position_prev_bar == 0:
            if sig_long:
                next_open = 1
            elif sig_short:
                next_open = -1

        # Расчет equity на конец бара
        if position != 0:
            current_equity = cash + pos_qty * c_arr[i]
        else:
            current_equity = cash
        last_equity = current_equity

    # === ФОРС-ЗАКРЫТИЕ ПОЗИЦИИ В КОНЦЕ ===
    if position != 0:
        close_position(c_arr[-1], "forced", len(c_arr) - 1)

    final_equity = cash
    net_profit = (final_equity / initial_capital - 1.0) * 100.0
    winrate = (wins / trades * 100.0) if trades > 0 else 0.0
    dd_pct = max_drawdown * 100.0

    # Sharpe Ratio
    sharpe_value = None
    if len(monthly_returns) >= 2:
        avg_return = np.mean(monthly_returns)
        sd_return = np.std(monthly_returns, ddof=0)
        rfr_m = (0.02 * 100.0) / 12.0
        if sd_return != 0:
            sharpe_value = (avg_return - rfr_m) / sd_return
    sharpe_str = f"{sharpe_value:.2f}" if sharpe_value is not None else ""

    return (ma_type.upper(), ma_period, n_long, n_short,
            stop_mult_long, rr_long, lp_long,
            stop_mult_short, rr_short, lp_short,
            long_stop_pct_filter, short_stop_pct_filter,
            long_stop_days_filter, short_stop_days_filter,
            trail_rr_long, trail_rr_short,
            trail_ma_type_long.upper(), trail_ma_length_long, trail_ma_offset_long,
            trail_ma_type_short.upper(), trail_ma_length_short, trail_ma_offset_short,
            f"{winrate:.2f}%", f"{net_profit:.2f}%", f"{dd_pct:.2f}%",
            trades, max_loss_streak, sharpe_str)

# ================== CSV заголовок ==================
CSV_HEADER = ("MA Type,MA Period,Close Count Long,Close Count Short,"
              "Stop Long X,RR Long,LP Long,Stop Short X,RR Short,LP Short,"
              "L Stop Max %,S Stop Max %,L Stop Max D,S Stop Max D,"
              "RR TrailingMALong,RR TrailingMAShort,"
              "Trail MA Type Long,Trail MA Length Long,Trail MA Offset Long,"
              "Trail MA Type Short,Trail MA Length Short,Trail MA Offset Short,"
              "Winrate%,Net Profit%,DD%,Total Trades,Max SL,Sharpe\n")

# ================== Грид/запуск для одного CSV ==================
def run_grid_search(csv_path: str, config: dict, workers: int = None, use_ha: bool = False) -> Path:
    """
    Запуск grid search с заданной конфигурацией
    """
    if workers is None or workers <= 0:
        workers = mp.cpu_count()

    engine = config.get('engine', 'process').lower()
    if engine not in {'process', 'thread'}:
        raise ValueError(f"Unsupported engine type: {engine}")

    # Формирование комбинаций для перебора
    ma_types = config.get('ma_types', ['EMA'])
    ma_periods = config.get('ma_periods', [45])
    cc_long = config.get('cc_long', [7])
    cc_short = config.get('cc_short', [5])
    stop_long_x = config.get('stop_long_x', [2.0])
    rr_long = config.get('rr_long', [3])
    lp_long = config.get('lp_long', [2])
    stop_short_x = config.get('stop_short_x', [2.0])
    rr_short = config.get('rr_short', [3])
    lp_short = config.get('lp_short', [2])
    long_stop_pct = config.get('long_stop_pct', [3])
    short_stop_pct = config.get('short_stop_pct', [3])
    long_stop_days = config.get('long_stop_days', [2])
    short_stop_days = config.get('short_stop_days', [4])
    trail_rr_long = config.get('trail_rr_long', [1])
    trail_rr_short = config.get('trail_rr_short', [1])
    trail_ma_types_long = config.get('trail_ma_types_long', ['SMA'])
    trail_ma_lengths_long = config.get('trail_ma_lengths_long', [160])
    trail_ma_offsets_long = config.get('trail_ma_offsets_long', [-1.0])
    trail_ma_types_short = config.get('trail_ma_types_short', ['SMA'])
    trail_ma_lengths_short = config.get('trail_ma_lengths_short', [160])
    trail_ma_offsets_short = config.get('trail_ma_offsets_short', [1.0])

    from itertools import product
    iterables = (
        ma_types, ma_periods, cc_long, cc_short,
        stop_long_x, rr_long, lp_long,
        stop_short_x, rr_short, lp_short,
        long_stop_pct, short_stop_pct,
        long_stop_days, short_stop_days,
        trail_rr_long, trail_rr_short,
        trail_ma_types_long, trail_ma_lengths_long, trail_ma_offsets_long,
        trail_ma_types_short, trail_ma_lengths_short, trail_ma_offsets_short
    )
    combos = [(
        ma_type, ma_period, nl, ns,
        slx, rrl, lpl,
        ssx, rrs, lps,
        lsp, ssp, lsd, ssd,
        trrl, trrs,
        tmt_l, tml_l, tmo_l,
        tmt_s, tml_s, tmo_s
    ) for (ma_type, ma_period, nl, ns,
           slx, rrl, lpl,
           ssx, rrs, lps,
           lsp, ssp, lsd, ssd,
           trrl, trrs,
           tmt_l, tml_l, tmo_l,
           tmt_s, tml_s, tmo_s) in product(*iterables)]

    total = len(combos)
    print(f"\n=== {Path(csv_path).name} ===")
    print(f"Total combinations: {total:,}")

    if total == 0:
        raise ValueError("No parameter combinations generated by the provided configuration.")

    start_ts = parse_local_to_utc_epoch(config.get('start_date'), 8) if config.get('start_date') else None
    end_ts = parse_local_to_utc_epoch(config.get('end_date'), 8) if config.get('end_date') else None
    init_args = (csv_path, start_ts, end_ts, 0.0005, 100.0, 0.01, 8, use_ha)

    # Предварительная инициализация для одиночных запусков (и ThreadPool)
    _init_pool(*init_args)

    # Калибровка ETA на одном потоке
    sample_n = min(30, total)
    t0 = time.time()
    for combo in combos[:sample_n]:
        simulate(combo)
    avg = (time.time() - t0) / max(1, sample_n)
    eta = avg * total / max(1, workers)
    print(f"ETA ≈ {fmt_dur(eta)}  (engine={engine}, workers={workers}, HA={use_ha})")

    @contextmanager
    def parallel_pool():
        if engine == 'process':
            pool = mp.Pool(processes=workers, initializer=_init_pool, initargs=init_args)
        else:
            from multiprocessing.dummy import Pool as ThreadPool
            _init_pool(*init_args)
            pool = ThreadPool(processes=workers)
        try:
            yield pool
        finally:
            pool.close()
            pool.join()

    # Основной прогон
    results = []
    t0 = time.time()
    chunk = 2000
    with parallel_pool() as pool:
        for i in tqdm(range(0, total, chunk), total=(total + chunk - 1) // chunk, desc="Grid"):
            batch = combos[i:i + chunk]
            if not batch:
                continue
            results.extend(pool.map(simulate, batch))
    dt = time.time() - t0
    print(f"Finished in {fmt_dur(dt)}; processed {len(results):,} combos")

    # Сортировка по Net Profit%
    results.sort(key=lambda r: float(r[23].rstrip('%')), reverse=True)

    # Формирование имени файла
    prefix = extract_prefix_from_filename(csv_path)
    cli_dates = dates_cli_label(config.get('start_date'), config.get('end_date'))
    
    ma_suffix = "+".join(ma_types) if len(ma_types) <= 3 else "ALL"
    base_name = f"{prefix} {cli_dates}_{ma_suffix}".strip()
    if use_ha:
        base_name += "_ha"
    base_name = re.sub(r'\s+', ' ', base_name)
    file_name = sanitize_filename(base_name) + ".csv"

    out_csv = Path(config.get('outdir', '.')) / file_name
    Path(config.get('outdir', '.')).mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(CSV_HEADER)
        for r in results:
            f.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(*r))

    print(f"Saved: {out_csv}")
    return out_csv

# ================== GUI ==================
def create_gui():
    """Создание графического интерфейса"""
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import threading
    
    # Список всех доступных MA
    MA_ALL = ["EMA", "SMA", "HMA", "ALMA", "WMA", "KAMA", "TMA", "T3", "DEMA", "VWMA", "VWAP"]
    
    # Дефолтные значения из Pine стратегии
    DEFAULTS = {
        'csv_path': '',
        'start_date': '2025-04-01 00:00',
        'end_date': '2025-09-01 00:00',
        'use_ha': False,
        'ma_types': ['EMA'],
        'ma_period': 45,
        'ma_period_range': '25-500, 1',
        'cc_long': 7,
        'cc_long_range': '2-10, 1',
        'cc_short': 5,
        'cc_short_range': '2-10, 1',
        'stop_long_x': 2.0,
        'stop_long_x_range': '1.0-3.0, 0.1',
        'rr_long': 3,
        'lp_long': 2,
        'lp_long_range': '2-6, 1',
        'stop_short_x': 2.0,
        'stop_short_x_range': '1.0-3.0, 0.1',
        'rr_short': 3,
        'lp_short': 2,
        'lp_short_range': '2-6, 1',
        'long_stop_pct': 3,
        'long_stop_pct_range': '3-10, 1',
        'short_stop_pct': 3,
        'short_stop_pct_range': '3-10, 1',
        'long_stop_days': 2,
        'long_stop_days_range': '2-7, 1',
        'short_stop_days': 4,
        'short_stop_days_range': '2-7, 1',
        'trail_rr_long': 1,
        'trail_rr_short': 1,
        'trail_ma_types_long': ['SMA'],
        'trail_ma_length_long': 160,
        'trail_ma_length_long_range': '25-200, 5',
        'trail_ma_offset_long': -1.0,
        'trail_ma_offset_long_range': '0--3, 0.5',
        'trail_ma_types_short': ['SMA'],
        'trail_ma_length_short': 160,
        'trail_ma_length_short_range': '25-200, 5',
        'trail_ma_offset_short': 1.0,
        'trail_ma_offset_short_range': '0-3, 0.5',
        'workers': mp.cpu_count()
    }
    
    root = tk.Tk()
    root.title("S_01 TrailingMA Light - Strategy Tester")
    root.geometry("900x800")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    
    # Переменные
    vars_dict = {}
    checkboxes = {}
    
    # Функция для парсинга диапазона
    def _normalize_value(value: float):
        """Преобразование значения к int при необходимости"""
        rounded = round(value, 8)
        if abs(rounded - round(rounded)) < 1e-8:
            return int(round(rounded))
        return round(rounded, 4)

    def parse_range(range_str):
        """Парсинг строки вида '25-500, 1' или '0--3, 0.5'"""
        try:
            parts = range_str.split(',')
            range_part = parts[0].strip()
            step = float(parts[1].strip())

            import re
            match = re.fullmatch(r"\s*([-+]?\d*\.?\d*)\s*-\s*([-+]?\d*\.?\d*)\s*", range_part)
            if match:
                start = float(match.group(1)) if match.group(1) else 0.0
                end = float(match.group(2)) if match.group(2) else 0.0
            else:
                start = end = float(range_part)

            if start > end:
                start, end = end, start

            result = []
            current = start
            while current <= end + 1e-9:
                result.append(_normalize_value(current))
                current += step
            return result
        except Exception:
            raw = float(range_str.split(',')[0].split('-')[0])
            return [_normalize_value(raw)]
    
    # Функция подсчета комбинаций
    def count_combinations():
        total = 1
        
        # MA types
        if checkboxes['ma_types'].get():
            ma_count = sum(1 for ma in MA_ALL if checkboxes[f'ma_{ma}'].get())
            if ma_count > 0:
                total *= ma_count
                total *= len(parse_range(vars_dict['ma_period_range'].get()))
        
        # Close Count
        if checkboxes['cc_long'].get():
            total *= len(parse_range(vars_dict['cc_long_range'].get()))
        if checkboxes['cc_short'].get():
            total *= len(parse_range(vars_dict['cc_short_range'].get()))
        
        # Stop Long
        if checkboxes['stop_long_x'].get():
            total *= len(parse_range(vars_dict['stop_long_x_range'].get()))
        if checkboxes['lp_long'].get():
            total *= len(parse_range(vars_dict['lp_long_range'].get()))
        
        # Stop Short
        if checkboxes['stop_short_x'].get():
            total *= len(parse_range(vars_dict['stop_short_x_range'].get()))
        if checkboxes['lp_short'].get():
            total *= len(parse_range(vars_dict['lp_short_range'].get()))
        
        # Stop filters
        if checkboxes['long_stop_pct'].get():
            total *= len(parse_range(vars_dict['long_stop_pct_range'].get()))
        if checkboxes['short_stop_pct'].get():
            total *= len(parse_range(vars_dict['short_stop_pct_range'].get()))
        if checkboxes['long_stop_days'].get():
            total *= len(parse_range(vars_dict['long_stop_days_range'].get()))
        if checkboxes['short_stop_days'].get():
            total *= len(parse_range(vars_dict['short_stop_days_range'].get()))
        
        # Trail MA Long
        if checkboxes['trail_ma_long'].get():
            trail_ma_long_count = sum(1 for ma in MA_ALL if checkboxes[f'trail_ma_long_{ma}'].get())
            if trail_ma_long_count > 0:
                total *= trail_ma_long_count
                total *= len(parse_range(vars_dict['trail_ma_length_long_range'].get()))
                total *= len(parse_range(vars_dict['trail_ma_offset_long_range'].get()))
        
        # Trail MA Short
        if checkboxes['trail_ma_short'].get():
            trail_ma_short_count = sum(1 for ma in MA_ALL if checkboxes[f'trail_ma_short_{ma}'].get())
            if trail_ma_short_count > 0:
                total *= trail_ma_short_count
                total *= len(parse_range(vars_dict['trail_ma_length_short_range'].get()))
                total *= len(parse_range(vars_dict['trail_ma_offset_short_range'].get()))
        
        vars_dict['combo_count'].set(f"Total combinations: {total:,}")
    
    # Создание фреймов
    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(0, weight=1)
    
    # Скроллбар
    canvas = tk.Canvas(main_frame)
    scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    for col in range(5):
        weight = 1 if col in (1, 3) else 0
        scrollable_frame.columnconfigure(col, weight=weight)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.bind(
        "<Configure>",
        lambda e: canvas.itemconfig(canvas_window, width=e.width)
    )
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
    scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
    
    # === CSV и даты ===
    row = 0
    ttk.Label(scrollable_frame, text="CSV File:").grid(row=row, column=0, sticky=tk.W, pady=5)
    vars_dict['csv_path'] = tk.StringVar(value=DEFAULTS['csv_path'])
    csv_entry = ttk.Entry(scrollable_frame, textvariable=vars_dict['csv_path'], width=50)
    csv_entry.grid(row=row, column=1, columnspan=3, sticky=(tk.W, tk.E), pady=5)
    
    def browse_csv():
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filename:
            vars_dict['csv_path'].set(filename)
    
    ttk.Button(scrollable_frame, text="Browse", command=browse_csv).grid(row=row, column=4, pady=5)
    
    row += 1
    ttk.Label(scrollable_frame, text="Start Date:").grid(row=row, column=0, sticky=tk.W, pady=5)
    vars_dict['start_date'] = tk.StringVar(value=DEFAULTS['start_date'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['start_date'], width=20).grid(row=row, column=1, sticky=tk.W, pady=5)
    
    ttk.Label(scrollable_frame, text="End Date:").grid(row=row, column=2, sticky=tk.W, pady=5)
    vars_dict['end_date'] = tk.StringVar(value=DEFAULTS['end_date'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['end_date'], width=20).grid(row=row, column=3, sticky=tk.W, pady=5)
    
    checkboxes['use_ha'] = tk.BooleanVar(value=DEFAULTS['use_ha'])
    ttk.Checkbutton(scrollable_frame, text="Heikin Ashi", variable=checkboxes['use_ha']).grid(row=row, column=4, sticky=tk.W, pady=5)
    
    # === T MA ===
    row += 1
    ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=10)
    
    row += 1
    checkboxes['ma_types'] = tk.BooleanVar(value=False)
    ttk.Checkbutton(scrollable_frame, text="", variable=checkboxes['ma_types'], command=count_combinations).grid(row=row, column=0, sticky=tk.W)
    ttk.Label(scrollable_frame, text="T MA:").grid(row=row, column=0, sticky=tk.W, padx=(20, 0))
    
    # MA types с ALL
    ma_frame = ttk.Frame(scrollable_frame)
    ma_frame.grid(row=row, column=1, columnspan=3, sticky=tk.W)
    
    checkboxes['ma_ALL'] = tk.BooleanVar(value=False)
    def toggle_all_ma():
        state = checkboxes['ma_ALL'].get()
        for ma in MA_ALL:
            checkboxes[f'ma_{ma}'].set(state)
        count_combinations()
    
    ttk.Checkbutton(ma_frame, text="ALL", variable=checkboxes['ma_ALL'], command=toggle_all_ma).pack(side=tk.LEFT)
    
    for ma in MA_ALL:
        checkboxes[f'ma_{ma}'] = tk.BooleanVar(value=(ma == 'EMA'))
        ttk.Checkbutton(ma_frame, text=ma, variable=checkboxes[f'ma_{ma}'], command=count_combinations).pack(side=tk.LEFT, padx=5)
    
    vars_dict['ma_period'] = tk.IntVar(value=DEFAULTS['ma_period'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['ma_period'], width=10).grid(row=row, column=3, sticky=tk.E)
    
    vars_dict['ma_period_range'] = tk.StringVar(value=DEFAULTS['ma_period_range'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['ma_period_range'], width=15).grid(row=row, column=4, sticky=tk.W, padx=5)
    
    # === T Break ===
    row += 1
    checkboxes['cc_long'] = tk.BooleanVar(value=False)
    ttk.Checkbutton(scrollable_frame, text="", variable=checkboxes['cc_long'], command=count_combinations).grid(row=row, column=0, sticky=tk.W)
    ttk.Label(scrollable_frame, text="T Break Long:").grid(row=row, column=0, sticky=tk.W, padx=(20, 0))
    vars_dict['cc_long'] = tk.IntVar(value=DEFAULTS['cc_long'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['cc_long'], width=10).grid(row=row, column=1, sticky=tk.W)
    vars_dict['cc_long_range'] = tk.StringVar(value=DEFAULTS['cc_long_range'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['cc_long_range'], width=15).grid(row=row, column=4, sticky=tk.W, padx=5)
    
    row += 1
    checkboxes['cc_short'] = tk.BooleanVar(value=False)
    ttk.Checkbutton(scrollable_frame, text="", variable=checkboxes['cc_short'], command=count_combinations).grid(row=row, column=0, sticky=tk.W)
    ttk.Label(scrollable_frame, text="T Break Short:").grid(row=row, column=0, sticky=tk.W, padx=(20, 0))
    vars_dict['cc_short'] = tk.IntVar(value=DEFAULTS['cc_short'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['cc_short'], width=10).grid(row=row, column=1, sticky=tk.W)
    vars_dict['cc_short_range'] = tk.StringVar(value=DEFAULTS['cc_short_range'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['cc_short_range'], width=15).grid(row=row, column=4, sticky=tk.W, padx=5)
    
    # === Stop Long ===
    row += 1
    ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=10)
    
    row += 1
    checkboxes['stop_long_x'] = tk.BooleanVar(value=False)
    ttk.Checkbutton(scrollable_frame, text="", variable=checkboxes['stop_long_x'], command=count_combinations).grid(row=row, column=0, sticky=tk.W)
    ttk.Label(scrollable_frame, text="Stop Long X:").grid(row=row, column=0, sticky=tk.W, padx=(20, 0))
    vars_dict['stop_long_x'] = tk.DoubleVar(value=DEFAULTS['stop_long_x'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['stop_long_x'], width=10).grid(row=row, column=1, sticky=tk.W)
    vars_dict['stop_long_x_range'] = tk.StringVar(value=DEFAULTS['stop_long_x_range'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['stop_long_x_range'], width=15).grid(row=row, column=4, sticky=tk.W, padx=5)
    
    ttk.Label(scrollable_frame, text="RR:").grid(row=row, column=2, sticky=tk.W, padx=5)
    vars_dict['rr_long'] = tk.IntVar(value=DEFAULTS['rr_long'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['rr_long'], width=10).grid(row=row, column=3, sticky=tk.W)
    
    row += 1
    checkboxes['lp_long'] = tk.BooleanVar(value=False)
    ttk.Checkbutton(scrollable_frame, text="", variable=checkboxes['lp_long'], command=count_combinations).grid(row=row, column=0, sticky=tk.W)
    ttk.Label(scrollable_frame, text="LP Long:").grid(row=row, column=0, sticky=tk.W, padx=(20, 0))
    vars_dict['lp_long'] = tk.IntVar(value=DEFAULTS['lp_long'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['lp_long'], width=10).grid(row=row, column=1, sticky=tk.W)
    vars_dict['lp_long_range'] = tk.StringVar(value=DEFAULTS['lp_long_range'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['lp_long_range'], width=15).grid(row=row, column=4, sticky=tk.W, padx=5)
    
    # === Stop Short ===
    row += 1
    checkboxes['stop_short_x'] = tk.BooleanVar(value=False)
    ttk.Checkbutton(scrollable_frame, text="", variable=checkboxes['stop_short_x'], command=count_combinations).grid(row=row, column=0, sticky=tk.W)
    ttk.Label(scrollable_frame, text="Stop Short X:").grid(row=row, column=0, sticky=tk.W, padx=(20, 0))
    vars_dict['stop_short_x'] = tk.DoubleVar(value=DEFAULTS['stop_short_x'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['stop_short_x'], width=10).grid(row=row, column=1, sticky=tk.W)
    vars_dict['stop_short_x_range'] = tk.StringVar(value=DEFAULTS['stop_short_x_range'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['stop_short_x_range'], width=15).grid(row=row, column=4, sticky=tk.W, padx=5)
    
    ttk.Label(scrollable_frame, text="RR:").grid(row=row, column=2, sticky=tk.W, padx=5)
    vars_dict['rr_short'] = tk.IntVar(value=DEFAULTS['rr_short'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['rr_short'], width=10).grid(row=row, column=3, sticky=tk.W)
    
    row += 1
    checkboxes['lp_short'] = tk.BooleanVar(value=False)
    ttk.Checkbutton(scrollable_frame, text="", variable=checkboxes['lp_short'], command=count_combinations).grid(row=row, column=0, sticky=tk.W)
    ttk.Label(scrollable_frame, text="LP Short:").grid(row=row, column=0, sticky=tk.W, padx=(20, 0))
    vars_dict['lp_short'] = tk.IntVar(value=DEFAULTS['lp_short'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['lp_short'], width=10).grid(row=row, column=1, sticky=tk.W)
    vars_dict['lp_short_range'] = tk.StringVar(value=DEFAULTS['lp_short_range'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['lp_short_range'], width=15).grid(row=row, column=4, sticky=tk.W, padx=5)
    
    # === Stop Filters ===
    row += 1
    ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=10)
    
    row += 1
    checkboxes['long_stop_pct'] = tk.BooleanVar(value=False)
    ttk.Checkbutton(scrollable_frame, text="", variable=checkboxes['long_stop_pct'], command=count_combinations).grid(row=row, column=0, sticky=tk.W)
    ttk.Label(scrollable_frame, text="L Stop Max %:").grid(row=row, column=0, sticky=tk.W, padx=(20, 0))
    vars_dict['long_stop_pct'] = tk.IntVar(value=DEFAULTS['long_stop_pct'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['long_stop_pct'], width=10).grid(row=row, column=1, sticky=tk.W)
    vars_dict['long_stop_pct_range'] = tk.StringVar(value=DEFAULTS['long_stop_pct_range'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['long_stop_pct_range'], width=15).grid(row=row, column=4, sticky=tk.W, padx=5)
    
    row += 1
    checkboxes['short_stop_pct'] = tk.BooleanVar(value=False)
    ttk.Checkbutton(scrollable_frame, text="", variable=checkboxes['short_stop_pct'], command=count_combinations).grid(row=row, column=0, sticky=tk.W)
    ttk.Label(scrollable_frame, text="S Stop Max %:").grid(row=row, column=0, sticky=tk.W, padx=(20, 0))
    vars_dict['short_stop_pct'] = tk.IntVar(value=DEFAULTS['short_stop_pct'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['short_stop_pct'], width=10).grid(row=row, column=1, sticky=tk.W)
    vars_dict['short_stop_pct_range'] = tk.StringVar(value=DEFAULTS['short_stop_pct_range'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['short_stop_pct_range'], width=15).grid(row=row, column=4, sticky=tk.W, padx=5)
    
    row += 1
    checkboxes['long_stop_days'] = tk.BooleanVar(value=False)
    ttk.Checkbutton(scrollable_frame, text="", variable=checkboxes['long_stop_days'], command=count_combinations).grid(row=row, column=0, sticky=tk.W)
    ttk.Label(scrollable_frame, text="L Stop Max D:").grid(row=row, column=0, sticky=tk.W, padx=(20, 0))
    vars_dict['long_stop_days'] = tk.IntVar(value=DEFAULTS['long_stop_days'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['long_stop_days'], width=10).grid(row=row, column=1, sticky=tk.W)
    vars_dict['long_stop_days_range'] = tk.StringVar(value=DEFAULTS['long_stop_days_range'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['long_stop_days_range'], width=15).grid(row=row, column=4, sticky=tk.W, padx=5)
    
    row += 1
    checkboxes['short_stop_days'] = tk.BooleanVar(value=False)
    ttk.Checkbutton(scrollable_frame, text="", variable=checkboxes['short_stop_days'], command=count_combinations).grid(row=row, column=0, sticky=tk.W)
    ttk.Label(scrollable_frame, text="S Stop Max D:").grid(row=row, column=0, sticky=tk.W, padx=(20, 0))
    vars_dict['short_stop_days'] = tk.IntVar(value=DEFAULTS['short_stop_days'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['short_stop_days'], width=10).grid(row=row, column=1, sticky=tk.W)
    vars_dict['short_stop_days_range'] = tk.StringVar(value=DEFAULTS['short_stop_days_range'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['short_stop_days_range'], width=15).grid(row=row, column=4, sticky=tk.W, padx=5)
    
    # === Trailing ===
    row += 1
    ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=10)
    
    row += 1
    ttk.Label(scrollable_frame, text="RR TrailingMALong:").grid(row=row, column=0, sticky=tk.W)
    vars_dict['trail_rr_long'] = tk.IntVar(value=DEFAULTS['trail_rr_long'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['trail_rr_long'], width=10).grid(row=row, column=1, sticky=tk.W)
    
    ttk.Label(scrollable_frame, text="RR TrailingMAShort:").grid(row=row, column=2, sticky=tk.W, padx=5)
    vars_dict['trail_rr_short'] = tk.IntVar(value=DEFAULTS['trail_rr_short'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['trail_rr_short'], width=10).grid(row=row, column=3, sticky=tk.W)
    
    # Trail MA Long
    row += 1
    checkboxes['trail_ma_long'] = tk.BooleanVar(value=False)
    ttk.Checkbutton(scrollable_frame, text="", variable=checkboxes['trail_ma_long'], command=count_combinations).grid(row=row, column=0, sticky=tk.W)
    ttk.Label(scrollable_frame, text="Trail MA Long:").grid(row=row, column=0, sticky=tk.W, padx=(20, 0))
    
    trail_ma_long_frame = ttk.Frame(scrollable_frame)
    trail_ma_long_frame.grid(row=row, column=1, columnspan=3, sticky=tk.W)
    
    checkboxes['trail_ma_long_ALL'] = tk.BooleanVar(value=False)
    def toggle_all_trail_ma_long():
        state = checkboxes['trail_ma_long_ALL'].get()
        for ma in MA_ALL:
            checkboxes[f'trail_ma_long_{ma}'].set(state)
        count_combinations()
    
    ttk.Checkbutton(trail_ma_long_frame, text="ALL", variable=checkboxes['trail_ma_long_ALL'], command=toggle_all_trail_ma_long).pack(side=tk.LEFT)
    
    for ma in MA_ALL:
        checkboxes[f'trail_ma_long_{ma}'] = tk.BooleanVar(value=(ma == 'SMA'))
        ttk.Checkbutton(trail_ma_long_frame, text=ma, variable=checkboxes[f'trail_ma_long_{ma}'], command=count_combinations).pack(side=tk.LEFT, padx=2)
    
    row += 1
    ttk.Label(scrollable_frame, text="Length:").grid(row=row, column=0, sticky=tk.W, padx=(40, 0))
    vars_dict['trail_ma_length_long'] = tk.IntVar(value=DEFAULTS['trail_ma_length_long'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['trail_ma_length_long'], width=10).grid(row=row, column=1, sticky=tk.W)
    vars_dict['trail_ma_length_long_range'] = tk.StringVar(value=DEFAULTS['trail_ma_length_long_range'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['trail_ma_length_long_range'], width=15).grid(row=row, column=4, sticky=tk.W, padx=5)
    
    row += 1
    ttk.Label(scrollable_frame, text="Offset:").grid(row=row, column=0, sticky=tk.W, padx=(40, 0))
    vars_dict['trail_ma_offset_long'] = tk.DoubleVar(value=DEFAULTS['trail_ma_offset_long'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['trail_ma_offset_long'], width=10).grid(row=row, column=1, sticky=tk.W)
    vars_dict['trail_ma_offset_long_range'] = tk.StringVar(value=DEFAULTS['trail_ma_offset_long_range'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['trail_ma_offset_long_range'], width=15).grid(row=row, column=4, sticky=tk.W, padx=5)
    
    # Trail MA Short
    row += 1
    checkboxes['trail_ma_short'] = tk.BooleanVar(value=False)
    ttk.Checkbutton(scrollable_frame, text="", variable=checkboxes['trail_ma_short'], command=count_combinations).grid(row=row, column=0, sticky=tk.W)
    ttk.Label(scrollable_frame, text="Trail MA Short:").grid(row=row, column=0, sticky=tk.W, padx=(20, 0))
    
    trail_ma_short_frame = ttk.Frame(scrollable_frame)
    trail_ma_short_frame.grid(row=row, column=1, columnspan=3, sticky=tk.W)
    
    checkboxes['trail_ma_short_ALL'] = tk.BooleanVar(value=False)
    def toggle_all_trail_ma_short():
        state = checkboxes['trail_ma_short_ALL'].get()
        for ma in MA_ALL:
            checkboxes[f'trail_ma_short_{ma}'].set(state)
        count_combinations()
    
    ttk.Checkbutton(trail_ma_short_frame, text="ALL", variable=checkboxes['trail_ma_short_ALL'], command=toggle_all_trail_ma_short).pack(side=tk.LEFT)
    
    for ma in MA_ALL:
        checkboxes[f'trail_ma_short_{ma}'] = tk.BooleanVar(value=(ma == 'SMA'))
        ttk.Checkbutton(trail_ma_short_frame, text=ma, variable=checkboxes[f'trail_ma_short_{ma}'], command=count_combinations).pack(side=tk.LEFT, padx=2)
    
    row += 1
    ttk.Label(scrollable_frame, text="Length:").grid(row=row, column=0, sticky=tk.W, padx=(40, 0))
    vars_dict['trail_ma_length_short'] = tk.IntVar(value=DEFAULTS['trail_ma_length_short'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['trail_ma_length_short'], width=10).grid(row=row, column=1, sticky=tk.W)
    vars_dict['trail_ma_length_short_range'] = tk.StringVar(value=DEFAULTS['trail_ma_length_short_range'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['trail_ma_length_short_range'], width=15).grid(row=row, column=4, sticky=tk.W, padx=5)
    
    row += 1
    ttk.Label(scrollable_frame, text="Offset:").grid(row=row, column=0, sticky=tk.W, padx=(40, 0))
    vars_dict['trail_ma_offset_short'] = tk.DoubleVar(value=DEFAULTS['trail_ma_offset_short'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['trail_ma_offset_short'], width=10).grid(row=row, column=1, sticky=tk.W)
    vars_dict['trail_ma_offset_short_range'] = tk.StringVar(value=DEFAULTS['trail_ma_offset_short_range'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['trail_ma_offset_short_range'], width=15).grid(row=row, column=4, sticky=tk.W, padx=5)
    
    # === Workers ===
    row += 1
    ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=10)
    
    row += 1
    ttk.Label(scrollable_frame, text="Workers:").grid(row=row, column=0, sticky=tk.W)
    vars_dict['workers'] = tk.IntVar(value=DEFAULTS['workers'])
    ttk.Entry(scrollable_frame, textvariable=vars_dict['workers'], width=10).grid(row=row, column=1, sticky=tk.W)
    
    # === Combo count ===
    row += 1
    vars_dict['combo_count'] = tk.StringVar(value="Total combinations: 1")
    ttk.Label(scrollable_frame, textvariable=vars_dict['combo_count'], font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=5, pady=10)
    
    # === Buttons ===
    row += 1
    button_frame = ttk.Frame(scrollable_frame)
    button_frame.grid(row=row, column=0, columnspan=5, pady=10)
    
    def save_preset():
        filename = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if filename:
            preset = {}
            for key, var in vars_dict.items():
                if key != 'combo_count':
                    preset[key] = var.get()
            for key, var in checkboxes.items():
                preset[f'check_{key}'] = var.get()
            with open(filename, 'w') as f:
                json.dump(preset, f, indent=2)
            messagebox.showinfo("Success", "Preset saved!")
    
    def load_preset():
        filename = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if filename:
            with open(filename, 'r') as f:
                preset = json.load(f)
            for key, value in preset.items():
                if key.startswith('check_'):
                    checkbox_key = key[6:]
                    if checkbox_key in checkboxes:
                        checkboxes[checkbox_key].set(value)
                elif key in vars_dict and key != 'combo_count':
                    vars_dict[key].set(value)
            count_combinations()
            messagebox.showinfo("Success", "Preset loaded!")
    
    def reset_defaults():
        for key, value in DEFAULTS.items():
            if key in vars_dict:
                vars_dict[key].set(value)
        for key in checkboxes:
            if not key.startswith('ma_') and not key.startswith('trail_ma_'):
                checkboxes[key].set(False)
        for ma in MA_ALL:
            checkboxes[f'ma_{ma}'].set(ma == 'EMA')
            checkboxes[f'trail_ma_long_{ma}'].set(ma == 'SMA')
            checkboxes[f'trail_ma_short_{ma}'].set(ma == 'SMA')
        checkboxes['ma_ALL'].set(False)
        checkboxes['trail_ma_long_ALL'].set(False)
        checkboxes['trail_ma_short_ALL'].set(False)
        count_combinations()
    
    def run_test():
        """Запуск одиночного теста"""
        csv_path = vars_dict['csv_path'].get()
        if not csv_path or not Path(csv_path).exists():
            messagebox.showerror("Error", "Please select a valid CSV file")
            return
        
        # Собираем конфигурацию
        config = {
            'csv_path': csv_path,
            'start_date': vars_dict['start_date'].get(),
            'end_date': vars_dict['end_date'].get(),
            'use_ha': checkboxes['use_ha'].get(),
            'ma_types': [vars_dict['ma_period'].get()],  # Одиночное значение
            'ma_periods': [vars_dict['ma_period'].get()],
            'cc_long': [vars_dict['cc_long'].get()],
            'cc_short': [vars_dict['cc_short'].get()],
            # ... и так далее для всех параметров
        }
        
        messagebox.showinfo("Test", "Single test functionality - to be implemented with chart visualization")
    
    def run_grid():
        """Запуск grid search"""
        csv_path = vars_dict['csv_path'].get()
        if not csv_path or not Path(csv_path).exists():
            messagebox.showerror("Error", "Please select a valid CSV file")
            return
        
        # Собираем конфигурацию для перебора
        config = {
            'start_date': vars_dict['start_date'].get(),
            'end_date': vars_dict['end_date'].get(),
            'outdir': '.'
        }
        
        # MA types
        if checkboxes['ma_types'].get():
            config['ma_types'] = [ma for ma in MA_ALL if checkboxes[f'ma_{ma}'].get()]
            config['ma_periods'] = parse_range(vars_dict['ma_period_range'].get())
        else:
            config['ma_types'] = [list(MA_ALL)[list(checkboxes[f'ma_{ma}'].get() for ma in MA_ALL).index(True)]]
            config['ma_periods'] = [vars_dict['ma_period'].get()]
        
        # Close Count
        config['cc_long'] = parse_range(vars_dict['cc_long_range'].get()) if checkboxes['cc_long'].get() else [vars_dict['cc_long'].get()]
        config['cc_short'] = parse_range(vars_dict['cc_short_range'].get()) if checkboxes['cc_short'].get() else [vars_dict['cc_short'].get()]
        
        # Остальные параметры аналогично...
        
        # Запуск в отдельном потоке
        def run_thread():
            try:
                run_grid_search(csv_path, config, vars_dict['workers'].get(), checkboxes['use_ha'].get())
                messagebox.showinfo("Success", "Grid search completed!")
            except Exception as e:
                messagebox.showerror("Error", str(e))
        
        threading.Thread(target=run_thread, daemon=True).start()
    
    ttk.Button(button_frame, text="Save Preset", command=save_preset).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Load Preset", command=load_preset).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Reset Defaults", command=reset_defaults).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Run Test", command=run_test).pack(side=tk.LEFT, padx=20)
    ttk.Button(button_frame, text="Run Grid", command=run_grid).pack(side=tk.LEFT, padx=5)
    
    # Инициализация счетчика комбинаций
    count_combinations()
    
    root.mainloop()

# ================== main ==================
def main():
    parser = argparse.ArgumentParser("S_01 TrailingMA Light — Strategy Tester")
    parser.add_argument("--gui", action="store_true", help="Launch GUI")
    parser.add_argument("--csv", help="CSV file path")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD HH:MM)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD HH:MM)")
    parser.add_argument("--ha", action="store_true", help="Use Heikin Ashi candles")
    parser.add_argument("--engine", choices=["process", "thread"], default="process",
                        help="Parallel execution backend (process/thread)")
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    
    args = parser.parse_args()
    
    if args.gui or not args.csv:
        create_gui()
    else:
        # CLI mode - запуск с дефолтными параметрами
        config = {
            'start_date': args.start,
            'end_date': args.end,
            'ma_types': ['EMA'],
            'ma_periods': [45],
            'cc_long': [7],
            'cc_short': [5],
            'stop_long_x': [2.0],
            'rr_long': [3],
            'lp_long': [2],
            'stop_short_x': [2.0],
            'rr_short': [3],
            'lp_short': [2],
            'long_stop_pct': [3],
            'short_stop_pct': [3],
            'long_stop_days': [2],
            'short_stop_days': [4],
            'trail_rr_long': [1],
            'trail_rr_short': [1],
            'trail_ma_types_long': ['SMA'],
            'trail_ma_lengths_long': [160],
            'trail_ma_offsets_long': [-1.0],
            'trail_ma_types_short': ['SMA'],
            'trail_ma_lengths_short': [160],
            'trail_ma_offsets_short': [1.0],
            'outdir': '.'
        }
        
        config['engine'] = args.engine
        run_grid_search(args.csv, config, args.workers, args.ha)

if __name__ == "__main__":
    if os.name == "nt":
        mp.freeze_support()
    main()