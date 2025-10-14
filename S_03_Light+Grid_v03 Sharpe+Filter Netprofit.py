# -*- coding: utf-8 -*-
"""
S_03 Reversal — FAST grid search (DD closed-only)
+ --ha: индикаторы/сигналы на Heikin Ashi, исполнение по обычным свечам
+ --csv: файл | директория | glob (*.csv)
+ ИМЯ ФАЙЛА: <Префикс из CSV> <YYYY.MM.DD-YYYY.MM.DD>_<MA1+MA2+...|ALL>[_ha].csv

  (пример: OKX_COREUSDT.P, 30 2024.07.01-2025.08.01_T3+KAMA_ha.csv
           OKX_COREUSDT.P, 30 2024.07.01-2025.08.01_ALL.csv)
+ Новый тип MA: HMAG (bug-compatible Hull MA на «перевёрнутом» WMA)

ДОБАВЛЕНО В ПРЕДЫДУЩЕЙ ВЕРСИИ:
- --ma <тип [периоды]>: фиксация периодов (например, "t3 64", "t3 64,65,66", "t3  64-66").
  Если периоды не указаны — используется полный диапазон 25..500 (как раньше).
- --cc L,S: фиксация Close Count Long/Short (например, --cc 3,4).
- Исключение отдельных MA при переборе: пример `--ma all -HMAG -T3`
  (исключит HMAG и T3, остальные из списка ALL будут перебираться).

ДОБАВЛЕНО В ЭТОЙ ВЕРСИИ (строго по ТЗ):
- --filter netprofit N  → в итоговый CSV попадают только строки с Net Profit% >= N.
"""

import os
import re
import glob
import math
import time
import argparse
import multiprocessing as mp
from typing import Tuple, List, Dict, Optional
from pathlib import Path

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
month_arr: np.ndarray = None   # Added: array of month indices for each timestamp

# Массивы для расчёта индикаторов/сигналов (обычные или HA):
sc_arr: np.ndarray = None   # close для сигналов
sh_arr: np.ndarray = None   # high  для сигналов
sl_arr: np.ndarray = None   # low   для сигналов

commission_rate: float = 0.0005
initial_capital: float = 100.0
contract_size: float = 0.01
vwap_tz_offset_hours: int = 8
ma_cache: Dict[Tuple[str, int], np.ndarray] = {}

DD_MODE_CLOSED: bool = True  # DD только по закрытым сделкам

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
    from datetime import datetime, timedelta
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
    """
    Возвращает 'Биржа_Тикер, Таймфрейм' из имени файла.
    Берём всё до блока дат 'YYYY.MM.DD-YYYY.MM.DD', иначе — выражение '..., <число>'.
    """
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
    r"""Удаляет запрещённые в Windows символы:  \\ / : * ? " < > |  и обрезает пробелы/точки в конце."""
    pattern = re.compile(r'[\\/:*?"<>|]')
    s = pattern.sub('', s)
    return s.rstrip(' .')

def expand_csv_argument(arg: str) -> List[str]:
    """--csv: файл, директория или glob-шаблон. Возвращает список .csv путей."""
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

# ================== «Баг-совместимые» WMA/HMA (HMAG) ==================
def wma_bug(arr: np.ndarray, n: int) -> np.ndarray:
    """
    Перевёрнутый WMA: больший вес у старых баров окна.
    Эквивалентность: WMA_bug(src, n) = 2 * SMA(src, n) - WMA(src, n)
    Округление периода: n -> max(1, round(n)).
    Возвращает массив той же длины с NaN на первых n-1 барах.
    """
    n = max(1, int(round(n)))
    if n > arr.shape[0]:
        return np.full(arr.shape[0], np.nan)
    s = sma(arr, n)   # NaN до n-1
    w = wma(arr, n)   # NaN до n-1
    return 2.0 * s - w

def hma_bug(arr: np.ndarray, n: int) -> np.ndarray:
    """
    HMAG на базе wma_bug:
      HMAG(n) = WMA_bug( 2*WMA_bug(src, round(n/2)) - WMA_bug(src, n), round(sqrt(n)) )
    Округления: round(...), затем max(1, ...). Корректный сдвиг NaN как у HMA.
    """
    n = max(1, int(round(n)))
    n_half = max(1, int(round(n / 2.0)))
    n_sqrt = max(1, int(round(math.sqrt(n))))
    w1 = wma_bug(arr, n_half)
    w2 = wma_bug(arr, n)
    start = n - 1
    diff = 2.0 * w1 - w2
    tail = diff[start:]
    if tail.shape[0] < n_sqrt:
        return np.full(arr.shape[0], np.nan)
    w3 = wma_bug(tail, n_sqrt)
    valid = w3[n_sqrt - 1:]
    out = np.full(arr.shape[0], np.nan)
    out[start + (n_sqrt - 1): start + (n_sqrt - 1) + valid.shape[0]] = valid
    return out

def compute_ma(ma_type: str, length: int) -> np.ndarray:
    key = ma_type.upper()
    if key == "SMA":   return sma(sc_arr, length)
    if key == "EMA":   return ema(sc_arr, length)
    if key == "WMA":   return wma(sc_arr, length)
    if key == "HMA":   return hma(sc_arr, length)
    if key == "HMAG":  return hma_bug(sc_arr, length)  # <--- HMAG
    if key == "ALMA":  return alma(sc_arr, length, 0.85, 6.0)
    if key == "KAMA":  return kama(sc_arr, length, 2, 30)
    if key == "TMA":   return tma_triangular(sc_arr, length)
    if key == "T3":    return t3(sc_arr, length, 0.7)
    if key == "DEMA":  return dema(sc_arr, length)
    if key == "VWMA":  return vwma(sc_arr, v_arr, length)
    if key == "VWAP":  return vwap(t_arr, sh_arr, sl_arr, sc_arr, v_arr, vwap_tz_offset_hours)
    raise ValueError(f"Unsupported MA: {ma_type}")

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

    # Compute month array for each timestamp (to detect month boundaries for Sharpe)
    month_arr = pd.to_datetime(df['time'], unit='s').dt.month.to_numpy(np.int32)

# ================== Симуляция одной комбинации ==================
def simulate(params: Tuple[str, int, int, int]) -> Tuple:
    ma_type, ma_period, n_long, n_short = params

    # Get or compute the moving average series for this MA type/period
    key = (ma_type.upper(), ma_period)
    ma = ma_cache.get(key)
    if ma is None:
        ma = compute_ma(ma_type, ma_period)
        ma_cache[key] = ma

    # Initialize simulation state
    cash = float(initial_capital)
    pos_qty = 0.0
    position = 0
    entry_price = 0.0
    entry_fee = 0.0
    prev_position = 0

    count_long = 0
    count_short = 0
    trades = 0
    wins = 0
    loss_streak = 0
    max_loss_streak = 0

    # DD по закрытым сделкам
    peak = initial_capital
    max_drawdown = 0.0

    next_close = 0
    next_open = 0

    # Sharpe ratio tracking variables
    month_start_equity = initial_capital
    monthly_returns: List[float] = []
    last_equity = initial_capital

    for i in range(o_arr.shape[0]):
        # Check for new month boundary
        if i > 0 and month_arr[i] != month_arr[i-1]:
            if month_start_equity > 0:
                monthly_returns.append((last_equity / month_start_equity - 1.0) * 100.0)
            month_start_equity = last_equity

        # закрытие на open (close position if signal)
        if next_close != 0 and position != 0:
            px = o_arr[i]
            exit_val = abs(pos_qty) * px
            fee = exit_val * commission_rate
            if position == 1:
                cash += pos_qty * px - fee
                pnl = pos_qty * (px - entry_price) - entry_fee - fee
            else:
                cash -= abs(pos_qty) * px + fee
                pnl = abs(pos_qty) * (entry_price - px) - entry_fee - fee
            trades += 1
            if pnl > 0:
                wins += 1
                loss_streak = 0
            else:
                loss_streak += 1
                max_loss_streak = max(max_loss_streak, loss_streak)
            pos_qty = 0.0
            position = 0
            entry_fee = 0.0
            entry_price = 0.0
            next_close = 0
            prev_position = 0

            eq = cash
            if eq > peak:
                peak = eq
            else:
                drop = (eq - peak) / peak
                if drop < max_drawdown:
                    max_drawdown = drop

        # открытие на open (open new position if signal)
        if next_open != 0 and position == 0:
            px = o_arr[i]
            qty = math.floor((cash / px) / contract_size) * contract_size
            if qty > 0:
                if next_open == 1:
                    cost = qty * px
                    fee = cost * commission_rate
                    cash -= (cost + fee)
                    pos_qty = qty
                    position = 1
                    entry_price = px
                    entry_fee = fee
                else:
                    proceeds = qty * px
                    fee = proceeds * commission_rate
                    cash += (proceeds - fee)
                    pos_qty = -qty
                    position = -1
                    entry_price = px
                    entry_fee = fee
                prev_position = position
            next_open = 0

        # сигналы по массивам сигналов (std или HA)
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

        sig_long = (count_long >= n_long)
        sig_short = (count_short >= n_short)

        next_close = 0
        next_open = 0
        if position == 1 and sig_short:
            next_close = 1
        elif position == -1 and sig_long:
            next_close = -1

        if position == 0 and prev_position == 0:
            if sig_long:
                next_open = 1
            elif sig_short:
                next_open = -1

        if position == 0:
            prev_position = 0

        # Calculate equity at end of this bar (mark-to-market for open positions)
        if position != 0:
            current_equity = cash + pos_qty * c_arr[i]
        else:
            current_equity = cash
        last_equity = current_equity

    # форс-закрытие на последнем close (close any open position at the end)
    if position != 0:
        px = c_arr[-1]
        exit_val = abs(pos_qty) * px
        fee = exit_val * commission_rate
        if position == 1:
            cash += pos_qty * px - fee
            pnl = pos_qty * (px - entry_price) - entry_fee - fee
        else:
            cash -= abs(pos_qty) * px + fee
            pnl = abs(pos_qty) * (entry_price - px) - entry_fee - fee
        trades += 1
        if pnl > 0:
            wins += 1
        else:
            loss_streak += 1
            max_loss_streak = max(max_loss_streak, loss_streak)
        eq = cash
        if eq > peak:
            peak = eq
        else:
            drop = (eq - peak) / peak
            if drop < max_drawdown:
                max_drawdown = drop

    final_equity = cash
    net_profit = (final_equity / initial_capital - 1.0) * 100.0
    winrate = (wins / trades * 100.0) if trades > 0 else 0.0
    dd_pct = max_drawdown * 100.0

    # Sharpe Ratio calculation (monthly returns of equity)
    sharpe_value = None
    if len(monthly_returns) >= 2:
        avg_return = np.mean(monthly_returns)
        sd_return = np.std(monthly_returns, ddof=0)    # population stdev
        rfr_m = (0.02 * 100.0) / 12.0                  # 2% annual -> monthly %
        if sd_return != 0:
            sharpe_value = (avg_return - rfr_m) / sd_return
    sharpe_str = f"{sharpe_value:.2f}" if sharpe_value is not None else ""

    return (ma_type.upper(), ma_period, n_long, n_short,
            f"{winrate:.2f}%", f"{net_profit:.2f}%", f"{dd_pct:.2f}%",
            trades, max_loss_streak, sharpe_str)

# ================== Парсинг --ma и --cc ==================
MA_ALL = ["EMA","SMA","HMA","HMAG","ALMA","WMA","KAMA","TMA","T3","DEMA","VWMA","VWAP"]
CSV_HEADER = "MA Type,MA Period,Close Count Long,Close Count Short,Winrate%,Net Profit%,DD%,Total Trades,Max SL,Sharpe\n"

def parse_period_list(spec: str) -> List[int]:
    """Поддерживает '64', '64,65,66', '64-66' и комбинации '60-62,64,70-72'."""
    spec = spec.replace(' ', '')
    out: List[int] = []
    for part in spec.split(','):
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            a = int(a); b = int(b)
            if a > b: a, b = b, a
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    return sorted(dict.fromkeys(out))

def parse_ma_tokens(tokens: List[str]) -> Tuple[Dict[str, Optional[List[int]]], List[str], bool]:
    """
    Разбор --ma.
    Поддержка:
      - 'ALL' (перебор всех типов);
      - исключения: '-T3', '-HMAG' (обычно вместе с ALL: 'ALL -T3 -HMAG');
      - явные типы с/без периодов: 'T3 64-66 EMA 100,120 VWMA'.
    Возвращает:
      ma_map: { 'T3': [список периодов] или None (если 'все периоды') }
      ma_types_for_name: список типов для имени файла
      is_all: True если в аргументах присутствовал 'ALL' (даже с исключениями)
    """
    if not tokens:
        return {m: None for m in MA_ALL}, MA_ALL[:], True

    # нормализация: заменим запятые/табуляции на пробелы, схлопнем множественные пробелы
    s = re.sub(r'[,\t;]+', ' ', " ".join(tokens).strip())
    s = re.sub(r'\s+', ' ', s).strip()
    if not s:
        return {m: None for m in MA_ALL}, MA_ALL[:], True

    parts = s.split(' ')
    has_all = any(p.upper() == "ALL" for p in parts)

    excludes: set = set()
    # Спецификации по конкретным МА (периоды)
    specs: Dict[str, Optional[List[int]]] = {}
    cur: Optional[str] = None
    buf: List[str] = []

    def flush_current():
        nonlocal cur, buf, specs
        if cur is None:
            return
        if buf:
            specs[cur] = parse_period_list(" ".join(buf))
        else:
            specs[cur] = None
        buf = []
        cur = None

    for tok in parts:
        up = tok.upper()
        if up == "ALL":
            flush_current()
            continue
        if up.startswith('-'):  # исключение: -T3, -HMAG ...
            flush_current()
            name = up[1:]
            if name in MA_ALL:
                excludes.add(name)
            else:
                raise SystemExit(f"Unknown MA to exclude: '{tok}'")
            continue
        if up in MA_ALL:
            flush_current()
            cur = up
            buf = []
            # отметим присутствие; периоды зададим в flush, если будут
            if cur not in specs:
                specs[cur] = None
            continue
        # иначе это часть периодов
        if cur is None:
            raise SystemExit(f"Invalid --ma syntax near '{tok}'. Start with a MA name or 'ALL'.")
        buf.append(tok)
    flush_current()

    # Сформировать итоговый набор МА
    if has_all:
        chosen = [m for m in MA_ALL if m not in excludes]
        # Спецификации с периодами (если заданы) переопределяют None
        ma_map: Dict[str, Optional[List[int]]] = {m: None for m in chosen}
        for m, per in specs.items():
            if m in ma_map and per is not None:
                ma_map[m] = per
        # ma_types_for_name: при ALL (даже с исключениями) оставим 'ALL' в имени файла
        return ma_map, chosen, True
    else:
        # без ALL — только то, что явно перечислено (никаких исключений)
        if not specs:
            raise SystemExit("No valid MA was parsed from --ma.")
        chosen = [m for m in specs.keys()]
        ma_map = dict(specs)
        return ma_map, chosen, False

# ================== Грид/запуск для одного CSV ==================
def run_for_one_csv(csv_path: str, args, ma_map: Dict[str, Optional[List[int]]], ma_types_for_name: List[str]) -> Path:
    # Close Count: либо фиксируем через --cc, либо диапазон 2..10
    if args.cc is not None:
        try:
            cc_l, cc_s = args.cc.replace(' ', '').split(',')
            cc_long = [int(cc_l)]
            cc_short = [int(cc_s)]
        except Exception:
            raise SystemExit("--cc must be like 'L,S', e.g. --cc 3,4")
    else:
        cc_long = range(2, 11)
        cc_short = range(2, 11)

    # грид-комбинации
    combos: List[Tuple[str, int, int, int]] = []
    for ma in ma_map.keys():
        if ma == "VWAP":
            periods = [0]  # период не используется
        else:
            periods = ma_map[ma] if ma_map[ma] is not None else range(25, 501)
        for p in periods:
            for nl in cc_long:
                for ns in cc_short:
                    combos.append((ma, int(p), int(nl), int(ns)))

    total = len(combos)
    print(f"\n=== {Path(csv_path).name} ===")
    print(f"Total combinations: {total:,}")

    start_ts = parse_local_to_utc_epoch(args.start, 8) if args.start else None
    end_ts   = parse_local_to_utc_epoch(args.end,   8) if args.end   else None

    # выбор пула
    if args.engine == "process":
        pool_factory = mp.Pool
        init_kwargs = dict(initializer=_init_pool,
                           initargs=(csv_path, start_ts, end_ts,
                                     args.commission, 100.0, args.contract_size, args.vwap_tz, args.ha))
    else:
        from multiprocessing.dummy import Pool as ThreadPool
        pool_factory = ThreadPool
        _init_pool(csv_path, start_ts, end_ts, args.commission, 100.0, args.contract_size, args.vwap_tz, args.ha)
        init_kwargs = {}

    # калибровка ETA
    sample_n = min(30, total)
    t0 = time.time()
    with pool_factory(processes=args.workers, **init_kwargs) as pool:
        pool.map(simulate, combos[:sample_n])
    avg = (time.time() - t0) / max(1, sample_n)
    eta = avg * total / max(1, args.workers)
    print(f"ETA ≈ {fmt_dur(eta)}  (engine={args.engine}, workers={args.workers}, HA={args.ha})")

    # основной прогон
    results: List[Tuple] = []
    t0 = time.time()
    with pool_factory(processes=args.workers, **init_kwargs) as pool:
        CHUNK = 2000
        for i in tqdm(range(0, total, CHUNK), total=(total+CHUNK-1)//CHUNK, desc="Grid"):
            results.extend(pool.map(simulate, combos[i:i+CHUNK]))
    dt = time.time() - t0
    print(f"Finished in {fmt_dur(dt)}; processed {len(results):,} combos")

    # --- ФИЛЬТР NetProfit перед сортировкой (минимальные изменения, максимум скорости) ---
    if args.filter and args.filter[0].lower() == "netprofit":
        try:
            thr = float(args.filter[1])
        except Exception:
            raise SystemExit("--filter netprofit <N>  (пример: --filter netprofit 0)")
        before = len(results)
        results = [r for r in results if float(r[5].rstrip('%')) >= thr]
        after = len(results)
        print(f"Filter: Net Profit% ≥ {thr:.2f} → kept {after:,}/{before:,}")

    # сортировка и сохранение (по Net Profit%)
    results.sort(key=lambda r: float(r[5].rstrip('%')), reverse=True)

    prefix = extract_prefix_from_filename(csv_path)
    cli_dates = dates_cli_label(args.start, args.end)

    # имя файла: если --ma ALL (включая вариант с исключениями) — ставим _ALL
    if getattr(args, "ma_all", False):
        ma_suffix = "ALL"
    else:
        ma_suffix = "+".join(ma_types_for_name) if ma_types_for_name else "ALL"

    base_name = f"{prefix} {cli_dates}_{ma_suffix}".strip()
    if args.ha:
        base_name += "_ha"
    base_name = re.sub(r'\s+', ' ', base_name)
    file_name = sanitize_filename(base_name) + ".csv"

    out_csv = Path(args.outdir) / file_name
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(CSV_HEADER)
        for r in results:
            f.write("{},{},{},{},{},{},{},{},{},{}\n".format(*r))

    print(f"Saved: {out_csv}")
    return out_csv

# ================== main ==================
def main():
    parser = argparse.ArgumentParser("S_03 Reversal — FAST grid (DD closed-only), multi-CSV + Heikin Ashi option + HMAG")
    parser.add_argument("--csv", required=True,
                        help="Файл CSV, директория или glob-шаблон (напр. C:\\data\\*.csv)")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--commission", type=float, default=0.0005)
    parser.add_argument("--engine", choices=["process","thread"], default="process")
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--contract-size", type=float, default=0.01)
    parser.add_argument("--vwap-tz", type=int, default=8)
    # nargs='+' — чтобы можно было писать:
    #   --ma all -HMAG -T3
    parser.add_argument("--ma", nargs="+", default=["ALL"],
                        help=("ALL или список: 't3 64-66 ema 100,120 vwma'. "
                              "Исключения с ALL: '-T3', '-HMAG'. "
                              "Если периоды не указаны — полный диапазон 25..500."))
    parser.add_argument("--cc", default=None, help="Фиксация Close Count как 'L,S', напр. --cc 3,4")
    parser.add_argument("--ha", action="store_true", help="Сигналы/индикаторы по Heikin Ashi; исполнение по обычным свечам")
    # ---- ДОБАВЛЕННЫЙ ФЛАГ ФИЛЬТРА ----
    parser.add_argument("--filter", nargs=2, metavar=("WHAT","N"),
                        help="Пример: --filter netprofit 0  (убрать строки с Net Profit% < 0)")
    args = parser.parse_args()

    # разбор --ma (с поддержкой исключений)
    ma_map, ma_types_for_name, is_all = parse_ma_tokens(args.ma)
    args.ma_all = is_all  # флаг для имени файла (_ALL)

    # разворачиваем --csv в список файлов
    csv_paths = expand_csv_argument(args.csv)
    print(f"Found CSV files: {len(csv_paths)}")
    for p in csv_paths:
        print(" -", p)

    # прогон по каждому CSV
    for csv_path in csv_paths:
        run_for_one_csv(csv_path, args, ma_map, ma_types_for_name)

if __name__ == "__main__":
    if os.name == "nt":
        mp.freeze_support()
    main()
