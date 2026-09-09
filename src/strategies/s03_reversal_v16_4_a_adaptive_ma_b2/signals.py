"""Causal Adaptive MA filters and always-combined S03 entry signals."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd
from numba import njit

from core.engine_v2.contracts import Signals
from core.engine_v2.dataprep import build_signal_execution_data
from indicators import ma as shared_ma


@njit(cache=True)
def _recursive_filter(close, length, dsma=False):
    period = length * 0.5 if dsma else length
    a = math.exp(-math.sqrt(2.0) * math.pi / period)
    c2 = 2.0 * a * math.cos(math.sqrt(2.0) * math.pi / period)
    c3 = -a * a
    c1 = 1.0 - c2 - c3
    result = close.copy()
    filt = np.zeros(len(close))
    for i in range(len(close)):
        if not dsma:
            if i >= 3:
                result[i] = c1 * (close[i] + close[i-1]) / 2.0 + c2 * result[i-1] + c3 * result[i-2]
            continue
        if i >= 2:
            previous = close[i-1] - close[i-3] if i >= 3 else 0.0
            filt[i] = c1 * (close[i] - close[i-2] + previous) / 2.0 + c2 * filt[i-1] + c3 * filt[i-2]
        if i >= length + 4:
            square_sum = 0.0
            for j in range(i-length+1, i+1):
                square_sum += filt[j] * filt[j]
            rms = math.sqrt(square_sum / length)
            scaled = filt[i] / rms if rms != 0.0 else 0.0
            alpha = abs(scaled) * 5.0 / length
            result[i] = alpha * close[i] + (1.0-alpha) * result[i-1]
    return result


@njit(cache=True)
def _frama(close, high, low, length):
    n = max(2, length + length % 2)
    half = n // 2
    result = close.copy()
    for i in range(n, len(close)):
        n1 = (np.max(high[i-half+1:i+1]) - np.min(low[i-half+1:i+1])) / half
        n2 = (np.max(high[i-n+1:i-half+1]) - np.min(low[i-n+1:i-half+1])) / half
        n3 = (np.max(high[i-n+1:i+1]) - np.min(low[i-n+1:i+1])) / n
        dimension = (math.log(n1+n2)-math.log(n3))/math.log(2.0) if n1 > 0 and n2 > 0 and n3 > 0 else 1.0
        alpha = max(0.01, min(1.0, math.exp(-4.6*(dimension-1.0))))
        result[i] = alpha * close[i] + (1.0-alpha) * result[i-1]
    return result


def moving_average(df: pd.DataFrame, ma_type: str, length: int) -> np.ndarray:
    close = df["Close"].to_numpy(dtype=float)
    if ma_type == "KAMA":
        return shared_ma.kama(df["Close"].astype(float), length).to_numpy(dtype=float)
    if ma_type == "SuperSmoother":
        return _recursive_filter(close, length)
    if ma_type == "DSMA":
        return _recursive_filter(close, length, True)
    if ma_type == "FRAMA":
        return _frama(close, df["High"].to_numpy(dtype=float), df["Low"].to_numpy(dtype=float), length)
    raise ValueError(f"Unsupported maType3: {ma_type}")


@njit(cache=True)
def close_counters(close, average):
    longs = np.zeros(len(close), dtype=np.int64)
    shorts = np.zeros(len(close), dtype=np.int64)
    long_count = short_count = 0
    for i in range(len(close)):
        long_count = long_count + 1 if close[i] > average[i] else 0
        short_count = short_count + 1 if close[i] < average[i] else 0
        longs[i], shorts[i] = long_count, short_count
    return longs, shorts


@njit(cache=True)
def band_states(close, high, low, average, long_pct, short_pct):
    states = np.zeros(len(close), dtype=np.int8)
    state = 0
    for i in range(len(close)):
        upper = average[i] * (1.0 + long_pct / 100.0)
        lower = average[i] * (1.0 - short_pct / 100.0)
        if high[i] >= upper and low[i] <= lower:
            state = 1 if close[i] > average[i] else -1
        elif high[i] > upper and close[i] > upper:
            state = 1
        elif low[i] < lower and close[i] < lower:
            state = -1
        states[i] = state
    return states


def build_execution_data_batch(df: pd.DataFrame, params_list: list[Mapping[str, Any]]):
    """Reuse each MA, counter pair and band state only within this call/chunk."""
    close, high, low = (df[name].to_numpy(dtype=float) for name in ("Close", "High", "Low"))
    averages, counters, bands = {}, {}, {}
    result = []
    for params in params_list:
        key = (params["maType3"], params["maLength3"])
        if key not in averages:
            averages[key] = moving_average(df, *key)
            counters[key] = close_counters(close, averages[key])
        band_key = (*key, params["tBandLongPct"], params["tBandShortPct"])
        if band_key not in bands:
            bands[band_key] = band_states(close, high, low, averages[key], *band_key[-2:])
        longs, shorts = counters[key]
        state = bands[band_key]
        signals = Signals(
            long_entries=(longs >= params["closeCountLong"]) & (state == 1),
            short_entries=(shorts >= params["closeCountShort"]) & (state == -1),
        )
        result.append(build_signal_execution_data(df, signals=signals))
    return result
