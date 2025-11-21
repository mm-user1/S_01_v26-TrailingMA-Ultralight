"""
Technical Indicators Library

This module contains implementations of various technical indicators
used across trading strategies.

All functions are pure (stateless) and operate on pandas Series.
"""

import math
from typing import Optional

import numpy as np
import pandas as pd

FACTOR_T3 = 0.7
FAST_KAMA = 2
SLOW_KAMA = 30
DEFAULT_ATR_PERIOD = 14
VALID_MA_TYPES = {
    "SMA",
    "EMA",
    "HMA",
    "WMA",
    "VWMA",
    "VWAP",
    "ALMA",
    "DEMA",
    "KAMA",
    "TMA",
    "T3",
}


def sma(series: pd.Series, length: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(length, min_periods=length).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=length, adjust=False).mean()


def wma(series: pd.Series, length: int) -> pd.Series:
    """Weighted Moving Average."""
    weights = np.arange(1, length + 1, dtype=float)
    return series.rolling(length, min_periods=length).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def hma(series: pd.Series, length: int) -> pd.Series:
    """Hull Moving Average."""
    if length <= 0:
        return pd.Series(np.nan, index=series.index)
    half_length = max(1, length // 2)
    sqrt_length = max(1, int(math.sqrt(length)))
    return wma(2 * wma(series, half_length) - wma(series, length), sqrt_length)


def vwma(series: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    """Volume-Weighted Moving Average."""
    weighted = (series * volume).rolling(length, min_periods=length).sum()
    vol_sum = volume.rolling(length, min_periods=length).sum()
    return weighted / vol_sum


def alma(series: pd.Series, length: int, offset: float = 0.85, sigma: float = 6) -> pd.Series:
    """Arnaud Legoux Moving Average."""
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
    """Double Exponential Moving Average."""
    e1 = ema(series, length)
    e2 = ema(e1, length)
    return 2 * e1 - e2


def kama(series: pd.Series, length: int) -> pd.Series:
    """Kaufman Adaptive Moving Average."""
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
        kama_values[i] = prev + a * (price - prev)
    return pd.Series(kama_values, index=series.index)


def tma(series: pd.Series, length: int) -> pd.Series:
    """Triangular Moving Average."""
    if length <= 0:
        return pd.Series(np.nan, index=series.index)
    first = sma(series, math.ceil(length / 2))
    return sma(first, math.floor(length / 2) + 1)


def gd(series: pd.Series, length: int) -> pd.Series:
    """Generalized DEMA helper for T3 calculation."""
    ema1 = ema(series, length)
    ema2 = ema(ema1, length)
    return ema1 * (1 + FACTOR_T3) - ema2 * FACTOR_T3


def t3(series: pd.Series, length: int) -> pd.Series:
    """Tillson T3 Moving Average."""
    if length <= 0:
        return pd.Series(np.nan, index=series.index)
    return gd(gd(gd(series, length), length), length)


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Volume-Weighted Average Price."""
    typical = (high + low + close) / 3
    tp_vol = typical * volume
    cumulative = tp_vol.cumsum()
    cumulative_vol = volume.cumsum()
    return cumulative / cumulative_vol


def get_ma(
    series: pd.Series,
    ma_type: str,
    length: int,
    volume: Optional[pd.Series] = None,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
) -> pd.Series:
    """Unified moving average interface that routes to specific MA implementations."""
    ma_type = ma_type.upper()
    if ma_type not in VALID_MA_TYPES:
        raise ValueError(f"Unsupported MA type: {ma_type}")
    if ma_type != "VWAP" and length == 0:
        return pd.Series(np.nan, index=series.index)
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
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()
