import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from backtesting import _stats

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


CSVSource = Union[str, Path, IO[str], IO[bytes]]


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
class StrategyResult:
    net_profit_pct: float
    max_drawdown_pct: float
    total_trades: int
    trades: List[TradeRecord]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "net_profit_pct": self.net_profit_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "total_trades": self.total_trades,
        }


@dataclass
class StrategyParams:
    use_backtester: bool
    use_date_filter: bool
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
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
    risk_per_trade_pct: float
    contract_size: float
    commission_rate: float = 0.0005
    atr_period: int = DEFAULT_ATR_PERIOD

    @staticmethod
    def _parse_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        value_str = str(value).strip().lower()
        if value_str in {"true", "1", "yes", "y", "on"}:
            return True
        if value_str in {"false", "0", "no", "n", "off"}:
            return False
        return default

    @staticmethod
    def _parse_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_int(value: Any, default: int) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_timestamp(value: Any) -> Optional[pd.Timestamp]:
        if value in (None, ""):
            return None
        try:
            ts = pd.Timestamp(value)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            return ts
        except (ValueError, TypeError):
            return None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "StrategyParams":
        payload = payload or {}

        ma_type = str(payload.get("maType", "EMA")).upper()
        if ma_type not in VALID_MA_TYPES:
            raise ValueError(f"Unsupported MA type: {ma_type}")
        trail_ma_long_type = str(payload.get("trailLongType", "SMA")).upper()
        if trail_ma_long_type not in VALID_MA_TYPES:
            raise ValueError(f"Unsupported trail MA long type: {trail_ma_long_type}")
        trail_ma_short_type = str(payload.get("trailShortType", "SMA")).upper()
        if trail_ma_short_type not in VALID_MA_TYPES:
            raise ValueError(f"Unsupported trail MA short type: {trail_ma_short_type}")

        return cls(
            use_backtester=cls._parse_bool(payload.get("backtester", True), True),
            use_date_filter=cls._parse_bool(payload.get("dateFilter", True), True),
            start=cls._parse_timestamp(payload.get("start")),
            end=cls._parse_timestamp(payload.get("end")),
            ma_type=ma_type,
            ma_length=max(cls._parse_int(payload.get("maLength", 45), 0), 0),
            close_count_long=max(cls._parse_int(payload.get("closeCountLong", 7), 0), 0),
            close_count_short=max(cls._parse_int(payload.get("closeCountShort", 5), 0), 0),
            stop_long_atr=cls._parse_float(payload.get("stopLongX", 2.0), 2.0),
            stop_long_rr=cls._parse_float(payload.get("stopLongRR", 3.0), 3.0),
            stop_long_lp=max(cls._parse_int(payload.get("stopLongLP", 2), 0), 1),
            stop_short_atr=cls._parse_float(payload.get("stopShortX", 2.0), 2.0),
            stop_short_rr=cls._parse_float(payload.get("stopShortRR", 3.0), 3.0),
            stop_short_lp=max(cls._parse_int(payload.get("stopShortLP", 2), 0), 1),
            stop_long_max_pct=max(cls._parse_float(payload.get("stopLongMaxPct", 3.0), 3.0), 0.0),
            stop_short_max_pct=max(cls._parse_float(payload.get("stopShortMaxPct", 3.0), 3.0), 0.0),
            stop_long_max_days=max(cls._parse_int(payload.get("stopLongMaxDays", 2), 0), 0),
            stop_short_max_days=max(cls._parse_int(payload.get("stopShortMaxDays", 4), 0), 0),
            trail_rr_long=max(cls._parse_float(payload.get("trailRRLong", 1.0), 1.0), 0.0),
            trail_rr_short=max(cls._parse_float(payload.get("trailRRShort", 1.0), 1.0), 0.0),
            trail_ma_long_type=trail_ma_long_type,
            trail_ma_long_length=max(cls._parse_int(payload.get("trailLongLength", 160), 0), 0),
            trail_ma_long_offset=cls._parse_float(payload.get("trailLongOffset", -1.0), -1.0),
            trail_ma_short_type=trail_ma_short_type,
            trail_ma_short_length=max(cls._parse_int(payload.get("trailShortLength", 160), 0), 0),
            trail_ma_short_offset=cls._parse_float(payload.get("trailShortOffset", 1.0), 1.0),
            risk_per_trade_pct=max(cls._parse_float(payload.get("riskPerTrade", 2.0), 2.0), 0.0),
            contract_size=max(cls._parse_float(payload.get("contractSize", 0.01), 0.01), 0.0),
            commission_rate=max(cls._parse_float(payload.get("commissionRate", 0.0005), 0.0005), 0.0),
            atr_period=max(cls._parse_int(payload.get("atrPeriod", DEFAULT_ATR_PERIOD), DEFAULT_ATR_PERIOD), 1),
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["start"] = self.start.isoformat() if self.start is not None else None
        data["end"] = self.end.isoformat() if self.end is not None else None
        return data


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


def get_ma(
    series: pd.Series,
    ma_type: str,
    length: int,
    volume: Optional[pd.Series] = None,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
) -> pd.Series:
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


def load_data(csv_source: CSVSource) -> pd.DataFrame:
    df = pd.read_csv(csv_source)
    if "time" not in df.columns:
        raise ValueError("CSV must include a 'time' column with timestamps in seconds")
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True, errors="coerce")
    if df["time"].isna().all():
        raise ValueError("Failed to parse timestamps from 'time' column")
    df = df.set_index("time").sort_index()
    expected_cols = {"open", "high", "low", "close", "Volume", "volume"}
    available_cols = set(df.columns)
    price_cols = {"open", "high", "low", "close"}
    if not price_cols.issubset({col.lower() for col in available_cols}):
        raise ValueError("CSV must include open, high, low, close columns")
    volume_col = None
    for col in ("Volume", "volume", "VOL", "vol"):
        if col in df.columns:
            volume_col = col
            break
    if volume_col is None:
        raise ValueError("CSV must include a volume column")
    renamed = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        volume_col: "Volume",
    }
    normalized_cols = {col: renamed.get(col.lower(), col) for col in df.columns}
    df = df.rename(columns=normalized_cols)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    equity_curve = equity_curve.ffill()
    drawdown = 1 - equity_curve / equity_curve.cummax()
    _, peak_dd = _stats.compute_drawdown_duration_peaks(drawdown)
    if peak_dd.isna().all():
        return 0.0
    return peak_dd.max() * 100


def run_strategy(df: pd.DataFrame, params: StrategyParams) -> StrategyResult:
    if params.use_backtester is False:
        raise ValueError("Backtester is disabled in the provided parameters")

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    ma_series = get_ma(close, params.ma_type, params.ma_length, volume, high, low)
    atr_series = atr(high, low, close, params.atr_period)
    lowest_long = low.rolling(params.stop_long_lp, min_periods=1).min()
    highest_short = high.rolling(params.stop_short_lp, min_periods=1).max()

    trail_ma_long = get_ma(close, params.trail_ma_long_type, params.trail_ma_long_length, volume, high, low)
    trail_ma_short = get_ma(close, params.trail_ma_short_type, params.trail_ma_short_length, volume, high, low)
    if params.trail_ma_long_length > 0:
        trail_ma_long = trail_ma_long * (1 + params.trail_ma_long_offset / 100.0)
    if params.trail_ma_short_length > 0:
        trail_ma_short = trail_ma_short * (1 + params.trail_ma_short_offset / 100.0)

    times = df.index
    if params.use_date_filter:
        mask = np.ones(len(times), dtype=bool)
        if params.start is not None:
            mask &= times >= params.start
        if params.end is not None:
            mask &= times <= params.end
        time_in_range = mask
    else:
        time_in_range = np.ones(len(times), dtype=bool)

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
        ma_value = ma_series.iat[i]
        atr_value = atr_series.iat[i]
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

        exit_price: Optional[float] = None
        if position > 0:
            if (
                not trail_activated_long
                and not math.isnan(entry_price)
                and not math.isnan(stop_price)
            ):
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
            if exit_price is None and entry_time_long is not None and params.stop_long_max_days > 0:
                days_in_trade = int(math.floor((time - entry_time_long).total_seconds() / 86400))
                if days_in_trade >= params.stop_long_max_days:
                    exit_price = c
            if exit_price is not None:
                gross_pnl = (exit_price - entry_price) * position_size
                exit_commission = exit_price * position_size * params.commission_rate
                realized_equity += gross_pnl - exit_commission
                trades.append(
                    TradeRecord(
                        direction="long",
                        entry_time=entry_time_long,
                        exit_time=time,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        size=position_size,
                        net_pnl=gross_pnl - exit_commission - entry_commission,
                    )
                )
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
            if (
                not trail_activated_short
                and not math.isnan(entry_price)
                and not math.isnan(stop_price)
            ):
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
            if exit_price is None and entry_time_short is not None and params.stop_short_max_days > 0:
                days_in_trade = int(math.floor((time - entry_time_short).total_seconds() / 86400))
                if days_in_trade >= params.stop_short_max_days:
                    exit_price = c
            if exit_price is not None:
                gross_pnl = (entry_price - exit_price) * position_size
                exit_commission = exit_price * position_size * params.commission_rate
                realized_equity += gross_pnl - exit_commission
                trades.append(
                    TradeRecord(
                        direction="short",
                        entry_time=entry_time_short,
                        exit_time=time,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        size=position_size,
                        net_pnl=gross_pnl - exit_commission - entry_commission,
                    )
                )
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
            stop_size = atr_value * params.stop_long_atr
            long_stop_price = lowest_value - stop_size
            long_stop_distance = c - long_stop_price
            if long_stop_distance > 0:
                long_stop_pct = (long_stop_distance / c) * 100
                if long_stop_pct <= params.stop_long_max_pct or params.stop_long_max_pct <= 0:
                    risk_cash = realized_equity * (params.risk_per_trade_pct / 100)
                    qty = risk_cash / long_stop_distance if long_stop_distance != 0 else 0
                    if params.contract_size > 0:
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
                        entry_commission = entry_price * position_size * params.commission_rate
                        realized_equity -= entry_commission

        if can_open_short and position == 0:
            stop_size = atr_value * params.stop_short_atr
            short_stop_price = highest_value + stop_size
            short_stop_distance = short_stop_price - c
            if short_stop_distance > 0:
                short_stop_pct = (short_stop_distance / c) * 100
                if short_stop_pct <= params.stop_short_max_pct or params.stop_short_max_pct <= 0:
                    risk_cash = realized_equity * (params.risk_per_trade_pct / 100)
                    qty = risk_cash / short_stop_distance if short_stop_distance != 0 else 0
                    if params.contract_size > 0:
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
                        entry_commission = entry_price * position_size * params.commission_rate
                        realized_equity -= entry_commission

        mark_to_market = realized_equity
        if position > 0 and not math.isnan(entry_price):
            mark_to_market += (c - entry_price) * position_size
        elif position < 0 and not math.isnan(entry_price):
            mark_to_market += (entry_price - c) * position_size
        realized_curve.append(realized_equity)
        prev_position = position

    equity_series = pd.Series(realized_curve, index=df.index[: len(realized_curve)])
    net_profit_pct = ((realized_equity - equity) / equity) * 100
    max_drawdown_pct = compute_max_drawdown(equity_series)
    total_trades = len(trades)

    return StrategyResult(
        net_profit_pct=net_profit_pct,
        max_drawdown_pct=max_drawdown_pct,
        total_trades=total_trades,
        trades=trades,
    )
