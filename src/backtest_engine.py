import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from backtesting import _stats
from indicators import DEFAULT_ATR_PERIOD, VALID_MA_TYPES
from strategy_registry import StrategyRegistry


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


def prepare_dataset_with_warmup(
    df: pd.DataFrame,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    params: StrategyParams
) -> tuple[pd.DataFrame, int]:
    """
    Trim dataset with warmup period for MA calculations.

    Args:
        df: Full OHLCV DataFrame with datetime index
        start: Start date for trading (None = use all data)
        end: End date for trading (None = use all data)
        params: Strategy parameters to calculate required warmup

    Returns:
        Tuple of (trimmed_df, trade_start_idx)
        - trimmed_df: DataFrame with warmup + trading period
        - trade_start_idx: Index where trading should begin (warmup ends)
    """
    # Calculate required warmup based on largest MA length
    max_ma_length = max(
        params.ma_length,
        params.trail_ma_long_length,
        params.trail_ma_short_length
    )

    # Dynamic warmup: at least 500 bars or 1.5x the longest MA
    required_warmup = max(500, int(max_ma_length * 1.5))

    # If no date filtering, use entire dataset
    if start is None and end is None:
        return df.copy(), 0

    # Find indices for start and end dates
    times = df.index

    # Determine start index
    if start is not None:
        # Find first index >= start
        start_mask = times >= start
        if not start_mask.any():
            # Start date is after all data
            print(f"Warning: Start date {start} is after all available data")
            return df.iloc[0:0].copy(), 0  # Return empty df
        start_idx = int(start_mask.argmax())
    else:
        start_idx = 0

    # Determine end index
    if end is not None:
        # Find last index <= end
        end_mask = times <= end
        if not end_mask.any():
            # End date is before all data
            print(f"Warning: End date {end} is before all available data")
            return df.iloc[0:0].copy(), 0  # Return empty df
        # Get the last True value
        end_idx = len(end_mask) - 1 - int(end_mask[::-1].argmax())
        end_idx += 1  # Include the end bar
    else:
        end_idx = len(df)

    # Calculate warmup start (go back from start_idx)
    warmup_start_idx = max(0, start_idx - required_warmup)

    # Check if we have enough data
    actual_warmup = start_idx - warmup_start_idx
    if actual_warmup < required_warmup:
        print(f"Warning: Insufficient warmup data. Need {required_warmup} bars, "
              f"only have {actual_warmup} bars available")

    # Trim the dataframe
    trimmed_df = df.iloc[warmup_start_idx:end_idx].copy()

    # Trade start index is where actual trading begins (after warmup)
    trade_start_idx = start_idx - warmup_start_idx

    return trimmed_df, trade_start_idx


def run_strategy_v2(
    df: pd.DataFrame,
    strategy,
    trade_start_idx: int = 0,
    cached_data: Optional[Dict[str, Any]] = None,
) -> StrategyResult:
    """Universal backtest entrypoint for BaseStrategy implementations."""

    if hasattr(strategy, "trade_start_idx"):
        strategy.trade_start_idx = trade_start_idx

    result = strategy.simulate(df, cached_data=cached_data)

    trades: List[TradeRecord] = []
    for trade in result.get("trades", []):
        entry_idx = trade.get("entry_idx")
        exit_idx = trade.get("exit_idx")
        entry_time = df.index[entry_idx] if entry_idx is not None else df.index[0]
        exit_time = df.index[exit_idx] if exit_idx is not None else df.index[-1]

        trades.append(
            TradeRecord(
                direction=trade.get("direction", ""),
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=trade.get("entry_price", math.nan),
                exit_price=trade.get("exit_price", math.nan),
                size=trade.get("size", 0.0),
                net_pnl=trade.get("net_pnl", 0.0),
            )
        )

    return StrategyResult(
        net_profit_pct=result.get("net_profit_pct", 0.0),
        max_drawdown_pct=result.get("max_drawdown_pct", 0.0),
        total_trades=result.get("total_trades", 0),
        trades=trades,
    )


def run_strategy(df: pd.DataFrame, params: StrategyParams, trade_start_idx: int = 0) -> StrategyResult:
    """Backward-compatible wrapper that routes to the strategy module."""

    param_dict = {
        "useBacktester": params.use_backtester,
        "dateFilter": params.use_date_filter,
        "startDate": params.start,
        "endDate": params.end,
        "maType": params.ma_type,
        "maLength": params.ma_length,
        "closeCountLong": params.close_count_long,
        "closeCountShort": params.close_count_short,
        "stopLongAtr": params.stop_long_atr,
        "stopLongRr": params.stop_long_rr,
        "stopLongLp": params.stop_long_lp,
        "stopShortAtr": params.stop_short_atr,
        "stopShortRr": params.stop_short_rr,
        "stopShortLp": params.stop_short_lp,
        "stopLongMaxPct": params.stop_long_max_pct,
        "stopShortMaxPct": params.stop_short_max_pct,
        "stopLongMaxDays": params.stop_long_max_days,
        "stopShortMaxDays": params.stop_short_max_days,
        "trailRrLong": params.trail_rr_long,
        "trailMaLongType": params.trail_ma_long_type,
        "trailMaLongLength": params.trail_ma_long_length,
        "trailMaLongOffset": params.trail_ma_long_offset,
        "trailRrShort": params.trail_rr_short,
        "trailMaShortType": params.trail_ma_short_type,
        "trailMaShortLength": params.trail_ma_short_length,
        "trailMaShortOffset": params.trail_ma_short_offset,
        "riskPerTradePct": params.risk_per_trade_pct,
        "contractSize": params.contract_size,
        "commissionRate": params.commission_rate,
        "atrPeriod": params.atr_period,
    }

    strategy = StrategyRegistry.get_strategy_instance("s01_trailing_ma", param_dict)
    return run_strategy_v2(df, strategy, trade_start_idx=trade_start_idx)
