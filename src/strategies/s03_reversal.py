"""
S_03 Reversal v07 Light Strategy

A reversal trading strategy that is always in the market (long or short).
Uses multi-MA confirmation with close count filters to switch positions.

Key Characteristics:
- Always in market (no flat periods)
- No stop-loss or take-profit
- Position reverses on opposite signal
- Uses full equity for position sizing
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from indicators import VALID_MA_TYPES, get_ma
from strategies.base_strategy import BaseStrategy


class S03Reversal(BaseStrategy):
    """S_03 Reversal strategy implementation."""

    STRATEGY_ID = "s03_reversal"
    STRATEGY_NAME = "S_03 Reversal v07 Light"
    VERSION = "7"

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)

        self.ma_fast_type = str(params["maFastType"]).upper()
        self.ma_fast_length = int(params["maFastLength"])

        self.ma_slow_type = str(params["maSlowType"]).upper()
        self.ma_slow_length = int(params["maSlowLength"])

        self.ma_trend_type = str(params["maTrendType"]).upper()
        self.ma_trend_length = int(params["maTrendLength"])

        self.close_count_long = int(params["closeCountLong"])
        self.close_count_short = int(params["closeCountShort"])

        self.date_filter = bool(params.get("dateFilter", False))
        self.start_date = self._parse_date(params.get("startDate"))
        self.end_date = self._parse_date(params.get("endDate"))

        self.trade_start_idx: int = 0
        self.time_in_range: Optional[np.ndarray] = None

        self.equity_pct = float(params.get("equityPct", 100.0))
        self.contract_size = float(params.get("contractSize", 0.0))
        self.commission_rate = float(params.get("commissionRate", 0.0005))
        self.use_backtester = bool(params.get("useBacktester", True))

        # Normalize commission key for BaseStrategy
        self.params["commission_rate"] = self.commission_rate

        self.df: Optional[pd.DataFrame] = None
        self.close: Optional[np.ndarray] = None
        self.high: Optional[np.ndarray] = None
        self.low: Optional[np.ndarray] = None
        self.open: Optional[np.ndarray] = None
        self.volume: Optional[np.ndarray] = None

        self.ma_fast: Optional[np.ndarray] = None
        self.ma_slow: Optional[np.ndarray] = None
        self.ma_trend: Optional[np.ndarray] = None

        self.counter_close_long = 0
        self.counter_close_short = 0

    @classmethod
    def get_param_definitions(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "useBacktester": {
                "type": "bool",
                "default": True,
                "description": "Enable/disable backtester",
            },
            "dateFilter": {
                "type": "bool",
                "default": True,
                "description": "Enable date filtering",
            },
            "startDate": {
                "type": "str",
                "default": "2025-06-15",
                "description": "Start date (YYYY-MM-DD)",
            },
            "endDate": {
                "type": "str",
                "default": "2025-11-15",
                "description": "End date (YYYY-MM-DD)",
            },
            "maFastType": {
                "type": "categorical",
                "choices": sorted(list(VALID_MA_TYPES)),
                "default": "SMA",
                "description": "Fast MA type for signals",
            },
            "maFastLength": {
                "type": "int",
                "default": 100,
                "min": 1,
                "max": 1000,
                "step": 1,
                "description": "Fast MA length",
            },
            "maSlowType": {
                "type": "categorical",
                "choices": sorted(list(VALID_MA_TYPES)),
                "default": "SMA",
                "description": "Slow MA type for signals",
            },
            "maSlowLength": {
                "type": "int",
                "default": 0,
                "min": 0,
                "max": 2000,
                "step": 1,
                "description": "Slow MA length",
            },
            "maTrendType": {
                "type": "categorical",
                "choices": sorted(list(VALID_MA_TYPES)),
                "default": "SMA",
                "description": "Trend filter MA type",
            },
            "maTrendLength": {
                "type": "int",
                "default": 100,
                "min": 1,
                "max": 5000,
                "step": 1,
                "description": "Trend filter MA length",
            },
            "closeCountLong": {
                "type": "int",
                "default": 4,
                "min": 1,
                "max": 100,
                "step": 1,
                "description": "Bars above fast MA to allow long",
            },
            "closeCountShort": {
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 100,
                "step": 1,
                "description": "Bars below fast MA to allow short",
            },
            "equityPct": {
                "type": "float",
                "default": 100.0,
                "min": 0.0,
                "max": 100.0,
                "step": 0.1,
                "description": "Percent of equity to allocate per trade",
            },
            "contractSize": {
                "type": "float",
                "default": 0.01,
                "min": 0.0,
                "max": 1000000.0,
                "step": 0.0001,
                "description": "Contract size for rounding position",
            },
            "commissionRate": {
                "type": "float",
                "default": 0.0005,
                "min": 0.0,
                "max": 1.0,
                "step": 0.0001,
                "description": "Commission rate per trade",
            },
        }

    def _validate_params(self) -> None:
        required = [
            "maFastType",
            "maFastLength",
            "maSlowType",
            "maSlowLength",
            "maTrendType",
            "maTrendLength",
            "closeCountLong",
            "closeCountShort",
        ]
        for key in required:
            if key not in self.params:
                raise ValueError(f"Missing parameter: {key}")

        for key in ("maFastType", "maSlowType", "maTrendType"):
            value = str(self.params[key]).upper()
            if value not in VALID_MA_TYPES:
                raise ValueError(f"Unsupported MA type: {value}")
            self.params[key] = value

        int_fields = ["maFastLength", "maSlowLength", "maTrendLength", "closeCountLong", "closeCountShort"]
        for key in int_fields:
            self.params[key] = int(float(self.params[key]))
            if self.params[key] < 0:
                raise ValueError(f"{key} must be non-negative")

        for key in ("equityPct", "contractSize", "commissionRate"):
            if key in self.params:
                self.params[key] = float(self.params[key])

        if "dateFilter" in self.params:
            self.params["dateFilter"] = bool(self.params.get("dateFilter", False))

    @staticmethod
    def _parse_date(value: Any) -> Optional[pd.Timestamp]:
        if value in (None, ""):
            return None
        ts = pd.to_datetime(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts

    @classmethod
    def get_cache_requirements(
        cls, param_combinations: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        if not param_combinations:
            param_defs = cls.get_param_definitions()
            ma_configs = {
                (param_defs["maFastType"]["default"], param_defs["maFastLength"]["default"]),
                (param_defs["maTrendType"]["default"], param_defs["maTrendLength"]["default"]),
            }
            slow_length = param_defs["maSlowLength"]["default"]
            if slow_length > 0:
                ma_configs.add((param_defs["maSlowType"]["default"], slow_length))
            return {"ma_types_and_lengths": list(ma_configs), "needs_atr": False}

        ma_configs = set()
        for combo in param_combinations:
            ma_configs.add((combo["maFastType"], combo["maFastLength"]))
            ma_configs.add((combo["maTrendType"], combo["maTrendLength"]))
            if combo.get("maSlowLength", 0) > 0:
                ma_configs.add((combo["maSlowType"], combo["maSlowLength"]))
        return {"ma_types_and_lengths": list(ma_configs), "needs_atr": False}

    def _prepare_data(self, df: pd.DataFrame, cached_data: Optional[Dict] = None) -> None:
        self.df = df
        self.close = df["Close"].to_numpy(dtype=float)
        self.open = df["Open"].to_numpy(dtype=float)
        self.high = df["High"].to_numpy(dtype=float)
        self.low = df["Low"].to_numpy(dtype=float)
        self.volume = df["Volume"].to_numpy(dtype=float)

        if cached_data and "ma_cache" in cached_data:
            ma_cache = cached_data["ma_cache"]
            fast_key = (self.ma_fast_type, self.ma_fast_length)
            trend_key = (self.ma_trend_type, self.ma_trend_length)

            self.ma_fast = ma_cache.get(fast_key)
            self.ma_trend = ma_cache.get(trend_key)
            if self.ma_fast is None or self.ma_trend is None:
                raise ValueError(
                    f"Missing MA cache values for keys: {fast_key}, {trend_key}"
                )
            if self.ma_slow_length > 0:
                slow_key = (self.ma_slow_type, self.ma_slow_length)
                self.ma_slow = ma_cache.get(slow_key)
                if self.ma_slow is None:
                    raise ValueError(f"Missing MA cache values for key: {slow_key}")
            else:
                self.ma_slow = np.full(len(df), np.nan)
        else:
            close_series = df["Close"].astype(float)
            volume_series = df["Volume"]
            high_series = df["High"]
            low_series = df["Low"]

            self.ma_fast = get_ma(
                close_series,
                self.ma_fast_type,
                self.ma_fast_length,
                volume=volume_series,
                high=high_series,
                low=low_series,
            ).to_numpy()
            if self.ma_slow_length > 0:
                self.ma_slow = get_ma(
                    close_series,
                    self.ma_slow_type,
                    self.ma_slow_length,
                    volume=volume_series,
                    high=high_series,
                    low=low_series,
                ).to_numpy()
            else:
                self.ma_slow = np.full(len(df), np.nan)
            self.ma_trend = get_ma(
                close_series,
                self.ma_trend_type,
                self.ma_trend_length,
                volume=volume_series,
                high=high_series,
                low=low_series,
            ).to_numpy()

        self.counter_close_long = 0
        self.counter_close_short = 0

        if self.date_filter:
            time_mask = np.zeros(len(df), dtype=bool)
            start_idx = int(self.trade_start_idx) if self.trade_start_idx is not None else 0
            use_date_bounds = start_idx == 0
            if use_date_bounds and self.start_date is not None:
                start_idx = int(df.index.searchsorted(self.start_date))

            time_mask[start_idx:] = True

            if self.end_date is not None:
                end_idx = int(df.index.searchsorted(self.end_date, side="right"))
                time_mask[end_idx:] = False

            self.time_in_range = time_mask
        else:
            self.time_in_range = np.ones(len(df), dtype=bool)

    def allows_reversal(self) -> bool:
        return True

    def should_long(self, idx: int) -> bool:
        if idx < 1 or not self.use_backtester:
            return False

        if self.time_in_range is not None and not self.time_in_range[idx]:
            self.counter_close_long = 0
            return False

        if self.ma_fast is None or np.isnan(self.ma_fast[idx]):
            self.counter_close_long = 0
            return False

        if self.close[idx] > self.ma_fast[idx]:
            self.counter_close_long += 1
            self.counter_close_short = 0
        else:
            self.counter_close_long = 0

        return self.counter_close_long >= self.close_count_long

    def should_short(self, idx: int) -> bool:
        if idx < 1 or not self.use_backtester:
            return False

        if self.time_in_range is not None and not self.time_in_range[idx]:
            self.counter_close_short = 0
            return False

        if self.ma_fast is None or np.isnan(self.ma_fast[idx]):
            self.counter_close_short = 0
            return False

        if self.close[idx] < self.ma_fast[idx]:
            self.counter_close_short += 1
            self.counter_close_long = 0
        else:
            self.counter_close_short = 0

        return self.counter_close_short >= self.close_count_short

    def calculate_entry(self, idx: int, direction: str) -> Tuple[float, float, float]:
        if idx >= len(self.close):
            return math.nan, math.nan, math.nan

        entry_price = self.close[idx]
        return entry_price, math.nan, math.nan

    def calculate_position_size(
        self, idx: int, direction: str, entry_price: float, stop_price: float, equity: float
    ) -> float:
        if entry_price <= 0:
            return 0.0

        position_value = equity * (self.equity_pct / 100.0)
        size = position_value / entry_price
        if self.contract_size > 0:
            size = math.floor(size / self.contract_size) * self.contract_size
        return size

    def should_exit(
        self, idx: int, position_info: Dict[str, Any]
    ) -> Tuple[bool, Optional[float], str]:
        if self.time_in_range is not None and not self.time_in_range[idx]:
            return True, self.close[idx], "date_filter"
        return False, None, ""

