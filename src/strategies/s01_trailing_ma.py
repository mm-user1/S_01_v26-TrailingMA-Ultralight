import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from indicators import VALID_MA_TYPES, atr, get_ma
from strategies.base_strategy import BaseStrategy


class S01TrailingMA(BaseStrategy):
    """S_01 TrailingMA v26 Ultralight strategy implementation."""

    STRATEGY_ID = "s01_trailing_ma"
    STRATEGY_NAME = "S_01 TrailingMA v26 Ultralight"
    VERSION = "26"

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)

        # Caches
        self._ma_trend: Optional[np.ndarray] = None
        self._atr: Optional[np.ndarray] = None
        self._trail_ma_long: Optional[np.ndarray] = None
        self._trail_ma_short: Optional[np.ndarray] = None
        self._lowest: Optional[np.ndarray] = None
        self._highest: Optional[np.ndarray] = None

        # State
        self.counter_close_trend_long = 0
        self.counter_close_trend_short = 0
        self.counter_trade_long = 0
        self.counter_trade_short = 0
        self.trail_price_long = math.nan
        self.trail_price_short = math.nan
        self.trail_activated_long = False
        self.trail_activated_short = False
        self.trade_start_idx: int = 0
        self.time_in_range: Optional[np.ndarray] = None
        self.entry_time_long: Optional[pd.Timestamp] = None
        self.entry_time_short: Optional[pd.Timestamp] = None

    # ════════════════════════════════════════════════════════════════
    # Parameter handling
    # ════════════════════════════════════════════════════════════════
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
            "maType": {
                "type": "categorical",
                "choices": sorted(list(VALID_MA_TYPES)),
                "default": "SMA",
                "description": "Trend MA type",
            },
            "maLength": {
                "type": "int",
                "default": 300,
                "min": 0,
                "max": 5000,
                "step": 1,
                "description": "Trend MA length",
            },
            "closeCountLong": {
                "type": "int",
                "default": 9,
                "min": 0,
                "max": 100,
                "step": 1,
                "description": "Bars above MA to allow long",
            },
            "closeCountShort": {
                "type": "int",
                "default": 5,
                "min": 0,
                "max": 100,
                "step": 1,
                "description": "Bars below MA to allow short",
            },
            "stopLongAtr": {
                "type": "float",
                "default": 2.0,
                "min": 0.0,
                "max": 20.0,
                "step": 0.1,
                "description": "ATR multiplier for long stop",
            },
            "stopLongRr": {
                "type": "float",
                "default": 3.0,
                "min": 0.0,
                "max": 20.0,
                "step": 0.1,
                "description": "Reward/risk for long target",
            },
            "stopLongLp": {
                "type": "int",
                "default": 2,
                "min": 1,
                "max": 500,
                "step": 1,
                "description": "Lookback bars for long stop",
            },
            "stopLongMaxPct": {
                "type": "float",
                "default": 7.0,
                "min": 0.0,
                "max": 100.0,
                "step": 0.1,
                "description": "Max stop distance % for longs",
            },
            "stopLongMaxDays": {
                "type": "int",
                "default": 5,
                "min": 0,
                "max": 365,
                "step": 1,
                "description": "Max days in long trade",
            },
            "stopShortAtr": {
                "type": "float",
                "default": 2.0,
                "min": 0.0,
                "max": 20.0,
                "step": 0.1,
                "description": "ATR multiplier for short stop",
            },
            "stopShortRr": {
                "type": "float",
                "default": 3.0,
                "min": 0.0,
                "max": 20.0,
                "step": 0.1,
                "description": "Reward/risk for short target",
            },
            "stopShortLp": {
                "type": "int",
                "default": 2,
                "min": 1,
                "max": 500,
                "step": 1,
                "description": "Lookback bars for short stop",
            },
            "stopShortMaxPct": {
                "type": "float",
                "default": 10.0,
                "min": 0.0,
                "max": 100.0,
                "step": 0.1,
                "description": "Max stop distance % for shorts",
            },
            "stopShortMaxDays": {
                "type": "int",
                "default": 2,
                "min": 0,
                "max": 365,
                "step": 1,
                "description": "Max days in short trade",
            },
            "trailRrLong": {
                "type": "float",
                "default": 1.0,
                "min": 0.0,
                "max": 10.0,
                "step": 0.1,
                "description": "RR to activate long trail",
            },
            "trailMaLongType": {
                "type": "categorical",
                "choices": sorted(list(VALID_MA_TYPES)),
                "default": "EMA",
                "description": "Trailing MA type for longs",
            },
            "trailMaLongLength": {
                "type": "int",
                "default": 90,
                "min": 0,
                "max": 5000,
                "step": 1,
                "description": "Trailing MA length for longs",
            },
            "trailMaLongOffset": {
                "type": "float",
                "default": -0.5,
                "min": -50.0,
                "max": 50.0,
                "step": 0.1,
                "description": "Trailing MA offset % for longs",
            },
            "trailRrShort": {
                "type": "float",
                "default": 1.0,
                "min": 0.0,
                "max": 10.0,
                "step": 0.1,
                "description": "RR to activate short trail",
            },
            "trailMaShortType": {
                "type": "categorical",
                "choices": sorted(list(VALID_MA_TYPES)),
                "default": "EMA",
                "description": "Trailing MA type for shorts",
            },
            "trailMaShortLength": {
                "type": "int",
                "default": 190,
                "min": 0,
                "max": 5000,
                "step": 1,
                "description": "Trailing MA length for shorts",
            },
            "trailMaShortOffset": {
                "type": "float",
                "default": 2.0,
                "min": -50.0,
                "max": 50.0,
                "step": 0.1,
                "description": "Trailing MA offset % for shorts",
            },
            "riskPerTradePct": {
                "type": "float",
                "default": 2.0,
                "min": 0.0,
                "max": 100.0,
                "step": 0.1,
                "description": "Risk per trade (%)",
            },
            "contractSize": {
                "type": "float",
                "default": 0.01,
                "min": 0.0,
                "max": 1000.0,
                "step": 0.01,
                "description": "Contract size rounding",
            },
            "commissionRate": {
                "type": "float",
                "default": 0.0005,
                "min": 0.0,
                "max": 1.0,
                "step": 0.0001,
                "description": "Commission rate per trade",
            },
            "atrPeriod": {
                "type": "int",
                "default": 14,
                "min": 1,
                "max": 500,
                "step": 1,
                "description": "ATR period",
            },
        }

    def _validate_params(self) -> None:
        required = [
            "useBacktester",
            "dateFilter",
            "maType",
            "maLength",
            "closeCountLong",
            "closeCountShort",
            "stopLongAtr",
            "stopLongRr",
            "stopLongLp",
            "stopShortAtr",
            "stopShortRr",
            "stopShortLp",
            "stopLongMaxPct",
            "stopShortMaxPct",
            "stopLongMaxDays",
            "stopShortMaxDays",
            "trailRrLong",
            "trailMaLongType",
            "trailMaLongLength",
            "trailMaLongOffset",
            "trailRrShort",
            "trailMaShortType",
            "trailMaShortLength",
            "trailMaShortOffset",
            "riskPerTradePct",
            "contractSize",
            "commissionRate",
            "atrPeriod",
        ]
        for key in required:
            if key not in self.params:
                raise ValueError(f"Missing parameter: {key}")

        self.params["maType"] = str(self.params["maType"]).upper()
        self.params["trailMaLongType"] = str(self.params["trailMaLongType"]).upper()
        self.params["trailMaShortType"] = str(self.params["trailMaShortType"]).upper()

        for key in ("maType", "trailMaLongType", "trailMaShortType"):
            if self.params[key] not in VALID_MA_TYPES:
                raise ValueError(f"Unsupported MA type: {self.params[key]}")

        int_fields = [
            "maLength",
            "closeCountLong",
            "closeCountShort",
            "stopLongLp",
            "stopShortLp",
            "stopLongMaxDays",
            "stopShortMaxDays",
            "trailMaLongLength",
            "trailMaShortLength",
            "atrPeriod",
        ]
        for key in int_fields:
            self.params[key] = int(math.floor(float(self.params[key])))

        float_fields = [
            "stopLongAtr",
            "stopLongRr",
            "stopLongMaxPct",
            "stopShortAtr",
            "stopShortRr",
            "stopShortMaxPct",
            "trailRrLong",
            "trailMaLongOffset",
            "trailRrShort",
            "trailMaShortOffset",
            "riskPerTradePct",
            "contractSize",
            "commissionRate",
        ]
        for key in float_fields:
            self.params[key] = float(self.params[key])

        for date_key in ("startDate", "endDate"):
            value = self.params.get(date_key)
            if value in (None, ""):
                self.params[date_key] = None
            else:
                ts = pd.Timestamp(value)
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                else:
                    ts = ts.tz_convert("UTC")
                self.params[date_key] = ts

        self.params["useBacktester"] = bool(self.params.get("useBacktester", True))
        self.params["dateFilter"] = bool(self.params.get("dateFilter", True))

    # ════════════════════════════════════════════════════════════════
    # Data preparation
    # ════════════════════════════════════════════════════════════════
    def _prepare_data(self, df: pd.DataFrame, cached_data: Optional[Dict] = None) -> None:
        self.df = df
        self.close = df["Close"].to_numpy()
        self.high = df["High"].to_numpy()
        self.low = df["Low"].to_numpy()
        self.open = df["Open"].to_numpy()
        self.volume = df["Volume"].to_numpy()
        self.times = df.index

        ma_key = (self.params["maType"], self.params["maLength"])
        trail_long_key = (self.params["trailMaLongType"], self.params["trailMaLongLength"])
        trail_short_key = (
            self.params["trailMaShortType"],
            self.params["trailMaShortLength"],
        )

        if cached_data:
            self._ma_trend = cached_data["ma_cache"][ma_key]
            self._atr = cached_data["atr"][self.params["atrPeriod"]]
            self._trail_ma_long = cached_data["ma_cache"][trail_long_key]
            self._trail_ma_short = cached_data["ma_cache"][trail_short_key]
            self._lowest = cached_data["lowest"][self.params["stopLongLp"]]
            self._highest = cached_data["highest"][self.params["stopShortLp"]]
        else:
            ma_series = get_ma(
                df["Close"],
                self.params["maType"],
                self.params["maLength"],
                df["Volume"],
                df["High"],
                df["Low"],
            )
            self._ma_trend = ma_series.to_numpy()
            self._atr = atr(df["High"], df["Low"], df["Close"], self.params["atrPeriod"]).to_numpy()
            self._lowest = (
                df["Low"].rolling(self.params["stopLongLp"], min_periods=1).min().to_numpy()
            )
            self._highest = (
                df["High"].rolling(self.params["stopShortLp"], min_periods=1).max().to_numpy()
            )
            self._trail_ma_long = get_ma(
                df["Close"],
                self.params["trailMaLongType"],
                self.params["trailMaLongLength"],
                df["Volume"],
                df["High"],
                df["Low"],
            ).to_numpy()
            self._trail_ma_short = get_ma(
                df["Close"],
                self.params["trailMaShortType"],
                self.params["trailMaShortLength"],
                df["Volume"],
                df["High"],
                df["Low"],
            ).to_numpy()

        if self.params["trailMaLongLength"] > 0:
            self._trail_ma_long = self._trail_ma_long * (1 + self.params["trailMaLongOffset"] / 100.0)
        if self.params["trailMaShortLength"] > 0:
            self._trail_ma_short = self._trail_ma_short * (1 + self.params["trailMaShortOffset"] / 100.0)

        if self.params.get("dateFilter"):
            time_mask = np.zeros(len(df), dtype=bool)
            start_idx = int(self.trade_start_idx) if self.trade_start_idx is not None else 0
            time_mask[start_idx:] = True
            self.time_in_range = time_mask
        else:
            self.time_in_range = np.ones(len(df), dtype=bool)

    # ════════════════════════════════════════════════════════════════
    # Strategy logic
    # ════════════════════════════════════════════════════════════════
    def should_long(self, idx: int) -> bool:
        up_trend = self.counter_close_trend_long >= self.params["closeCountLong"]
        not_recent_long = self.counter_trade_long == 0
        valid_range = bool(self.time_in_range[idx]) if self.time_in_range is not None else True
        return (
            up_trend
            and not_recent_long
            and self.params.get("useBacktester", True)
            and valid_range
            and not np.isnan(self._atr[idx])
            and not np.isnan(self._lowest[idx])
        )

    def should_short(self, idx: int) -> bool:
        down_trend = self.counter_close_trend_short >= self.params["closeCountShort"]
        not_recent_short = self.counter_trade_short == 0
        valid_range = bool(self.time_in_range[idx]) if self.time_in_range is not None else True
        return (
            down_trend
            and not_recent_short
            and self.params.get("useBacktester", True)
            and valid_range
            and not np.isnan(self._atr[idx])
            and not np.isnan(self._highest[idx])
        )

    def calculate_entry(self, idx: int, direction: str) -> Tuple[float, float, float]:
        c = self.close[idx]
        atr_value = self._atr[idx]

        if direction == "long":
            lowest_value = self._lowest[idx]
            stop_size = atr_value * self.params["stopLongAtr"]
            stop_price = lowest_value - stop_size
            stop_distance = c - stop_price
            if stop_distance <= 0:
                return math.nan, math.nan, math.nan
            stop_pct = (stop_distance / c) * 100
            if (
                self.params["stopLongMaxPct"] > 0
                and stop_pct > self.params["stopLongMaxPct"]
            ):
                return math.nan, math.nan, math.nan
            target_price = c + stop_distance * self.params["stopLongRr"]
            return c, stop_price, target_price

        highest_value = self._highest[idx]
        stop_size = atr_value * self.params["stopShortAtr"]
        stop_price = highest_value + stop_size
        stop_distance = stop_price - c
        if stop_distance <= 0:
            return math.nan, math.nan, math.nan
        stop_pct = (stop_distance / c) * 100
        if self.params["stopShortMaxPct"] > 0 and stop_pct > self.params["stopShortMaxPct"]:
            return math.nan, math.nan, math.nan
        target_price = c - stop_distance * self.params["stopShortRr"]
        return c, stop_price, target_price

    def calculate_position_size(
        self, idx: int, direction: str, entry_price: float, stop_price: float, equity: float
    ) -> float:
        stop_distance = abs(entry_price - stop_price)
        if stop_distance == 0:
            return 0.0
        risk_cash = equity * (self.params["riskPerTradePct"] / 100)
        qty = risk_cash / stop_distance
        contract_size = self.params["contractSize"]
        if contract_size > 0:
            qty = math.floor(qty / contract_size) * contract_size
        return qty

    def should_exit(
        self, idx: int, position_info: Dict[str, Any]
    ) -> Tuple[bool, Optional[float], str]:
        direction = position_info["direction"]
        entry_price = position_info["entry_price"]
        stop_price = position_info["stop_price"]
        target_price = position_info["target_price"]
        reason = ""

        c = self.close[idx]
        h = self.high[idx]
        l = self.low[idx]
        time = self.times[idx]

        if direction > 0:
            if (
                not self.trail_activated_long
                and not math.isnan(entry_price)
                and not math.isnan(stop_price)
            ):
                activation_price = entry_price + (entry_price - stop_price) * self.params["trailRrLong"]
                if h >= activation_price:
                    self.trail_activated_long = True
                    if math.isnan(self.trail_price_long):
                        self.trail_price_long = stop_price
            trail_value = self._trail_ma_long[idx]
            if not math.isnan(self.trail_price_long) and not np.isnan(trail_value):
                if np.isnan(self.trail_price_long) or trail_value > self.trail_price_long:
                    self.trail_price_long = trail_value
            if self.trail_activated_long:
                if not math.isnan(self.trail_price_long) and l <= self.trail_price_long:
                    exit_price = h if self.trail_price_long > h else self.trail_price_long
                    reason = "trailing"
                    return True, exit_price, reason
            else:
                if l <= stop_price:
                    reason = "stop"
                    return True, stop_price, reason
                if h >= target_price:
                    reason = "target"
                    return True, target_price, reason
            if (
                self.entry_time_long is not None
                and self.params["stopLongMaxDays"] > 0
            ):
                days_in_trade = int(math.floor((time - self.entry_time_long).total_seconds() / 86400))
                if days_in_trade >= self.params["stopLongMaxDays"]:
                    reason = "max_days"
                    return True, c, reason
        else:
            if (
                not self.trail_activated_short
                and not math.isnan(entry_price)
                and not math.isnan(stop_price)
            ):
                activation_price = entry_price - (stop_price - entry_price) * self.params["trailRrShort"]
                if l <= activation_price:
                    self.trail_activated_short = True
                    if math.isnan(self.trail_price_short):
                        self.trail_price_short = stop_price
            trail_value = self._trail_ma_short[idx]
            if not math.isnan(self.trail_price_short) and not np.isnan(trail_value):
                if np.isnan(self.trail_price_short) or trail_value < self.trail_price_short:
                    self.trail_price_short = trail_value
            if self.trail_activated_short:
                if not math.isnan(self.trail_price_short) and h >= self.trail_price_short:
                    exit_price = l if self.trail_price_short < l else self.trail_price_short
                    reason = "trailing"
                    return True, exit_price, reason
            else:
                if h >= stop_price:
                    reason = "stop"
                    return True, stop_price, reason
                if l <= target_price:
                    reason = "target"
                    return True, target_price, reason
            if (
                self.entry_time_short is not None
                and self.params["stopShortMaxDays"] > 0
            ):
                days_in_trade = int(math.floor((time - self.entry_time_short).total_seconds() / 86400))
                if days_in_trade >= self.params["stopShortMaxDays"]:
                    reason = "max_days"
                    return True, c, reason

        return False, None, reason

    # ════════════════════════════════════════════════════════════════
    # Simulation loop
    # ════════════════════════════════════════════════════════════════
    def _run_simulation(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not self.params.get("useBacktester", True):
            raise ValueError("Backtester is disabled in the provided parameters")

        equity = 100.0
        realized_equity = equity
        position = 0
        prev_position = 0
        position_size = 0.0
        entry_idx_position = 0
        entry_price = math.nan
        stop_price = math.nan
        target_price = math.nan
        entry_commission = 0.0

        self.trail_price_long = math.nan
        self.trail_price_short = math.nan
        self.trail_activated_long = False
        self.trail_activated_short = False
        self.entry_time_long = None
        self.entry_time_short = None
        self.counter_close_trend_long = 0
        self.counter_close_trend_short = 0
        self.counter_trade_long = 0
        self.counter_trade_short = 0

        trades: List[Dict[str, Any]] = []
        equity_curve: List[float] = []

        commission_rate = self.params.get("commissionRate", 0.0)

        for idx in range(len(df)):
            time = self.times[idx]
            c = self.close[idx]
            h = self.high[idx]
            l = self.low[idx]
            ma_value = self._ma_trend[idx]
            atr_value = self._atr[idx]
            lowest_value = self._lowest[idx]
            highest_value = self._highest[idx]
            if not np.isnan(ma_value):
                if c > ma_value:
                    self.counter_close_trend_long += 1
                    self.counter_close_trend_short = 0
                elif c < ma_value:
                    self.counter_close_trend_short += 1
                    self.counter_close_trend_long = 0
                else:
                    self.counter_close_trend_long = 0
                    self.counter_close_trend_short = 0

            if position > 0:
                self.counter_trade_long = 1
                self.counter_trade_short = 0
            elif position < 0:
                self.counter_trade_long = 0
                self.counter_trade_short = 1

            exit_price: Optional[float] = None
            if position > 0:
                should_exit, exit_price, reason = self.should_exit(
                    idx,
                    {
                        "direction": position,
                        "entry_price": entry_price,
                        "stop_price": stop_price,
                        "target_price": target_price,
                        "entry_idx": entry_idx_position,
                        "size": position_size,
                    },
                )
                if should_exit and exit_price is not None:
                    gross_pnl = (exit_price - entry_price) * position_size
                    exit_commission = exit_price * position_size * commission_rate
                    realized_equity += gross_pnl - exit_commission
                    trades.append(
                        {
                            "direction": "long",
                            "entry_idx": 0,
                            "exit_idx": idx,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "size": position_size,
                            "net_pnl": gross_pnl - exit_commission - entry_commission,
                            "reason": reason,
                        }
                    )
                    position = 0
                    position_size = 0.0
                    entry_price = math.nan
                    stop_price = math.nan
                    target_price = math.nan
                    entry_idx_position = 0
                    self.trail_price_long = math.nan
                    self.trail_activated_long = False
                    self.entry_time_long = None
                    entry_commission = 0.0

            elif position < 0:
                should_exit, exit_price, reason = self.should_exit(
                    idx,
                    {
                        "direction": position,
                        "entry_price": entry_price,
                        "stop_price": stop_price,
                        "target_price": target_price,
                        "entry_idx": entry_idx_position,
                        "size": position_size,
                    },
                )
                if should_exit and exit_price is not None:
                    gross_pnl = (entry_price - exit_price) * position_size
                    exit_commission = exit_price * position_size * commission_rate
                    realized_equity += gross_pnl - exit_commission
                    trades.append(
                        {
                            "direction": "short",
                            "entry_idx": 0,
                            "exit_idx": idx,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "size": position_size,
                            "net_pnl": gross_pnl - exit_commission - entry_commission,
                            "reason": reason,
                        }
                    )
                    position = 0
                    position_size = 0.0
                    entry_price = math.nan
                    stop_price = math.nan
                    target_price = math.nan
                    entry_idx_position = 0
                    self.trail_price_short = math.nan
                    self.trail_activated_short = False
                    self.entry_time_short = None
                    entry_commission = 0.0

            up_trend = self.counter_close_trend_long >= self.params["closeCountLong"] and self.counter_trade_long == 0
            down_trend = self.counter_close_trend_short >= self.params["closeCountShort"] and self.counter_trade_short == 0

            can_open_long = (
                up_trend
                and position == 0
                and prev_position == 0
                and self.time_in_range[idx]
                and not np.isnan(atr_value)
                and not np.isnan(lowest_value)
            )
            can_open_short = (
                down_trend
                and position == 0
                and prev_position == 0
                and self.time_in_range[idx]
                and not np.isnan(atr_value)
                and not np.isnan(highest_value)
            )

            if can_open_long:
                stop_size = atr_value * self.params["stopLongAtr"]
                long_stop_price = lowest_value - stop_size
                long_stop_distance = c - long_stop_price
                if long_stop_distance > 0:
                    long_stop_pct = (long_stop_distance / c) * 100
                    if (
                        long_stop_pct <= self.params["stopLongMaxPct"]
                        or self.params["stopLongMaxPct"] <= 0
                    ):
                        risk_cash = realized_equity * (self.params["riskPerTradePct"] / 100)
                        qty = risk_cash / long_stop_distance if long_stop_distance != 0 else 0
                        if self.params["contractSize"] > 0:
                            qty = math.floor((qty / self.params["contractSize"])) * self.params["contractSize"]
                        if qty > 0:
                            position = 1
                            position_size = qty
                            entry_price = c
                            stop_price = long_stop_price
                            target_price = c + long_stop_distance * self.params["stopLongRr"]
                            entry_idx_position = idx
                            self.trail_price_long = long_stop_price
                            self.trail_activated_long = False
                            self.entry_time_long = time
                            entry_commission = entry_price * position_size * commission_rate
                            realized_equity -= entry_commission

            if can_open_short and position == 0:
                stop_size = atr_value * self.params["stopShortAtr"]
                short_stop_price = highest_value + stop_size
                short_stop_distance = short_stop_price - c
                if short_stop_distance > 0:
                    short_stop_pct = (short_stop_distance / c) * 100
                    if (
                        short_stop_pct <= self.params["stopShortMaxPct"]
                        or self.params["stopShortMaxPct"] <= 0
                    ):
                        risk_cash = realized_equity * (self.params["riskPerTradePct"] / 100)
                        qty = risk_cash / short_stop_distance if short_stop_distance != 0 else 0
                        if self.params["contractSize"] > 0:
                            qty = math.floor((qty / self.params["contractSize"])) * self.params["contractSize"]
                        if qty > 0:
                            position = -1
                            position_size = qty
                            entry_price = c
                            stop_price = short_stop_price
                            target_price = c - short_stop_distance * self.params["stopShortRr"]
                            entry_idx_position = idx
                            self.trail_price_short = short_stop_price
                            self.trail_activated_short = False
                            self.entry_time_short = time
                            entry_commission = entry_price * position_size * commission_rate
                            realized_equity -= entry_commission

            equity_curve.append(realized_equity)
            prev_position = position

        equity_series = pd.Series(equity_curve, index=self.df.index[: len(equity_curve)])
        from backtest_engine import compute_max_drawdown

        net_profit_pct = ((realized_equity - equity) / equity) * 100
        max_drawdown_pct = compute_max_drawdown(equity_series)
        total_trades = len(trades)

        return {
            "net_profit_pct": net_profit_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "total_trades": total_trades,
            "trades": trades,
            "equity_curve": equity_curve,
        }

    # ════════════════════════════════════════════════════════════════
    # Optimization cache requirements
    # ════════════════════════════════════════════════════════════════
    @classmethod
    def get_cache_requirements(cls, param_combinations: List[Dict]) -> Dict:
        ma_specs = set()
        long_lp_values = set()
        short_lp_values = set()
        atr_periods = set()

        for combo in param_combinations:
            ma_specs.add((combo["maType"], combo["maLength"]))
            ma_specs.add((combo["trailMaLongType"], combo["trailMaLongLength"]))
            ma_specs.add((combo["trailMaShortType"], combo["trailMaShortLength"]))
            long_lp_values.add(combo["stopLongLp"])
            short_lp_values.add(combo["stopShortLp"])
            atr_periods.add(combo.get("atrPeriod", 14))

        return {
            "ma_types_and_lengths": list(ma_specs),
            "long_lp_values": list(long_lp_values),
            "short_lp_values": list(short_lp_values),
            "needs_atr": True,
            "atr_periods": list(atr_periods),
        }

