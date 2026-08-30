"""Signal and data-preparation helpers for S06 v06-4-A2 B2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import pandas as pd

from core.engine_v2.contracts import Signals
from core.engine_v2.dataprep import build_execution_data
from core.engine_v2.kernel import ExecutionData
from strategies.s06_r_trend_v02_b2.signals import (
    ENTRY_MODES,
    S06B2Params,
    build_signal_and_initial_stop_arrays,
    pine_atr,
)


TRAIL_MODES = {
    "Off (Bracket)",
    "R Trail",
    "Chandelier Exit",
    "Fixed-AF SAR",
}
CONTRACT_SIZES = {
    0.0001,
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
    100000.0,
    1000000.0,
}


def normalize_parameter_aliases(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Collapse accepted Pine smoothing aliases into canonical Merlin names."""

    result = dict(payload or {})
    for canonical, alias in (
        ("fastSmooth", "fastSmoothing"),
        ("slowSmooth", "slowSmoothing"),
    ):
        if canonical not in result and alias in result:
            result[canonical] = result[alias]
        result.pop(alias, None)
    return result


def _finite_in_range(name: str, value: float, minimum: float, maximum: float) -> None:
    if not math.isfinite(value) or value < minimum or value > maximum:
        raise ValueError(f"{name} must be finite and between {minimum:g} and {maximum:g}.")


def _integer_value(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be an integer.")
    return int(numeric)


def _float_value(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc


@dataclass(frozen=True)
class S06V064A2Params:
    dateFilter: bool = True
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None
    entryMode: str = "Reversal @ Triangle"
    enableLong: bool = True
    enableShort: bool = True
    fastLength: int = 21
    fastSmooth: int = 7
    slowLength: int = 112
    slowSmooth: int = 3
    thresholdOS: int = 20
    thresholdOB: int = 20
    stopX: float = 2.0
    stopRR: float = 3.0
    stopLP: int = 2
    stopMaxPct: float = 4.0
    stopMaxDays: int = 4
    riskPerTrade: float = 2.0
    contractSize: float = 0.01
    trailMode: str = "R Trail"
    trailRR: float = 1.0
    trailDistanceR: float = 1.0
    chandelierATRLength: int = 14
    chandelierATRMult: float = 3.0
    sarSpeed: float = 0.01
    initialCapital: float = 100.0
    commissionPct: float = 0.05
    warmupBars: int = 1000

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "S06V064A2Params":
        d = normalize_parameter_aliases(payload)
        trail_mode = str(d.get("trailMode", cls.trailMode))

        def active(name: str, default: Any, modes: set[str]) -> Any:
            return d.get(name, default) if trail_mode in modes else default

        params = cls(
            dateFilter=S06B2Params._coerce_bool(d.get("dateFilter"), cls.dateFilter),
            start=S06B2Params._parse_timestamp(d.get("start")),
            end=S06B2Params._parse_timestamp(d.get("end")),
            entryMode=str(d.get("entryMode", cls.entryMode)),
            enableLong=S06B2Params._coerce_bool(d.get("enableLong"), cls.enableLong),
            enableShort=S06B2Params._coerce_bool(d.get("enableShort"), cls.enableShort),
            fastLength=_integer_value("fastLength", d.get("fastLength", cls.fastLength)),
            fastSmooth=_integer_value("fastSmooth", d.get("fastSmooth", cls.fastSmooth)),
            slowLength=_integer_value("slowLength", d.get("slowLength", cls.slowLength)),
            slowSmooth=_integer_value("slowSmooth", d.get("slowSmooth", cls.slowSmooth)),
            thresholdOS=_integer_value("thresholdOS", d.get("thresholdOS", cls.thresholdOS)),
            thresholdOB=_integer_value("thresholdOB", d.get("thresholdOB", cls.thresholdOB)),
            stopX=float(d.get("stopX", cls.stopX)),
            stopRR=_float_value("stopRR", active("stopRR", cls.stopRR, {"Off (Bracket)"})),
            stopLP=_integer_value("stopLP", d.get("stopLP", cls.stopLP)),
            stopMaxPct=float(d.get("stopMaxPct", cls.stopMaxPct)),
            stopMaxDays=_integer_value("stopMaxDays", d.get("stopMaxDays", cls.stopMaxDays)),
            riskPerTrade=float(d.get("riskPerTrade", cls.riskPerTrade)),
            contractSize=float(d.get("contractSize", cls.contractSize)),
            trailMode=trail_mode,
            trailRR=_float_value(
                "trailRR", active("trailRR", cls.trailRR, TRAIL_MODES - {"Off (Bracket)"})
            ),
            trailDistanceR=_float_value(
                "trailDistanceR", active("trailDistanceR", cls.trailDistanceR, {"R Trail"})
            ),
            chandelierATRLength=_integer_value(
                "chandelierATRLength",
                active("chandelierATRLength", cls.chandelierATRLength, {"Chandelier Exit"}),
            ),
            chandelierATRMult=_float_value(
                "chandelierATRMult",
                active("chandelierATRMult", cls.chandelierATRMult, {"Chandelier Exit"}),
            ),
            sarSpeed=_float_value(
                "sarSpeed", active("sarSpeed", cls.sarSpeed, {"Fixed-AF SAR"})
            ),
            initialCapital=float(d.get("initialCapital", cls.initialCapital)),
            commissionPct=float(d.get("commissionPct", cls.commissionPct)),
            warmupBars=int(d.get("warmupBars", cls.warmupBars)),
        )
        params.validate()
        return params

    def validate(self) -> None:
        if self.entryMode not in ENTRY_MODES:
            raise ValueError(f"Invalid entryMode '{self.entryMode}'.")
        if self.trailMode not in TRAIL_MODES:
            raise ValueError(f"Invalid trailMode '{self.trailMode}'.")
        if self.fastLength < 2 or self.slowLength < 2:
            raise ValueError("fastLength and slowLength must be at least 2.")
        if self.fastSmooth < 1 or self.slowSmooth < 1:
            raise ValueError("fastSmooth and slowSmooth must be at least 1.")
        if not 1 <= self.thresholdOS <= 50 or not 1 <= self.thresholdOB <= 50:
            raise ValueError("thresholdOS and thresholdOB must be between 1 and 50.")
        _finite_in_range("stopX", self.stopX, 1.0, 3.0)
        if self.stopLP <= 0:
            raise ValueError("stopLP must be a positive integer.")
        _finite_in_range("stopMaxPct", self.stopMaxPct, 2.0, 8.0)
        if self.stopMaxDays not in {2, 4, 6}:
            raise ValueError("stopMaxDays must be 2, 4, or 6.")
        for name in ("riskPerTrade", "initialCapital"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite value greater than zero.")
        if self.contractSize not in CONTRACT_SIZES:
            raise ValueError("contractSize must be one of the supported Pine contract sizes.")
        if not math.isfinite(self.commissionPct) or self.commissionPct < 0.0:
            raise ValueError("commissionPct must be finite and non-negative.")
        if self.warmupBars < 0:
            raise ValueError("warmupBars must be non-negative.")
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("start must not be after end.")

        if self.trailMode == "Off (Bracket)":
            _finite_in_range("stopRR", self.stopRR, 1.5, 3.0)
        else:
            _finite_in_range("trailRR", self.trailRR, 1.0, 3.0)
        if self.trailMode == "R Trail":
            _finite_in_range("trailDistanceR", self.trailDistanceR, 0.5, 3.0)
        elif self.trailMode == "Chandelier Exit":
            if self.chandelierATRLength not in {14, 28}:
                raise ValueError("chandelierATRLength must be 14 or 28.")
            _finite_in_range("chandelierATRMult", self.chandelierATRMult, 2.0, 6.0)
        elif self.trailMode == "Fixed-AF SAR":
            _finite_in_range("sarSpeed", self.sarSpeed, 0.005, 0.02)

    def shared_signal_params(self) -> S06B2Params:
        """Project only the fields consumed by the shared S06 preparation."""

        return S06B2Params(
            entryMode=self.entryMode,
            enableLong=self.enableLong,
            enableShort=self.enableShort,
            fastLength=self.fastLength,
            fastSmooth=self.fastSmooth,
            slowLength=self.slowLength,
            slowSmooth=self.slowSmooth,
            thresholdOS=self.thresholdOS,
            thresholdOB=self.thresholdOB,
            stopLP=self.stopLP,
        )


def build_s06_v064a2_execution_data(
    df: pd.DataFrame,
    params: S06V064A2Params,
) -> ExecutionData:
    arrays = build_signal_and_initial_stop_arrays(df, params.shared_signal_params())
    signals = Signals(
        long_entries=arrays["long_signal"],
        short_entries=arrays["short_signal"],
    )
    chandelier_atr = (
        pine_atr(df, params.chandelierATRLength)
        if params.trailMode == "Chandelier Exit"
        else None
    )
    return build_execution_data(
        df,
        signals=signals,
        atr=arrays["atr"],
        rolling_low=arrays["rolling_low"],
        rolling_high=arrays["rolling_high"],
        chandelier_atr=chandelier_atr,
    )


__all__ = [
    "CONTRACT_SIZES",
    "S06V064A2Params",
    "TRAIL_MODES",
    "build_s06_v064a2_execution_data",
    "normalize_parameter_aliases",
]
