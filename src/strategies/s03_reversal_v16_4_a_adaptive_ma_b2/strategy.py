"""Thin adapter to the certified generic signal-reversal engine."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from core.engine_v2.profile import parse_execution_profile
from core.engine_v2.runner import run_v2_strategy
from core.engine_v2.runtime_contract import normalize_v2_runtime_field_value
from strategies.base import BaseStrategy
from .signals import build_execution_data_batch

SIGNAL_CACHE_PARAM_NAMES = (
    "maType3", "maLength3", "closeCountLong", "closeCountShort",
    "tBandLongPct", "tBandShortPct",
)
DATAPREP_CACHE_PARAM_NAMES = SIGNAL_CACHE_PARAM_NAMES


@lru_cache(maxsize=1)
def _config():
    return json.loads(Path(__file__).with_name("config.json").read_text(encoding="utf-8"))


def load_config():
    return deepcopy(_config())


@lru_cache(maxsize=1)
def load_profile():
    return parse_execution_profile(_config())


def normalized_params(params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    supplied = dict(params or {})
    for alias, name in (("useDateFilter", "dateFilter"), ("startDate", "start"), ("endDate", "end")):
        if alias in supplied and name not in supplied:
            supplied[name] = supplied[alias]
    result = {}
    for name, spec in _config()["parameters"].items():
        value = supplied.get(name, spec["default"])
        kind = spec["type"]
        if spec["role"] == "runtime":
            value = normalize_v2_runtime_field_value(name, value, user_boundary=False)
        elif kind == "bool":
            value = normalize_v2_runtime_field_value("dateFilter", value, path=name)
        elif kind in {"int", "float"}:
            if isinstance(value, bool):
                raise ValueError(f"{name} must be {kind}.")
            value = float(value)
            if not math.isfinite(value) or (kind == "int" and not value.is_integer()):
                raise ValueError(f"{name} must be a finite {kind}.")
            if not spec["min"] <= value <= spec["max"]:
                raise ValueError(f"{name} must be between {spec['min']} and {spec['max']}.")
            value = int(value) if kind == "int" else value
        elif value not in spec["options"]:
            raise ValueError(f"Unsupported {name}: {value}")
        result[name] = value
    if result["start"] and result["end"] and pd.Timestamp(result["start"]) > pd.Timestamp(result["end"]):
        raise ValueError("start must not be after end.")
    return result


def _truncate_at_end(df, params):
    if params["dateFilter"] and params["end"] is not None:
        return df.loc[df.index <= pd.Timestamp(params["end"])]
    return df


def build_v2_execution_data(df, params):
    parsed = normalized_params(params)
    return build_execution_data_batch(_truncate_at_end(df, parsed), [parsed])[0]


def build_v2_execution_data_batch(df, params_list):
    parsed = [normalized_params(params) for params in params_list]
    if not parsed:
        return []
    keys = {(p["dateFilter"], p["end"] if p["dateFilter"] else None) for p in parsed}
    if len(keys) == 1:
        return build_execution_data_batch(_truncate_at_end(df, parsed[0]), parsed)
    return [build_execution_data_batch(_truncate_at_end(df, p), [p])[0] for p in parsed]


class S03ReversalV164AAdaptiveMAB2(BaseStrategy):
    STRATEGY_ID = "s03_reversal_v16_4_a_adaptive_ma_b2"
    STRATEGY_NAME = "S03 Reversal v16-4-A Adaptive MA B2"
    STRATEGY_VERSION = "v16-4-a-b2"

    @staticmethod
    def run(df, params, trade_start_idx=0):
        parsed = normalized_params(params)
        return run_v2_strategy(
            data=build_v2_execution_data(df, parsed), profile=load_profile(),
            params=parsed, trade_start_idx=trade_start_idx,
        ).strategy_result
