from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.engine_v2.compiled_kernel import (
    OUTPUT_COLUMN_COUNT,
    OUTPUT_FINAL_BALANCE,
    OUTPUT_TOTAL_TRADES,
    build_stacked_execution_data,
    evaluate_compiled_batch,
    evaluate_compiled_stacked_batch,
)
from core.engine_v2.contracts import Signals
from core.engine_v2.kernel import ExecutionData, run_reference_kernel
from core.engine_v2.profile import ProfileValidationError, parse_execution_profile
from core.engine_v2.runner import build_kernel_config
from core.grid_v2 import (
    GridV2Settings,
    GridV2StrategyHooks,
    build_grid_v2_plan,
    estimate_grid_v2_cache,
)


def _parameter(type_, default, *, values=None):
    optimize = {"enabled": False}
    if values is not None:
        optimize = {"enabled": True, "values": values}
    return {"type": type_, "default": default, "role": "execution", "optimize": optimize}


def _profile_config():
    return {
        "id": "stateful_trail_fixture",
        "engine": "v2",
        "execution": {
            "entryOrder": "market_next_open",
            "stop": "atr_swing",
            "sizing": "risk_per_trade",
            "maxDays": False,
            "margin": "off",
            "boundary": "strict_close",
            "priceRounding": "none",
            "variantSelector": {
                "param": "trailKind",
                "mapping": {
                    "none": "bracket",
                    "ma": "ma",
                    "r_distance": "r_distance",
                    "chandelier": "chandelier",
                    "fixed_af_sar": "fixed_af_sar",
                },
            },
            "variants": {
                "bracket": {"target": "rr", "trail": "none"},
                "ma": {"target": "none", "trail": "ma", "trailActivation": "rr"},
                "r_distance": {
                    "target": "none",
                    "trail": "r_distance",
                    "trailActivation": "rr",
                },
                "chandelier": {
                    "target": "none",
                    "trail": "chandelier",
                    "trailActivation": "rr",
                },
                "fixed_af_sar": {
                    "target": "none",
                    "trail": "fixed_af_sar",
                    "trailActivation": "rr",
                },
            },
        },
        "parameters": {
            "trailKind": _parameter("select", "none"),
            "stopX": _parameter("float", 0.0),
            "stopLP": _parameter("int", 1),
            "stopMaxPct": _parameter("float", 20.0),
            "stopRR": _parameter("float", 2.0),
            "trailRR": _parameter("float", 1.0),
            "trailMAType": _parameter("select", "SMA"),
            "trailMALength": _parameter("int", 2),
            "trailMAOffsetEx": _parameter("float", 0.0),
            "trailDistanceR": _parameter("float", 1.0),
            "chandelierATRLength": _parameter("int", 2, values=[2, 3]),
            "chandelierATRMult": _parameter("float", 2.0),
            "sarSpeed": _parameter("float", 0.2),
            "riskPerTrade": _parameter("float", 100.0),
            "contractSize": _parameter("float", 1.0),
        },
    }


def _data(*, include_chandelier=True):
    length = 6
    return ExecutionData(
        timestamps=tuple(pd.date_range("2025-01-01", periods=length, freq="30min", tz="UTC")),
        open=np.array([100.0, 100.0, 104.0, 108.0, 106.0, 107.0]),
        high=np.array([100.0, 106.0, 110.0, 112.0, 109.0, 108.0]),
        low=np.array([95.0, 99.0, 103.0, 106.0, 104.0, 105.0]),
        close=np.array([100.0, 104.0, 108.0, 110.0, 106.0, 107.0]),
        signals=Signals(
            long_entries=np.array([True, False, False, False, False, False]),
            short_entries=np.zeros(length, dtype=bool),
        ),
        atr=np.zeros(length),
        rolling_low=np.array([95.0, 99.0, 103.0, 106.0, 104.0, 105.0]),
        rolling_high=np.array([100.0, 106.0, 110.0, 112.0, 109.0, 108.0]),
        trail_long=np.array([np.nan, 101.0, 102.0, 103.0, 104.0, 105.0]),
        trail_short=np.full(length, np.nan),
        chandelier_atr=(
            np.array([np.nan, np.nan, 2.0, 2.0, 2.0, 2.0])
            if include_chandelier
            else None
        ),
    )


def _params(mode):
    profile = parse_execution_profile(_profile_config())
    values = dict(profile.parameter_defaults)
    values["trailKind"] = mode
    return values


@pytest.mark.parametrize("mode", ["none", "ma", "r_distance", "chandelier", "fixed_af_sar"])
def test_compiled_stateful_modes_match_reference(mode):
    profile = parse_execution_profile(_profile_config())
    params = _params(mode)
    data = _data()
    compiled = evaluate_compiled_batch(
        data=data,
        profile=profile,
        params_batch=[params],
        trade_start_idx=0,
        n_workers=1,
    ).outputs[0]
    reference = run_reference_kernel(data, build_kernel_config(profile=profile, params=params))

    assert compiled.shape == (OUTPUT_COLUMN_COUNT,)
    assert compiled[OUTPUT_TOTAL_TRADES] == len(reference.trades)
    assert compiled[OUTPUT_FINAL_BALANCE] == pytest.approx(reference.balance_curve[-1])


@pytest.mark.parametrize("workers", [1, 2])
def test_mixed_stacked_batch_matches_grouped_with_optional_chandelier_transport(workers):
    profile = parse_execution_profile(_profile_config())
    modes = ["none", "ma", "r_distance", "chandelier", "fixed_af_sar"]
    params_batch = [_params(mode) for mode in modes]
    grouped = evaluate_compiled_batch(
        data=_data(),
        profile=profile,
        params_batch=params_batch,
        trade_start_idx=0,
        n_workers=workers,
    ).outputs
    stacked_data = build_stacked_execution_data(
        [_data(include_chandelier=False), _data(include_chandelier=True)],
        [0, 0, 0, 1, 0],
    )
    stacked = evaluate_compiled_stacked_batch(
        stacked_data=stacked_data,
        profile=profile,
        params_batch=params_batch,
        trade_start_idx=0,
        n_workers=workers,
    ).outputs

    np.testing.assert_allclose(stacked, grouped, rtol=0.0, atol=0.0, equal_nan=True)
    assert stacked_data.chandelier_atr.shape == (1, 6)
    assert stacked_data.chandelier_data_index.tolist() == [-1, 0]


@pytest.mark.parametrize("bad", [0, -1, 1.5, float("nan"), float("inf"), True])
def test_chandelier_length_declaration_requires_positive_integers(bad):
    config = _profile_config()
    declaration = config["parameters"]["chandelierATRLength"]
    declaration["default"] = bad
    declaration["optimize"] = {"enabled": False}

    with pytest.raises(ProfileValidationError, match="chandelierATRLength"):
        parse_execution_profile(config)


def test_inactive_stateful_parameters_are_not_read():
    profile = parse_execution_profile(_profile_config())
    params = _params("none")
    params.update(
        trailDistanceR="ignored",
        chandelierATRLength="ignored",
        chandelierATRMult="ignored",
        sarSpeed="ignored",
    )

    output = evaluate_compiled_batch(
        data=_data(include_chandelier=False),
        profile=profile,
        params_batch=[params],
        trade_start_idx=0,
    )
    assert output.outputs.shape == (1, OUTPUT_COLUMN_COUNT)


def test_compiled_chandelier_rejects_missing_optional_row_before_dispatch():
    profile = parse_execution_profile(_profile_config())
    with pytest.raises(ValueError, match="chandelier_atr"):
        evaluate_compiled_batch(
            data=_data(include_chandelier=False),
            profile=profile,
            params_batch=[_params("chandelier")],
            trade_start_idx=0,
        )

    stacked_data = build_stacked_execution_data(
        [_data(include_chandelier=False)],
        [0],
    )
    with pytest.raises(ValueError, match="chandelier_atr"):
        evaluate_compiled_stacked_batch(
            stacked_data=stacked_data,
            profile=profile,
            params_batch=[_params("chandelier")],
            trade_start_idx=0,
        )


def test_cache_estimate_accounts_only_for_active_chandelier_rows_and_mapping():
    config = _profile_config()
    df = pd.DataFrame({"close": np.arange(6, dtype=float)})
    hooks = GridV2StrategyHooks(build_execution_data=lambda _df, _params: _data())
    chandelier_plan = build_grid_v2_plan(
        config,
        settings=GridV2Settings(enabled_variants=("chandelier",)),
    )
    bracket_plan = build_grid_v2_plan(
        config,
        settings=GridV2Settings(enabled_variants=("bracket",)),
    )

    chandelier = estimate_grid_v2_cache(chandelier_plan, df, 0, hooks)
    bracket = estimate_grid_v2_cache(bracket_plan, df, 0, hooks)

    assert chandelier.chandelier_combo_count == 2
    assert chandelier.chandelier_atr_nbytes == 2 * 6 * 8
    assert chandelier.chandelier_mapping_nbytes == 2 * 4
    assert bracket.chandelier_combo_count == 0
    assert bracket.chandelier_atr_nbytes == 0
    assert bracket.chandelier_mapping_nbytes == 0
    assert bracket.bytes_per_dataprep_combo == 6 * 5 * 8
