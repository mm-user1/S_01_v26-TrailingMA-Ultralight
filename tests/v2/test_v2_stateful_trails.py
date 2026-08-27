from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from core.engine_v2.compiled_kernel import (
    OUTPUT_COLUMN_COUNT,
    OUTPUT_FINAL_BALANCE,
    OUTPUT_FLAGS,
    OUTPUT_GROSS_LOSS,
    OUTPUT_GROSS_PROFIT,
    OUTPUT_INVALID_STOP_DISTANCE_COUNT,
    OUTPUT_LIQUIDATION_COUNT,
    OUTPUT_LOSING_TRADES,
    OUTPUT_MARGIN_REJECT_COUNT,
    OUTPUT_MAX_DRAWDOWN_PCT,
    OUTPUT_MAX_NOTIONAL,
    OUTPUT_MAX_REQUIRED_LEVERAGE,
    OUTPUT_NET_PROFIT_PCT,
    OUTPUT_NO_CAPITAL_HALT,
    OUTPUT_REJECTED_FILL_COUNT,
    OUTPUT_ROMAD,
    OUTPUT_TOTAL_TRADES,
    OUTPUT_WINNING_TRADES,
    OUTPUT_WIN_RATE_PCT,
    OUTPUT_ZERO_SIZE_ENTRY_COUNT,
    build_stacked_execution_data,
    evaluate_compiled_batch,
    evaluate_compiled_stacked_batch,
)
from core.engine_v2.contracts import Signals
from core.engine_v2.kernel import ExecutionData, run_reference_kernel
from core.engine_v2.profile import ProfileValidationError, parse_execution_profile
from core.engine_v2.runner import build_kernel_config, run_v2_strategy
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


def _stateful_parity_data(mode, direction):
    if mode == "r_distance":
        open_ = [100.0, 100.0, 104.0]
        high = [100.0, 106.0, 105.0]
        low = [95.0, 99.0, 101.0]
        close = [100.0, 104.0, 102.0]
        chandelier_atr = None
    elif mode == "chandelier":
        open_ = [100.0, 100.0, 104.0, 108.0]
        high = [100.0, 106.0, 110.0, 109.0]
        low = [95.0, 99.0, 103.0, 105.0]
        close = [100.0, 104.0, 108.0, 106.0]
        chandelier_atr = [np.nan, 1.0, 2.0, 2.0]
    else:
        open_ = [100.0, 100.0, 104.0, 108.0, 106.0]
        high = [100.0, 105.0, 110.0, 112.0, 107.0]
        low = [95.0, 99.0, 103.0, 106.0, 102.0]
        close = [100.0, 104.0, 108.0, 110.0, 103.0]
        chandelier_atr = None

    if direction == "short":
        open_ = [200.0 - value for value in open_]
        close = [200.0 - value for value in close]
        high, low = (
            [200.0 - value for value in low],
            [200.0 - value for value in high],
        )
    length = len(open_)
    return ExecutionData(
        timestamps=tuple(pd.date_range("2025-01-01", periods=length, freq="30min", tz="UTC")),
        open=np.asarray(open_),
        high=np.asarray(high),
        low=np.asarray(low),
        close=np.asarray(close),
        signals=Signals(
            long_entries=np.asarray([direction == "long"] + [False] * (length - 1)),
            short_entries=np.asarray([direction == "short"] + [False] * (length - 1)),
        ),
        atr=np.zeros(length),
        rolling_low=np.asarray(low),
        rolling_high=np.asarray(high),
        trail_long=np.full(length, np.nan),
        trail_short=np.full(length, np.nan),
        chandelier_atr=(
            np.asarray(chandelier_atr) if chandelier_atr is not None else None
        ),
    )


def _assert_stable_compiled_outputs_match_reference(compiled, reference):
    basic = reference.basic_metrics
    advanced = reference.advanced_metrics
    guardrails = reference.guardrail_summary
    floating_pairs = (
        (OUTPUT_FINAL_BALANCE, reference.strategy_result.balance_curve[-1]),
        (OUTPUT_NET_PROFIT_PCT, basic.net_profit_pct),
        (OUTPUT_GROSS_PROFIT, basic.gross_profit),
        (OUTPUT_GROSS_LOSS, basic.gross_loss),
        (OUTPUT_WIN_RATE_PCT, basic.win_rate),
        (OUTPUT_MAX_DRAWDOWN_PCT, basic.max_drawdown_pct),
        (OUTPUT_MAX_REQUIRED_LEVERAGE, guardrails.max_required_leverage),
        (OUTPUT_MAX_NOTIONAL, guardrails.max_notional),
    )
    for index, expected in floating_pairs:
        assert compiled[index] == pytest.approx(expected, rel=1e-12, abs=1e-12)
    if advanced.romad is not None:
        assert compiled[OUTPUT_ROMAD] == pytest.approx(
            advanced.romad, rel=1e-12, abs=1e-12
        )

    assert int(compiled[OUTPUT_TOTAL_TRADES]) == basic.total_trades
    assert int(compiled[OUTPUT_WINNING_TRADES]) == basic.winning_trades
    assert int(compiled[OUTPUT_LOSING_TRADES]) == basic.losing_trades
    assert int(compiled[OUTPUT_INVALID_STOP_DISTANCE_COUNT]) == guardrails.invalid_stop_distance_count
    assert int(compiled[OUTPUT_ZERO_SIZE_ENTRY_COUNT]) == guardrails.zero_size_entry_count
    assert int(compiled[OUTPUT_REJECTED_FILL_COUNT]) == guardrails.rejected_fill_count
    assert int(compiled[OUTPUT_MARGIN_REJECT_COUNT]) == guardrails.margin_reject_count
    assert int(compiled[OUTPUT_LIQUIDATION_COUNT]) == guardrails.liquidation_count
    assert bool(compiled[OUTPUT_NO_CAPITAL_HALT]) is guardrails.no_capital_halt
    assert int(compiled[OUTPUT_FLAGS]) == guardrails.flags


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


@pytest.mark.parametrize("mode", ["r_distance", "chandelier", "fixed_af_sar"])
@pytest.mark.parametrize("direction", ["long", "short"])
def test_compiled_stateful_trade_transitions_match_reference(mode, direction):
    profile = parse_execution_profile(_profile_config())
    params = _params(mode)
    params["commissionPct"] = 2.0
    if mode == "r_distance":
        params["trailDistanceR"] = 0.8
    data = _stateful_parity_data(mode, direction)

    compiled = evaluate_compiled_batch(
        data=data,
        profile=profile,
        params_batch=[params],
        trade_start_idx=0,
        n_workers=1,
        compute_max_drawdown_mtm=True,
    )
    reference = run_v2_strategy(
        data=data,
        profile=profile,
        params=params,
        trade_start_idx=0,
        compute_max_drawdown_mtm=True,
    )

    _assert_stable_compiled_outputs_match_reference(compiled.outputs[0], reference)
    assert reference.max_drawdown_mtm_pct is not None
    assert compiled.max_drawdown_mtm_pct is not None
    assert compiled.max_drawdown_mtm_pct.shape == (1,)
    assert compiled.max_drawdown_mtm_pct[0] == pytest.approx(
        reference.max_drawdown_mtm_pct,
        rel=1e-12,
        abs=1e-12,
    )
    assert reference.strategy_result.trades[0].exit_price == pytest.approx(
        {"r_distance": 102.0, "chandelier": 106.0, "fixed_af_sar": 102.4}[mode]
        if direction == "long"
        else {"r_distance": 98.0, "chandelier": 94.0, "fixed_af_sar": 97.6}[mode]
    )
    assert reference.strategy_result.trades[0].exit_time == data.timestamps[-1]

    if mode == "fixed_af_sar":
        activation_only = ExecutionData(
            **{
                **data.__dict__,
                "timestamps": data.timestamps[:2],
                "open": data.open[:2],
                "high": data.high[:2],
                "low": data.low[:2],
                "close": data.close[:2],
                "signals": Signals(
                    long_entries=data.signals.long_entries[:2],
                    short_entries=data.signals.short_entries[:2],
                ),
                "atr": data.atr[:2],
                "rolling_low": data.rolling_low[:2],
                "rolling_high": data.rolling_high[:2],
                "trail_long": data.trail_long[:2],
                "trail_short": data.trail_short[:2],
            }
        )
        activation = run_reference_kernel(
            activation_only,
            replace(
                build_kernel_config(profile=profile, params=params),
                boundary_mode="none",
            ),
        )
        assert activation.standing_state.trail_stop == 100.0


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
        compute_max_drawdown_mtm=True,
    )
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
        compute_max_drawdown_mtm=True,
    )

    np.testing.assert_allclose(
        stacked.outputs,
        grouped.outputs,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    assert grouped.max_drawdown_mtm_pct is not None
    assert stacked.max_drawdown_mtm_pct is not None
    np.testing.assert_allclose(
        stacked.max_drawdown_mtm_pct,
        grouped.max_drawdown_mtm_pct,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
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


def test_active_chandelier_rejects_non_mapping_optimize_metadata():
    config = _profile_config()
    config["parameters"]["chandelierATRLength"]["optimize"] = "bad-shape"

    with pytest.raises(ProfileValidationError) as exc_info:
        parse_execution_profile(config)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "V2_INVALID_PROFILE"
    assert diagnostic.path == "parameters.chandelierATRLength.optimize"
    assert ".optimize" in diagnostic.message
    assert "mapping when present" in diagnostic.message


@pytest.mark.parametrize(
    "optimize",
    [
        pytest.param("omitted", id="omitted"),
        pytest.param(None, id="null"),
        pytest.param({"enabled": False}, id="disabled-mapping"),
        pytest.param({"enabled": True, "values": [1, 2, 3]}, id="enabled-values"),
    ],
)
def test_active_chandelier_accepts_valid_optimize_metadata(optimize):
    config = _profile_config()
    declaration = config["parameters"]["chandelierATRLength"]
    if optimize == "omitted":
        declaration.pop("optimize")
    else:
        declaration["optimize"] = optimize

    assert parse_execution_profile(config).strategy_id == "stateful_trail_fixture"


def test_inactive_chandelier_preserves_non_mapping_optimize_behavior():
    config = _profile_config()
    del config["execution"]["variants"]["chandelier"]
    del config["execution"]["variantSelector"]["mapping"]["chandelier"]
    config["parameters"]["chandelierATRLength"]["optimize"] = "bad-shape"

    assert parse_execution_profile(config).strategy_id == "stateful_trail_fixture"


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
