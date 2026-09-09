from __future__ import annotations

import math
from dataclasses import fields, replace

import numpy as np
import pandas as pd
import pytest

from core.engine_v2.compiled_kernel import (
    OUTPUT_COLUMN_COUNT,
    build_calendar_month_ids,
    build_stacked_execution_data,
    evaluate_compiled_batch,
    evaluate_compiled_stacked_batch,
)
from core.engine_v2.contracts import Signals
from core.engine_v2.dataprep import build_signal_execution_data
from core.engine_v2 import compiled_kernel_signal as signal_compiled
from core.metrics import build_utc_day_ids
from core.engine_v2.kernel import ExecutionData
from core.engine_v2.runner import run_v2_strategy
from core.grid_v2 import (
    GridV2Settings,
    GridV2StrategyHooks,
    build_grid_v2_plan,
    estimate_grid_v2_cache,
    execute_grid_v2_candidates,
)
from strategies.s03_reversal_v11_regime_er_b2 import strategy as s03_strategy
from strategies.s03_reversal_v16_4_a_adaptive_ma_b2 import strategy as adaptive_strategy
from strategies.s06_r_trend_v02_b2 import strategy as s06_strategy
from strategies.s06_r_trend_v02_b2.strategy import load_config

from s03_regime_er_test_helpers import (
    REFERENCE_A,
    merged_reference_params as s03_params,
    prepared_reference_dataset as s03_dataset,
)
from s06_b2_test_helpers import merged_reference_params, prepared_reference_dataset


def mtm_oracle(equity, evaluation_start, initial_evaluation_equity):
    """Independent definition: maximum close-equity decline from the running peak."""

    observed = [float(value) for value in equity[max(0, int(evaluation_start)) :]]
    if not observed or any(not math.isfinite(value) for value in observed):
        return float("nan")
    peak = float(initial_evaluation_equity)
    drawdowns = []
    for value in observed:
        peak = max(peak, value)
        drawdowns.append(100.0 * (peak - value) / peak)
    return max(drawdowns)


@pytest.mark.parametrize(
    ("equity", "start", "expected"),
    [
        ([100.0, 100.0, 100.0], 0, 0.0),
        ([100.0, 80.0, 100.0], 0, 20.0),
        ([100.0, 120.0, 90.0], 0, 25.0),
        ([100.0, 120.0, 60.0], 0, 50.0),
        ([100.0, -20.0], 0, 120.0),
        ([1.0, 999.0, 100.0, 90.0], 2, 10.0),
        ([100.0], 0, 0.0),
        ([100.0, 90.0], 1, 10.0),
    ],
)
def test_independent_mtm_oracle_contract(equity, start, expected):
    assert mtm_oracle(equity, start, 100.0) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("equity", "start"),
    [([], 0), ([100.0], 1), ([100.0, np.nan, 90.0], 0), ([100.0, np.inf], 0)],
)
def test_independent_mtm_oracle_empty_and_sticky_nan(equity, start):
    assert math.isnan(mtm_oracle(equity, start, 100.0))


def _real_inputs():
    params = merged_reference_params("reference_b_trend_bracket")
    dataframe, trade_start_idx = prepared_reference_dataset()
    data = s06_strategy.build_v2_execution_data(dataframe, params)
    return params, dataframe, trade_start_idx, data


def _empty_data() -> ExecutionData:
    empty_float = np.empty(0, dtype=np.float64)
    empty_bool = np.empty(0, dtype=np.bool_)
    return ExecutionData(
        timestamps=(),
        open=empty_float,
        high=empty_float,
        low=empty_float,
        close=empty_float,
        signals=Signals(long_entries=empty_bool, short_entries=empty_bool),
        atr=empty_float,
        rolling_low=empty_float,
        rolling_high=empty_float,
        trail_long=empty_float,
        trail_short=empty_float,
    )


def test_reference_and_compiled_mtm_only_request_match_independent_oracle():
    params, _, trade_start_idx, data = _real_inputs()
    reference = run_v2_strategy(
        data=data,
        profile=s06_strategy.load_profile(),
        params=params,
        trade_start_idx=trade_start_idx,
        compute_max_drawdown_mtm=True,
    )
    expected = mtm_oracle(reference.kernel_result.equity_curve, trade_start_idx, 100.0)
    compiled = evaluate_compiled_batch(
        data=data,
        profile=s06_strategy.load_profile(),
        params_batch=[params],
        trade_start_idx=trade_start_idx,
        n_workers=1,
        compute_sharpe=False,
        compute_sharpe_daily=False,
        compute_sqn=False,
        compute_max_drawdown_mtm=True,
    )

    assert expected > 0.0
    assert reference.max_drawdown_mtm_pct == pytest.approx(expected, abs=1e-12)
    assert compiled.max_drawdown_mtm_pct is not None
    assert compiled.max_drawdown_mtm_pct.shape == (1,)
    assert compiled.max_drawdown_mtm_pct.nbytes == np.dtype(np.float64).itemsize
    assert compiled.max_drawdown_mtm_pct[0] == pytest.approx(expected, abs=1e-12)


def test_compiled_requested_empty_paths_are_initialized_nan():
    params, _, _, data = _real_inputs()
    profile = s06_strategy.load_profile()
    empty = evaluate_compiled_batch(
        data=_empty_data(),
        profile=profile,
        params_batch=[params],
        trade_start_idx=0,
        compute_max_drawdown_mtm=True,
    )
    no_evaluation = evaluate_compiled_batch(
        data=data,
        profile=profile,
        params_batch=[params],
        trade_start_idx=len(data.close),
        compute_max_drawdown_mtm=True,
    )

    assert empty.max_drawdown_mtm_pct is not None
    assert no_evaluation.max_drawdown_mtm_pct is not None
    assert math.isnan(empty.max_drawdown_mtm_pct[0])
    assert math.isnan(no_evaluation.max_drawdown_mtm_pct[0])


def test_disabled_parallel_batch_has_no_sidecar_and_preserves_26_columns_bitwise():
    params, _, trade_start_idx, data = _real_inputs()
    params_batch = [params, dict(params), dict(params)]
    disabled = evaluate_compiled_batch(
        data=data,
        profile=s06_strategy.load_profile(),
        params_batch=params_batch,
        trade_start_idx=trade_start_idx,
        n_workers=2,
    )
    enabled = evaluate_compiled_batch(
        data=data,
        profile=s06_strategy.load_profile(),
        params_batch=params_batch,
        trade_start_idx=trade_start_idx,
        n_workers=2,
        compute_max_drawdown_mtm=True,
    )

    assert disabled.max_drawdown_mtm_pct is None
    assert disabled.outputs.shape == enabled.outputs.shape == (3, OUTPUT_COLUMN_COUNT)
    assert np.array_equal(disabled.outputs, enabled.outputs, equal_nan=True)
    assert enabled.max_drawdown_mtm_pct is not None
    assert enabled.max_drawdown_mtm_pct.shape == (3,)


def test_grouped_and_stacked_requested_sidecars_are_equal():
    params, _, trade_start_idx, data = _real_inputs()
    grouped = evaluate_compiled_batch(
        data=data,
        profile=s06_strategy.load_profile(),
        params_batch=[params],
        trade_start_idx=trade_start_idx,
        compute_max_drawdown_mtm=True,
    )
    stacked_data = build_stacked_execution_data([data], [0])
    stacked = evaluate_compiled_stacked_batch(
        stacked_data=stacked_data,
        profile=s06_strategy.load_profile(),
        params_batch=[params],
        trade_start_idx=trade_start_idx,
        compute_max_drawdown_mtm=True,
    )

    assert grouped.max_drawdown_mtm_pct is not None
    assert stacked.max_drawdown_mtm_pct is not None
    assert np.array_equal(grouped.outputs, stacked.outputs, equal_nan=True)
    assert np.array_equal(
        grouped.max_drawdown_mtm_pct,
        stacked.max_drawdown_mtm_pct,
        equal_nan=True,
    )


def test_grid_typed_transport_and_exact_sidecar_cache_accounting():
    params, dataframe, trade_start_idx, _ = _real_inputs()
    settings = GridV2Settings(
        enabled_variants=("bracket",),
        enabled_axes=(),
        slow_enrich_selected=False,
    )
    plan = build_grid_v2_plan(load_config(), settings, base_params=params)
    hooks = GridV2StrategyHooks.from_strategy(s06_strategy)
    disabled = estimate_grid_v2_cache(plan, dataframe, trade_start_idx, hooks)
    enabled = estimate_grid_v2_cache(
        plan,
        dataframe,
        trade_start_idx,
        hooks,
        compute_max_drawdown_mtm=True,
    )
    result = execute_grid_v2_candidates(
        plan,
        dataframe,
        trade_start_idx,
        hooks,
        compute_max_drawdown_mtm=True,
    )

    assert disabled.mtm_sidecar_nbytes == 0
    assert enabled.mtm_sidecar_nbytes == plan.deduped_candidate_count * 8
    assert enabled.output_column_count == disabled.output_column_count == OUTPUT_COLUMN_COUNT
    assert enabled.bytes_per_output_candidate == disabled.bytes_per_output_candidate == 26 * 8
    assert math.isfinite(result.rows[0].max_drawdown_mtm_pct)
    assert result.metadata["compute_max_drawdown_mtm"] is True
    assert result.metadata["mtm_sidecar_nbytes"] == 8


def test_older_signal_reversal_supports_mtm_and_preserves_outputs():
    params = s03_params(REFERENCE_A)
    dataframe, trade_start_idx = s03_dataset()
    plan = build_grid_v2_plan(s03_strategy.load_config(),
        GridV2Settings(enabled_axes=(), slow_enrich_selected=False), base_params=params)
    hooks = GridV2StrategyHooks.from_strategy(s03_strategy)
    off = execute_grid_v2_candidates(plan, dataframe, trade_start_idx, hooks)
    on = execute_grid_v2_candidates(plan, dataframe, trade_start_idx, hooks, compute_max_drawdown_mtm=True)
    reference = run_v2_strategy(data=s03_strategy.build_v2_execution_data(dataframe, params),
        profile=plan.profile, params=params, trade_start_idx=trade_start_idx, compute_max_drawdown_mtm=True)
    expected = mtm_oracle(reference.kernel_result.equity_curve, trade_start_idx, params["initialCapital"])
    assert on.metadata["compiled_batch_used"]
    assert on.rows[0].max_drawdown_mtm_pct == pytest.approx(expected, rel=1e-9, abs=1e-12)
    assert reference.max_drawdown_mtm_pct == pytest.approx(expected, rel=1e-9, abs=1e-12)
    _assert_existing_row_equal(on.rows[0], off.rows[0])


def _assert_existing_row_equal(left, right):
    for field in fields(left):
        if field.name == "max_drawdown_mtm_pct":
            continue
        a, b = getattr(left, field.name), getattr(right, field.name)
        if isinstance(a, float) and math.isnan(a):
            assert math.isnan(b), field.name
        else:
            assert a == b, field.name


def _signal_fixture(side="long", *, close=None, entry=0, exit_at=None):
    values = np.array(close if close is not None else
                      ([100, 80, 90, 110, 100] if side == "long" else [100, 120, 110, 90, 100]), dtype=float)
    opens = np.full(len(values), 100.0)
    frame = pd.DataFrame({"Open": opens, "High": np.maximum(opens, values),
                          "Low": np.minimum(opens, values), "Close": values,
                          "Volume": np.full(len(values), 1000.0)},
                         index=pd.date_range("2025-01-01", periods=len(values), freq="30min", tz="UTC"))
    long, short, exits = (np.zeros(len(values), dtype=bool) for _ in range(3))
    if entry is not None:
        (long if side == "long" else short)[entry] = True
    if exit_at is not None:
        exits[exit_at] = True
    data = build_signal_execution_data(frame, signals=Signals(long_entries=long, short_entries=short,
        long_exits=exits if side == "long" else None, short_exits=exits if side == "short" else None))
    return data


def _signal_params(**overrides):
    params = {k: v["default"] for k, v in adaptive_strategy.load_config()["parameters"].items()}
    params.update(initialCapital=100.0, commissionPct=0.1, contractSize=0.0001,
                  positionPct=100.0, dateFilter=False)
    params.update(overrides)
    return params


def _signal_stack(rows, mapping):
    return signal_compiled.build_signal_stacked_execution_data(rows, mapping,
        month_ids=build_calendar_month_ids(rows[0].timestamps), day_ids=build_utc_day_ids(rows[0].timestamps))


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("emergency", [False, True])
@pytest.mark.parametrize("all_metrics", [False, True])
def test_signal_mtm_oracle_commissions_stops_final_close_and_default_off(side, emergency, all_metrics):
    data = _signal_fixture(side)
    params = _signal_params(useEmergencySL=emergency, emergencySlPct=10.0)
    profile = adaptive_strategy.load_profile()
    kwargs = dict(data=data, profile=profile, params=params, trade_start_idx=0)
    reference = run_v2_strategy(**kwargs, compute_max_drawdown_mtm=True)
    off_reference = run_v2_strategy(**kwargs)
    assert reference.kernel_result.trades == off_reference.kernel_result.trades
    np.testing.assert_array_equal(reference.kernel_result.equity_curve, off_reference.kernel_result.equity_curve)
    expected = mtm_oracle(reference.kernel_result.equity_curve, 0, 100.0)
    if not emergency:
        # One unit, entry and strict-final exit at 100; 0.1 commission on each fill.
        np.testing.assert_allclose(reference.kernel_result.equity_curve, [100, 79.9, 89.9, 109.9, 99.8], rtol=0, atol=1e-12)
        assert expected == pytest.approx(20.1)
        assert reference.kernel_result.trades[0].entry_time == data.timestamps[1]
        assert reference.kernel_result.trades[0].exit_time == data.timestamps[-1]
    assert reference.max_drawdown_mtm_pct == pytest.approx(expected, rel=1e-9, abs=1e-12)
    kwargs = dict(stacked_data=_signal_stack([data], [0]), profile=profile, params_batch=[params],
                  trade_start_idx=0, compute_sharpe=all_metrics, compute_sharpe_daily=all_metrics,
                  compute_sqn=all_metrics)
    on = signal_compiled.evaluate_compiled_signal_stacked_batch(**kwargs, compute_max_drawdown_mtm=True)
    off = signal_compiled.evaluate_compiled_signal_stacked_batch(**kwargs)
    assert off.max_drawdown_mtm_pct is None
    assert on.outputs.shape == (1, 26)
    np.testing.assert_array_equal(on.outputs, off.outputs)
    assert on.max_drawdown_mtm_pct[0] == pytest.approx(expected, rel=1e-9, abs=1e-12)
    if not emergency:
        from core.engine_v2.compiled_kernel import OUTPUT_MAX_DRAWDOWN_PCT
        assert on.outputs[0, OUTPUT_MAX_DRAWDOWN_PCT] < expected


def test_signal_pending_exit_executes_next_open_before_close_observation():
    data = _signal_fixture(exit_at=1)
    params = _signal_params()
    reference = run_v2_strategy(data=data, profile=adaptive_strategy.load_profile(), params=params,
                                compute_max_drawdown_mtm=True)
    np.testing.assert_allclose(reference.kernel_result.equity_curve, [100, 79.9, 99.8, 99.8, 99.8], atol=1e-12)
    assert reference.kernel_result.trades[0].exit_time == data.timestamps[2]
    compiled = signal_compiled.evaluate_compiled_signal_stacked_batch(stacked_data=_signal_stack([data], [0]),
        profile=adaptive_strategy.load_profile(), params_batch=[params], trade_start_idx=0, compute_max_drawdown_mtm=True)
    assert compiled.max_drawdown_mtm_pct[0] == pytest.approx(20.1, abs=1e-12)


@pytest.mark.parametrize("start,entry", [(0, None), (2, None), (2, 2), (5, None), (7, None)])
def test_signal_warmup_flat_and_empty_evaluation(start, entry):
    data = _signal_fixture(close=[1000, 1000, 100, 80, 100], entry=entry)
    params = _signal_params(commissionPct=0)
    if start <= len(data.close):
        reference = run_v2_strategy(data=data, profile=adaptive_strategy.load_profile(), params=params,
                                    trade_start_idx=start, compute_max_drawdown_mtm=True)
        expected = mtm_oracle(reference.kernel_result.equity_curve, start, 100)
    compiled = signal_compiled.evaluate_compiled_signal_stacked_batch(stacked_data=_signal_stack([data], [0]),
        profile=adaptive_strategy.load_profile(), params_batch=[params], trade_start_idx=start, compute_max_drawdown_mtm=True)
    if start >= 5:
        assert math.isnan(compiled.max_drawdown_mtm_pct[0])
        if start == 5:
            assert math.isnan(reference.max_drawdown_mtm_pct)
    else:
        assert compiled.max_drawdown_mtm_pct[0] == pytest.approx(expected, rel=1e-9, abs=1e-12)
        assert expected == pytest.approx(20.0 if entry == 2 else 0.0)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("requested", [False, True])
def test_signal_zero_candidates_both_entry_points(packed, requested):
    data = _signal_fixture()
    kwargs = (dict(packed_config_arrays=signal_compiled._empty_signal_config_arrays(0)) if packed else dict(params_batch=[]))
    result = signal_compiled.evaluate_compiled_signal_stacked_batch(stacked_data=_signal_stack([data], []),
        profile=adaptive_strategy.load_profile(), trade_start_idx=0, compute_max_drawdown_mtm=requested, **kwargs)
    assert result.outputs.shape == (0, 26)
    if requested:
        assert result.max_drawdown_mtm_pct.shape == (0,) and result.max_drawdown_mtm_pct.dtype == np.float64
    else:
        assert result.max_drawdown_mtm_pct is None


def test_signal_zero_bars_initializes_requested_sidecar():
    result = signal_compiled.evaluate_compiled_signal_stacked_batch(stacked_data=_signal_stack([_empty_data()], [0]),
        profile=adaptive_strategy.load_profile(), params_batch=[_signal_params()], trade_start_idx=0, compute_max_drawdown_mtm=True)
    assert math.isnan(result.max_drawdown_mtm_pct[0])


def test_signal_negative_equity_drawdown_exceeds_100_percent():
    data = _signal_fixture("short", close=[100, 250, 100, 100])
    params = _signal_params(commissionPct=0)
    reference = run_v2_strategy(data=data, profile=adaptive_strategy.load_profile(), params=params, compute_max_drawdown_mtm=True)
    np.testing.assert_array_equal(reference.kernel_result.equity_curve, [100, -50, 100, 100])
    compiled = signal_compiled.evaluate_compiled_signal_stacked_batch(stacked_data=_signal_stack([data], [0]),
        profile=adaptive_strategy.load_profile(), params_batch=[params], trade_start_idx=0, compute_max_drawdown_mtm=True)
    assert compiled.max_drawdown_mtm_pct[0] == reference.max_drawdown_mtm_pct == mtm_oracle([100,-50,100,100], 0, 100) == 150


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_signal_nonfinite_observation_is_sticky_at_numerical_seam(invalid):
    from core.engine_v2.runner import _max_drawdown_mtm_from_equity
    data = _signal_fixture()
    stack = _signal_stack([data], [0])
    close = stack.close.copy()
    close[2] = invalid
    # Inject after valid OHLC preparation, at the numerical accumulator seam.
    stack = replace(stack, close=close)
    result = signal_compiled.evaluate_compiled_signal_stacked_batch(stacked_data=stack,
        profile=adaptive_strategy.load_profile(), params_batch=[_signal_params()], trade_start_idx=0, compute_max_drawdown_mtm=True)
    assert math.isnan(result.max_drawdown_mtm_pct[0])
    assert math.isnan(_max_drawdown_mtm_from_equity([100, 79.9, invalid, 109.9, 99.8], 0, 100))


@pytest.mark.parametrize("packed", [False, True])
def test_signal_repeated_data_mapping_and_real_threads_preserve_sidecar(packed):
    numba = signal_compiled.numba
    previous = numba.get_num_threads()
    if previous < 2:
        pytest.skip("Two actual Numba threads unavailable.")
    long = _signal_fixture()
    short = replace(long, signals=Signals(long_entries=np.zeros(5, dtype=bool),
                                         short_entries=np.array([True, False, False, False, False])))
    flat = replace(long, signals=Signals(long_entries=np.zeros(5, dtype=bool), short_entries=np.zeros(5, dtype=bool)))
    rows, mapping = [long, short, flat], [1, 0, 1, 2, 0]
    params = [_signal_params()] * len(mapping)
    profile = adaptive_strategy.load_profile()
    kwargs = (dict(packed_config_arrays=signal_compiled._pack_signal_config_arrays(profile, params)) if packed
              else dict(params_batch=params))
    results = []
    for threads in (1, 2):
        result = signal_compiled.evaluate_compiled_signal_stacked_batch(stacked_data=_signal_stack(rows, mapping),
            profile=profile, trade_start_idx=0, n_workers=threads, compute_max_drawdown_mtm=True, **kwargs)
        assert numba.get_num_threads() == previous
        results.append(result)
    expected = [mtm_oracle(run_v2_strategy(data=rows[i], profile=profile, params=params[j]).kernel_result.equity_curve, 0, 100)
                for j, i in enumerate(mapping)]
    assert len(set(expected)) == 3
    np.testing.assert_allclose(results[0].max_drawdown_mtm_pct, expected, rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(results[0].max_drawdown_mtm_pct, results[1].max_drawdown_mtm_pct)
    np.testing.assert_array_equal(results[0].outputs, results[1].outputs)


def test_signal_thread_state_is_restored_on_compiled_failure(monkeypatch):
    previous = signal_compiled.numba.get_num_threads()
    def fail(*args):
        assert signal_compiled.numba.get_num_threads() == 1
        raise RuntimeError("injected numerical failure")
    monkeypatch.setattr(signal_compiled, "_COMPILED_SIGNAL_STACKED_BATCH_LOOP", fail)
    with pytest.raises(RuntimeError, match="injected numerical failure"):
        signal_compiled.evaluate_compiled_signal_stacked_batch(stacked_data=_signal_stack([_signal_fixture()], [0]),
            profile=adaptive_strategy.load_profile(), params_batch=[_signal_params()], trade_start_idx=0,
            n_workers=1, compute_max_drawdown_mtm=True)
    assert signal_compiled.numba.get_num_threads() == previous


def test_adaptive_signal_chunked_reordered_indices_and_exact_sidecar_accounting():
    n = 512
    close = 100 + 15 * np.sin(np.arange(n) / 8)
    frame = pd.DataFrame(dict(Open=close, High=close+1, Low=close-1, Close=close, Volume=np.ones(n)),
                         index=pd.date_range("2025-01-01", periods=n, freq="30min", tz="UTC"))
    plan = build_grid_v2_plan(adaptive_strategy.load_config(),
        GridV2Settings(enabled_axes=("maType3", "maLength3"), slow_enrich_selected=False),
        base_params=_signal_params())
    hooks = GridV2StrategyHooks.from_strategy(adaptive_strategy)
    indices = (39, 0, 10, 30, 1, 20)
    off = estimate_grid_v2_cache(plan, frame, 100, hooks, candidate_indices=indices)
    on = estimate_grid_v2_cache(plan, frame, 100, hooks, candidate_indices=indices, compute_max_drawdown_mtm=True)
    single = estimate_grid_v2_cache(plan, frame, 100, hooks, candidate_indices=(0,), compute_max_drawdown_mtm=True)
    assert on.mtm_sidecar_nbytes == len(indices)*8 and off.mtm_sidecar_nbytes == 0
    assert on.estimated_total_mb - off.estimated_total_mb == pytest.approx(len(indices)*8/(1024**2), abs=1e-12)
    limited = replace(plan, settings=replace(plan.settings, max_signal_cache_mb=(single.estimated_total_mb+on.estimated_total_mb)/2))
    kwargs = dict(candidate_indices=indices, compute_max_drawdown_mtm=True)
    full = execute_grid_v2_candidates(plan, frame, 100, hooks, **kwargs)
    chunked = execute_grid_v2_candidates(limited, frame, 100, hooks, **kwargs)
    reference = execute_grid_v2_candidates(replace(plan, settings=replace(plan.settings, prefer_compiled=False)), frame, 100, hooks, **kwargs)
    assert chunked.metadata["chunk_count"] > 1 and chunked.metadata["compiled_batch_used"]
    assert chunked.metadata["mtm_sidecar_nbytes"] == 8*len(indices)
    # Grid intentionally publishes selected rows in authoritative plan order.
    assert [r.candidate_id for r in chunked.rows] == [plan.candidate_table.candidate_id_for_index(i) for i in sorted(indices)]
    for a,b,c in zip(full.rows, chunked.rows, reference.rows):
        _assert_existing_row_equal(a,b)
        assert a.max_drawdown_mtm_pct == b.max_drawdown_mtm_pct
        assert b.max_drawdown_mtm_pct == pytest.approx(c.max_drawdown_mtm_pct, rel=1e-9, abs=1e-12)


def test_all_480_real_candidates_are_thread_deterministic_for_mtm_sidecar():
    params, dataframe, trade_start_idx, _ = _real_inputs()
    plan = build_grid_v2_plan(
        load_config(),
        GridV2Settings(
            enabled_variants=("bracket",),
            slow_enrich_selected=False,
            compiled_workers=1,
        ),
        base_params=params,
    )
    assert plan.deduped_candidate_count == 480
    hooks = GridV2StrategyHooks.from_strategy(s06_strategy)
    one = execute_grid_v2_candidates(
        plan,
        dataframe,
        trade_start_idx,
        hooks,
        compute_max_drawdown_mtm=True,
    )
    two = execute_grid_v2_candidates(
        replace(plan, settings=replace(plan.settings, compiled_workers=2)),
        dataframe,
        trade_start_idx,
        hooks,
        compute_max_drawdown_mtm=True,
    )

    one_values = np.asarray([row.max_drawdown_mtm_pct for row in one.rows])
    two_values = np.asarray([row.max_drawdown_mtm_pct for row in two.rows])
    assert one_values.shape == two_values.shape == (480,)
    assert np.array_equal(one_values, two_values, equal_nan=True)
