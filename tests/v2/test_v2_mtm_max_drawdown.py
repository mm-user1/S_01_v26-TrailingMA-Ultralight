from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from core.engine_v2.compiled_kernel import (
    OUTPUT_COLUMN_COUNT,
    build_stacked_execution_data,
    evaluate_compiled_batch,
    evaluate_compiled_stacked_batch,
)
from core.engine_v2.contracts import Signals
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


def test_requested_signal_reversal_fails_before_execution(monkeypatch):
    params = s03_params(REFERENCE_A)
    dataframe, trade_start_idx = s03_dataset()
    plan = build_grid_v2_plan(
        s03_strategy.load_config(),
        GridV2Settings(enabled_axes=(), slow_enrich_selected=False),
        base_params=params,
    )
    hooks = GridV2StrategyHooks.from_strategy(s03_strategy)
    monkeypatch.setattr(
        "core.grid_v2._candidate_cache_keys",
        lambda *args, **kwargs: pytest.fail("execution planning must not start"),
    )

    with pytest.raises(ValueError, match="position-family execution topology"):
        execute_grid_v2_candidates(
            plan,
            dataframe,
            trade_start_idx,
            hooks,
            compute_max_drawdown_mtm=True,
        )


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
