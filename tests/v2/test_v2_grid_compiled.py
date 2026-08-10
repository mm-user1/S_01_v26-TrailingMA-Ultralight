from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
import os

import numpy as np
import pandas as pd
import pytest

import core.engine_v2.compiled_kernel as compiled_kernel_module
from core.engine_v2.compiled_kernel import (
    OUTPUT_COLUMN_COUNT,
    OUTPUT_FINAL_BALANCE,
    OUTPUT_GROSS_LOSS,
    OUTPUT_GROSS_PROFIT,
    OUTPUT_LOSING_TRADES,
    OUTPUT_MAX_CONSECUTIVE_LOSSES,
    OUTPUT_MAX_DRAWDOWN_PCT,
    OUTPUT_NET_PROFIT_PCT,
    OUTPUT_PROFIT_FACTOR,
    OUTPUT_ROMAD,
    OUTPUT_SHARPE_DAILY,
    OUTPUT_SHARPE_DAILY_ACTIVE_DAYS,
    OUTPUT_SHARPE_DAILY_OBSERVATIONS,
    OUTPUT_SHARPE_RATIO,
    OUTPUT_SQN,
    OUTPUT_TOTAL_TRADES,
    OUTPUT_WINNING_TRADES,
    OUTPUT_WIN_RATE_PCT,
    _validated_worker_count,
    _timestamp_ns,
    _timestamps_ns,
    _pack_config_arrays,
    build_calendar_month_ids,
    build_stacked_execution_data,
    compiled_batch_available,
    evaluate_compiled_batch,
    evaluate_compiled_stacked_batch,
    pack_compiled_config_arrays_from_rows,
)
from core.engine_v2.contracts import Signals
from core.engine_v2.kernel import ExecutionData
from core.engine_v2.profile import parse_execution_profile
from core.engine_v2.runner import run_v2_strategy
from core.grid_engine import _grid_v2_result_from_row, _grid_v2_slow_result
from core.metrics import _advanced_metric_view, _calculate_monthly_returns, build_utc_day_ids
from core.grid_v2 import (
    GridV2Settings,
    GridV2StrategyHooks,
    _optional_compiled_count,
    _pack_table_config_arrays,
    build_grid_v2_plan,
    deterministic_candidate_subset_indices,
    estimate_grid_v2_cache,
    execute_grid_v2_candidates,
)
from strategies.s06_r_trend_v02_b2 import strategy as s06_b2_strategy
from strategies.s06_r_trend_v02_b2.strategy import load_config, normalized_params

from s06_b2_test_helpers import merged_reference_params, prepared_reference_dataset

COMPILED_SUBSET_LIMIT = 240


pytestmark = pytest.mark.skipif(
    not compiled_batch_available(),
    reason="Numba compiled V2 Grid path is unavailable in this process.",
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1, 1), (np.int64(6), 6), ("6", 6), (2.0, 2)],
)
def test_compiled_grid_v2_worker_count_accepts_integer_values(value, expected):
    assert _validated_worker_count(value) == expected


@pytest.mark.parametrize(
    "value",
    [0, -1, np.int64(0), 2.9, np.nan, np.inf, True, False, "2.9", "abc", ""],
)
def test_compiled_grid_v2_worker_count_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="positive integer"):
        _validated_worker_count(value)


@pytest.mark.parametrize(
    "values",
    [
        pd.date_range("2025-01-01", periods=3, freq="30min", tz="UTC"),
        pd.date_range("2025-01-01", periods=3, freq="30min"),
        np.array(
            ["2025-01-01T00:00:00.000000000", "2025-01-01T00:30:00.123456789", "NaT"],
            dtype="datetime64[ns]",
        ),
        tuple(pd.date_range("2025-01-01", periods=3, freq="30min", tz="UTC")),
        [
            pd.Timestamp("2025-01-01T00:00:00Z"),
            pd.Timestamp("2025-01-01 00:30:00"),
            np.datetime64("2025-01-01T01:00:00.123456789"),
            pd.NaT,
            np.datetime64("NaT"),
            None,
            "",
        ],
    ],
)
def test_compiled_timestamp_vectorization_matches_scalar_conversion(values):
    expected = np.array([_timestamp_ns(value, 0) for value in values], dtype=np.int64)

    assert np.array_equal(_timestamps_ns(values), expected)


@pytest.fixture(scope="module")
def prepared_data():
    return prepared_reference_dataset()


@pytest.fixture(scope="module")
def hooks():
    return GridV2StrategyHooks.from_strategy(s06_b2_strategy)


def _assert_float_equal(actual, expected):
    actual = float(actual)
    if expected is None:
        assert math.isnan(actual)
        return
    expected = float(expected)
    if math.isnan(expected):
        assert math.isnan(actual)
    elif math.isinf(expected):
        assert math.isinf(actual) and (actual > 0.0) == (expected > 0.0)
    else:
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12)


def _assert_rows_equal(compiled_row, reference_row):
    assert compiled_row.candidate_id == reference_row.candidate_id
    assert compiled_row.total_trades == reference_row.total_trades
    assert compiled_row.winning_trades == reference_row.winning_trades
    assert compiled_row.losing_trades == reference_row.losing_trades
    assert compiled_row.max_consecutive_losses == reference_row.max_consecutive_losses
    _assert_float_equal(compiled_row.net_profit_pct, reference_row.net_profit_pct)
    _assert_float_equal(compiled_row.max_drawdown_pct, reference_row.max_drawdown_pct)
    _assert_float_equal(compiled_row.romad, reference_row.romad)
    _assert_float_equal(compiled_row.profit_factor, reference_row.profit_factor)
    _assert_float_equal(compiled_row.win_rate_pct, reference_row.win_rate_pct)
    _assert_float_equal(compiled_row.gross_profit, reference_row.gross_profit)
    _assert_float_equal(compiled_row.gross_loss, reference_row.gross_loss)
    _assert_float_equal(compiled_row.final_balance, reference_row.final_balance)
    _assert_float_equal(compiled_row.sharpe_ratio, reference_row.sharpe_ratio)
    _assert_float_equal(compiled_row.sqn, reference_row.sqn)
    _assert_float_equal(compiled_row.sharpe_daily, reference_row.sharpe_daily)
    assert compiled_row.sharpe_daily_observations == reference_row.sharpe_daily_observations
    assert compiled_row.sharpe_daily_active_days == reference_row.sharpe_daily_active_days


def test_compiled_output_contract_appends_conditional_metrics():
    assert OUTPUT_SHARPE_RATIO == 21
    assert OUTPUT_SQN == 22
    assert OUTPUT_SHARPE_DAILY == 23
    assert OUTPUT_SHARPE_DAILY_OBSERVATIONS == 24
    assert OUTPUT_SHARPE_DAILY_ACTIVE_DAYS == 25
    assert OUTPUT_COLUMN_COUNT == 26


@pytest.mark.parametrize("value", [math.inf, -1.0, 1.5, float(np.iinfo(np.int64).max) * 2.0])
def test_compiled_daily_diagnostic_materialization_rejects_invalid_counts(value):
    with pytest.raises(ValueError, match="must be NaN|out of range"):
        _optional_compiled_count(value, "sharpe_daily_observations")


def test_compiled_daily_diagnostic_materialization_preserves_none_and_zero():
    assert _optional_compiled_count(math.nan, "sharpe_daily_observations") is None
    assert _optional_compiled_count(0.0, "sharpe_daily_observations") == 0


def test_calendar_month_ids_are_contiguous_int32_and_preserve_transitions():
    index = pd.DatetimeIndex(
        [
            "2024-12-31T23:30:00Z",
            "2025-01-01T00:00:00Z",
            "2025-01-31T23:30:00Z",
            "2025-02-01T00:00:00Z",
        ]
    )

    actual = build_calendar_month_ids(index)

    assert actual.dtype == np.int32
    assert actual.flags.c_contiguous
    assert actual.tolist() == [2024 * 12 + 12, 2025 * 12 + 1, 2025 * 12 + 1, 2025 * 12 + 2]


def test_grouped_compiled_sharpe_requires_complete_month_ids_before_dispatch(monkeypatch):
    data = _data(
        open_=[100.0, 101.0],
        high=[101.0, 102.0],
        low=[99.0, 100.0],
        close=[100.5, 101.5],
    )
    profile = parse_execution_profile(load_config())
    dispatched_month_ids = []

    def fake_loop(*args):
        dispatched_month_ids.append(args[5])
        args[-1].fill(np.nan)

    monkeypatch.setattr(compiled_kernel_module, "_COMPILED_BATCH_LOOP", fake_loop)
    empty = np.empty(0, dtype=np.int32)

    with pytest.raises(ValueError, match="month_ids matching the execution bars"):
        evaluate_compiled_batch(
            data=data,
            profile=profile,
            params_batch=[_edge_params()],
            trade_start_idx=0,
            compute_sharpe=True,
            month_ids=empty,
        )
    assert dispatched_month_ids == []

    with pytest.raises(ValueError, match="length must be zero or match"):
        evaluate_compiled_batch(
            data=data,
            profile=profile,
            params_batch=[_edge_params()],
            trade_start_idx=0,
            compute_sharpe=False,
            month_ids=np.array([2025 * 12 + 1], dtype=np.int32),
        )

    evaluate_compiled_batch(
        data=data,
        profile=profile,
        params_batch=[_edge_params()],
        trade_start_idx=0,
        compute_sharpe=False,
        month_ids=empty,
    )
    complete = build_calendar_month_ids(data.timestamps)
    evaluate_compiled_batch(
        data=data,
        profile=profile,
        params_batch=[_edge_params()],
        trade_start_idx=0,
        compute_sharpe=True,
        month_ids=complete,
    )

    assert len(dispatched_month_ids) == 2
    assert dispatched_month_ids[0].size == 0
    assert dispatched_month_ids[1].dtype == np.int32
    assert dispatched_month_ids[1].flags.c_contiguous
    assert dispatched_month_ids[1].size == len(data.timestamps)


def test_grouped_compiled_daily_sharpe_requires_complete_day_ids_before_dispatch(monkeypatch):
    data = _data(
        open_=[100.0, 101.0],
        high=[101.0, 102.0],
        low=[99.0, 100.0],
        close=[100.5, 101.5],
    )
    profile = parse_execution_profile(load_config())
    dispatched_day_ids = []

    def fake_loop(*args):
        dispatched_day_ids.append(args[6])
        args[-1].fill(np.nan)

    monkeypatch.setattr(compiled_kernel_module, "_COMPILED_BATCH_LOOP", fake_loop)
    with pytest.raises(ValueError, match="day_ids matching the execution bars"):
        evaluate_compiled_batch(
            data=data,
            profile=profile,
            params_batch=[merged_reference_params("reference_b_trend_bracket")],
            trade_start_idx=0,
            compute_sharpe_daily=True,
            day_ids=np.empty(0, dtype=np.int32),
        )
    assert dispatched_day_ids == []

    evaluate_compiled_batch(
        data=data,
        profile=profile,
        params_batch=[merged_reference_params("reference_b_trend_bracket")],
        trade_start_idx=0,
        compute_sharpe_daily=True,
        day_ids=build_utc_day_ids(data.timestamps),
    )
    assert dispatched_day_ids[0].dtype == np.int32
    assert dispatched_day_ids[0].size == len(data.timestamps)


def test_position_stacked_sharpe_requires_complete_month_ids_before_dispatch(monkeypatch):
    data = _data(
        open_=[100.0, 101.0],
        high=[101.0, 102.0],
        low=[99.0, 100.0],
        close=[100.5, 101.5],
    )
    profile = parse_execution_profile(load_config())
    params = [_edge_params()]
    dispatched_month_ids = []

    def fake_loop(*args):
        dispatched_month_ids.append(args[5])
        args[-1].fill(np.nan)

    monkeypatch.setattr(compiled_kernel_module, "_COMPILED_STACKED_BATCH_LOOP", fake_loop)
    empty_stacked = build_stacked_execution_data([data], [0])

    with pytest.raises(ValueError, match="month_ids matching the execution bars"):
        evaluate_compiled_stacked_batch(
            stacked_data=empty_stacked,
            profile=profile,
            params_batch=params,
            trade_start_idx=0,
            compute_sharpe=True,
        )
    assert dispatched_month_ids == []

    malformed_stacked = replace(
        empty_stacked,
        month_ids=np.array([2025 * 12 + 1], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="length must be zero or match"):
        evaluate_compiled_stacked_batch(
            stacked_data=malformed_stacked,
            profile=profile,
            params_batch=params,
            trade_start_idx=0,
            compute_sharpe=False,
        )

    evaluate_compiled_stacked_batch(
        stacked_data=empty_stacked,
        profile=profile,
        params_batch=params,
        trade_start_idx=0,
        compute_sharpe=False,
    )
    complete_stacked = build_stacked_execution_data(
        [data],
        [0],
        month_ids=build_calendar_month_ids(data.timestamps),
    )
    evaluate_compiled_stacked_batch(
        stacked_data=complete_stacked,
        profile=profile,
        params_batch=params,
        trade_start_idx=0,
        compute_sharpe=True,
    )

    assert len(dispatched_month_ids) == 2
    assert dispatched_month_ids[0].size == 0
    assert dispatched_month_ids[1].size == len(data.timestamps)


def test_position_cache_estimate_counts_fixed_output_and_optional_month_ids(prepared_data, hooks):
    df, trade_start_idx = prepared_data
    plan = build_grid_v2_plan(
        _config_with_rounding("none"),
        GridV2Settings(
            enabled_variants=("bracket",),
            enabled_axes=("stopX",),
            prefer_compiled=True,
            top_n=0,
        ),
        base_params=merged_reference_params("reference_b_trend_bracket"),
    )
    indices = (0, 1, 2)

    disabled = estimate_grid_v2_cache(plan, df, trade_start_idx, hooks, indices)
    sharpe = estimate_grid_v2_cache(
        plan,
        df,
        trade_start_idx,
        hooks,
        indices,
        compute_sharpe=True,
    )
    daily = estimate_grid_v2_cache(
        plan,
        df,
        trade_start_idx,
        hooks,
        indices,
        compute_sharpe_daily=True,
    )

    assert disabled.output_column_count == 26
    assert disabled.bytes_per_output_candidate == 26 * 8
    assert disabled.estimated_output_mb * 1024 * 1024 == len(indices) * 26 * 8
    assert disabled.month_id_nbytes == 0
    assert sharpe.month_id_nbytes == len(df) * 4
    assert daily.day_id_nbytes == len(df) * 4
    assert daily.month_id_nbytes == 0
    assert sharpe.bytes_per_shared_market_bar == disabled.bytes_per_shared_market_bar + 4
    assert (sharpe.estimated_shared_market_mb - disabled.estimated_shared_market_mb) * 1024 * 1024 == pytest.approx(len(df) * 4)


def _config_with_rounding(price_rounding: str) -> dict:
    config = deepcopy(load_config())
    config["execution"]["priceRounding"] = price_rounding
    return config


def _required_subset_indices(plan) -> tuple[int, ...]:
    first_by_variant: dict[str, int] = {}
    default_like: list[int] = []
    defaults = load_config()["parameters"]
    default_params = {
        name: spec.get("default")
        for name, spec in defaults.items()
        if isinstance(spec, dict) and "default" in spec
    }
    for candidate in plan.candidates:
        first_by_variant.setdefault(candidate.variant_name, candidate.candidate_id - 1)
        if all(candidate.params.get(name) == default_params.get(name) for name in candidate.axis_param_names):
            default_like.append(candidate.candidate_id - 1)
    return (
        0,
        len(plan.candidates) - 1,
        *first_by_variant.values(),
        *default_like,
    )


@pytest.mark.parametrize("price_rounding", ["none", "tick_outward"])
def test_table_config_packer_matches_mapping_packer_for_certified_topologies(price_rounding):
    base_params = merged_reference_params("reference_b_trend_bracket")
    plan = build_grid_v2_plan(
        _config_with_rounding(price_rounding),
        GridV2Settings(top_n=0),
        base_params=base_params,
    )
    indices = (0, 479, 480, 18_435, 48_479)
    params_batch = [
        normalized_params(plan.candidate_table.params_for_index(index))
        for index in indices
    ]
    expected = _pack_config_arrays(plan.profile, params_batch, trade_start_idx=1000)

    def get_value(row_index: int, name: str, default):
        candidate_index = indices[row_index]
        if plan.candidate_table.has_param_for_index(candidate_index, name):
            return plan.candidate_table.param_value_for_index(candidate_index, name)
        return default

    def get_modes(row_index: int):
        return plan.candidate_table.modes_for_index(indices[row_index])

    callback_actual = pack_compiled_config_arrays_from_rows(
        row_count=len(indices),
        get_value=get_value,
        get_modes=get_modes,
        trade_start_idx=1000,
    )
    actual = _pack_table_config_arrays(plan, indices, trade_start_idx=1000)

    assert expected.keys() == actual.keys()
    for name in expected:
        assert np.array_equal(actual[name], expected[name], equal_nan=True), name
        assert np.array_equal(callback_actual[name], expected[name], equal_nan=True), name


@pytest.mark.parametrize("price_rounding", ["none", "tick_outward"])
def test_compiled_grid_v2_batch_matches_reference_batch_for_certification_subset(
    prepared_data,
    hooks,
    price_rounding,
):
    assert os.environ.get("NUMBA_DISABLE_JIT") not in {"1", "true", "True"}
    df, trade_start_idx = prepared_data
    base_params = merged_reference_params("reference_b_trend_bracket")
    config = _config_with_rounding(price_rounding)
    compiled_plan = build_grid_v2_plan(
        config,
        GridV2Settings(
            prefer_compiled=True,
            top_n=6,
        ),
        base_params=base_params,
    )
    reference_plan = build_grid_v2_plan(
        config,
        GridV2Settings(
            prefer_compiled=False,
            top_n=6,
        ),
        base_params=base_params,
    )
    indices = deterministic_candidate_subset_indices(
        len(compiled_plan.candidates),
        COMPILED_SUBSET_LIMIT,
        required_indices=_required_subset_indices(compiled_plan),
    )

    compiled = execute_grid_v2_candidates(compiled_plan, df, trade_start_idx, hooks, indices)
    reference = execute_grid_v2_candidates(reference_plan, df, trade_start_idx, hooks, indices)

    assert compiled.metadata["backend_kind"] == "compiled_numba"
    assert compiled.metadata["compiled_batch_used"] is True
    assert reference.metadata["backend_kind"] == "reference"
    assert len(compiled.rows) == COMPILED_SUBSET_LIMIT
    assert {row.variant_name for row in compiled.rows} == {"bracket", "trail"}
    for compiled_row, reference_row in zip(compiled.rows, reference.rows):
        _assert_rows_equal(compiled_row, reference_row)


def test_compiled_grid_v2_worker_count_is_deterministic(prepared_data, hooks):
    df, trade_start_idx = prepared_data
    base_params = merged_reference_params("reference_b_trend_bracket")
    config = _config_with_rounding("none")
    settings = {
        "enabled_variants": ("bracket",),
        "enabled_axes": ("stopX", "stopRR"),
        "prefer_compiled": True,
        "top_n": 0,
    }
    one_worker_plan = build_grid_v2_plan(
        config,
        GridV2Settings(**settings, compiled_workers=1),
        base_params=base_params,
    )
    many_worker_plan = build_grid_v2_plan(
        config,
        GridV2Settings(**settings, compiled_workers=2),
        base_params=base_params,
    )
    indices = (0, 1, 5, len(one_worker_plan.candidates) - 1)

    one_worker = execute_grid_v2_candidates(one_worker_plan, df, trade_start_idx, hooks, indices)
    many_workers = execute_grid_v2_candidates(many_worker_plan, df, trade_start_idx, hooks, indices)

    assert one_worker.metadata["compiled_workers"] == 1
    assert many_workers.metadata["compiled_workers"] == 2
    for left, right in zip(one_worker.rows, many_workers.rows):
        _assert_rows_equal(left, right)


@pytest.mark.parametrize(
    ("compute_sharpe", "compute_sharpe_daily", "compute_sqn"),
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (True, True, False),
        (False, True, True),
        (True, True, True),
    ],
)
def test_position_fast_metrics_are_request_gated_and_match_reference(
    prepared_data,
    hooks,
    compute_sharpe,
    compute_sharpe_daily,
    compute_sqn,
):
    df, trade_start_idx = prepared_data
    base_params = merged_reference_params("reference_b_trend_bracket")
    common = dict(
        enabled_variants=("bracket",),
        enabled_axes=("stopX", "stopRR"),
        top_n=0,
    )
    compiled_plan = build_grid_v2_plan(
        _config_with_rounding("none"),
        GridV2Settings(**common, prefer_compiled=True, compiled_workers=2),
        base_params=base_params,
    )
    reference_plan = build_grid_v2_plan(
        _config_with_rounding("none"),
        GridV2Settings(**common, prefer_compiled=False),
        base_params=base_params,
    )
    indices = (0, 1, 5)

    compiled = execute_grid_v2_candidates(
        compiled_plan,
        df,
        trade_start_idx,
        hooks,
        indices,
        compute_sharpe=compute_sharpe,
        compute_sharpe_daily=compute_sharpe_daily,
        compute_sqn=compute_sqn,
    )
    reference = execute_grid_v2_candidates(
        reference_plan,
        df,
        trade_start_idx,
        hooks,
        indices,
        compute_sharpe=compute_sharpe,
        compute_sharpe_daily=compute_sharpe_daily,
        compute_sqn=compute_sqn,
    )

    assert compiled.metadata["compute_sharpe"] is compute_sharpe
    assert compiled.metadata["compute_sharpe_daily"] is compute_sharpe_daily
    assert compiled.metadata["compute_sqn"] is compute_sqn
    assert compiled.metadata["month_id_nbytes"] == (len(df) * 4 if compute_sharpe else 0)
    assert compiled.cache_estimate.month_id_nbytes == (len(df) * 4 if compute_sharpe else 0)
    assert compiled.metadata["day_id_nbytes"] == (len(df) * 4 if compute_sharpe_daily else 0)
    assert compiled.cache_estimate.day_id_nbytes == (len(df) * 4 if compute_sharpe_daily else 0)
    for compiled_row, reference_row in zip(compiled.rows, reference.rows):
        _assert_rows_equal(compiled_row, reference_row)
        assert math.isfinite(compiled_row.sharpe_ratio) is compute_sharpe
        assert math.isfinite(compiled_row.sqn) is compute_sqn
        assert (compiled_row.sharpe_daily_observations is not None) is compute_sharpe_daily
        assert (compiled_row.sharpe_daily_active_days is not None) is compute_sharpe_daily
    if not compute_sharpe:
        assert compiled.rows[0].sharpe_ratio is compiled.rows[1].sharpe_ratio
        assert reference.rows[0].sharpe_ratio is reference.rows[1].sharpe_ratio
    if not compute_sqn:
        assert compiled.rows[0].sqn is compiled.rows[1].sqn
        assert reference.rows[0].sqn is reference.rows[1].sqn
    materialized = _grid_v2_result_from_row(compiled.rows[0], metric_tier="compiled_fast")
    assert (math.isfinite(materialized.sharpe_ratio) if compute_sharpe else materialized.sharpe_ratio is None)
    assert (math.isfinite(materialized.sqn) if compute_sqn else materialized.sqn is None)
    assert (
        math.isfinite(materialized.sharpe_daily)
        if compute_sharpe_daily
        else materialized.sharpe_daily is None
    )
    assert materialized.fast_metrics["sharpe_ratio"] == materialized.sharpe_ratio
    assert materialized.fast_metrics["sqn"] == materialized.sqn
    assert materialized.fast_metrics["sharpe_daily"] == materialized.sharpe_daily


def test_position_bracket_compiled_sharpe_matches_reference_after_real_warmup(
    prepared_data,
    hooks,
):
    df, trade_start_idx = prepared_data
    plan = build_grid_v2_plan(
        _config_with_rounding("none"),
        GridV2Settings(
            enabled_variants=("bracket",),
            enabled_axes=("stopX", "stopRR"),
            prefer_compiled=True,
            top_n=0,
        ),
        base_params=merged_reference_params("reference_b_trend_bracket"),
    )
    compiled = execute_grid_v2_candidates(
        plan,
        df,
        trade_start_idx,
        hooks,
        (0,),
        compute_sharpe=True,
    )
    candidate = plan.candidate_for_index(0)
    params = hooks.normalize_params(dict(candidate.params))
    data = hooks.build_execution_data(df, params)
    reference = run_v2_strategy(
        data=data,
        profile=plan.profile,
        params=params,
        trade_start_idx=trade_start_idx,
    ).strategy_result
    view = _advanced_metric_view(reference)
    monthly_returns = _calculate_monthly_returns(
        view.equity_observations,
        view.timestamps,
        initial_equity=view.initial_equity,
    )
    evaluation_months = {
        (timestamp.year, timestamp.month) for timestamp in view.timestamps
    }
    prepared_months = {(timestamp.year, timestamp.month) for timestamp in df.index}
    compiled_row = compiled.rows[0]

    assert trade_start_idx == 1000
    assert df.index[0] < df.index[trade_start_idx]
    assert compiled_row.variant_name == "bracket"
    assert compiled_row.total_trades == reference.total_trades > 0
    assert math.isfinite(compiled_row.sharpe_ratio)
    assert compiled_row.sharpe_ratio == reference.sharpe_ratio
    assert reference.metric_start_idx == trade_start_idx
    assert reference.metric_initial_equity == 100.0
    assert len(reference.timestamps) == len(reference.equity_curve) == len(df)
    assert view.timestamps[0] == df.index[trade_start_idx]
    assert len(monthly_returns) == len(evaluation_months) < len(prepared_months)


def test_position_selected_reference_validation_compares_daily_ratio_and_counts(
    prepared_data,
    hooks,
):
    df, trade_start_idx = prepared_data
    plan = build_grid_v2_plan(
        _config_with_rounding("none"),
        GridV2Settings(
            enabled_variants=("bracket",),
            enabled_axes=("stopX", "stopRR"),
            prefer_compiled=True,
            top_n=0,
        ),
        base_params=merged_reference_params("reference_b_trend_bracket"),
    )
    fast_run = execute_grid_v2_candidates(
        plan,
        df,
        trade_start_idx,
        hooks,
        (0,),
        compute_sharpe_daily=True,
    )

    selected = _grid_v2_slow_result(
        plan=plan,
        df=df,
        trade_start_idx=trade_start_idx,
        hooks=hooks,
        row=fast_run.rows[0],
        compute_sharpe_daily=True,
        validation_tolerances={},  # Historical Grid Settings payload without additive keys.
        fail_on_error=True,
    )

    assert selected.validation_status == "passed"
    assert selected.validation_diffs["sharpe_daily"]["passed"] is True
    assert selected.validation_diffs["sharpe_daily_observations"]["passed"] is True
    assert selected.validation_diffs["sharpe_daily_active_days"]["passed"] is True


def test_sampled_position_grid_compiled_rows_match_reference(prepared_data, hooks):
    df, trade_start_idx = prepared_data
    base_params = merged_reference_params("reference_b_trend_bracket")
    common = {
        "enabled_variants": ("bracket",),
        "enabled_axes": ("stopX", "stopRR"),
        "planning_policy": "sampled",
        "requested_budget": 7,
        "seed": 77,
        "top_n": 2,
    }
    compiled_plan = build_grid_v2_plan(
        _config_with_rounding("none"),
        GridV2Settings(**common, prefer_compiled=True),
        base_params=base_params,
    )
    reference_plan = build_grid_v2_plan(
        _config_with_rounding("none"),
        GridV2Settings(**common, prefer_compiled=False),
        base_params=base_params,
    )

    compiled = execute_grid_v2_candidates(compiled_plan, df, trade_start_idx, hooks)
    reference = execute_grid_v2_candidates(reference_plan, df, trade_start_idx, hooks)

    assert compiled_plan.planned_candidate_count == 7
    assert compiled_plan.candidate_table.semantic_keys_by_row == reference_plan.candidate_table.semantic_keys_by_row
    assert compiled.metadata["compiled_config_packing"] == "mapping"
    for compiled_row, reference_row in zip(compiled.rows, reference.rows):
        _assert_rows_equal(compiled_row, reference_row)


def test_compiled_grid_v2_stacked_batch_matches_grouped_batch(prepared_data, hooks):
    df, trade_start_idx = prepared_data
    base_params = merged_reference_params("reference_b_trend_bracket")
    plan = build_grid_v2_plan(
        _config_with_rounding("none"),
        GridV2Settings(
            enabled_variants=("bracket",),
            enabled_axes=("stopX", "stopLP"),
            prefer_compiled=True,
            top_n=0,
        ),
        base_params=base_params,
    )
    candidates = tuple(plan.candidates[:6])
    params_batch = [
        hooks.normalize_params(dict(candidate.params)) if hooks.normalize_params else candidate.params
        for candidate in candidates
    ]
    data_rows = []
    row_by_stop_lp = {}
    data_index = []
    for candidate, params in zip(candidates, params_batch):
        stop_lp = candidate.params["stopLP"]
        if stop_lp not in row_by_stop_lp:
            row_by_stop_lp[stop_lp] = len(data_rows)
            data_rows.append(hooks.build_execution_data(df, params))
        data_index.append(row_by_stop_lp[stop_lp])

    grouped_outputs = []
    for params, row_index in zip(params_batch, data_index):
        grouped_outputs.append(
            evaluate_compiled_batch(
                data=data_rows[row_index],
                profile=plan.profile,
                params_batch=[params],
                trade_start_idx=trade_start_idx,
            ).outputs[0]
        )
    grouped = np.vstack(grouped_outputs)
    stacked_data = build_stacked_execution_data(data_rows, data_index)
    stacked = evaluate_compiled_stacked_batch(
        stacked_data=stacked_data,
        profile=plan.profile,
        params_batch=params_batch,
        trade_start_idx=trade_start_idx,
        n_workers=2,
    )

    assert stacked.execution_mode == "stacked"
    assert stacked_data.row_count == 2
    assert np.array_equal(stacked.outputs, grouped, equal_nan=True)


def test_compiled_grid_v2_stacked_batch_rejects_shared_market_mismatch():
    data_a = _data(
        open_=[100.0, 101.0],
        high=[101.0, 102.0],
        low=[99.0, 100.0],
        close=[100.5, 101.5],
    )
    data_b = _data(
        open_=[100.0, 102.0],
        high=[101.0, 102.0],
        low=[99.0, 100.0],
        close=[100.5, 101.5],
    )

    with pytest.raises(ValueError, match="shared OHLC/timestamps"):
        build_stacked_execution_data([data_a, data_b], [0, 1])


def _data(
    *,
    open_,
    high,
    low,
    close,
    long=None,
    short=None,
    atr=None,
    rolling_low=None,
    rolling_high=None,
    trail_long=None,
    trail_short=None,
):
    length = len(open_)
    return ExecutionData(
        timestamps=tuple(pd.date_range("2025-01-01", periods=length, freq="30min", tz="UTC")),
        open=np.array(open_, dtype=float),
        high=np.array(high, dtype=float),
        low=np.array(low, dtype=float),
        close=np.array(close, dtype=float),
        signals=Signals(
            long_entries=np.array(long if long is not None else [False] * length, dtype=bool),
            short_entries=np.array(short if short is not None else [False] * length, dtype=bool),
        ),
        atr=np.array(atr if atr is not None else [0.0] * length, dtype=float),
        rolling_low=np.array(rolling_low if rolling_low is not None else low, dtype=float),
        rolling_high=np.array(rolling_high if rolling_high is not None else high, dtype=float),
        trail_long=np.array(trail_long if trail_long is not None else [np.nan] * length, dtype=float),
        trail_short=np.array(trail_short if trail_short is not None else [np.nan] * length, dtype=float),
    )


def _edge_params(**overrides):
    params = normalized_params(
        {
            "entryMode": "Reversal @ Triangle",
            "enableLong": True,
            "enableShort": True,
            "fastLength": 21,
            "fastSmooth": 7,
            "slowLength": 112,
            "slowSmooth": 3,
            "thresholdOS": 20,
            "thresholdOB": 20,
            "stopX": 0.0,
            "stopRR": 2.0,
            "stopLP": 2,
            "stopMaxPct": 10.0,
            "stopMaxDays": 4,
            "riskPerTrade": 100.0,
            "contractSize": 1.0,
            "useTrailMA": False,
            "trailRR": 1.0,
            "trailMAType": "SMA",
            "trailMALength": 150,
            "trailMAOffsetEx": 0.0,
            "initialCapital": 100.0,
            "commissionPct": 0.0,
            "tickSize": 0.01,
            "dateFilter": False,
        }
    )
    params.update(overrides)
    return params


def test_empty_compiled_result_initializes_appended_metric_columns():
    data = _data(open_=[], high=[], low=[], close=[])
    profile = parse_execution_profile(load_config())

    values = evaluate_compiled_batch(
        data=data,
        profile=profile,
        params_batch=[_edge_params()],
        trade_start_idx=0,
        compute_sharpe=True,
        compute_sqn=True,
    ).outputs[0]

    assert values.shape == (26,)
    assert math.isnan(values[OUTPUT_SHARPE_RATIO])
    assert math.isnan(values[OUTPUT_SQN])
    assert math.isnan(values[OUTPUT_SHARPE_DAILY])
    assert math.isnan(values[OUTPUT_SHARPE_DAILY_OBSERVATIONS])
    assert math.isnan(values[OUTPUT_SHARPE_DAILY_ACTIVE_DAYS])


def test_position_daily_final_day_and_zero_trade_diagnostics_are_independent_of_ratio():
    data = _data(
        open_=[100.0, 100.0],
        high=[100.0, 100.0],
        low=[100.0, 100.0],
        close=[100.0, 100.0],
    )
    data = replace(
        data,
        timestamps=tuple(pd.DatetimeIndex(["2025-01-01T12:00:00Z", "2025-01-03T12:00:00Z"])),
    )
    profile = parse_execution_profile(load_config())

    disabled = evaluate_compiled_batch(
        data=data,
        profile=profile,
        params_batch=[_edge_params()],
        trade_start_idx=0,
    ).outputs[0]
    daily = evaluate_compiled_batch(
        data=data,
        profile=profile,
        params_batch=[_edge_params()],
        trade_start_idx=0,
        compute_sharpe_daily=True,
    ).outputs[0]
    invalid = evaluate_compiled_batch(
        data=data,
        profile=profile,
        params_batch=[_edge_params(initialCapital=0.0)],
        trade_start_idx=0,
        compute_sharpe_daily=True,
    ).outputs[0]

    assert math.isnan(disabled[OUTPUT_SHARPE_DAILY])
    assert math.isnan(disabled[OUTPUT_SHARPE_DAILY_OBSERVATIONS])
    assert math.isnan(disabled[OUTPUT_SHARPE_DAILY_ACTIVE_DAYS])
    assert math.isnan(daily[OUTPUT_SHARPE_DAILY])
    assert daily[OUTPUT_SHARPE_DAILY_OBSERVATIONS] == 2.0
    assert daily[OUTPUT_SHARPE_DAILY_ACTIVE_DAYS] == 0.0
    assert math.isnan(invalid[OUTPUT_SHARPE_DAILY])
    assert math.isnan(invalid[OUTPUT_SHARPE_DAILY_OBSERVATIONS])
    assert math.isnan(invalid[OUTPUT_SHARPE_DAILY_ACTIVE_DAYS])


def test_position_daily_empty_evaluation_matches_canonical_unavailable_tuple():
    data = _data(
        open_=[100.0, 100.0],
        high=[100.0, 100.0],
        low=[100.0, 100.0],
        close=[100.0, 100.0],
    )
    profile = parse_execution_profile(load_config())
    params = _edge_params()
    trade_start_idx = len(data.timestamps)

    compiled = evaluate_compiled_batch(
        data=data,
        profile=profile,
        params_batch=[params],
        trade_start_idx=trade_start_idx,
        compute_sharpe_daily=True,
    ).outputs[0]
    canonical = run_v2_strategy(
        data=data,
        profile=profile,
        params=params,
        trade_start_idx=trade_start_idx,
        compute_sharpe_daily=True,
    ).advanced_metrics

    compiled_tuple = (
        None if math.isnan(compiled[OUTPUT_SHARPE_DAILY]) else compiled[OUTPUT_SHARPE_DAILY],
        _optional_compiled_count(
            compiled[OUTPUT_SHARPE_DAILY_OBSERVATIONS],
            "sharpe_daily_observations",
        ),
        _optional_compiled_count(
            compiled[OUTPUT_SHARPE_DAILY_ACTIVE_DAYS],
            "sharpe_daily_active_days",
        ),
    )
    canonical_tuple = (
        canonical.sharpe_daily,
        canonical.sharpe_daily_observations,
        canonical.sharpe_daily_active_days,
    )
    assert compiled_tuple == canonical_tuple == (None, None, None)


@pytest.mark.parametrize(("trade_count", "expect_defined"), [(29, False), (30, True)])
def test_intrabar_target_exits_update_sqn_once_at_29_30_boundary(trade_count, expect_defined):
    length = trade_count * 2
    long_entries = [index % 2 == 0 for index in range(length)]
    rolling_low = [
        97.0 - float((index // 2) % 3) if index % 2 == 0 else 99.0
        for index in range(length)
    ]
    data = _data(
        open_=[100.0] * length,
        high=[100.0 if index % 2 == 0 else 110.0 for index in range(length)],
        low=[rolling_low[index] if index % 2 == 0 else 99.0 for index in range(length)],
        close=[100.0 if index % 2 == 0 else 105.0 for index in range(length)],
        long=long_entries,
        rolling_low=rolling_low,
    )
    profile = parse_execution_profile(load_config())
    params = _edge_params(stopRR=1.0)

    compiled = evaluate_compiled_batch(
        data=data,
        profile=profile,
        params_batch=[params],
        trade_start_idx=0,
        compute_sqn=True,
    ).outputs[0]
    reference = run_v2_strategy(
        data=data,
        profile=profile,
        params=params,
        trade_start_idx=0,
    ).strategy_result

    assert int(compiled[OUTPUT_TOTAL_TRADES]) == reference.total_trades == trade_count
    if expect_defined:
        assert math.isfinite(compiled[OUTPUT_SQN])
        _assert_float_equal(compiled[OUTPUT_SQN], reference.sqn)
    else:
        assert math.isnan(compiled[OUTPUT_SQN])
        assert reference.sqn is None


def test_sharpe_assigns_first_new_month_bar_to_new_month():
    data = _data(
        open_=[100.0, 100.0, 105.0],
        high=[100.0, 101.0, 106.0],
        low=[97.0, 99.0, 104.0],
        close=[100.0, 101.0, 105.0],
        long=[True, False, False],
        rolling_low=[97.0, 99.0, 104.0],
    )
    data = replace(
        data,
        timestamps=(
            pd.Timestamp("2025-01-31T23:30:00Z"),
            pd.Timestamp("2025-02-01T00:00:00Z"),
            pd.Timestamp("2025-03-01T00:00:00Z"),
        ),
    )
    profile = parse_execution_profile(load_config())
    params = _edge_params(stopRR=10.0)

    compiled = evaluate_compiled_batch(
        data=data,
        profile=profile,
        params_batch=[params],
        trade_start_idx=0,
        compute_sharpe=True,
    ).outputs[0]
    reference = run_v2_strategy(
        data=data,
        profile=profile,
        params=params,
        trade_start_idx=0,
    ).strategy_result
    metric_view = _advanced_metric_view(reference)
    monthly_returns = _calculate_monthly_returns(
        metric_view.equity_observations,
        metric_view.timestamps,
        initial_equity=metric_view.initial_equity,
    )

    assert len(monthly_returns) == 3
    assert monthly_returns[-1] != 0.0
    _assert_float_equal(compiled[OUTPUT_SHARPE_RATIO], reference.sharpe_ratio)


def _assert_compiled_matches_direct_reference(data: ExecutionData, params: dict):
    profile = parse_execution_profile(load_config())
    grouped = evaluate_compiled_batch(
        data=data,
        profile=profile,
        params_batch=[params],
        trade_start_idx=0,
    ).outputs[0]
    stacked = evaluate_compiled_stacked_batch(
        stacked_data=build_stacked_execution_data([data], [0]),
        profile=profile,
        params_batch=[params],
        trade_start_idx=0,
    ).outputs[0]
    assert np.array_equal(grouped, stacked, equal_nan=True)
    compiled = stacked
    reference = run_v2_strategy(
        data=data,
        profile=profile,
        params=params,
        trade_start_idx=0,
    ).strategy_result

    _assert_float_equal(compiled[OUTPUT_NET_PROFIT_PCT], reference.net_profit_pct)
    _assert_float_equal(compiled[OUTPUT_MAX_DRAWDOWN_PCT], reference.max_drawdown_pct)
    _assert_float_equal(compiled[OUTPUT_ROMAD], reference.romad)
    _assert_float_equal(compiled[OUTPUT_PROFIT_FACTOR], reference.profit_factor)
    reference_win_rate = (
        float(reference.winning_trades) / float(reference.total_trades) * 100.0
        if reference.total_trades
        else 0.0
    )
    _assert_float_equal(compiled[OUTPUT_WIN_RATE_PCT], reference_win_rate)
    _assert_float_equal(compiled[OUTPUT_GROSS_PROFIT], reference.gross_profit)
    _assert_float_equal(compiled[OUTPUT_GROSS_LOSS], reference.gross_loss)
    _assert_float_equal(compiled[OUTPUT_FINAL_BALANCE], reference.balance_curve[-1])
    assert int(compiled[OUTPUT_TOTAL_TRADES]) == reference.total_trades
    assert int(compiled[OUTPUT_WINNING_TRADES]) == reference.winning_trades
    assert int(compiled[OUTPUT_LOSING_TRADES]) == reference.losing_trades

    consecutive = 0
    max_consecutive = 0
    for trade in reference.trades:
        if trade.net_pnl <= 0.0:
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0
    assert int(compiled[OUTPUT_MAX_CONSECUTIVE_LOSSES]) == max_consecutive


def test_compiled_grid_v2_edge_cases_match_direct_reference_runner():
    no_trade = _data(
        open_=[100.0, 100.0, 100.0],
        high=[100.0, 101.0, 101.0],
        low=[99.0, 99.0, 99.0],
        close=[100.0, 100.0, 100.0],
    )
    _assert_compiled_matches_direct_reference(no_trade, _edge_params())

    zero_loss = _data(
        open_=[100.0, 100.0],
        high=[100.0, 106.0],
        low=[97.0, 99.0],
        close=[100.0, 105.0],
        long=[True, False],
        rolling_low=[97.0, 99.0],
    )
    _assert_compiled_matches_direct_reference(zero_loss, _edge_params(stopRR=1.0))

    max_days_strict_boundary = _data(
        open_=[100.0, 100.0, 101.0],
        high=[100.0, 101.0, 101.0],
        low=[97.0, 99.0, 99.0],
        close=[100.0, 100.0, 102.0],
        long=[True, False, False],
        rolling_low=[97.0, 99.0, 99.0],
    )
    _assert_compiled_matches_direct_reference(
        max_days_strict_boundary,
        _edge_params(stopRR=10.0, stopMaxDays=1.0 / 48.0),
    )

    episodic_drawdown = _data(
        open_=[100.0, 100.0, 100.0, 108.0, 108.0],
        high=[100.0, 106.0, 100.0, 108.0, 109.0],
        low=[97.0, 99.0, 97.0, 102.0, 102.0],
        close=[100.0, 105.0, 100.0, 108.0, 102.0],
        long=[True, False, True, False, False],
        rolling_low=[97.0, 99.0, 97.0, 102.0, 102.0],
    )
    _assert_compiled_matches_direct_reference(episodic_drawdown, _edge_params(stopRR=1.0))
