"""Reference-only certification tests for UTC Daily Sharpe."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from core.backtest_engine import StrategyResult, TradeRecord
from core.metrics import (
    _advanced_metric_view,
    _calculate_daily_returns,
    _calculate_sharpe_daily_value,
    _utc_day_ids,
    calculate_advanced,
)
from core.walkforward_engine import WalkForwardEngine


def _loop_daily_returns(equity, timestamps, anchor):
    """Independent literal oracle; deliberately shares no production grouping code."""
    normalized = []
    for timestamp in timestamps:
        parsed = pd.Timestamp(timestamp)
        parsed = parsed.tz_localize("UTC") if parsed.tzinfo is None else parsed.tz_convert("UTC")
        normalized.append(parsed)

    closes = []
    current_date = normalized[0].date()
    previous_equity = None
    for value, timestamp in zip(equity, normalized):
        if timestamp.date() != current_date:
            closes.append(float(previous_equity))
            current_date = timestamp.date()
        previous_equity = value
    closes.append(float(previous_equity))

    returns = []
    opening = float(anchor)
    for close in closes:
        returns.append(close / opening - 1.0)
        opening = close
    return np.asarray(returns, dtype=float)


def _result(equity, timestamps, *, anchor=100.0, trades=True, metric_start_idx=0):
    trade_records = [TradeRecord(net_pnl=1.0)] if trades else []
    return StrategyResult(
        trades=trade_records,
        equity_curve=list(equity),
        balance_curve=list(equity),
        timestamps=list(timestamps),
        metric_start_idx=metric_start_idx,
        metric_initial_equity=anchor,
    )


def test_hand_computable_daily_sharpe_uses_fractional_rf_ddof0_and_sqrt365():
    returns = np.asarray([0.01, -0.02, 0.03], dtype=float)
    risk_free_rate = 0.0365
    expected = math.sqrt(365.0) * (
        float(np.mean(returns)) - risk_free_rate / 365.0
    ) / float(np.std(returns, ddof=0))

    assert _calculate_sharpe_daily_value(returns, risk_free_rate) == pytest.approx(expected)
    assert _calculate_sharpe_daily_value(-returns, risk_free_rate) < 0.0
    assert _calculate_sharpe_daily_value([0.01]) is None
    assert _calculate_sharpe_daily_value([0.01, 0.01]) is None
    assert _calculate_sharpe_daily_value([0.01, math.nan]) is None


def test_utc_day_ids_normalize_naive_and_aware_timestamps_exactly():
    timestamps = [
        "2025-01-01T23:30:00",
        "2025-01-02T01:30:00+02:00",
        "2025-01-02T00:00:00Z",
    ]
    ids = _utc_day_ids(timestamps)

    assert ids.dtype == np.int32
    assert ids.tolist() == [20089, 20089, 20090]


@pytest.mark.parametrize(
    "representation",
    [
        pytest.param("datetime_index_ns", id="datetime-index-ns"),
        pytest.param("datetime_index_us", id="datetime-index-us"),
        pytest.param("datetime_index_s", id="datetime-index-s"),
        pytest.param("numpy_s", id="numpy-datetime64-s"),
        pytest.param("timestamp_list", id="production-timestamp-list"),
    ],
)
def test_utc_day_ids_are_resolution_and_representation_independent(representation):
    timestamp_text = [
        "1969-12-31T23:59:59",
        "1970-01-01T00:00:00",
        "2025-01-01T23:59:59",
        "2025-01-02T00:00:00",
    ]
    datetime_index = pd.DatetimeIndex(timestamp_text)
    if representation == "datetime_index_ns":
        timestamps = datetime_index.as_unit("ns")
    elif representation == "datetime_index_us":
        timestamps = datetime_index.as_unit("us")
    elif representation == "datetime_index_s":
        timestamps = datetime_index.as_unit("s")
    elif representation == "numpy_s":
        timestamps = np.asarray(timestamp_text, dtype="datetime64[s]")
    else:
        timestamps = [pd.Timestamp(value) for value in timestamp_text]

    ids = _utc_day_ids(timestamps)

    assert ids.dtype == np.int32
    assert ids.tolist() == [-1, 0, 20089, 20090]


@pytest.mark.parametrize(
    "timestamps",
    [[pd.NaT], ["not-a-timestamp"]],
)
def test_daily_timestamp_failures_are_actionable(timestamps):
    with pytest.raises(ValueError, match="Daily Sharpe timestamps"):
        _utc_day_ids(timestamps)


def test_enabled_metric_rejects_bad_timestamp_before_monthly_calendar_work():
    result = _result([101.0], ["not-a-timestamp"])

    with pytest.raises(ValueError, match="Daily Sharpe timestamps"):
        calculate_advanced(result, compute_sharpe_daily=True)


def test_vectorized_day_closes_match_independent_loop_and_retain_edges_and_gaps():
    timestamps = pd.to_datetime(
        [
            "2025-01-01T10:00:00Z",
            "2025-01-01T23:30:00Z",
            "2025-01-02T00:00:00Z",
            "2025-01-02T23:30:00Z",
            "2025-01-04T12:00:00Z",
        ]
    )
    equity = [101.0, 102.0, 103.0, 104.0, 108.0]

    actual = _calculate_daily_returns(equity, timestamps, initial_equity=100.0)
    oracle = _loop_daily_returns(equity, timestamps, 100.0)

    assert actual is not None
    assert actual == pytest.approx(oracle)
    assert actual == pytest.approx([0.02, 104.0 / 102.0 - 1.0, 108.0 / 104.0 - 1.0])
    assert float(np.prod(1.0 + actual)) == pytest.approx(1.08, rel=1e-12, abs=1e-12)


def test_source_order_is_preserved_instead_of_sorted_or_repaired():
    timestamps = pd.to_datetime(
        ["2025-01-02T12:00:00Z", "2025-01-01T12:00:00Z", "2025-01-01T13:00:00Z"]
    )
    equity = [110.0, 99.0, 101.0]

    actual = _calculate_daily_returns(equity, timestamps, initial_equity=100.0)

    assert actual == pytest.approx([0.10, 101.0 / 110.0 - 1.0])


@pytest.mark.parametrize(
    ("equity", "anchor"),
    [
        ([math.nan, 101.0], 100.0),
        ([math.inf, 101.0], 100.0),
        ([0.0, 101.0], 100.0),
        ([101.0, 102.0], 0.0),
        ([101.0, 102.0], math.nan),
    ],
)
def test_invalid_required_values_suppress_the_complete_daily_series(equity, anchor):
    timestamps = pd.to_datetime(["2025-01-01T12:00:00Z", "2025-01-02T12:00:00Z"])
    assert _calculate_daily_returns(equity, timestamps, initial_equity=anchor) is None


def test_diagnostics_are_gated_and_independent_of_ratio_availability():
    timestamps = pd.to_datetime(
        ["2025-01-01T12:00:00Z", "2025-01-02T12:00:00Z", "2025-01-03T12:00:00Z"]
    )
    result = _result([100.0, 100.0 + 5e-11, 100.0], timestamps, trades=False)

    disabled = calculate_advanced(result)
    enabled = calculate_advanced(result, compute_sharpe_daily=True)

    assert disabled.sharpe_daily is None
    assert disabled.sharpe_daily_observations is None
    assert disabled.sharpe_daily_active_days is None
    assert enabled.sharpe_daily is None
    assert enabled.sharpe_daily_observations == 3
    assert enabled.sharpe_daily_active_days == 0


def test_no_active_day_gate_and_no_completed_trade_gate_only_the_ratio():
    timestamps = pd.to_datetime(
        ["2025-01-01T12:00:00Z", "2025-01-02T12:00:00Z", "2025-01-03T12:00:00Z"]
    )
    equity = [100.0, 101.0, 100.5]

    with_trade = calculate_advanced(
        _result(equity, timestamps, trades=True), compute_sharpe_daily=True
    )
    without_trade = calculate_advanced(
        _result(equity, timestamps, trades=False), compute_sharpe_daily=True
    )

    assert with_trade.sharpe_daily is not None
    assert with_trade.sharpe_daily_observations == 3
    assert with_trade.sharpe_daily_active_days == 2
    assert without_trade.sharpe_daily is None
    assert without_trade.sharpe_daily_observations == 3
    assert without_trade.sharpe_daily_active_days == 2


def test_metric_boundary_excludes_warmup_days_and_uses_canonical_anchor():
    timestamps = pd.to_datetime(
        [
            "2024-12-30T12:00:00Z",
            "2024-12-31T12:00:00Z",
            "2025-01-01T12:00:00Z",
            "2025-01-02T12:00:00Z",
        ]
    )
    result = _result(
        [70.0, 80.0, 102.0, 101.0],
        timestamps,
        anchor=100.0,
        metric_start_idx=2,
    )

    advanced = calculate_advanced(result, compute_sharpe_daily=True)
    expected = _calculate_sharpe_daily_value([0.02, 101.0 / 102.0 - 1.0])

    assert advanced.sharpe_daily == pytest.approx(expected)
    assert advanced.sharpe_daily_observations == 2
    assert advanced.sharpe_daily_active_days == 2


def test_daily_sharpe_is_exactly_invariant_to_technical_warmup_prefix():
    evaluation_timestamps = pd.to_datetime(
        [
            "2025-03-01T00:00:00Z",
            "2025-03-31T23:30:00Z",
            "2025-04-01T00:00:00Z",
            "2025-05-15T00:00:00Z",
        ]
    )
    evaluation_equity = [99.5, 105.0, 102.0, 115.0]
    snapshots = []
    for prefix_len, frequency in ((0, "30min"), (100, "15min"), (1000, "1D")):
        prefix_timestamps = (
            pd.date_range(
                end=evaluation_timestamps[0] - pd.Timedelta(frequency),
                periods=prefix_len,
                freq=frequency,
            )
            if prefix_len
            else pd.DatetimeIndex([])
        )
        result = _result(
            [100.0] * prefix_len + evaluation_equity,
            list(prefix_timestamps) + list(evaluation_timestamps),
            metric_start_idx=prefix_len,
        )
        view = _advanced_metric_view(result)
        advanced = calculate_advanced(result, compute_sharpe_daily=True)

        assert view.initial_equity == 100.0
        assert np.array_equal(view.equity_observations, evaluation_equity)
        assert list(view.timestamps) == list(evaluation_timestamps)
        snapshots.append(
            (
                advanced.sharpe_daily,
                advanced.sharpe_daily_observations,
                advanced.sharpe_daily_active_days,
            )
        )

    assert snapshots[1] == snapshots[0]
    assert snapshots[2] == snapshots[0]


def test_nonfinite_nonclosing_intraday_equity_invalidates_complete_daily_series():
    timestamps = pd.to_datetime(
        [
            "2025-01-30T12:00:00Z",
            "2025-01-30T18:00:00Z",
            "2025-01-30T23:00:00Z",
            "2025-01-31T23:00:00Z",
            "2025-02-01T12:00:00Z",
            "2025-02-28T23:00:00Z",
            "2025-03-01T12:00:00Z",
        ]
    )
    result = _result([101.0, math.nan, 102.0, 103.0, 104.0, 105.0, 106.0], timestamps)

    disabled = calculate_advanced(result)
    enabled = calculate_advanced(result, compute_sharpe_daily=True)

    assert enabled.sharpe_daily is None
    assert enabled.sharpe_daily_observations is None
    assert enabled.sharpe_daily_active_days is None
    for name, expected in disabled.to_dict().items():
        if name.startswith("sharpe_daily"):
            continue
        actual = getattr(enabled, name)
        if isinstance(expected, float) and math.isnan(expected):
            assert isinstance(actual, float) and math.isnan(actual)
        else:
            assert actual == expected


def test_explicit_anchor_captures_first_bar_loss_and_inclusive_midnight_last_day():
    timestamps = pd.to_datetime(
        [
            "2025-01-01T23:30:00Z",
            "2025-01-02T00:00:00Z",
            "2025-01-02T23:30:00Z",
            "2025-01-03T00:00:00Z",
        ]
    )
    returns = _calculate_daily_returns(
        [99.0, 98.0, 101.0, 100.0],
        timestamps,
        initial_equity=100.0,
    )

    assert returns == pytest.approx([-0.01, 101.0 / 99.0 - 1.0, 100.0 / 101.0 - 1.0])
    assert len(returns) == 3


def test_legacy_boundary_zero_uses_first_observation_as_deterministic_anchor():
    timestamps = pd.to_datetime(
        ["2025-01-01T12:00:00Z", "2025-01-01T23:00:00Z", "2025-01-02T12:00:00Z"]
    )
    result = StrategyResult(
        trades=[TradeRecord(net_pnl=1.0)],
        equity_curve=[90.0, 99.0, 101.0],
        balance_curve=[90.0, 99.0, 101.0],
        timestamps=list(timestamps),
    )

    advanced = calculate_advanced(result, compute_sharpe_daily=True)
    expected = _calculate_sharpe_daily_value([0.10, 101.0 / 99.0 - 1.0])

    assert advanced.sharpe_daily == pytest.approx(expected)
    assert advanced.sharpe_daily_observations == 2


@pytest.mark.parametrize("frequency", ["15min", "30min", "1h"])
def test_supported_intraday_frequencies_emit_one_return_per_observed_utc_day(frequency):
    timestamps = pd.date_range("2025-01-01T06:00:00Z", "2025-01-03T00:00:00Z", freq=frequency)
    equity = np.linspace(100.0, 103.0, len(timestamps))

    returns = _calculate_daily_returns(equity, timestamps, initial_equity=100.0)

    assert returns is not None
    assert len(returns) == 3
    assert float(np.prod(1.0 + returns)) == pytest.approx(1.03, rel=1e-12, abs=1e-12)


def test_invalid_daily_denominator_does_not_change_monthly_skip_semantics():
    timestamps = pd.to_datetime(
        ["2025-01-31T12:00:00Z", "2025-02-01T12:00:00Z", "2025-02-02T12:00:00Z"]
    )
    result = _result([0.0, 101.0, 102.0], timestamps)

    enabled = calculate_advanced(result, compute_sharpe_daily=True)
    disabled = calculate_advanced(result)

    assert enabled.sharpe_daily is None
    assert enabled.sharpe_daily_observations is None
    assert enabled.sharpe_daily_active_days is None
    assert enabled.sharpe_ratio == disabled.sharpe_ratio
    assert enabled.sortino_ratio == disabled.sortino_ratio


def test_sparse_delayed_oos_prefix_does_not_synthesize_missing_calendar_days():
    live = _result(
        [101.0, 102.0],
        pd.to_datetime(["2025-01-11T00:00:00Z", "2025-01-12T00:00:00Z"]),
        trades=False,
    )
    prefixed = WalkForwardEngine._prepend_flat_prefix(
        live,
        scheduled_start=pd.Timestamp("2025-01-01T00:00:00Z"),
        live_start=pd.Timestamp("2025-01-11T00:00:00Z"),
    )

    advanced = calculate_advanced(prefixed, compute_sharpe_daily=True)

    assert prefixed.timestamps == list(
        pd.to_datetime(
            ["2025-01-01T00:00:00Z", "2025-01-11T00:00:00Z", "2025-01-12T00:00:00Z"]
        )
    )
    assert advanced.sharpe_daily_observations == 3
    assert advanced.sharpe_daily_active_days == 2


def test_sparse_flat_wfa_result_has_only_endpoint_observations_and_zero_active_days():
    flat = WalkForwardEngine._build_flat_result(
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-03-01T00:00:00Z"),
    )

    advanced = calculate_advanced(flat, compute_sharpe_daily=True)

    assert len(flat.timestamps) == 2
    assert advanced.sharpe_daily is None
    assert advanced.sharpe_daily_observations == 2
    assert advanced.sharpe_daily_active_days == 0
