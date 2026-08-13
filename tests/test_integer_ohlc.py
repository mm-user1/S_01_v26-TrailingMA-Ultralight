from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd
import pytest

from core import metrics
from core.backtest_engine import load_data
from strategies.s06_r_trend_v02 import strategy as s06_v1
from strategies.s06_r_trend_v02.strategy import S06Params, S06RTrendV02
from strategies.s06_r_trend_v02_b2.signals import (
    S06B2Params,
    build_indicator_arrays,
    build_s06_b2_execution_data,
)
from strategies.s06_r_trend_v02_regime_trendlines_b2.signals import (
    S06RegimeTLParams,
    build_regime_tl_execution_data,
)


OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _csv_text(rows: list[tuple[object, object, object, object, object, object]]) -> str:
    lines = ["time,open,high,low,close,Volume"]
    lines.extend(",".join(str(value) for value in row) for row in rows)
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize(
    "rows",
    [
        [
            (1735691400, 101, 103, 99, 102, 11),
            (1735689600, 100, 102, 98, 101, 10),
        ],
        [
            (1735691400, 101.5, 103.5, 99.5, 102.5, 11.5),
            (1735689600, 100.5, 102.5, 98.5, 101.5, 10.5),
        ],
        [
            (1735691400, 101, 103.5, 99, 102.5, 11),
            (1735689600, 100, 102.5, 98, 101.5, 10.5),
        ],
        [
            (1735691400, "1.01e2", "1.03e2", "9.9e1", "1.02e2", "1.1e1"),
            (1735689600, "1.00e2", "1.02e2", "9.8e1", "1.01e2", "1.0e1"),
        ],
    ],
    ids=["integer", "float", "mixed", "numeric_strings"],
)
def test_load_data_returns_sorted_utc_float64_ohlcv(rows):
    loaded = load_data(StringIO(_csv_text(rows)))

    assert list(loaded.columns) == OHLCV_COLUMNS
    assert loaded.index.name == "time"
    assert str(loaded.index.dtype) == "datetime64[ns, UTC]"
    assert loaded.index.is_monotonic_increasing
    assert len(loaded) == 2
    assert all(loaded[column].dtype == np.dtype("float64") for column in OHLCV_COLUMNS)
    assert loaded.iloc[0].tolist() == [float(value) for value in rows[1][1:]]


def test_load_data_rejects_non_numeric_ohlcv_text():
    rows = [(1735689600, "abc", 102, 98, 101, 10)]

    with pytest.raises(ValueError, match="could not convert string to float"):
        load_data(StringIO(_csv_text(rows)))


def test_load_data_preserves_existing_blank_and_na_token_behavior():
    text = (
        "time,open,high,low,close,Volume\n"
        "1735689600,100,102,98,101,\n"
        "1735691400,101,103,99,NA,11\n"
    )

    loaded = load_data(StringIO(text))

    assert all(loaded[column].dtype == np.dtype("float64") for column in OHLCV_COLUMNS)
    assert np.isnan(loaded.iloc[0]["Volume"])
    assert np.isnan(loaded.iloc[1]["Close"])


def _integer_frame(length: int = 240) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=length, freq="30min", tz="UTC")
    x = np.arange(length, dtype=np.int64)
    close = 100 + ((x * 7) % 19)
    return pd.DataFrame(
        {
            "Open": close + ((x % 3) - 1),
            "High": close + 3,
            "Low": close - 3,
            "Close": close,
            "Volume": 1000 + x,
        },
        index=index,
    )


def _assert_float_array_equivalence(integer_values, float_values) -> None:
    assert integer_values.dtype == np.dtype("float64")
    assert float_values.dtype == np.dtype("float64")
    np.testing.assert_allclose(integer_values, float_values, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_s06_b2_integer_and_float_preparation_are_equivalent():
    integer_df = _integer_frame()
    float_df = integer_df.astype("float64")
    params = S06B2Params()

    integer_arrays = build_indicator_arrays(integer_df, params)
    float_arrays = build_indicator_arrays(float_df, params)
    for name in ("atr", "rolling_low", "rolling_high", "trail_long", "trail_short"):
        _assert_float_array_equivalence(integer_arrays[name], float_arrays[name])
    for name in ("long_signal", "short_signal"):
        np.testing.assert_array_equal(integer_arrays[name], float_arrays[name])

    integer_data = build_s06_b2_execution_data(integer_df, params)
    float_data = build_s06_b2_execution_data(float_df, params)
    for name in ("open", "high", "low", "close", "atr", "rolling_low", "rolling_high", "trail_long", "trail_short"):
        _assert_float_array_equivalence(getattr(integer_data, name), getattr(float_data, name))
    np.testing.assert_array_equal(integer_data.signals.long_entries, float_data.signals.long_entries)
    np.testing.assert_array_equal(integer_data.signals.short_entries, float_data.signals.short_entries)


def test_s06_regime_tl_inherits_integer_ohlc_safety():
    integer_df = _integer_frame()
    float_df = integer_df.astype("float64")
    params = S06RegimeTLParams()

    integer_data = build_regime_tl_execution_data(integer_df, params)
    float_data = build_regime_tl_execution_data(float_df, params)

    for name in ("open", "high", "low", "close", "atr", "rolling_low", "rolling_high", "trail_long", "trail_short"):
        _assert_float_array_equivalence(getattr(integer_data, name), getattr(float_data, name))
    np.testing.assert_array_equal(integer_data.signals.long_entries, float_data.signals.long_entries)
    np.testing.assert_array_equal(integer_data.signals.short_entries, float_data.signals.short_entries)


def _v1_trade_signature(result) -> tuple:
    return tuple(
        (
            trade.direction,
            trade.entry_time,
            trade.exit_time,
            trade.entry_price,
            trade.exit_price,
            trade.size,
            trade.net_pnl,
        )
        for trade in result.trades
    )


def test_s06_v1_integer_and_float_preparation_execution_and_metrics_are_equivalent():
    integer_df = _integer_frame()
    float_df = integer_df.astype("float64")
    parsed = S06Params(use_date_filter=False, contractSize=1.0)

    integer_arrays = s06_v1._build_strategy_arrays(integer_df, parsed)
    float_arrays = s06_v1._build_strategy_arrays(float_df, parsed)
    for name in ("atr", "lowest", "highest", "trail_long", "trail_short"):
        _assert_float_array_equivalence(getattr(integer_arrays, name), getattr(float_arrays, name))
    np.testing.assert_array_equal(integer_arrays.long_signal, float_arrays.long_signal)
    np.testing.assert_array_equal(integer_arrays.short_signal, float_arrays.short_signal)

    params = {
        "dateFilter": False,
        "contractSize": 1.0,
        "initialCapital": 100.0,
    }
    integer_result = S06RTrendV02.run(integer_df, params, trade_start_idx=0)
    float_result = S06RTrendV02.run(float_df, params, trade_start_idx=0)
    assert _v1_trade_signature(integer_result) == _v1_trade_signature(float_result)
    integer_metrics = metrics.calculate_basic(integer_result, initial_balance=100.0)
    float_metrics = metrics.calculate_basic(float_result, initial_balance=100.0)
    assert integer_metrics == float_metrics


def test_wfa_style_float_csv_round_trip_is_value_dtype_index_equivalent():
    expected = _integer_frame(8).astype("float64")
    csv_buffer = StringIO()
    persisted = expected.reset_index(names="time")
    persisted["time"] = persisted["time"].astype("int64") // 1_000_000_000
    persisted.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"}).to_csv(
        csv_buffer,
        index=False,
    )
    csv_buffer.seek(0)

    reloaded = load_data(csv_buffer)

    pd.testing.assert_frame_equal(
        reloaded,
        expected.rename_axis("time"),
        check_dtype=True,
        check_freq=False,
    )
