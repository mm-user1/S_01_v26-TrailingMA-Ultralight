from __future__ import annotations

import math

import numpy as np
import pytest

from strategies.s03_reversal_v10 import fast_grid as s03_v10
from strategies.s03_reversal_v11 import fast_grid as s03_v11
from strategies.s06_r_trend_v02 import fast_grid as s06


def _s03_metrics(backend, closes):
    close = np.asarray(closes, dtype=np.float64)
    high = close.copy()
    low = close.copy()
    zeros_i32 = np.zeros(close.size, dtype=np.int32)
    ma = close.copy()
    ma[-2] = close[-2] - 1.0
    ma[-1] = close[-1] + 1.0
    common = (
        high,
        low,
        zeros_i32,
        zeros_i32,
        ma,
        0,
        False,
        0.0,
        True,
        False,
        1,
        1,
        0.0,
        0.0,
    )
    tail = (1.0, 100.0, 0.0, False, False, False, False, 0.02)
    if backend is s03_v10:
        return backend._S03_FAST_LOOP(close, *common, *tail)
    return backend._S03_FAST_LOOP(
        close,
        close,
        *common,
        False,
        10.0,
        16,
        *tail,
    )


def _s06_metrics(closes):
    close = np.asarray(closes, dtype=np.float64)
    count = close.size
    open_values = close.copy()
    open_values[-1] = close[-2]
    high = np.maximum(open_values, close)
    low = np.minimum(open_values, close)
    signal_long = np.zeros(count, dtype=np.bool_)
    signal_long[count - 2] = True
    empty_trace_i64 = np.empty(count, dtype=np.int64)
    empty_trace_f64 = np.empty(count, dtype=np.float64)
    return s06._S06_FAST_LOOP(
        open_values,
        high,
        low,
        close,
        np.arange(count, dtype=np.int64) * 3_600_000_000_000,
        np.zeros(count, dtype=np.int32),
        np.zeros(count, dtype=np.int32),
        np.zeros(count, dtype=np.float64),
        np.zeros(count, dtype=np.float64),
        np.full(count, 200.0, dtype=np.float64),
        signal_long,
        np.zeros(count, dtype=np.bool_),
        np.full(count, np.nan, dtype=np.float64),
        0,
        False,
        0,
        np.iinfo(np.int64).max,
        True,
        False,
        False,
        0.0,
        10.0,
        100.0,
        999,
        100.0,
        1.0,
        1.0,
        0.0,
        100.0,
        0.0,
        False,
        False,
        False,
        False,
        0.02,
        False,
        empty_trace_i64,
        empty_trace_i64.copy(),
        empty_trace_i64.copy(),
        empty_trace_f64,
        empty_trace_f64.copy(),
        empty_trace_f64.copy(),
        empty_trace_f64.copy(),
    )


@pytest.mark.skipif(
    not (s03_v10.NUMBA_AVAILABLE and s03_v11.NUMBA_AVAILABLE and s06.NUMBA_AVAILABLE),
    reason="Numba Fast Grid backends are required.",
)
@pytest.mark.parametrize("closes", [[100.0, 99.0], [100.0, 100.0, 100.0, 99.0]])
def test_all_v1_fast_backends_include_single_observation_terminal_drawdown(closes):
    outputs = [_s03_metrics(s03_v10, closes), _s03_metrics(s03_v11, closes), _s06_metrics(closes)]

    for values in outputs:
        assert values[0] == pytest.approx(-1.0, rel=1e-12, abs=1e-12)
        assert values[1] == pytest.approx(1.0, rel=1e-12, abs=1e-12)
        assert values[9] == pytest.approx(-1.0, rel=1e-12, abs=1e-12)
        assert math.isfinite(values[1])
