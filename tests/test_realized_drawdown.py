from __future__ import annotations

import math

import pandas as pd
import pytest

from core.backtest_engine import StrategyResult
from core.engine_v2.metrics_kernel import compute_core_metrics_from_balance_and_trades
from core.metrics import calculate_advanced, calculate_basic, enrich_strategy_result


def _result(balance_curve, *, equity_curve=None, metric_start_idx=0, metric_initial_equity=None):
    count = len(balance_curve)
    return StrategyResult(
        trades=[],
        balance_curve=list(balance_curve),
        equity_curve=list(balance_curve if equity_curve is None else equity_curve),
        timestamps=list(pd.date_range("2025-01-01", periods=count, freq="h", tz="UTC")),
        metric_start_idx=metric_start_idx,
        metric_initial_equity=metric_initial_equity,
    )


def _oracle(balance_curve):
    running_peak = None
    max_pct = 0.0
    max_abs = 0.0
    for raw_balance in balance_curve:
        balance = float(raw_balance)
        if not math.isfinite(balance):
            continue
        if running_peak is None or balance > running_peak:
            running_peak = balance
            continue
        drawdown = running_peak - balance
        max_abs = max(max_abs, drawdown)
        if running_peak > 0.0:
            max_pct = max(max_pct, drawdown / running_peak * 100.0)
    return max_pct, max_abs


@pytest.mark.parametrize(
    "balance_curve",
    [
        [100.0, 90.0, 105.0, 80.0],
        [100.0, 99.0],
        [100.0, 100.0, 100.0, 99.0],
        [100.0, 95.0, 90.0],
        [100.0, 80.0, 100.0, 90.0],
        [100.0, 90.0, 100.0, 70.0],
        [100.0, 100.0, 100.0],
        [100.0, 110.0, 120.0],
        [],
        [math.nan, math.inf, -math.inf],
        [math.nan, 100.0, math.nan, 90.0, math.inf, 95.0],
        [-10.0, -20.0, 10.0, 5.0],
        [100.0, 50.0, 1000.0, 900.0],
    ],
)
def test_canonical_and_v2_reference_match_independent_full_path_oracle(balance_curve):
    expected_pct, expected_abs = _oracle(balance_curve)
    slow = calculate_basic(_result(balance_curve))
    reference = compute_core_metrics_from_balance_and_trades(balance_curve, [])

    assert slow.max_drawdown_pct == pytest.approx(expected_pct, rel=1e-12, abs=1e-12)
    assert slow.max_drawdown == pytest.approx(expected_abs, rel=1e-12, abs=1e-12)
    assert reference.max_drawdown_pct == pytest.approx(expected_pct, rel=1e-12, abs=1e-12)
    assert reference.max_drawdown == pytest.approx(expected_abs, rel=1e-12, abs=1e-12)


def test_percentage_and_absolute_drawdown_have_independent_owning_peaks():
    basic = calculate_basic(_result([100.0, 50.0, 1000.0, 900.0]))

    assert basic.max_drawdown_pct == pytest.approx(50.0)
    assert basic.max_drawdown == pytest.approx(100.0)


def test_initial_balance_and_advanced_metric_boundary_do_not_own_realized_drawdown():
    result = _result(
        [100.0, 90.0, 105.0, 80.0],
        metric_start_idx=3,
        metric_initial_equity=999.0,
    )
    basic = calculate_basic(result, initial_balance=1000.0)

    assert basic.net_profit_pct == pytest.approx(-92.0)
    assert basic.max_drawdown_pct == pytest.approx(25.0 / 105.0 * 100.0)
    assert basic.max_drawdown == pytest.approx(25.0)


@pytest.mark.parametrize("prefix_length", [0, 1, 10, 1000])
def test_flat_technical_warmup_prefix_does_not_change_realized_drawdown(prefix_length):
    balance_curve = [100.0] * prefix_length + [100.0, 90.0, 105.0, 80.0]
    basic = calculate_basic(_result(balance_curve))

    assert basic.max_drawdown_pct == pytest.approx(25.0 / 105.0 * 100.0)
    assert basic.max_drawdown == pytest.approx(25.0)


def test_realized_drawdown_ignores_equity_curve_and_unrealized_pnl():
    balance_curve = [100.0, 100.0, 90.0]
    calm = calculate_basic(_result(balance_curve, equity_curve=[100.0, 100.0, 90.0]))
    volatile = calculate_basic(_result(balance_curve, equity_curve=[100.0, 10.0, 90.0]))

    assert volatile.max_drawdown_pct == calm.max_drawdown_pct == pytest.approx(10.0)
    assert volatile.max_drawdown == calm.max_drawdown == pytest.approx(10.0)


def test_selected_slow_payload_receives_corrected_drawdown_and_romad():
    result = _result([100.0, 90.0, 105.0, 80.0])

    enrich_strategy_result(result, initial_balance=100.0)

    assert result.max_drawdown_pct == pytest.approx(25.0 / 105.0 * 100.0)
    assert result.max_drawdown == pytest.approx(25.0)
    assert result.romad == pytest.approx(-0.84)


@pytest.mark.parametrize(
    ("balance_curve", "initial_balance", "expected_romad"),
    [
        ([100.0, 90.0, 105.0, 80.0], 100.0, -0.84),
        ([100.0, 110.0], 100.0, 1000.0),
        ([100.0, 110.0], 200.0, 0.0),
    ],
)
def test_romad_uses_corrected_percentage_and_preserves_zero_dd_convention(
    balance_curve,
    initial_balance,
    expected_romad,
):
    advanced = calculate_advanced(_result(balance_curve), initial_balance=initial_balance)

    assert advanced.romad == pytest.approx(expected_romad, rel=1e-12, abs=1e-12)
