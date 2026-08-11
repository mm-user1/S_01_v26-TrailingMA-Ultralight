import math

import pytest

from core.engine_v2.metrics_kernel import compute_core_metrics_from_balance_and_trades


@pytest.mark.parametrize(
    ("balance_curve", "expected_pct", "expected_abs"),
    [
        ([100.0, 90.0, 105.0, 80.0], 25.0 / 105.0 * 100.0, 25.0),
        ([100.0, 99.0], 1.0, 1.0),
        ([100.0, 100.0, 100.0, 99.0], 1.0, 1.0),
        ([100.0, 95.0, 90.0], 10.0, 10.0),
        ([100.0, 80.0, 100.0, 90.0], 20.0, 20.0),
        ([100.0, 90.0, 100.0, 70.0], 30.0, 30.0),
        ([100.0, 100.0, 100.0], 0.0, 0.0),
        ([100.0, 110.0, 120.0], 0.0, 0.0),
        ([], 0.0, 0.0),
        ([math.nan, math.inf, -math.inf], 0.0, 0.0),
        ([math.nan, 100.0, math.nan, 90.0, math.inf, 95.0], 10.0, 10.0),
        ([-10.0, -20.0, 10.0, 5.0], 50.0, 10.0),
        ([100.0, 50.0, 1000.0, 900.0], 50.0, 100.0),
    ],
)
def test_reference_realized_drawdown_scans_complete_path(
    balance_curve,
    expected_pct,
    expected_abs,
):
    metrics = compute_core_metrics_from_balance_and_trades(balance_curve, [])

    assert metrics.max_drawdown_pct == pytest.approx(expected_pct, rel=1e-12, abs=1e-12)
    assert metrics.max_drawdown == pytest.approx(expected_abs, rel=1e-12, abs=1e-12)


def test_reference_initial_balance_does_not_seed_drawdown_peak():
    metrics = compute_core_metrics_from_balance_and_trades(
        [100.0, 90.0],
        [],
        initial_balance=1000.0,
    )

    assert metrics.net_profit_pct == pytest.approx(-91.0)
    assert metrics.max_drawdown_pct == pytest.approx(10.0)
    assert metrics.max_drawdown == pytest.approx(10.0)
    assert metrics.romad == pytest.approx(-9.1)


@pytest.mark.parametrize(
    ("balance_curve", "initial_balance", "expected_romad"),
    [
        ([100.0, 90.0, 105.0, 80.0], 100.0, -0.84),
        ([100.0, 110.0], 100.0, 1000.0),
        ([100.0, 110.0], 200.0, 0.0),
    ],
)
def test_reference_romad_uses_corrected_percentage_and_zero_dd_convention(
    balance_curve,
    initial_balance,
    expected_romad,
):
    metrics = compute_core_metrics_from_balance_and_trades(
        balance_curve,
        [],
        initial_balance=initial_balance,
    )

    assert metrics.romad == pytest.approx(expected_romad, rel=1e-12, abs=1e-12)
