import numpy as np


from core.metrics import calculate_higher_moments_from_monthly_returns
from core.post_process import calculate_dsr, calculate_expected_max_sharpe


def test_calculate_expected_max_sharpe_basic():
    """Expected max SR increases with number of trials."""
    sr0_10 = calculate_expected_max_sharpe(0.0, 1.0, 10)
    sr0_100 = calculate_expected_max_sharpe(0.0, 1.0, 100)
    sr0_1000 = calculate_expected_max_sharpe(0.0, 1.0, 1000)
    assert sr0_10 is not None and sr0_100 is not None and sr0_1000 is not None
    assert sr0_10 < sr0_100 < sr0_1000


def test_calculate_dsr_high_sharpe_track_length():
    """High Sharpe with longer track record should have higher DSR."""
    dsr_long = calculate_dsr(2.0, 0.5, 0.0, 3.0, 36)
    dsr_short = calculate_dsr(2.0, 0.5, 0.0, 3.0, 12)
    assert dsr_long is not None and dsr_short is not None
    assert dsr_long > dsr_short


def test_calculate_higher_moments_normal():
    """Normal returns should have near-zero skew and ~3 raw kurtosis."""
    np.random.seed(42)
    returns = np.random.normal(0, 1, 1000).tolist()
    skew, kurt = calculate_higher_moments_from_monthly_returns(returns)
    assert skew is not None and kurt is not None
    assert abs(skew) < 0.5
    assert 2.0 < kurt < 4.5
