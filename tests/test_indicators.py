"""Parity tests for extracted indicators (Phase 5)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


from core import backtest_engine
from indicators import ma, volatility


DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "OKX_LINKUSDT.P, 15 2025.05.01-2025.11.20.csv"


# Fixtures
@pytest.fixture(scope="module")
def test_df():
    if not DATA_PATH.exists():
        pytest.skip(f"Test data not found: {DATA_PATH}")
    return backtest_engine.load_data(str(DATA_PATH))


@pytest.fixture(scope="module")
def test_series(test_df):
    return test_df["Close"]


@pytest.fixture(scope="module")
def test_volume(test_df):
    return test_df["Volume"]


@pytest.fixture(scope="module")
def test_ohlcv(test_df):
    return test_df[["High", "Low", "Close", "Volume"]]


class TestMAEdgeCases:
    """Test MA functions with edge cases."""

    def test_sma_zero_length(self, test_series):
        result = ma.sma(test_series, 0)
        assert result.isna().all()

    def test_ema_single_value(self):
        series = pd.Series([100.0])
        result = ma.ema(series, 1)
        assert not result.isna().any()
        assert result.iloc[0] == 100.0


class TestVolatilityParity:
    """Test parity for volatility indicators."""

    def test_atr_parity(self, test_ohlcv):
        period = 14
        old_atr = backtest_engine.atr(
            test_ohlcv["High"],
            test_ohlcv["Low"],
            test_ohlcv["Close"],
            period,
        )
        new_atr = volatility.atr(
            test_ohlcv["High"],
            test_ohlcv["Low"],
            test_ohlcv["Close"],
            period,
        )
        pd.testing.assert_series_equal(old_atr, new_atr)


class TestAllMATypes:
    """Test all MA types work via get_ma()."""

    def test_all_ma_types_work(self, test_series, test_volume, test_ohlcv):
        length = 20
        for ma_type in ma.VALID_MA_TYPES:
            if ma_type == "VWMA":
                result = ma.get_ma(test_series, ma_type, length, volume=test_volume)
            elif ma_type == "VWAP":
                result = ma.get_ma(
                    test_ohlcv["Close"],
                    ma_type,
                    length,
                    volume=test_ohlcv["Volume"],
                    high=test_ohlcv["High"],
                    low=test_ohlcv["Low"],
                )
            else:
                result = ma.get_ma(test_series, ma_type, length)

            assert result is not None
            assert len(result) == len(test_series)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
