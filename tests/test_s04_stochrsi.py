from pathlib import Path

import pandas as pd
import pytest

from dataclasses import asdict

from core import backtest_engine, metrics
from indicators.oscillators import rsi, stoch_rsi
from strategies.s04_stochrsi.strategy import S04Params, S04StochRSI


PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "OKX_LINKUSDT.P, 15 2025.05.01-2025.11.20.csv"
TRADING_START = pd.Timestamp("2025-06-01", tz="UTC")
TRADING_END = pd.Timestamp("2025-10-01", tz="UTC")


@pytest.fixture(scope="module")
def test_data():
    if not DATA_PATH.exists():
        pytest.skip(f"Test data not found: {DATA_PATH}")
    df = backtest_engine.load_data(str(DATA_PATH))
    return df.loc[:TRADING_END]


def test_rsi_calculation(test_data):
    slice_df = test_data.loc[TRADING_START:TRADING_END]
    rsi_values = rsi(slice_df["Close"], 16)

    assert len(rsi_values) == len(slice_df)
    assert rsi_values.min() >= 0.0
    assert rsi_values.max() <= 100.0
    assert not rsi_values.iloc[-100:].isna().all()


def test_stoch_rsi_calculation(test_data):
    slice_df = test_data.loc[TRADING_START:TRADING_END]
    stoch_values = stoch_rsi(slice_df["Close"], 16, 16)

    assert len(stoch_values) == len(slice_df)
    assert stoch_values.min() >= 0.0
    assert stoch_values.max() <= 100.0


def test_s04_basic_run(test_data):
    params = asdict(S04Params(startDate=TRADING_START, endDate=TRADING_END))

    result = S04StochRSI.run(test_data, params, trade_start_idx=0)

    assert result is not None
    assert isinstance(result.trades, list)
    assert len(result.equity_curve) == len(test_data)
    assert len(result.balance_curve) == len(test_data)


def test_s04_reference_performance(test_data):
    params = {
        "rsiLen": 16,
        "stochLen": 16,
        "kLen": 3,
        "dLen": 3,
        "obLevel": 75.0,
        "osLevel": 15.0,
        "extLookback": 23,
        "confirmBars": 14,
        "riskPerTrade": 2.0,
        "contractSize": 0.01,
        "initialCapital": 100.0,
        "commissionPct": 0.05,
        "startDate": TRADING_START,
        "endDate": TRADING_END,
    }

    result = S04StochRSI.run(test_data, params, trade_start_idx=0)

    basic = metrics.calculate_basic(result)

    expected_net_profit_pct = 113.26
    expected_max_dd_pct = 10.99
    expected_total_trades = 52

    tolerance = 0.05

    assert abs(basic.net_profit_pct - expected_net_profit_pct) / expected_net_profit_pct <= tolerance, (
        f"Net Profit mismatch: {basic.net_profit_pct}% vs expected {expected_net_profit_pct}%"
    )

    assert abs(basic.max_drawdown_pct - expected_max_dd_pct) / expected_max_dd_pct <= tolerance, (
        f"Max DD mismatch: {basic.max_drawdown_pct}% vs expected {expected_max_dd_pct}%"
    )

    assert abs(basic.total_trades - expected_total_trades) <= 3, (
        f"Total trades mismatch: {basic.total_trades} vs expected {expected_total_trades}"
    )
