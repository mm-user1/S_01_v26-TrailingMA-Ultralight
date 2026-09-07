"""
Regression tests for S01 Trailing MA strategy.

These tests ensure that the S01 strategy behavior remains unchanged during migration.
The baseline results were generated with the legacy implementation and serve as the
"golden standard" for validation.

Test Strategy:
1. Load baseline results from data/baseline/
2. Run current S01 implementation with same parameters
3. Compare results with appropriate tolerances
4. Fail if results diverge beyond tolerance

Tolerance Configuration:
- net_profit_pct: ±0.01% (floating point tolerance)
- max_drawdown_pct: ±0.01%
- total_trades: exact match (±0)
- trade entry/exit times: exact match
- trade PnL: ±0.0001 (floating point epsilon)
"""

import pytest
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from core.backtest_engine import load_data, prepare_dataset_with_warmup
from strategies.s01_trailing_ma.strategy import S01Params, S01TrailingMA


# Tolerance configuration
TOLERANCE_CONFIG = {
    "net_profit_pct": 0.01,      # ±0.01%
    "max_drawdown_pct": 0.01,    # ±0.01%
    "total_trades": 0,            # exact match
    "trade_pnl": 0.0001,          # floating point epsilon
    "sharpe_ratio": 0.001,        # ±0.001
}

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
BASELINE_DIR = PROJECT_ROOT / "data" / "baseline"
METRICS_FILE = BASELINE_DIR / "s01_metrics.json"
TRADES_FILE = BASELINE_DIR / "s01_trades.csv"


@pytest.fixture(scope="module")
def baseline_metrics() -> Dict[str, Any]:
    """Load baseline metrics from JSON."""
    if not METRICS_FILE.exists():
        pytest.skip(f"Baseline metrics not found: {METRICS_FILE}")

    with open(METRICS_FILE, "r") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def baseline_trades() -> pd.DataFrame:
    """Load baseline trades from CSV."""
    if not TRADES_FILE.exists():
        pytest.skip(f"Baseline trades not found: {TRADES_FILE}")

    return pd.read_csv(TRADES_FILE)


@pytest.fixture(scope="module")
def test_data() -> pd.DataFrame:
    """Load test dataset using official load_data function."""
    csv_path = PROJECT_ROOT / "data" / "raw" / "OKX_LINKUSDT.P, 15 2025.05.01-2025.11.20.csv"

    if not csv_path.exists():
        pytest.skip(f"Test data not found: {csv_path}")

    # Use official load_data function to ensure consistency
    df = load_data(str(csv_path))

    return df


@pytest.fixture(scope="module")
def current_result(baseline_metrics, test_data):
    """Run current S01 implementation with baseline parameters."""
    # Extract parameters from baseline
    params_dict = baseline_metrics["parameters"]

    # Parse parameters
    params = S01Params.from_dict(params_dict)

    # Prepare dataset with warmup (same as baseline generation)
    start_ts = pd.Timestamp(params_dict["start"], tz="UTC")
    end_ts = pd.Timestamp(params_dict["end"], tz="UTC")
    warmup_bars = baseline_metrics["warmup_bars"]

    df_prepared, trade_start_idx = prepare_dataset_with_warmup(
        test_data, start_ts, end_ts, warmup_bars
    )

    # Run strategy
    result = S01TrailingMA.run(df_prepared, asdict(params), trade_start_idx)

    return result


@pytest.mark.regression
class TestS01Regression:
    """Regression tests for S01 strategy baseline behavior."""

    def test_baseline_files_exist(self):
        """Verify baseline files exist before running tests."""
        assert BASELINE_DIR.exists(), "Baseline directory not found"
        assert METRICS_FILE.exists(), "Baseline metrics file not found"
        # Trades file is optional if there were no trades
        # We check its existence in trade comparison tests if needed

    def test_net_profit_matches(self, baseline_metrics, current_result):
        """Test that net profit % matches baseline within tolerance."""
        baseline_value = baseline_metrics["net_profit_pct"]
        current_value = current_result.net_profit_pct

        tolerance = TOLERANCE_CONFIG["net_profit_pct"]
        diff = abs(current_value - baseline_value)

        assert diff <= tolerance, (
            f"Net Profit diverged from baseline:\n"
            f"  Baseline: {baseline_value:.2f}%\n"
            f"  Current:  {current_value:.2f}%\n"
            f"  Diff:     {diff:.4f}% (tolerance: ±{tolerance}%)"
        )

    def test_max_drawdown_matches(self, baseline_metrics, current_result):
        """Test that max drawdown % matches baseline within tolerance."""
        baseline_value = baseline_metrics["max_drawdown_pct"]
        current_value = current_result.max_drawdown_pct

        tolerance = TOLERANCE_CONFIG["max_drawdown_pct"]
        diff = abs(current_value - baseline_value)

        assert diff <= tolerance, (
            f"Max Drawdown diverged from baseline:\n"
            f"  Baseline: {baseline_value:.2f}%\n"
            f"  Current:  {current_value:.2f}%\n"
            f"  Diff:     {diff:.4f}% (tolerance: ±{tolerance}%)"
        )

    def test_total_trades_matches(self, baseline_metrics, current_result):
        """Test that total trades count matches baseline exactly."""
        baseline_value = baseline_metrics["total_trades"]
        current_value = current_result.total_trades

        assert current_value == baseline_value, (
            f"Total trades count mismatch:\n"
            f"  Baseline: {baseline_value}\n"
            f"  Current:  {current_value}"
        )

    def test_trade_count_consistency(self, baseline_trades, current_result):
        """Test that number of trade records matches total_trades."""
        if TRADES_FILE.exists():
            baseline_count = len(baseline_trades)
            current_count = len(current_result.trades)

            assert current_count == baseline_count, (
                f"Trade record count mismatch:\n"
                f"  Baseline records: {baseline_count}\n"
                f"  Current records:  {current_count}"
            )

    def test_sharpe_ratio_matches(self, baseline_metrics, current_result):
        """Test that Sharpe ratio matches baseline within tolerance."""
        baseline_value = baseline_metrics.get("sharpe_ratio")
        current_value = current_result.sharpe_ratio

        # Both should be either None or have values
        if baseline_value is None:
            assert current_value is None, "Sharpe ratio should be None in current result"
            return

        tolerance = TOLERANCE_CONFIG["sharpe_ratio"]
        diff = abs(current_value - baseline_value)

        assert diff <= tolerance, (
            f"Sharpe ratio diverged from baseline:\n"
            f"  Baseline: {baseline_value:.4f}\n"
            f"  Current:  {current_value:.4f}\n"
            f"  Diff:     {diff:.6f} (tolerance: ±{tolerance})"
        )

    def test_trade_directions_match(self, baseline_trades, current_result):
        """Test that trade directions (Long/Short) match baseline."""
        if not TRADES_FILE.exists():
            pytest.skip("No baseline trades file")

        if len(baseline_trades) == 0:
            assert len(current_result.trades) == 0, "Expected no trades"
            return

        for i, (baseline_row, current_trade) in enumerate(zip(
            baseline_trades.itertuples(index=False),
            current_result.trades
        )):
            assert current_trade.direction == baseline_row.direction, (
                f"Trade {i} direction mismatch:\n"
                f"  Baseline: {baseline_row.direction}\n"
                f"  Current:  {current_trade.direction}"
            )

    def test_trade_entry_times_match(self, baseline_trades, current_result):
        """Test that trade entry times match baseline exactly."""
        if not TRADES_FILE.exists():
            pytest.skip("No baseline trades file")

        if len(baseline_trades) == 0:
            return

        for i, (baseline_row, current_trade) in enumerate(zip(
            baseline_trades.itertuples(index=False),
            current_result.trades
        )):
            baseline_time = pd.Timestamp(baseline_row.entry_time)
            current_time = current_trade.entry_time

            assert current_time == baseline_time, (
                f"Trade {i} entry time mismatch:\n"
                f"  Baseline: {baseline_time}\n"
                f"  Current:  {current_time}"
            )

    def test_trade_exit_times_match(self, baseline_trades, current_result):
        """Test that trade exit times match baseline exactly."""
        if not TRADES_FILE.exists():
            pytest.skip("No baseline trades file")

        if len(baseline_trades) == 0:
            return

        for i, (baseline_row, current_trade) in enumerate(zip(
            baseline_trades.itertuples(index=False),
            current_result.trades
        )):
            baseline_time = pd.Timestamp(baseline_row.exit_time)
            current_time = current_trade.exit_time

            assert current_time == baseline_time, (
                f"Trade {i} exit time mismatch:\n"
                f"  Baseline: {baseline_time}\n"
                f"  Current:  {current_time}"
            )

    def test_trade_pnl_matches(self, baseline_trades, current_result):
        """Test that trade PnL values match baseline within floating point tolerance."""
        if not TRADES_FILE.exists():
            pytest.skip("No baseline trades file")

        if len(baseline_trades) == 0:
            return

        tolerance = TOLERANCE_CONFIG["trade_pnl"]

        for i, (baseline_row, current_trade) in enumerate(zip(
            baseline_trades.itertuples(index=False),
            current_result.trades
        )):
            baseline_pnl = baseline_row.net_pnl
            current_pnl = current_trade.net_pnl

            diff = abs(current_pnl - baseline_pnl)

            assert diff <= tolerance, (
                f"Trade {i} PnL mismatch:\n"
                f"  Baseline: {baseline_pnl:.6f}\n"
                f"  Current:  {current_pnl:.6f}\n"
                f"  Diff:     {diff:.8f} (tolerance: ±{tolerance})"
            )

    def test_advanced_metrics_present(self, baseline_metrics, current_result):
        """Test that advanced metrics are calculated when present in baseline."""
        advanced_metrics = [
            "sharpe_ratio",
            "profit_factor",
            "romad",
            "sqn",
            "ulcer_index",
            "consistency_score"
        ]

        for metric in advanced_metrics:
            baseline_value = baseline_metrics.get(metric)
            current_value = getattr(current_result, metric)

            # If baseline has value, current should too
            if baseline_value is not None:
                assert current_value is not None, (
                    f"{metric} was present in baseline but missing in current result"
                )


@pytest.mark.regression
@pytest.mark.slow
class TestS01RegressionConsistency:
    """Additional consistency tests for regression validation."""

    def test_multiple_runs_produce_same_results(self, test_data, baseline_metrics):
        """Test that running the strategy multiple times produces identical results."""
        params_dict = baseline_metrics["parameters"]
        params = S01Params.from_dict(params_dict)

        start_ts = pd.Timestamp(params_dict["start"], tz="UTC")
        end_ts = pd.Timestamp(params_dict["end"], tz="UTC")
        warmup_bars = baseline_metrics["warmup_bars"]

        df_prepared, trade_start_idx = prepare_dataset_with_warmup(
            test_data, start_ts, end_ts, warmup_bars
        )

        # Run strategy twice
        result1 = S01TrailingMA.run(df_prepared, asdict(params), trade_start_idx)
        result2 = S01TrailingMA.run(df_prepared, asdict(params), trade_start_idx)

        # Results should be identical
        assert result1.net_profit_pct == result2.net_profit_pct
        assert result1.max_drawdown_pct == result2.max_drawdown_pct
        assert result1.total_trades == result2.total_trades
        assert len(result1.trades) == len(result2.trades)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "regression"])
