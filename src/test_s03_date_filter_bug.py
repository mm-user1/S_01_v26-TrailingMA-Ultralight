#!/usr/bin/env python3
"""
Test script to reproduce and verify S_03 date filter bug fix.

Bug: S_03 applies date filter twice in optimization mode, causing incorrect results.
Expected: Same results in CLI mode and optimization mode with identical parameters.
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from strategies.s03_reversal import S03Reversal
from backtest_engine import load_data


def test_s03_cli_baseline():
    """Test S_03 with default parameters (CLI mode - no caching)."""
    print("\n" + "="*80)
    print("TEST 1: S_03 CLI Baseline (No Cache)")
    print("="*80)

    csv_path = "../data/OKX_LINKUSDT.P, 15 2025.05.01-2025.11.20.csv"
    df = load_data(csv_path)

    # Default parameters from tests.md
    params = {
        "useBacktester": True,
        "dateFilter": True,
        "startDate": "2025-06-15",
        "endDate": "2025-11-15",
        "maFastType": "SMA",
        "maFastLength": 100,
        "maSlowType": "SMA",
        "maSlowLength": 0,  # Disabled
        "maTrendType": "SMA",
        "maTrendLength": 100,
        "closeCountLong": 4,
        "closeCountShort": 5,
        "equityPct": 100.0,
        "contractSize": 0.01,
        "commissionRate": 0.0005,
    }

    strategy = S03Reversal(params)
    result = strategy.simulate(df, cached_data=None)

    print(f"Net Profit:     {result['net_profit_pct']:.2f}%")
    print(f"Max Drawdown:   {result['max_drawdown_pct']:.2f}%")
    print(f"Total Trades:   {result['total_trades']}")
    print(f"Win Rate:       {result.get('win_rate', 0):.2f}%")

    expected = {
        "net_profit_pct": 83.56,
        "max_drawdown_pct": 35.34,
        "total_trades": 224,
    }

    # Check with tolerance
    tolerance = 0.01
    matches = True
    for key, expected_val in expected.items():
        actual_val = result[key]
        if key == "total_trades":
            if actual_val != expected_val:
                print(f"❌ {key}: expected {expected_val}, got {actual_val}")
                matches = False
        else:
            if abs(actual_val - expected_val) > tolerance:
                print(f"❌ {key}: expected {expected_val}, got {actual_val}")
                matches = False

    if matches:
        print("✅ CLI baseline test PASSED")
    else:
        print("❌ CLI baseline test FAILED")

    return result


def test_s03_optimization_mode():
    """Test S_03 in optimization mode (with cached MAs and trade_start_idx set)."""
    print("\n" + "="*80)
    print("TEST 2: S_03 Optimization Mode (With Cache)")
    print("="*80)

    csv_path = "../data/OKX_LINKUSDT.P, 15 2025.05.01-2025.11.20.csv"
    df = load_data(csv_path)

    # Same parameters
    params = {
        "useBacktester": True,
        "dateFilter": True,
        "startDate": "2025-06-15",
        "endDate": "2025-11-15",
        "maFastType": "SMA",
        "maFastLength": 100,
        "maSlowType": "SMA",
        "maSlowLength": 0,
        "maTrendType": "SMA",
        "maTrendLength": 100,
        "closeCountLong": 4,
        "closeCountShort": 5,
        "equityPct": 100.0,
        "contractSize": 0.01,
        "commissionRate": 0.0005,
    }

    # Simulate what optimizer does:
    # 1. Pre-compute MAs
    from indicators import get_ma

    close_series = df["Close"]
    volume_series = df["Volume"]
    high_series = df["High"]
    low_series = df["Low"]

    ma_fast = get_ma(close_series, "SMA", 100, volume=volume_series, high=high_series, low=low_series).to_numpy()
    ma_trend = get_ma(close_series, "SMA", 100, volume=volume_series, high=high_series, low=low_series).to_numpy()

    cached_data = {
        "ma_cache": {
            ("SMA", 100): ma_fast,
        }
    }

    # 2. Calculate trade_start_idx (warmup period for MA=100)
    # Optimizer would set this based on max MA length
    warmup_period = 100
    trade_start_idx = warmup_period

    # Create strategy and manually set trade_start_idx
    strategy = S03Reversal(params)
    strategy.trade_start_idx = trade_start_idx

    # Run simulation with cache
    result = strategy.simulate(df, cached_data=cached_data)

    print(f"Net Profit:     {result['net_profit_pct']:.2f}%")
    print(f"Max Drawdown:   {result['max_drawdown_pct']:.2f}%")
    print(f"Total Trades:   {result['total_trades']}")
    print(f"Win Rate:       {result.get('win_rate', 0):.2f}%")
    print(f"Trade Start Idx: {trade_start_idx}")

    # Should match CLI baseline!
    expected = {
        "net_profit_pct": 83.56,
        "max_drawdown_pct": 35.34,
        "total_trades": 224,
    }

    tolerance = 0.01
    matches = True
    for key, expected_val in expected.items():
        actual_val = result[key]
        if key == "total_trades":
            if actual_val != expected_val:
                print(f"❌ {key}: expected {expected_val}, got {actual_val}")
                matches = False
        else:
            if abs(actual_val - expected_val) > tolerance:
                print(f"❌ {key}: expected {expected_val}, got {actual_val}")
                matches = False

    if matches:
        print("✅ Optimization mode test PASSED")
    else:
        print("❌ Optimization mode test FAILED - BUG STILL PRESENT")

    return result


def test_s01_regression():
    """Ensure S_01 still works after changes (regression test)."""
    print("\n" + "="*80)
    print("TEST 3: S_01 Regression Test")
    print("="*80)

    from strategies.s01_trailing_ma import S01TrailingMA

    csv_path = "../data/OKX_LINKUSDT.P, 15 2025.05.01-2025.11.20.csv"
    df = load_data(csv_path)

    # Get default parameters
    param_defs = S01TrailingMA.get_param_definitions()
    params = {k: v["default"] for k, v in param_defs.items()}

    strategy = S01TrailingMA(params)
    result = strategy.simulate(df, cached_data=None)

    print(f"Net Profit:     {result['net_profit_pct']:.2f}%")
    print(f"Max Drawdown:   {result['max_drawdown_pct']:.2f}%")
    print(f"Total Trades:   {result['total_trades']}")

    expected = {
        "net_profit_pct": 230.75,
        "max_drawdown_pct": 20.03,
        "total_trades": 93,
    }

    tolerance = 0.01
    matches = True
    for key, expected_val in expected.items():
        actual_val = result[key]
        if key == "total_trades":
            if actual_val != expected_val:
                print(f"❌ {key}: expected {expected_val}, got {actual_val}")
                matches = False
        else:
            if abs(actual_val - expected_val) > tolerance:
                print(f"❌ {key}: expected {expected_val}, got {actual_val}")
                matches = False

    if matches:
        print("✅ S_01 regression test PASSED")
    else:
        print("❌ S_01 regression test FAILED")

    return matches


if __name__ == "__main__":
    print("\n" + "█"*80)
    print("S_03 DATE FILTER BUG TEST SUITE")
    print("█"*80)

    try:
        # Test 1: CLI baseline
        cli_result = test_s03_cli_baseline()

        # Test 2: Optimization mode (should match CLI after fix)
        opt_result = test_s03_optimization_mode()

        # Test 3: S_01 regression
        s01_ok = test_s01_regression()

        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)

        cli_ok = abs(cli_result['net_profit_pct'] - 83.56) < 0.01
        opt_ok = abs(opt_result['net_profit_pct'] - 83.56) < 0.01

        if cli_ok and opt_ok and s01_ok:
            print("✅ ALL TESTS PASSED - Bug is fixed!")
            sys.exit(0)
        else:
            print("❌ SOME TESTS FAILED - Bug still present or new regression")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ TEST SUITE CRASHED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
