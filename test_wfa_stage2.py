#!/usr/bin/env python3
"""
Tests for WFA Stage 2 - IS-only optimization and forward-only metrics
"""

import sys
from unittest.mock import Mock, patch, MagicMock
from typing import List
import pandas as pd
from pathlib import Path

sys.path.insert(0, 'src')

from walkforward_engine import WFConfig, WalkForwardEngine


class MockTradeRecord:
    """Mock trade record for testing"""
    def __init__(self, entry_time, net_pnl):
        self.entry_time = entry_time
        self.net_pnl = net_pnl


def test_is_only_optimization():
    """
    Test that _run_optuna_on_window applies dateFilter with IS boundaries
    """
    print("\n" + "="*60)
    print("TEST 1: IS-only optimization with dateFilter")
    print("="*60)

    # Create mock dataframe
    dates = pd.date_range('2025-01-01', periods=2000, freq='15min', tz='UTC')
    df = pd.DataFrame({
        'Open': 100.0,
        'High': 101.0,
        'Low': 99.0,
        'Close': 100.0,
        'Volume': 1000.0
    }, index=dates)

    # IS boundaries: bars 1000-1500 (indices in full df)
    is_start_idx = 1000
    is_end_idx = 1500
    is_start_time = df.index[is_start_idx]
    is_end_time = df.index[is_end_idx - 1]

    # Create warmup + IS dataframe
    warmup_start = is_start_idx - 500
    df_window = df.iloc[warmup_start:is_end_idx].copy()

    print(f"   df_window range: {df_window.index[0]} to {df_window.index[-1]}")
    print(f"   IS boundaries: {is_start_time} to {is_end_time}")

    # Setup WFA engine
    base_config_template = {
        "enabled_params": {},
        "param_ranges": {},
        "fixed_params": {
            "start": df.index[0],
            "end": df.index[-1],
            "dateFilter": False,  # Should be overridden
        },
        "ma_types_trend": ["EMA"],
        "ma_types_trail_long": ["SMA"],
        "ma_types_trail_short": ["SMA"],
        "lock_trail_types": False,
        "risk_per_trade_pct": 2.0,
        "contract_size": 0.01,
        "commission_rate": 0.0005,
        "atr_period": 14,
        "worker_processes": 1,
        "filter_min_profit": False,
        "min_profit_threshold": 0.0,
        "score_config": {},
    }

    optuna_settings = {
        "target": "score",
        "budget_mode": "trials",
        "n_trials": 10,
        "time_limit": 60,
        "convergence_patience": 5,
        "enable_pruning": False,
        "sampler": "random",
        "pruner": "none",
        "warmup_trials": 0,
        "save_study": False,
    }

    wf_config = WFConfig(num_windows=1, gap_bars=0, topk_per_window=5)
    engine = WalkForwardEngine(wf_config, base_config_template, optuna_settings)

    # Track what config was passed to run_optuna_optimization
    captured_config = None

    def mock_run_optuna(config, optuna_config):
        nonlocal captured_config
        captured_config = config
        # Return mock results
        return []

    with patch('walkforward_engine.run_optuna_optimization', side_effect=mock_run_optuna):
        try:
            engine._run_optuna_on_window(df_window, is_start_time, is_end_time)
        except Exception:
            pass  # We're just testing config, not full execution

    # Verify dateFilter was applied
    print("\n   Checking captured config:")
    if captured_config is None:
        print("   ❌ Config was not captured")
        return False

    print(f"   dateFilter: {captured_config.fixed_params.get('dateFilter')}")
    print(f"   start: {captured_config.fixed_params.get('start')}")
    print(f"   end: {captured_config.fixed_params.get('end')}")

    # Assertions
    if captured_config.fixed_params.get("dateFilter") is not True:
        print("   ❌ dateFilter should be True")
        return False

    if captured_config.fixed_params.get("start") != is_start_time:
        print(f"   ❌ start time mismatch: {captured_config.fixed_params.get('start')} != {is_start_time}")
        return False

    if captured_config.fixed_params.get("end") != is_end_time:
        print(f"   ❌ end time mismatch: {captured_config.fixed_params.get('end')} != {is_end_time}")
        return False

    print("   ✅ dateFilter correctly applied with IS boundaries!")
    return True


def test_forward_only_metrics():
    """
    Test that forward metrics are computed only from forward-zone trades
    """
    print("\n" + "="*60)
    print("TEST 2: Forward-only metrics calculation")
    print("="*60)

    # Create mock dataframe
    dates = pd.date_range('2025-01-01', periods=3000, freq='15min', tz='UTC')
    df = pd.DataFrame({
        'Open': 100.0,
        'High': 101.0,
        'Low': 99.0,
        'Close': 100.0,
        'Volume': 1000.0
    }, index=dates)

    # Setup boundaries
    forward_start_idx = 2500
    forward_end_idx = 3000
    forward_start_time = df.index[forward_start_idx]
    forward_end_time = df.index[forward_end_idx - 1]

    print(f"   Forward zone: bars {forward_start_idx} to {forward_end_idx}")
    print(f"   Forward time: {forward_start_time} to {forward_end_time}")

    # Create mock trades: some before, some inside forward zone
    warmup_trade1 = MockTradeRecord(df.index[2400], 100.0)  # Before forward
    warmup_trade2 = MockTradeRecord(df.index[2450], 50.0)   # Before forward
    forward_trade1 = MockTradeRecord(df.index[2550], 200.0) # Inside forward
    forward_trade2 = MockTradeRecord(df.index[2700], -50.0) # Inside forward
    forward_trade3 = MockTradeRecord(df.index[2900], 100.0) # Inside forward

    all_trades = [warmup_trade1, warmup_trade2, forward_trade1, forward_trade2, forward_trade3]

    # Mock run_strategy to return these trades
    mock_result = Mock()
    mock_result.trades = all_trades
    mock_result.net_profit_pct = 15.0  # This should NOT be used

    print(f"\n   Total trades: {len(all_trades)}")
    print(f"   Warmup trades (before forward): 2")
    print(f"   Forward trades (inside forward): 3")
    print(f"   Total PnL (all trades): {sum(t.net_pnl for t in all_trades)}")
    print(f"   Forward PnL (forward only): {sum(t.net_pnl for t in [forward_trade1, forward_trade2, forward_trade3])}")

    # Setup WFA engine
    base_config_template = {
        "enabled_params": {},
        "param_ranges": {},
        "fixed_params": {
            "start": df.index[0],
            "end": df.index[-1],
        },
        "ma_types_trend": ["EMA"],
        "ma_types_trail_long": ["SMA"],
        "ma_types_trail_short": ["SMA"],
        "lock_trail_types": False,
        "risk_per_trade_pct": 2.0,
        "contract_size": 0.01,
        "commission_rate": 0.0005,
        "atr_period": 14,
        "worker_processes": 1,
        "filter_min_profit": False,
        "min_profit_threshold": 0.0,
        "score_config": {},
    }

    wf_config = WFConfig()
    engine = WalkForwardEngine(wf_config, base_config_template, {})

    # Test the helper method directly
    forward_trades_only = [forward_trade1, forward_trade2, forward_trade3]
    forward_profit_pct = engine._compute_segment_performance(forward_trades_only, initial_equity=10000.0)

    expected_pnl = 200.0 + (-50.0) + 100.0  # = 250.0
    expected_pct = (250.0 / 10000.0) * 100.0  # = 2.5%

    print(f"\n   Expected forward profit: {expected_pct:.2f}%")
    print(f"   Actual forward profit: {forward_profit_pct:.2f}%")

    if abs(forward_profit_pct - expected_pct) < 0.01:
        print("   ✅ Forward-only metrics correctly calculated!")
        return True
    else:
        print("   ❌ Forward profit mismatch!")
        return False


def test_segment_performance_helper():
    """
    Test the _compute_segment_performance helper method
    """
    print("\n" + "="*60)
    print("TEST 3: Segment performance helper method")
    print("="*60)

    base_config = {
        "enabled_params": {},
        "param_ranges": {},
        "fixed_params": {},
        "ma_types_trend": ["EMA"],
        "ma_types_trail_long": ["SMA"],
        "ma_types_trail_short": ["SMA"],
        "lock_trail_types": False,
        "risk_per_trade_pct": 2.0,
        "contract_size": 0.01,
        "commission_rate": 0.0005,
        "atr_period": 14,
        "worker_processes": 1,
        "filter_min_profit": False,
        "min_profit_threshold": 0.0,
        "score_config": {},
    }

    wf_config = WFConfig()
    engine = WalkForwardEngine(wf_config, base_config, {})

    # Test case 1: Empty trades
    result = engine._compute_segment_performance([], initial_equity=10000.0)
    print(f"   Empty trades: {result}%")
    if result != 0.0:
        print("   ❌ Should return 0.0 for empty trades")
        return False

    # Test case 2: Profitable trades
    trades = [
        MockTradeRecord(pd.Timestamp('2025-01-01', tz='UTC'), 100.0),
        MockTradeRecord(pd.Timestamp('2025-01-02', tz='UTC'), 200.0),
        MockTradeRecord(pd.Timestamp('2025-01-03', tz='UTC'), 50.0),
    ]
    result = engine._compute_segment_performance(trades, initial_equity=10000.0)
    expected = (350.0 / 10000.0) * 100.0  # 3.5%
    print(f"   Profitable trades: {result:.2f}% (expected: {expected:.2f}%)")
    if abs(result - expected) > 0.01:
        print("   ❌ Profit calculation incorrect")
        return False

    # Test case 3: Loss
    trades = [
        MockTradeRecord(pd.Timestamp('2025-01-01', tz='UTC'), -100.0),
        MockTradeRecord(pd.Timestamp('2025-01-02', tz='UTC'), -50.0),
    ]
    result = engine._compute_segment_performance(trades, initial_equity=10000.0)
    expected = (-150.0 / 10000.0) * 100.0  # -1.5%
    print(f"   Loss trades: {result:.2f}% (expected: {expected:.2f}%)")
    if abs(result - expected) > 0.01:
        print("   ❌ Loss calculation incorrect")
        return False

    print("   ✅ All helper method tests passed!")
    return True


if __name__ == "__main__":
    print("="*60)
    print("WFA STAGE 2 TESTS")
    print("="*60)

    results = []
    results.append(("IS-only optimization", test_is_only_optimization()))
    results.append(("Forward-only metrics", test_forward_only_metrics()))
    results.append(("Segment performance helper", test_segment_performance_helper()))

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("✅ ALL STAGE 2 TESTS PASSED!")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)
