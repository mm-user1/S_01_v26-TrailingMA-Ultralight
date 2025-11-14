#!/usr/bin/env python3
"""
Smoke test for WFA warmup logic (Stage 1)
Tests that warmup history is properly preserved and WF zone starts at UI-specified date
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
from pathlib import Path
from walkforward_engine import WFConfig, WalkForwardEngine

def test_wfa_warmup():
    """Test that WFA properly handles warmup and logical boundaries"""

    # Load sample data
    data_file = Path("data/OKX_LINKUSDT.P, 15 2025.02.01-2025.09.09.csv")
    if not data_file.exists():
        print(f"❌ Test data file not found: {data_file}")
        return False

    print(f"📊 Loading test data from: {data_file}")

    # Read CSV
    df = pd.read_csv(data_file)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)

    print(f"   Total bars in file: {len(df)}")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")

    # Define logical test range (middle portion of data to allow warmup)
    # Let's use April-August for logical range
    logical_start = pd.Timestamp("2025-04-01 00:00:00", tz='UTC')
    logical_end = pd.Timestamp("2025-08-01 00:00:00", tz='UTC')

    print(f"\n🎯 Testing with logical range: {logical_start} to {logical_end}")

    # Simulate the warmup-aware filtering that happens in server.py
    main_start_idx = None
    main_end_idx = None

    for idx in range(len(df)):
        if df.index[idx] >= logical_start:
            main_start_idx = idx
            break

    for idx in range(len(df) - 1, -1, -1):
        if df.index[idx] <= logical_end:
            main_end_idx = idx
            break

    if main_start_idx is None or main_end_idx is None:
        print("❌ Could not find logical start/end in data")
        return False

    print(f"   Logical start index: {main_start_idx} ({df.index[main_start_idx]})")
    print(f"   Logical end index: {main_end_idx} ({df.index[main_end_idx]})")

    # Add warmup
    warmup_bars = 1000
    warmup_start_idx = max(0, main_start_idx - warmup_bars)

    print(f"   Warmup start index: {warmup_start_idx} ({df.index[warmup_start_idx]})")
    print(f"   Actual warmup bars: {main_start_idx - warmup_start_idx}")

    # Create warmup-aware dataframe
    df_wf = df.iloc[warmup_start_idx: main_end_idx + 1].copy()

    print(f"   WF dataframe size: {len(df_wf)} bars")
    print(f"   WF dataframe range: {df_wf.index[0]} to {df_wf.index[-1]}")

    # Create WFA config
    base_config_template = {
        "enabled_params": {},
        "param_ranges": {},
        "fixed_params": {
            "start": logical_start,
            "end": logical_end,
            "dateFilter": True,
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

    wf_config = WFConfig(num_windows=3, gap_bars=50, topk_per_window=5)
    engine = WalkForwardEngine(wf_config, base_config_template, optuna_settings)

    print(f"\n🔧 Testing WalkForwardEngine.split_data...")

    try:
        windows, fwd_start, fwd_end = engine.split_data(df_wf)

        print(f"   ✅ split_data succeeded")
        print(f"   Created {len(windows)} windows")
        print(f"   Forward zone: bars {fwd_start} to {fwd_end}")

        # Critical check: first window should start at or very close to logical_start
        first_window = windows[0]
        first_window_time = df_wf.index[first_window.is_start]

        print(f"\n📍 Critical Check - First Window Start:")
        print(f"   Expected (UI start): {logical_start}")
        print(f"   Actual (first IS):   {first_window_time}")
        print(f"   Index: {first_window.is_start}")

        # The first window should start at main_start_idx (relative to df_wf)
        # In df_wf, main_start_idx becomes (main_start_idx - warmup_start_idx)
        expected_idx_in_wf = main_start_idx - warmup_start_idx

        if first_window.is_start == expected_idx_in_wf:
            print(f"   ✅ First window starts at correct logical position!")
        else:
            print(f"   ⚠️  First window index mismatch!")
            print(f"   Expected index in df_wf: {expected_idx_in_wf}")
            print(f"   Actual index: {first_window.is_start}")
            return False

        # Check that warmup data is available before first window
        if first_window.is_start >= warmup_bars:
            print(f"   ✅ Sufficient warmup data available ({first_window.is_start} bars before first IS)")
        else:
            print(f"   ⚠️  Limited warmup: only {first_window.is_start} bars available")

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED - WFA warmup logic working correctly!")
        print("="*60)

        return True

    except Exception as e:
        print(f"   ❌ split_data failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("WFA WARMUP LOGIC TEST (Stage 1)")
    print("="*60 + "\n")

    success = test_wfa_warmup()

    sys.exit(0 if success else 1)
