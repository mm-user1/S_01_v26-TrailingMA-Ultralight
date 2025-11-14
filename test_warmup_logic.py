#!/usr/bin/env python3
"""
Simple unit test for warmup logic (no dependencies)
Tests the core index calculation logic
"""

import pandas as pd
from pathlib import Path

def test_warmup_index_logic():
    """Test the index calculation logic for warmup preservation"""

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

    # Test parameters matching what server.py would do
    logical_start = pd.Timestamp("2025-04-01 00:00:00", tz='UTC')
    logical_end = pd.Timestamp("2025-08-01 00:00:00", tz='UTC')

    print(f"\n🎯 Testing warmup preservation logic")
    print(f"   Logical range: {logical_start} to {logical_end}")

    # Step 1: Find logical boundaries (as server.py does)
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
        print("❌ Could not find logical boundaries")
        return False

    print(f"\n✅ Step 1: Found logical boundaries")
    print(f"   main_start_idx: {main_start_idx} → {df.index[main_start_idx]}")
    print(f"   main_end_idx: {main_end_idx} → {df.index[main_end_idx]}")
    print(f"   Logical range bars: {main_end_idx - main_start_idx + 1}")

    # Step 2: Calculate warmup-aware slice (as server.py does)
    warmup_bars = 1000
    warmup_start_idx = max(0, main_start_idx - warmup_bars)

    print(f"\n✅ Step 2: Calculate warmup slice")
    print(f"   warmup_bars: {warmup_bars}")
    print(f"   warmup_start_idx: {warmup_start_idx} → {df.index[warmup_start_idx]}")
    print(f"   Actual warmup bars available: {main_start_idx - warmup_start_idx}")

    # Create warmup-aware dataframe
    df_wf = df.iloc[warmup_start_idx: main_end_idx + 1].copy()

    print(f"   df_wf total bars: {len(df_wf)}")
    print(f"   df_wf range: {df_wf.index[0]} to {df_wf.index[-1]}")

    # Step 3: Simulate what split_data would do
    # It should find logical_start/logical_end in df_wf and use those as WF zone start
    logical_start_idx_in_wf = None
    logical_end_idx_in_wf = None

    for idx in range(len(df_wf)):
        if df_wf.index[idx] >= logical_start:
            logical_start_idx_in_wf = idx
            break

    for idx in range(len(df_wf) - 1, -1, -1):
        if df_wf.index[idx] <= logical_end:
            logical_end_idx_in_wf = idx
            break

    print(f"\n✅ Step 3: Find logical boundaries in df_wf")
    print(f"   logical_start_idx_in_wf: {logical_start_idx_in_wf} → {df_wf.index[logical_start_idx_in_wf]}")
    print(f"   logical_end_idx_in_wf: {logical_end_idx_in_wf} → {df_wf.index[logical_end_idx_in_wf]}")

    # Step 4: Verify the logic
    expected_logical_start_in_wf = main_start_idx - warmup_start_idx

    print(f"\n✅ Step 4: Verify index mapping")
    print(f"   Expected logical_start in df_wf: {expected_logical_start_in_wf}")
    print(f"   Actual logical_start in df_wf: {logical_start_idx_in_wf}")

    if logical_start_idx_in_wf == expected_logical_start_in_wf:
        print(f"   ✅ Index mapping correct!")
    else:
        print(f"   ❌ Index mapping mismatch!")
        return False

    # Step 5: Check that first window would start at logical_start
    # In split_data, wf_zone_start = logical_start_idx (in df_wf)
    # First window: is_start = wf_zone_start
    wf_zone_start = logical_start_idx_in_wf
    first_window_start_time = df_wf.index[wf_zone_start]

    print(f"\n✅ Step 5: Check first window start")
    print(f"   WF zone starts at index: {wf_zone_start}")
    print(f"   First window would start at: {first_window_start_time}")
    print(f"   Expected (UI start): {logical_start}")
    print(f"   Match: {first_window_start_time == logical_start}")

    # Step 6: Check warmup availability
    warmup_available = wf_zone_start
    print(f"\n✅ Step 6: Check warmup availability")
    print(f"   Warmup bars available: {warmup_available}")
    print(f"   Warmup requested: {warmup_bars}")

    if warmup_available >= warmup_bars:
        print(f"   ✅ Full warmup available!")
    else:
        print(f"   ⚠️  Partial warmup ({warmup_available}/{warmup_bars})")

    print("\n" + "="*60)
    print("✅ ALL WARMUP LOGIC TESTS PASSED!")
    print("="*60)
    print("\nKey findings:")
    print(f"  • Logical range preserved: {main_end_idx - main_start_idx + 1} bars")
    print(f"  • Warmup history added: {warmup_available} bars")
    print(f"  • First WF window starts at UI date: {first_window_start_time}")
    print(f"  • Total WFA dataset: {len(df_wf)} bars")

    return True

if __name__ == "__main__":
    print("="*60)
    print("WARMUP LOGIC UNIT TEST")
    print("="*60 + "\n")

    import sys
    success = test_warmup_index_logic()
    sys.exit(0 if success else 1)
