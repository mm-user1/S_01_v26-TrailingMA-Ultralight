#!/usr/bin/env python3
"""
Simplified Stage 2 tests - verify code changes without full dependencies
"""

import pandas as pd
import re

print("="*60)
print("WFA STAGE 2 - CODE VERIFICATION")
print("="*60)

def test_optuna_signature():
    """Verify _run_optuna_on_window has correct signature"""
    print("\n✅ TEST 1: _run_optuna_on_window signature")

    with open('src/walkforward_engine.py', 'r') as f:
        content = f.read()

    # Check signature includes is_start_time and is_end_time
    pattern = r'def _run_optuna_on_window\s*\(\s*self\s*,\s*df_window.*is_start_time.*is_end_time'
    if re.search(pattern, content, re.DOTALL):
        print("   ✅ Signature updated correctly")
        return True
    else:
        print("   ❌ Signature not found or incorrect")
        return False


def test_datefilter_applied():
    """Verify dateFilter is applied in _run_optuna_on_window"""
    print("\n✅ TEST 2: dateFilter application")

    with open('src/walkforward_engine.py', 'r') as f:
        content = f.read()

    # Check for dateFilter override
    if 'fixed_params["dateFilter"] = True' in content:
        print("   ✅ dateFilter set to True")
    else:
        print("   ❌ dateFilter not set")
        return False

    # Check for start/end override
    if 'fixed_params["start"] = is_start_time' in content:
        print("   ✅ start time override found")
    else:
        print("   ❌ start time override not found")
        return False

    if 'fixed_params["end"] = is_end_time' in content:
        print("   ✅ end time override found")
    else:
        print("   ❌ end time override not found")
        return False

    return True


def test_is_time_calculation():
    """Verify IS time boundaries are calculated in run_wf_optimization"""
    print("\n✅ TEST 3: IS time boundary calculation")

    with open('src/walkforward_engine.py', 'r') as f:
        content = f.read()

    # Check for is_start_time calculation
    if 'is_start_time = df.index[window.is_start]' in content:
        print("   ✅ is_start_time calculated from df.index")
    else:
        print("   ❌ is_start_time calculation not found")
        return False

    # Check for is_end_time calculation
    if 'is_end_time = df.index[window.is_end - 1]' in content:
        print("   ✅ is_end_time calculated from df.index")
    else:
        print("   ❌ is_end_time calculation not found")
        return False

    # Check that times are passed to _run_optuna_on_window
    if 'is_df_with_warmup, is_start_time, is_end_time' in content:
        print("   ✅ IS times passed to _run_optuna_on_window")
    else:
        print("   ❌ IS times not passed correctly")
        return False

    return True


def test_forward_time_calculation():
    """Verify forward time boundaries are calculated"""
    print("\n✅ TEST 4: Forward time boundary calculation")

    with open('src/walkforward_engine.py', 'r') as f:
        content = f.read()

    # Check for forward time calculations
    if 'forward_start_time = df.index[fwd_start]' in content:
        print("   ✅ forward_start_time calculated")
    else:
        print("   ❌ forward_start_time calculation not found")
        return False

    if 'forward_end_time = df.index[fwd_end - 1]' in content:
        print("   ✅ forward_end_time calculated")
    else:
        print("   ❌ forward_end_time calculation not found")
        return False

    return True


def test_forward_trade_filtering():
    """Verify forward trades are filtered by time"""
    print("\n✅ TEST 5: Forward trade filtering")

    with open('src/walkforward_engine.py', 'r') as f:
        content = f.read()

    # Check for trade filtering
    pattern = r'forward_trades\s*=\s*\[\s*trade\s+for\s+trade\s+in\s+result\.trades'
    if re.search(pattern, content):
        print("   ✅ Forward trades filtered from result.trades")
    else:
        print("   ❌ Forward trade filtering not found")
        return False

    # Check for time-based filtering
    if 'forward_start_time <= trade.entry_time <= forward_end_time' in content:
        print("   ✅ Time-based filtering applied")
    else:
        print("   ❌ Time-based filtering not found")
        return False

    return True


def test_segment_performance_method():
    """Verify _compute_segment_performance helper exists"""
    print("\n✅ TEST 6: Segment performance helper method")

    with open('src/walkforward_engine.py', 'r') as f:
        content = f.read()

    # Check for method definition
    if 'def _compute_segment_performance(self, trades' in content:
        print("   ✅ _compute_segment_performance method defined")
    else:
        print("   ❌ Helper method not found")
        return False

    # Check that it's used in forward test
    if 'self._compute_segment_performance(' in content:
        print("   ✅ Helper method is called")
    else:
        print("   ❌ Helper method not used")
        return False

    return True


def test_forward_uses_helper():
    """Verify forward test uses helper instead of result.net_profit_pct"""
    print("\n✅ TEST 7: Forward test uses computed metrics")

    with open('src/walkforward_engine.py', 'r') as f:
        content = f.read()

    # Find forward test section
    forward_section_match = re.search(
        r'# Step 4: Forward Test.*?forward_params\.append\(agg\.params\)',
        content,
        re.DOTALL
    )

    if not forward_section_match:
        print("   ❌ Forward test section not found")
        return False

    forward_section = forward_section_match.group(0)

    # Check that helper is used
    if '_compute_segment_performance' in forward_section:
        print("   ✅ Forward uses _compute_segment_performance")
    else:
        print("   ❌ Forward doesn't use helper method")
        return False

    # Check that forward_trades are used
    if 'forward_trades' in forward_section:
        print("   ✅ Forward uses filtered trades")
    else:
        print("   ❌ Forward doesn't use filtered trades")
        return False

    return True


# Run all tests
tests = [
    ("_run_optuna_on_window signature", test_optuna_signature),
    ("dateFilter application", test_datefilter_applied),
    ("IS time calculation", test_is_time_calculation),
    ("Forward time calculation", test_forward_time_calculation),
    ("Forward trade filtering", test_forward_trade_filtering),
    ("Segment performance helper", test_segment_performance_method),
    ("Forward uses helper", test_forward_uses_helper),
]

results = []
for name, test_func in tests:
    try:
        passed = test_func()
        results.append((name, passed))
    except Exception as e:
        print(f"   ❌ Test raised exception: {e}")
        results.append((name, False))

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
    print("\n✅ ALL STAGE 2 CODE VERIFICATIONS PASSED!")
    print("\nStage 2 implementation complete:")
    print("  • IS-only optimization via dateFilter")
    print("  • Forward-only metrics from filtered trades")
    print("  • Segment performance helper method")
    import sys
    sys.exit(0)
else:
    print("\n❌ SOME VERIFICATIONS FAILED")
    import sys
    sys.exit(1)
