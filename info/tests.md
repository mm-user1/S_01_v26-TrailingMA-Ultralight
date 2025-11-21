# Reference Tests - Multi-Strategy Migration

## Purpose

This file defines **reference tests** that MUST be run after each migration phase to ensure no behavior changes.

**Critical Rule:** If reference test fails after a migration phase → **DO NOT PROCEED** until fixed.

---

## How to Conduct Reference Test

### Method

1. **Use default parameters** from strategy's `get_param_definitions()`
2. **Use specified CSV file** with market data
3. **Run single backtest** (not optimization)
4. **Compare KPIs** with expected values below
5. **Match tolerance:** ±0.01% for percentages, exact match for trade count

### Command

```bash
cd src
python run_backtest.py --csv <CSV_FILE_PATH> --strategy <STRATEGY_ID>
```

Or via strategy class directly:

```python
from strategies.<strategy_module> import <StrategyClass>
from backtest_engine import load_data

df = load_data('<CSV_FILE_PATH>')

# Get default params
param_defs = StrategyClass.get_param_definitions()
params = {k: v['default'] for k, v in param_defs.items()}

# Run simulation
strategy = StrategyClass(params)
result = strategy.simulate(df)

print(f"Net Profit: {result['net_profit_pct']:.2f}%")
print(f"Max DD: {result['max_drawdown_pct']:.2f}%")
print(f"Total Trades: {result['total_trades']}")
```

### Validation

Compare output with "Expected Results" below:

- **Net Profit %:** Must match within ±0.01%
- **Max Drawdown %:** Must match within ±0.01%
- **Total Trades:** Must match exactly (integer)

**If any value differs → Migration phase has a bug!**

---

## S_01 TrailingMA v26 Ultralight

### Test Configuration

**CSV File:** `data/"OKX_LINKUSDT.P, 15 2025.05.01-2025.11.20.csv"`

**Strategy ID:** `s01_trailing_ma`

**Default Parameters:**

- Date range: from 2025-06-15 to 2025-11-15
- Parameters:
  MA Type = SMA
  MA Length = 300
  Close Count Long = 9
  Close Count Short = 5
  Stop Long ATR = 2.0
  Stop Long RR = 3
  Stop Long LP = 2
  Stop Short ATR = 2.0
  Stop Short RR = 3
  Stop Short LP = 2
  Stop Long Max % = 7.0
  Stop Short Max % = 10.0
  Stop Long Max Days = 5
  Stop Short Max Days = 2
  Trail RR Long = 1
  Trail RR Short = 1
  Trail MA Long Type = EMA
  Trail MA Long Length = 90
  Trail MA Long Offset = -0.5
  Trail MA Short Type = EMA
  Trail MA Short Length = 190
  Trail MA Short Offset = 2.0

### Expected Results

```
Strategy: S_01 TrailingMA v26 Ultralight
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Performance Metrics:
├─ Net Profit:        230.75%
├─ Max Drawdown:      20.03%
├─ Total Trades:      93
```

---

## S_03 Reversal v07 Light

### Test Configuration

**CSV File:** `data/"OKX_LINKUSDT.P, 15 2025.05.01-2025.11.20.csv"` *(same as S_01)*

**Strategy ID:** `s03_reversal`

**Default Parameters:**
- Date range: from 2025-06-15 to 2025-11-15
- Parameters:
  Fast MA Type = SMA
  Fast MA Length = 100
  Slow MA Length = 0 (disabled)
  Trend MA Type = SMA
  Trend MA Length = 100
  Close Count Long = 4
  Close Count Short = 5
  Equity % = 100
  Contract Size = 0.01
  Commission Rate = 0.0005

### Expected Results

```
Strategy: S_03 Reversal v07 Light
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Performance Metrics:
├─ Net Profit:        83.56%
├─ Max Drawdown:      35.34%
├─ Total Trades:      224
├─ Winning Trades:    70
├─ Losing Trades:     154
├─ Win Rate:          31.25%
├─ Profit Factor:     1.26
└─ Sharpe Ratio:      0.15

Special Checks (Reversal Strategy):
├─ Always in position: YES
├─ Gaps (flat periods): 0
└─ Position changes: 224
```

---

## Additional Test Scenarios

### Optimization Test (Phase 4+)

**Purpose:** Verify optimizer works with strategy

**Method:**
```bash
# Small grid to test optimizer
cd src
python -c "
from optimizer_engine import run_optimization, OptimizationConfig

config = OptimizationConfig(
    csv_file=open(../data/"OKX_LINKUSDT.P, 15 2025.05.01-2025.11.20.csv"),
    strategy_id='s01_trailing_ma',
    enabled_params={'maLength': True},
    param_ranges={'maLength': (30, 60, 15)},
    fixed_params={...},  # All other params at defaults
    worker_processes=2
)

results = run_optimization(config)
print(f'✅ Optimization completed: {len(results)} results')
"
```

**Expected:**
- Runs without errors
- Returns 3 results (30, 45, 60)
- Best result has Net Profit within ±0.5% of full optimization

---

## Performance Benchmarks

**Purpose:** Ensure refactoring doesn't degrade performance

### Single Backtest Speed

**Test:**

```bash
time python run_backtest.py --csv data/"OKX_LINKUSDT.P, 15 2025.05.01-2025.11.20.csv" --strategy s01_trailing_ma
```

**Expected:**
- Phase 1-3: < 3 seconds (should be similar to original)
- Phase 4-6: < 3 seconds (no performance loss)

### Optimization Speed (1000 combinations)

**Test:**
```python
# Grid with ~1000 combinations
enabled_params = {
    'maLength': True,
    'closeCountLong': True,
    'stopLongX': True
}
param_ranges = {
    'maLength': (30, 60, 5),       # 7 values
    'closeCountLong': (3, 10, 1),  # 8 values
    'stopLongX': (1.5, 3.5, 0.25)  # 9 values
}
# Total: 7 * 8 * 9 = 504 combinations

# Time it
import time
start = time.time()
results = run_optimization(config)
elapsed = time.time() - start
print(f'Time: {elapsed:.1f}s')
```

**Expected:**
- Phase 4: < 120 seconds with 6 workers (should match original)
- If slower → check caching is working

---

## Regression Test Checklist

Run after EVERY phase:

### Phase 1: Extract Indicators
- [ ] S_01 reference test passes ✅
- [ ] Backtest speed unchanged
- [ ] Imports work: `python -c "import indicators"`

### Phase 2: Create Base Strategy
- [ ] S_01 reference test passes ✅ (no changes to backtest yet)
- [ ] Imports work: `python -c "from strategies.base_strategy import BaseStrategy"`
- [ ] ABC enforcement works (cannot instantiate BaseStrategy)

### Phase 3: Extract S_01
- [ ] S_01 reference test passes ✅ **CRITICAL**
- [ ] Net Profit matches baseline
- [ ] Max DD matches baseline
- [ ] Trade count matches baseline
- [ ] Backtest via CLI works
- [ ] Backtest via old code works (backward compatibility)

### Phase 4: Refactor Optimizer
- [ ] S_01 reference test passes ✅
- [ ] Small optimization test (3 combos) works
- [ ] CSV export works
- [ ] Optimization speed acceptable

### Phase 5: Add S_03
- [ ] S_01 reference test STILL passes ✅ (no regression)
- [ ] S_03 baseline established ✅
- [ ] S_03 always in position (no flat gaps)
- [ ] S_03 reversal logic works
- [ ] Both strategies in StrategyRegistry

### Phase 6: Update UI/API
- [ ] S_01 reference test passes ✅
- [ ] S_03 reference test passes ✅
- [ ] UI: Strategy selector works
- [ ] UI: Both strategies run
- [ ] API: /api/strategies returns both
- [ ] CLI: --strategy flag works

---

## Troubleshooting Failed Tests

### Net Profit Differs by > 0.01%

**Common Causes:**
1. Counter logic changed (check increment/reset conditions)
2. Floating point rounding (ensure same precision)
3. Position sizing calculation changed
4. Commission calculation changed

**Debug Steps:**
```python
# Print trade-by-trade comparison
old_trades = run_old_version(...)
new_trades = run_new_version(...)

for i, (old, new) in enumerate(zip(old_trades, new_trades)):
    if abs(old.net_pnl - new.net_pnl) > 0.001:
        print(f"Trade {i}: OLD PnL={old.net_pnl:.4f}, NEW PnL={new.net_pnl:.4f}")
        print(f"  Entry: {old.entry_price} vs {new.entry_price}")
        print(f"  Exit: {old.exit_price} vs {new.exit_price}")
```

### Trade Count Differs

**Common Causes:**
1. Entry conditions changed
2. Counter logic changed
3. Stop/target hit logic changed

**Debug:**
- Print entry signals: `print(f"Bar {idx}: should_long={should_long}")`
- Compare signals bar-by-bar with old version

### Sharpe Ratio Differs

**Common Causes:**
1. Equity curve calculation changed
2. Returns calculation changed

**Debug:**
- Compare equity_curve arrays element-by-element
- Check if equity tracking handles unrealized PnL correctly

---

## Test Data Requirements

### CSV File Format

**Required Columns:**
- `Open`, `High`, `Low`, `Close`, `Volume`
- Index: DatetimeIndex

**Minimum Requirements:**
- At least 500 bars (for meaningful MA calculations)
- Includes both trending and ranging periods
- Realistic price/volume data

---

## Notes

**Why Reference Tests are Critical:**

During refactoring, it's easy to introduce subtle bugs:
- Off-by-one errors in loops
- Counter reset logic changes
- Floating point precision issues
- State management bugs

Reference tests catch these immediately, preventing:
- Accumulation of bugs across phases
- Hours of debugging later
- Incorrect baseline for future strategies

**Frequency:**
Run reference test:
- After EVERY migration phase
- After ANY change to strategy logic
- Before committing code
- Before merging PRs

---

**End of tests.md**
