# Migration Prompt 1: Extract Indicators Module

## Context

**Current Phase:** Phase 1 of 6
**Migration Checklist:** See `migration_checklist.md` - Phase 1
**Duration:** 0.5-1 day

**Goal:** Extract all technical indicator functions from `backtest_engine.py` into a new `indicators.py` module.

**Why:** This establishes a foundation for shared code between strategies. Indicators are pure functions (no state) that will be used by multiple strategies.

---

## Current State

The file `src/backtest_engine.py` contains 11 moving average calculation functions plus ATR:
- `sma()`, `ema()`, `hma()`, `wma()`, `vwma()`
- `alma()`, `kama()`, `tma()`, `t3()`, `dema()`
- `vwap()`, `get_ma()` (unified interface)
- `atr()` (Average True Range)

These functions are currently embedded in backtest_engine.py (lines ~188-331) but are general-purpose indicators that should be shared.

---

## Task

Create a new module `src/indicators.py` and move all indicator calculation functions there.

---

## Detailed Steps

### Step 1: Create indicators.py

Create a new file `/src/indicators.py` with the following structure:

```python
"""
Technical Indicators Library

This module contains implementations of various technical indicators
used across trading strategies.

All functions are pure (stateless) and operate on pandas Series.
"""

import numpy as np
import pandas as pd
from typing import Optional


def sma(series: pd.Series, length: int) -> pd.Series:
    """
    Simple Moving Average

    Args:
        series: Price series (usually Close)
        length: Period for calculation

    Returns:
        Series with SMA values
    """
    return series.rolling(window=length).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    """
    Exponential Moving Average

    Args:
        series: Price series
        length: Period for calculation

    Returns:
        Series with EMA values
    """
    return series.ewm(span=length, adjust=False).mean()


# ... (continue for all 11 MA types + ATR)
```

### Step 2: Copy Functions from backtest_engine.py

**Copy these functions to indicators.py:**

1. **sma** - Simple Moving Average
2. **ema** - Exponential Moving Average
3. **hma** - Hull Moving Average
4. **wma** - Weighted Moving Average
5. **vwma** - Volume-Weighted Moving Average
6. **alma** - Arnaud Legoux Moving Average
7. **kama** - Kaufman Adaptive Moving Average
8. **tma** - Triangular Moving Average
9. **t3** - T3 Moving Average
10. **dema** - Double Exponential Moving Average
11. **vwap** - Volume-Weighted Average Price
12. **get_ma** - Unified MA interface (routes to correct function based on type)
13. **atr** - Average True Range

**Important:**
- Copy the ENTIRE function body
- Preserve all docstrings
- Keep function signatures identical
- Do NOT modify logic

**Location in backtest_engine.py:**
- Search for `def sma(` to find start of indicators section
- Copy until the end of `atr()` function

### Step 3: Add Type Hints and Documentation

For each function, ensure:

```python
def function_name(
    series: pd.Series,
    length: int,
    # ... other params
) -> pd.Series:
    """
    Brief description.

    Args:
        series: Description
        length: Description

    Returns:
        Description of return value
    """
    # Implementation
```

### Step 4: Update backtest_engine.py

**At the top of backtest_engine.py**, add imports:

```python
from indicators import (
    sma, ema, hma, wma, vwma,
    alma, kama, tma, t3, dema,
    vwap, get_ma, atr
)
```

**Remove the old function definitions** from backtest_engine.py:
- Delete all 11 MA function bodies
- Delete ATR function body
- Keep only imports

**Verify:**
- Run `python -c "import backtest_engine"` - should work without errors
- No "NameError" or "ImportError"

### Step 5: Update optimizer_engine.py

**At the top of optimizer_engine.py**, add imports:

```python
from indicators import get_ma, atr
```

**Verify:**
- Run `python -c "import optimizer_engine"` - should work

### Step 6: Update optuna_engine.py

Check if optuna_engine.py imports from backtest_engine. If so:

```python
from indicators import get_ma, atr
```

---

## Acceptance Criteria

Before committing, verify:

1. **File Created:**
   - [ ] `/src/indicators.py` exists
   - [ ] Contains all 13 functions (11 MAs + get_ma + ATR)
   - [ ] All functions have docstrings and type hints

2. **Imports Updated:**
   - [ ] `backtest_engine.py` imports from `indicators`
   - [ ] `optimizer_engine.py` imports from `indicators`
   - [ ] No broken imports in any file

3. **Functions Removed:**
   - [ ] No duplicate function definitions
   - [ ] Old indicator functions deleted from `backtest_engine.py`

4. **No Logic Changes:**
   - [ ] Function bodies identical to originals
   - [ ] No behavior modifications

5. **Imports Work:**
   ```bash
   cd src
   python -c "import indicators"
   python -c "import backtest_engine"
   python -c "import optimizer_engine"
   python -c "import optuna_engine"
   ```
   All should succeed without errors.

---

## Testing

### Test 1: Import Test

```bash
cd src
python -c "
from indicators import sma, ema, get_ma, atr
import pandas as pd

# Test SMA
s = pd.Series([1, 2, 3, 4, 5])
result = sma(s, 3)
print(f'SMA test: {result.iloc[-1]:.2f}')  # Should be 4.0

# Test get_ma
result = get_ma(s, 'SMA', 3)
print(f'get_ma test: {result.iloc[-1]:.2f}')  # Should be 4.0

print('✅ Indicators module works!')
"
```

### Test 2: Reference Test (CRITICAL)

**Run the S_01 baseline test:**

```bash
cd src
python run_backtest.py --csv ../data/OKX_LINKUSDT.P,\ 15...csv
```

**Compare results with baseline recorded in tests.md:**

- Net Profit %: `Expected: _____ Got: _____` ✅ / ❌
- Max DD %: `Expected: _____ Got: _____` ✅ / ❌
- Total Trades: `Expected: _____ Got: _____` ✅ / ❌

**If ANY value differs:**
- ❌ DO NOT COMMIT
- Debug: Check if all functions copied correctly
- Verify no logic changes introduced
- Re-test until values match EXACTLY

**If all values match:**
- ✅ Proceed to commit

### Test 3: Optimization Test

Run a small optimization to verify optimizer still works:

```bash
cd src
python -c "
from optimizer_engine import run_optimization, OptimizationConfig
from backtest_engine import load_data

# Create minimal config
df = load_data('../data/OKX_LINKUSDT.P, 15...csv')

# If this runs without errors, optimizer is OK
print('✅ Optimizer imports work!')
"
```

---

## Commit Message

```bash
git add src/indicators.py src/backtest_engine.py src/optimizer_engine.py src/optuna_engine.py
git commit -m "Phase 1: Extract indicators module

- Create indicators.py with all MA functions and ATR
- Move 11 MA types + get_ma() + atr() from backtest_engine.py
- Update imports in backtest_engine.py, optimizer_engine.py, optuna_engine.py
- Reference test: S_01 results unchanged ✅

Files changed:
- NEW: src/indicators.py (13 functions)
- MODIFIED: src/backtest_engine.py (import from indicators)
- MODIFIED: src/optimizer_engine.py (import from indicators)
- MODIFIED: src/optuna_engine.py (import from indicators)"
```

---

## Troubleshooting

### Issue: ImportError: cannot import name 'sma'

**Cause:** Function name mismatch or typo.

**Solution:**
- Check `indicators.py` - is function named exactly `sma`?
- Check `backtest_engine.py` - is import exactly `from indicators import sma`?
- Case-sensitive: `sma` ≠ `SMA`

### Issue: Reference test results differ

**Cause:** Logic was accidentally modified during copy.

**Solution:**
- Use `diff` to compare old and new function bodies
- Ensure no whitespace/indentation changes that affect logic
- Check pandas/numpy version compatibility

### Issue: Circular import

**Cause:** `indicators.py` imports from `backtest_engine.py` or vice versa.

**Solution:**
- `indicators.py` should have NO imports from project modules
- Only external imports: `pandas`, `numpy`, `math`
- `backtest_engine.py` imports from `indicators.py` (one-way dependency)

---

## Next Steps

After Phase 1 is complete and committed:

**Checklist:**
- [ ] Reference test passes ✅
- [ ] Committed to git ✅
- [ ] Proceed to **Phase 2: Create Base Strategy Contract**
  - See `migration_prompt_2.md`
  - See `migration_checklist.md` - Phase 2

---

## Additional Notes

**Why pure functions?**

Indicators in `indicators.py` are pure functions:
- No global state
- No class attributes
- Input → Output
- Deterministic (same input = same output)

This makes them:
- Easy to test
- Easy to cache (in optimizer)
- Easy to reuse across strategies

**What belongs in indicators.py:**

✅ **YES:**
- Standard technical indicators (SMA, RSI, MACD, etc.)
- Indicators used by 2+ strategies
- Pure mathematical transformations

❌ **NO:**
- Strategy-specific logic
- Functions with state (counters, flags)
- Business logic (entry/exit conditions)

**When to add new indicators:**

If translating a Pine strategy that uses a new indicator:
1. Check if it's a standard indicator (RSI, Bollinger Bands, etc.) → add to `indicators.py`
2. Check if it's custom AND will be used by multiple strategies → add to `indicators.py`
3. If only used by one strategy → keep in strategy file as private method

---

**End of Migration Prompt 1**
