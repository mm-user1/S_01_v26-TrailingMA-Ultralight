# Instructions for AI Coding Agents (Claude Code / GPT Codex)

## Purpose

This document provides **critical instructions** for AI agents working on this codebase. These rules ensure consistency, correctness, and maintainability.

---

## Core Principles

### 1. **Preserve Exact Behavior**

When refactoring or migrating code:
- **NEVER** change logic unless explicitly requested
- Copy functions character-for-character when moving between files
- Preserve variable names, order of operations, and edge cases
- Run reference tests after EVERY change

### 2. **Test-Driven Development**

- Read `tests.md` BEFORE starting work
- Run reference test BEFORE making changes (establish baseline)
- Run reference test AFTER making changes (verify no regression)
- If test fails → **STOP** and debug before proceeding

### 3. **Follow Migration Checklist**

- Migration phases are in `migration_checklist.md`
- Each phase has a detailed prompt in `migration_prompt_N.md`
- Complete phases IN ORDER - do not skip ahead
- Check off items in checklist as you complete them

---

## Naming Conventions

### Python Code (Backend)

**Variables and Functions:**
- Use `snake_case` for variables: `ma_length`, `stop_price`, `counter_close_long`
- Use `snake_case` for functions: `calculate_entry()`, `should_long()`
- Use `snake_case` for methods: `_prepare_data()`, `_run_simulation()`

**Classes:**
- Use `PascalCase` for classes: `BaseStrategy`, `S01TrailingMA`, `StrategyRegistry`

**Constants:**
- Use `UPPER_SNAKE_CASE` for constants: `STRATEGY_ID`, `MAX_WORKERS`, `DEFAULT_CONFIG`

**Private:**
- Prefix with `_` for internal methods: `_validate_params()`, `_calculate_max_drawdown()`
- Prefix with `_` for internal variables: `_ma_cache`, `_atr_values`

### Frontend/API (JavaScript/JSON)

**API Parameters:**
- Use `camelCase` for API parameters and as keys in `get_param_definitions()`
- Examples: `maLength`, `stopLongX`, `closeCountLong`

**Strategy `get_param_definitions()` format:**
```python
@staticmethod
def get_param_definitions() -> Dict[str, Dict[str, Any]]:
    return {
        'maLength': {  # camelCase key (matches frontend/API)
            'default': 45,
            'type': 'int',
            'min': 1,
            'max': 500,
            'description': 'Moving average period'
        },
        'stopLongX': {
            'default': 2.0,
            'type': 'float',
            'min': 0.5,
            'max': 5.0,
            'description': 'Long stop ATR multiplier'
        }
        # ...
    }
```

**Note:** Keys in `get_param_definitions()` use camelCase for consistency with API. Inside strategy class, you can convert to snake_case if preferred:
```python
def __init__(self, params: Dict[str, Any]):
    super().__init__(params)
    self.ma_length = params['maLength']  # Convert to snake_case internally
    self.stop_long_x = params['stopLongX']
```

---

## File Organization

### Where to Put Code

**indicators.py:**
- ✅ Standard technical indicators (SMA, EMA, RSI, ATR, etc.)
- ✅ Indicators used by 2+ strategies
- ✅ Pure functions (no state, no side effects)
- ❌ Strategy-specific logic
- ❌ Functions with state

**strategies/<strategy>.py:**
- ✅ Strategy entry/exit logic
- ✅ Strategy-specific indicators (used by only this strategy)
- ✅ State management (counters, flags)
- ✅ Parameter definitions
- ❌ General-purpose utilities

**backtest_engine.py:**
- ✅ Generic simulation loop
- ✅ Equity tracking
- ✅ Metrics calculation (drawdown, sharpe, etc.)
- ✅ CSV loading
- ❌ Strategy-specific logic

**optimizer_engine.py / optuna_engine.py:**
- ✅ Parameter grid generation
- ✅ Worker pool management
- ✅ Caching infrastructure
- ❌ Strategy logic (call strategy.simulate())

---

## Strategy Implementation Guidelines

### When Creating a New Strategy

1. **Read Pine Script** (if translating from Pine)
   - See `pine_guidelines.md` for Pine→Python conventions

2. **Create `/src/strategies/sXX_strategy_name.py`:**
   ```python
   from .base_strategy import BaseStrategy
   from indicators import get_ma, atr  # Import only what's needed
   import math
   import numpy as np
   import pandas as pd
   ```

3. **Set Metadata:**
   ```python
   class SXX_StrategyName(BaseStrategy):
       STRATEGY_ID = "sxx_strategy_name"  # snake_case
       STRATEGY_NAME = "S_XX Strategy Name"  # Human-readable
       VERSION = "01"  # Strategy version
   ```

4. **Implement ALL Abstract Methods:**
   - `_validate_params()`
   - `should_long(idx)`
   - `should_short(idx)`
   - `calculate_entry(idx, direction)`
   - `calculate_position_size(...)`
   - `should_exit(idx, position_info)`
   - `_prepare_data(df, cached_data)`
   - `get_param_definitions()` - classmethod

5. **Initialize State in `__init__`:**
   ```python
   def __init__(self, params):
       super().__init__(params)

       # Indicator caches
       self._ma = None
       self._atr = None

       # Strategy state
       self.counter_long = 0
       self.counter_short = 0
       self.trail_price = math.nan
       self.trail_activated = False
   ```

6. **Handle Caching in `_prepare_data`:**
   ```python
   def _prepare_data(self, df, cached_data):
       # Store OHLCV data
       self.df = df
       self.close = df['Close'].to_numpy()
       # ... other arrays

       if cached_data:
           # Use pre-computed (optimization mode)
           self._ma = cached_data['ma_cache'][(self.params['maType'], self.params['maLength'])]
       else:
           # Compute on-the-fly (single backtest mode)
           from indicators import get_ma
           self._ma = get_ma(df['Close'], self.params['maType'], self.params['maLength']).to_numpy()
   ```

7. **Define Cache Requirements:**
   ```python
   @classmethod
   def get_cache_requirements(cls, param_combinations):
       ma_configs = set()
       for combo in param_combinations:
           ma_configs.add((combo['maType'], combo['maLength']))
       return {'ma_types_and_lengths': list(ma_configs), 'needs_atr': False}
   ```

8. **Register in StrategyRegistry:**
   ```python
   # In strategy_registry.py
   from strategies.sxx_strategy_name import SXX_StrategyName

   _strategies = {
       # ... existing
       "sxx_strategy_name": SXX_StrategyName,
   }
   ```

---

## Common Pitfalls and How to Avoid

### 1. Counter Management

**Problem:** Counters get out of sync with position state.

**Solution:**
- Update counters BEFORE checking conditions
- Reset counters at correct times
- For position-based counters (counter_trade_long/short), coordinate with _run_simulation

**Example:**
```python
def should_long(self, idx):
    c = self.close[idx]
    ma = self._ma[idx]

    # Update FIRST
    if c > ma:
        self.counter_close_long += 1
        self.counter_close_short = 0
    elif c < ma:
        self.counter_close_short += 1
        self.counter_close_long = 0
    else:  # Equal
        self.counter_close_long = 0
        self.counter_close_short = 0

    # THEN check
    return self.counter_close_long >= self.params['close_count_long']
```

### 2. Trailing Stop Logic

**Problem:** Trail price not updating correctly or activation logic wrong.

**Solution:**
- Initialize trail_price to `math.nan`
- Check activation BEFORE updating trail price
- Use `math.isnan()` to check for NaN
- Update trail_price with `max()` for long, `min()` for short

**Example:**
```python
# In should_exit() for long position
if not self.trail_activated:
    activation_price = entry + (entry - stop) * trail_rr
    if high >= activation_price:
        self.trail_activated = True
        self.trail_price = stop  # Initialize to stop

# Update trail price
trail_ma = self._trail_ma[idx]
if not math.isnan(trail_ma):
    if math.isnan(self.trail_price):
        self.trail_price = trail_ma
    else:
        self.trail_price = max(self.trail_price, trail_ma)

# Check exit
if self.trail_activated:
    if low <= self.trail_price:
        return (True, self.trail_price, 'trailing')
```

### 3. Position Sizing Rounding

**Problem:** Position size calculation doesn't match original.

**Solution:**
- Use exact same order of operations
- Round to contract size using `math.floor(qty / contract_size) * contract_size`
- NOT `round(qty / contract_size) * contract_size` (different rounding!)

**Example:**
```python
def calculate_position_size(self, idx, direction, entry, stop, equity):
    stop_distance = abs(entry - stop)
    if stop_distance == 0:
        return 0.0

    risk_cash = equity * (self.params['risk_pct'] / 100.0)
    qty = risk_cash / stop_distance

    # Round to contract size
    contract_size = self.params['contract_size']
    if contract_size > 0:
        qty = math.floor(qty / contract_size) * contract_size  # Use floor!

    return qty
```

### 4. Lookback Period Edge Cases

**Problem:** Index out of bounds when using lookback periods.

**Solution:**
- Handle bars < lookback period correctly
- Use slicing: `array[max(0, idx-lp+1):idx+1]`

**Example:**
```python
# Get lowest low over lookback period
lp = self.params['stop_long_lp']
start_idx = max(0, idx - lp + 1)
lowest = self.low[start_idx:idx+1].min()
```

### 5. NaN Handling

**Problem:** Operations on NaN values produce unexpected results.

**Solution:**
- Check for NaN before using values
- Use `math.isnan()` not `value == nan` (always False!)
- Return NaN to skip trades: `return (math.nan, math.nan, math.nan)`

**Example:**
```python
if math.isnan(ma_value) or math.isnan(atr_value):
    return False  # Can't trade without indicators

if math.isnan(entry_price):
    return 0.0  # Skip this entry
```

---

## Testing Requirements

### Unit Testing

When implementing a new strategy:

1. **Test parameter definitions:**
   ```python
   param_defs = Strategy.get_param_definitions()
   assert 'maLength' in param_defs
   assert param_defs['maLength']['type'] == 'int'
   assert param_defs['maLength']['default'] == 45
   ```

2. **Test cache requirements:**
   ```python
   combos = [{'maType': 'SMA', 'maLength': 50}, {'maType': 'EMA', 'maLength': 30}]
   cache_req = Strategy.get_cache_requirements(combos)
   assert ('SMA', 50) in cache_req['ma_types_and_lengths']
   assert ('EMA', 30) in cache_req['ma_types_and_lengths']
   ```

3. **Test with known data:**
   ```python
   # Create simple test data
   df = pd.DataFrame({
       'Close': [100, 101, 102, 103, 104],
       'High': [101, 102, 103, 104, 105],
       'Low': [99, 100, 101, 102, 103],
       'Open': [100, 101, 102, 103, 104],
       'Volume': [1000]*5
   }, index=pd.date_range('2025-01-01', periods=5, freq='1H'))

   strategy = Strategy({...params...})
   result = strategy.simulate(df)

   # Verify structure
   assert 'net_profit_pct' in result
   assert 'trades' in result
   assert isinstance(result['trades'], list)
   ```

### Integration Testing

1. **Run reference test** (see `tests.md`)
2. **Run small optimization** (3-10 combinations)
3. **Verify via UI/API** (manual testing)

---

## Performance Optimization

### DO:
- ✅ Pre-compute indicators in `_prepare_data()`
- ✅ Store indicators as numpy arrays (faster than Series)
- ✅ Use vectorized operations where possible
- ✅ Define cache requirements to enable optimizer caching

### DON'T:
- ❌ Recompute indicators every bar in should_long/should_short
- ❌ Use pandas operations in tight loops
- ❌ Create new DataFrame objects in simulation loop

**Example - GOOD:**
```python
def _prepare_data(self, df, cached_data):
    # Compute ONCE
    self._ma = get_ma(df['Close'], ...).to_numpy()

def should_long(self, idx):
    # Use cached value
    ma = self._ma[idx]  # Fast array lookup
```

**Example - BAD:**
```python
def should_long(self, idx):
    # Recomputing every bar!
    ma = get_ma(self.df['Close'], ...).iloc[idx]  # SLOW!
```

---

## Error Handling

### Parameter Validation

```python
def _validate_params(self):
    required = ['maLength', 'stopAtr', 'riskPct']
    for param in required:
        if param not in self.params:
            raise ValueError(f"Missing required parameter: {param}")

    # Type checks
    if not isinstance(self.params['maLength'], int):
        raise TypeError(f"maLength must be int, got {type(self.params['maLength'])}")

    # Range checks
    if self.params['maLength'] < 1 or self.params['maLength'] > 500:
        raise ValueError(f"maLength must be in [1, 500], got {self.params['maLength']}")
```

### Simulation Errors

**Fail Fast:**
- If indicators can't be computed → raise error immediately
- If parameters invalid → raise in `_validate_params()`
- Don't silently return zero/NaN and continue

**Clear Error Messages:**
```python
# GOOD
raise ValueError(f"Cannot compute MA with length={length} for dataframe with only {len(df)} rows")

# BAD
raise ValueError("Error")
```

---

## Debugging Strategies

### 1. Print Intermediate Values

```python
def should_long(self, idx):
    c = self.close[idx]
    ma = self._ma[idx]

    print(f"Bar {idx}: close={c:.2f}, ma={ma:.2f}, counter={self.counter_long}")

    # ... logic
```

### 2. Compare with Old Implementation

```python
# Run old and new side-by-side
old_result = old_run_strategy(df, params)
new_result = new_strategy.simulate(df)

for i, (old_trade, new_trade) in enumerate(zip(old_result.trades, new_result['trades'])):
    if old_trade.entry_idx != new_trade['entry_idx']:
        print(f"Trade {i}: Entry differs! Old={old_trade.entry_idx}, New={new_trade['entry_idx']}")
```

### 3. Isolate Components

Test methods in isolation:
```python
strategy = Strategy(params)
strategy._prepare_data(df, None)

# Test specific bar
idx = 100
long_signal = strategy.should_long(idx)
print(f"Should long at {idx}: {long_signal}")

# Test entry calculation
entry, stop, target = strategy.calculate_entry(idx, 'long')
print(f"Entry: {entry}, Stop: {stop}, Target: {target}")
```

---

## Documentation Standards

### Docstrings

**Classes:**
```python
class S01TrailingMA(BaseStrategy):
    """
    Trailing Moving Average strategy.

    Trend-following system with:
    - MA crossover entry
    - ATR-based stop loss
    - RR-based take profit
    - Trailing MA exit with activation threshold
    - Risk-based position sizing

    Parameters: 28 total (see get_param_definitions)
    """
```

**Methods:**
```python
def calculate_entry(self, idx: int, direction: str) -> Tuple[float, float, float]:
    """
    Calculate entry, stop, and target prices for new trade.

    Args:
        idx: Current bar index
        direction: "long" or "short"

    Returns:
        Tuple of (entry_price, stop_price, target_price)
        Returns (nan, nan, nan) if entry should be skipped

    Logic:
        - Entry: current close
        - Stop: lowest_low(lookback) - (ATR * multiplier)
        - Target: entry + (entry - stop) * RR_ratio
        - Skip if stop distance > max_stop_pct
    """
```

### Code Comments

**When to Comment:**
- Complex logic that's not obvious
- Tricky edge cases
- Counter-intuitive behavior
- Performance optimizations

**When NOT to Comment:**
- Obvious code (`x = 5  # Set x to 5`)
- Repeating function names (`def calculate_entry():  # Calculate entry`)

**Example:**
```python
# Check trailing activation threshold
# Note: Use >= not > to match Pine behavior exactly
if high >= activation_price:
    self.trail_activated = True

# Initialize trail price to stop price (not entry!)
# This ensures we never trail below break-even
self.trail_price = stop_price
```

---

## Version Control

### Commit Messages

**Format:**
```
Phase N: Short description

- Detailed change 1
- Detailed change 2
- Reference test: Result ✅

Optional: Files changed, metrics, etc.
```

**Example:**
```
Phase 3: Extract S_01 to strategy module

- Create S01TrailingMA class with all 28 parameters
- Implement all BaseStrategy abstract methods
- Add backward compatibility wrapper in backtest_engine
- Register S_01 in StrategyRegistry
- Reference test: Results IDENTICAL ✅

Files changed:
- NEW: src/strategies/s01_trailing_ma.py (650 lines)
- MODIFIED: src/backtest_engine.py (add wrapper)
- MODIFIED: src/strategy_registry.py (register S_01)

Verified:
- Net Profit: 18.42% (matches baseline)
- Max DD: 5.67% (matches baseline)
- Trades: 23 (matches baseline)
```

### Branch Strategy

- Main branch: `main`
- Migration work: `migration/multi-strategy`
- Feature branches: `feature/strategy-sxx`
- Hotfixes: `hotfix/fix-description`

---

## Communication with User

### When to Ask Questions

**Ask when:**
- Requirements unclear or ambiguous
- Multiple valid approaches exist
- Trade-offs need user decision
- Breaking change might be needed

**Don't ask when:**
- Implementation detail (you decide)
- Follow established patterns (use existing code as guide)
- Covered in documentation (read docs first)

### Progress Updates

**Provide after each phase:**
- Phase completed
- Reference test result
- Any issues encountered
- Ready to proceed to next phase

**Example:**
```
✅ Phase 3 Complete: Extract S_01 to Strategy Module

Reference Test Results:
- Net Profit: 18.42% ✅ (matches baseline exactly)
- Max DD: 5.67% ✅ (matches baseline exactly)
- Trades: 23 ✅ (matches baseline exactly)

Time: 2.5 hours
Issues: None
Ready for Phase 4: Refactor Optimizer
```

---

## Emergency Procedures

### If Reference Test Fails

1. **STOP immediately** - do not proceed
2. **Identify the diff** - which metric differs?
3. **Isolate the bug** - which method has the issue?
4. **Debug systematically** - use print statements, compare with original
5. **Fix and re-test** - verify fix resolves issue
6. **Only then proceed** to next step

### If Stuck on Bug > 2 Hours

1. **Document what you've tried**
2. **Show code diff** between working and broken version
3. **Show specific test failure**
4. **Ask user for guidance** with all context

---

## Summary Checklist

Before committing ANY code:

- [ ] Reference test passes (see tests.md)
- [ ] No console errors or warnings
- [ ] Code follows naming conventions
- [ ] Docstrings added to new functions/classes
- [ ] No commented-out code (delete it)
- [ ] No debug print statements (remove them)
- [ ] Imports organized (stdlib → third-party → local)
- [ ] Phase checklist item marked complete

---

**End of agents.md**
