# Migration Prompt 3: Extract S_01 to Strategy Module

## Context

**Phase:** 3 of 6 (CRITICAL - Preserve exact S_01 behavior)
**Previous:** Phase 2 ✅ (BaseStrategy created)
**Checklist:** `migration_checklist.md` - Phase 3
**Duration:** 2-3 days

**Goal:** Move S_01 TrailingMA logic from `backtest_engine.py` into `strategies/s01_trailing_ma.py` implementing BaseStrategy.

---

## Critical Success Criteria

**Reference test MUST match baseline EXACTLY:**
- Same Net Profit % (to 0.01% precision)
- Same Max Drawdown %
- Same number of trades
- Same trade entry/exit bars

**If ANY difference → DO NOT COMMIT until fixed.**

---

## Implementation Steps

### 1. Create S_01 Strategy File

`/src/strategies/s01_trailing_ma.py`:

**Class structure:**
```python
from .base_strategy import BaseStrategy
from indicators import get_ma, atr
import math
import pandas as pd
import numpy as np

class S01TrailingMA(BaseStrategy):
    STRATEGY_ID = "s01_trailing_ma"
    STRATEGY_NAME = "S_01 TrailingMA v26 Ultralight"
    VERSION = "26"
    
    def __init__(self, params):
        super().__init__(params)
        # Caches
        self._ma_trend = None
        self._atr = None
        self._trail_ma_long = None
        self._trail_ma_short = None
        self._lowest = None
        self._highest = None
        
        # State
        self.counter_close_trend_long = 0
        self.counter_close_trend_short = 0
        self.counter_trade_long = 0
        self.counter_trade_short = 0
        self.trail_price_long = math.nan
        self.trail_price_short = math.nan
        self.trail_activated_long = False
        self.trail_activated_short = False
```

### 2. Implement get_param_definitions()

**Map all 29 S_01 parameters:**
- Date filter: date_filter, start_date, end_date
- Trend MA: ma_type, ma_length
- Entry: close_count_long, close_count_short
- Long stops: stop_long_atr, stop_long_rr, stop_long_lp, stop_long_max_pct, stop_long_max_days
- Short stops: stop_short_atr, stop_short_rr, stop_short_lp, stop_short_max_pct, stop_short_max_days
- Trailing long: trail_rr_long, trail_ma_long_type, trail_ma_long_length, trail_ma_long_offset
- Trailing short: trail_rr_short, trail_ma_short_type, trail_ma_short_length, trail_ma_short_offset
- Risk: risk_per_trade_pct, contract_size, commission_rate, atr_period

**Format:**
```python
@classmethod
def get_param_definitions(cls):
    return {
        'maLength': {  # camelCase key for API/frontend
            'type': 'int',
            'default': 45,
            'min': 0,
            'max': 500,
            'step': 1,
            'description': 'Trend MA Length'
        },
        # ... all 28 parameters
    }
```

### 3. Implement _prepare_data()

**Handle both cached and on-the-fly computation:**
```python
def _prepare_data(self, df, cached_data):
    self.df = df
    self.close = df['Close'].to_numpy()
    self.high = df['High'].to_numpy()
    self.low = df['Low'].to_numpy()
    self.open = df['Open'].to_numpy()
    self.volume = df['Volume'].to_numpy()
    self.times = df.index
    
    if cached_data:
        # Use cache from optimizer
        ma_key = (self.params['maType'], self.params['maLength'])
        self._ma_trend = cached_data['ma_cache'][ma_key]

        trail_long_key = (self.params['trailMaLongType'], self.params['trailMaLongLength'])
        self._trail_ma_long = cached_data['ma_cache'][trail_long_key]
        # Apply offset
        self._trail_ma_long = self._trail_ma_long * (1 + self.params['trailMaLongOffset'] / 100.0)

        # Similar for trail_short, ATR, lowest, highest

    else:
        # Compute on-the-fly
        self._ma_trend = get_ma(df['Close'], self.params['maType'], self.params['maLength'], ...).to_numpy()

        trail_long_ma = get_ma(df['Close'], self.params['trailMaLongType'], self.params['trailMaLongLength'], ...)
        self._trail_ma_long = trail_long_ma.to_numpy() * (1 + self.params['trailMaLongOffset'] / 100.0)

        # Similar for other indicators

        self._atr = atr(df['High'], df['Low'], df['Close'], self.params['atrPeriod']).to_numpy()
        
        # Compute lowest/highest
        self._lowest = df['Low'].rolling(window=self.params['stopLongLp']).min().to_numpy()
        self._highest = df['High'].rolling(window=self.params['stopShortLp']).max().to_numpy()
```

### 4. Implement should_long()

**Move logic from backtest_engine.run_strategy():**
```python
def should_long(self, idx):
    c = self.close[idx]
    ma = self._ma_trend[idx]
    
    # Update counters (CRITICAL - preserve exact logic!)
    if not math.isnan(ma):
        if c > ma:
            self.counter_close_trend_long += 1
            self.counter_close_trend_short = 0
        elif c < ma:
            self.counter_close_trend_short += 1
            self.counter_close_trend_long = 0
        else:
            self.counter_close_trend_long = 0
            self.counter_close_trend_short = 0
    
    # Update trade counters based on current position
    # NOTE: This is tricky - need position info from _run_simulation
    # For now, use approach: counter_trade_long/short are managed outside
    
    # Check conditions
    up_trend = (
        self.counter_close_trend_long >= self.params['closeCountLong']
        and self.counter_trade_long == 0
    )

    return up_trend and not math.isnan(self._atr[idx])
```

**IMPORTANT NOTE:** Counter management requires coordination with _run_simulation. May need to pass position state to should_long/short or update counters in _run_simulation.

### 5. Implement calculate_entry()

```python
def calculate_entry(self, idx, direction):
    c = self.close[idx]
    atr_val = self._atr[idx]
    
    if direction == 'long':
        # Get lowest low over lookback period
        lp = self.params['stopLongLp']
        if idx < lp:
            lowest = self.low[:idx+1].min()
        else:
            lowest = self.low[idx-lp+1:idx+1].min()

        # Calculate stop
        stop = lowest - (atr_val * self.params['stopLongAtr'])
        stop_distance = c - stop

        # Check max stop %
        max_stop_pct = self.params['stopLongMaxPct']
        if max_stop_pct > 0:
            stop_pct = (stop_distance / c) * 100
            if stop_pct > max_stop_pct:
                return (math.nan, math.nan, math.nan)

        # Calculate target
        target = c + (stop_distance * self.params['stopLongRr'])

        return (c, stop, target)
    
    else:  # short
        # Similar logic with highest
        pass
```

### 6. Implement should_exit()

**Handle trailing stop logic:**
```python
def should_exit(self, idx, position_info):
    direction = position_info['direction']
    entry_price = position_info['entry_price']
    stop_price = position_info['stop_price']
    target_price = position_info['target_price']
    entry_idx = position_info['entry_idx']
    
    h = self.high[idx]
    l = self.low[idx]
    c = self.close[idx]
    
    if direction > 0:  # Long
        # Check trailing activation
        if not self.trail_activated_long:
            activation_price = entry_price + (entry_price - stop_price) * self.params['trailRrLong']
            if h >= activation_price:
                self.trail_activated_long = True
                self.trail_price_long = stop_price

        # Update trailing price
        trail_ma_val = self._trail_ma_long[idx]
        if not math.isnan(trail_ma_val):
            if math.isnan(self.trail_price_long) or trail_ma_val > self.trail_price_long:
                self.trail_price_long = trail_ma_val

        # Check exits
        if self.trail_activated_long:
            if l <= self.trail_price_long:
                exit_price = h if self.trail_price_long > h else self.trail_price_long
                return (True, exit_price, 'trailing')
        else:
            # Regular stop/target
            if l <= stop_price:
                return (True, stop_price, 'stop')
            if h >= target_price:
                return (True, target_price, 'target')

        # Max days
        max_days = self.params['stopLongMaxDays']
        if max_days > 0:
            days_in_trade = idx - entry_idx
            if days_in_trade >= max_days:
                return (True, c, 'max_days')
    
    # Similar for short
    
    return (False, None, '')
```

### 7. Implement get_cache_requirements()

```python
@classmethod
def get_cache_requirements(cls, param_combinations):
    ma_types_and_lengths = set()
    long_lp = set()
    short_lp = set()
    atr_periods = set()

    for combo in param_combinations:
        ma_types_and_lengths.add((combo.get('maType', 'EMA'), combo.get('maLength', 45)))
        ma_types_and_lengths.add((combo.get('trailMaLongType', 'SMA'), combo.get('trailMaLongLength', 160)))
        ma_types_and_lengths.add((combo.get('trailMaShortType', 'SMA'), combo.get('trailMaShortLength', 160)))

        long_lp.add(combo.get('stopLongLp', 2))
        short_lp.add(combo.get('stopShortLp', 2))
        atr_periods.add(combo.get('atrPeriod', 14))

    return {
        'ma_types_and_lengths': list(ma_types_and_lengths),
        'long_lp_values': list(long_lp),
        'short_lp_values': list(short_lp),
        'needs_atr': True
    }
```

### 8. Register in StrategyRegistry

In `strategy_registry.py`:
```python
from strategies.s01_trailing_ma import S01TrailingMA

class StrategyRegistry:
    _strategies = {
        "s01_trailing_ma": S01TrailingMA,
    }
```

### 9. Update backtest_engine.py

Add backward compatibility wrapper:
```python
from strategy_registry import StrategyRegistry

def run_strategy(df, params, trade_start_idx=0):
    """
    Backward compatible wrapper for S_01 strategy.
    
    Old code can still call this function.
    Internally uses new strategy system.
    """
    # Convert StrategyParams to dict
    if hasattr(params, '__dict__'):
        params_dict = params.__dict__
    else:
        params_dict = params
    
    # Get S_01 strategy
    strategy = StrategyRegistry.get_strategy_instance('s01_trailing_ma', params_dict)
    
    # Run simulation
    result = strategy.simulate(df, cached_data=None)
    
    # Convert to old StrategyResult format
    from backtest_engine import StrategyResult, TradeRecord
    
    trade_records = [
        TradeRecord(
            direction=t['direction'],
            entry_time=df.index[t['entry_idx']],
            entry_price=t['entry_price'],
            exit_time=df.index[t['exit_idx']],
            exit_price=t['exit_price'],
            size=t['size'],
            net_pnl=t['net_pnl']
        )
        for t in result['trades']
    ]
    
    return StrategyResult(
        net_profit_pct=result['net_profit_pct'],
        max_drawdown_pct=result['max_drawdown_pct'],
        total_trades=result['total_trades'],
        trades=trade_records
    )
```

---

## Testing (CRITICAL)

### Reference Test

```bash
cd src
python run_backtest.py --csv ../data/OKX_LINKUSDT.P,\ 15...csv
```

**Expected results (from baseline):**
- Net Profit %: `_____`
- Max DD %: `_____`
- Total Trades: `_____`

**If results differ:**

1. **Check counter logic** - most common source of bugs
2. **Print intermediate values:**
   ```python
   print(f"Bar {idx}: counter_long={self.counter_close_trend_long}, ma={ma}, close={c}")
   ```
3. **Compare with old run_strategy()** side-by-side
4. **Check floating point precision** - use same rounding as original

### Detailed Comparison Test

```python
# Run old and new side-by-side
from backtest_engine import load_data, run_strategy as run_old
from strategies.s01_trailing_ma import S01TrailingMA

df = load_data('../data/test.csv')

# Old way
from backtest_engine import StrategyParams
old_params = StrategyParams(ma_type='EMA', ma_length=45, ...)
old_result = run_old(df, old_params)

# New way
new_strategy = S01TrailingMA({'maType': 'EMA', 'maLength': 45, ...})
new_result = new_strategy.simulate(df)

print(f"Old profit: {old_result.net_profit_pct:.6f}%")
print(f"New profit: {new_result['net_profit_pct']:.6f}%")
print(f"Match: {abs(old_result.net_profit_pct - new_result['net_profit_pct']) < 0.01}")
```

---

## Common Debugging Issues

**Counter Desync:**
- Ensure counters update at correct time in loop
- Check reset conditions (exactly when to reset to 0)

**Trailing Stop Calculation:**
- Verify offset application: `ma * (1 + offset/100)` not `ma + offset`
- Check activation threshold calculation

**Position Sizing:**
- Ensure rounding to contract size identical to original
- Check order of operations in size calculation

**Lookback Periods:**
- Handle bars < lookback period correctly
- Use exact same slice: `low[max(0, idx-lp+1):idx+1].min()`

---

## Commit

```bash
git add src/strategies/s01_trailing_ma.py src/strategy_registry.py src/backtest_engine.py
git commit -m "Phase 3: Extract S_01 to strategy module

- Create S01TrailingMA class with all S_01 logic
- Implement all BaseStrategy abstract methods
- Add backward compatibility wrapper in backtest_engine
- Register S_01 in StrategyRegistry
- Reference test: Results IDENTICAL ✅

Verified:
- Net Profit matches baseline
- Max DD matches baseline  
- Trade count matches baseline"
```

---

## Next Phase

Proceed to **Phase 4: Refactor Optimizer**
- `migration_prompt_4.md`
- `migration_checklist.md` - Phase 4

---

**End of Prompt 3**
