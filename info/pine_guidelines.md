# PineScript → Python Translation Guidelines

## Purpose

This document provides **guidelines for writing PineScript strategies** that are easy to translate into Python for use in this backtesting platform.

**Target Audience:** Strategy developers writing Pine code that will be translated to Python.

---

## Core Principles

### 1. Clarity Over Cleverness

**Write explicit, readable code:**
```pine
// GOOD
float ma_fast = ta.sma(close, 10)
float ma_slow = ta.sma(close, 50)
bool long_condition = ma_fast > ma_slow

// AVOID (too condensed)
bool long_condition = ta.sma(close, 10) > ta.sma(close, 50)
```

**Why:** Explicit variables are easier to map to Python. Agent can clearly see each indicator computation.

---

### 2. Consistent Naming

**Use descriptive, standardized names:**
```pine
// GOOD - clear what it is
input int ma_length = 45
input float stop_atr_multiplier = 2.0
input int close_count_long = 7

// AVOID - ambiguous
input int period = 45
input float x = 2.0
input int count = 7
```

**Naming Convention:**
- **In PineScript**: Use `snake_case` (PineScript standard)
- **In Python get_param_definitions()**: Convert to `camelCase` (API/frontend compatibility)
- Calculated values: prefix with `calc_`
- Counters: prefix with `counter_`
- Conditions: prefix with `cond_`

---

## Parameter Naming Convention

### Inputs (Strategy Parameters)

**Format:** `<category>_<parameter>_<direction>`

**Examples:**
```pine
// Moving Averages
input string ma_type = "EMA"
input int ma_length = 45

input string ma1_type = "KAMA"
input int ma1_length = 15

input string ma2_type = "SMA"
input int ma2_length = 50

// Entry Parameters
input int close_count_long = 7
input int close_count_short = 5

// Stop Loss Parameters
input float stop_long_atr = 2.0         // ATR multiplier for long stops
input float stop_long_rr = 3.0          // Risk/Reward ratio
input int stop_long_lp = 2              // Lookback period
input float stop_long_max_pct = 3.0     // Max stop as % of entry
input int stop_long_max_days = 2        // Max days in trade

input float stop_short_atr = 2.0
input float stop_short_rr = 3.0
input int stop_short_lp = 2

// Trailing Parameters
input float trail_rr_long = 1.0         // Activation threshold (RR)
input string trail_ma_long_type = "SMA"
input int trail_ma_long_length = 160
input float trail_ma_long_offset = -1.0  // % offset

input float trail_rr_short = 1.0
input string trail_ma_short_type = "SMA"
input int trail_ma_short_length = 160
input float trail_ma_short_offset = 1.0

// Risk Management
input float risk_per_trade_pct = 2.0
input float contract_size = 0.01
input float commission_rate = 0.0004
```

**Translation to Python:**
```python
@classmethod
def get_param_definitions(cls):
    """
    Convert PineScript snake_case parameters to Python camelCase.

    PineScript: ma_type, ma_length, stop_long_atr
    Python API: maType, maLength, stopLongAtr
    """
    return {
        'maType': {  # camelCase key (converted from ma_type)
            'type': 'categorical',
            'choices': ['SMA', 'EMA', 'HMA', ...],
            'default': 'EMA',
            'description': 'Moving average type'
        },
        'maLength': {  # camelCase key (converted from ma_length)
            'type': 'int',
            'default': 45,
            'min': 1,
            'max': 500,
            'description': 'Moving average period'
        },
        'stopLongAtr': {  # camelCase (converted from stop_long_atr)
            'type': 'float',
            'default': 2.0,
            'min': 0.5,
            'max': 5.0,
            'description': 'ATR multiplier for long stops'
        },
        # ... etc
    }
```

**Conversion Rule:** `snake_case` → `camelCase`
- `ma_type` → `maType`
- `close_count_long` → `closeCountLong`
- `trail_ma_long_offset` → `trailMaLongOffset`

---

## Variable Naming Convention

### Calculated Values

**Prefix with `calc_`:**
```pine
// Moving Averages
calc_ma_trend = ta.sma(close, ma_length)
calc_ma1 = getMA(close, ma1_type, ma1_length)
calc_ma2 = getMA(close, ma2_type, ma2_length)
calc_ma3 = getMA(close, ma3_type, ma3_length)

// ATR
calc_atr = ta.atr(atr_period)

// Trailing MAs
calc_trail_ma_long = getMA(close, trail_ma_long_type, trail_ma_long_length) * (1 + trail_ma_long_offset / 100)
calc_trail_ma_short = getMA(close, trail_ma_short_type, trail_ma_short_length) * (1 + trail_ma_short_offset / 100)

// Lookbacks
calc_lowest_long = ta.lowest(low, stop_long_lp)
calc_highest_short = ta.highest(high, stop_short_lp)
```

**Python translation:** Calculated in `_prepare_data()` method
```python
self._ma_trend = get_ma(df['Close'], self.params['maType'], self.params['maLength']).to_numpy()
self._atr = atr(df['High'], df['Low'], df['Close'], self.params['atrPeriod']).to_numpy()
```

---

### Counters

**Prefix with `counter_`:**
```pine
var int counter_close_long = 0
var int counter_close_short = 0
var int counter_trade_long = 0
var int counter_trade_short = 0

// Update counters
if close > calc_ma_trend
    counter_close_long += 1
    counter_close_short := 0
else if close < calc_ma_trend
    counter_close_short += 1
    counter_close_long := 0
else
    counter_close_long := 0
    counter_close_short := 0
```

**Python translation:** Initialized in `__init__`, updated in `should_long/should_short`
```python
self.counter_close_long = 0
self.counter_close_short = 0
```

---

### Conditions

**Prefix with `cond_`:**
```pine
cond_up_trend = counter_close_long >= close_count_long
cond_down_trend = counter_close_short >= close_count_short

cond_ma_confirm_long = calc_ma1 > calc_ma2 and close > calc_ma3
cond_ma_confirm_short = calc_ma1 < calc_ma2 and close < calc_ma3

cond_can_open_long = cond_up_trend and counter_trade_long == 0 and not na(calc_atr)
cond_can_open_short = cond_down_trend and counter_trade_short == 0 and not na(calc_atr)
```

**Python translation:** Logic in `should_long/should_short`
```python
def should_long(self, idx):
    up_trend = self.counter_close_long >= self.params['close_count_long']
    can_open = up_trend and self.counter_trade_long == 0 and not math.isnan(self._atr[idx])
    return can_open
```

---

## Commenting for Translation

### Indicator Annotations

**Mark indicators that should go in `indicators.py`:**
```pine
// @indicator: Custom VWMA with special period
// @location: indicators.py
// @description: Volume-weighted MA with custom smoothing
customVWMA(src, length) =>
    sum_volume = math.sum(volume, length)
    sum_pv = math.sum(src * volume, length)
    sum_pv / sum_volume
```

**Agent will:**
1. See `@location: indicators.py` annotation
2. Add this function to `indicators.py` instead of strategy file
3. Import it in strategy: `from indicators import custom_vwma`

**If NOT marked:** Indicator stays in strategy file as private method.

---

### Code Blocks to Skip

**Mark code that should NOT be translated:**
```pine
// @skip-translation-start
// This is Pine-specific plotting code, skip in Python
plot(calc_ma_trend, "MA", color=color.blue)
plotshape(cond_can_open_long, "Long", shape.triangleup, location.belowbar)
// @skip-translation-end

// Continue with logic...
```

**Agent will ignore everything between `@skip-translation-start` and `@skip-translation-end`.**

---

### Strategy Type Annotations

**At top of file, specify strategy characteristics:**
```pine
// @strategy-type: reversal
// @description: Always in market, reverses position on opposite signal
// @has-stops: false
// @has-trailing: false
// @position-sizing: equity-percentage
```

**Tells agent:**
- `@strategy-type: reversal` → Set `allows_reversal() = True`
- `@has-stops: false` → Skip stop/target logic, return `(entry, nan, nan)`
- `@position-sizing: equity-percentage` → Use 100% equity sizing

**Example:**
```pine
// @strategy-type: stops-based
// @has-stops: true
// @has-trailing: true
// @position-sizing: risk-based
```

---

## Standardized Functions

### Moving Averages

**Use unified MA function:**
```pine
// Define once at top of strategy
getMA(src, ma_type, length) =>
    switch ma_type
        "SMA" => ta.sma(src, length)
        "EMA" => ta.ema(src, length)
        "HMA" => ta.hma(src, length)
        "WMA" => ta.wma(src, length)
        "VWMA" => ta.vwma(src, length)
        "KAMA" => customKAMA(src, length)  // If custom
        => ta.sma(src, length)  // Default

// Then use consistently
calc_ma_trend = getMA(close, ma_type, ma_length)
calc_ma1 = getMA(close, ma1_type, ma1_length)
```

**Python translation:** Uses `indicators.get_ma()`
```python
self._ma_trend = get_ma(df['Close'], self.params['ma_type'], self.params['ma_length']).to_numpy()
```

---

### Stop/Target Calculation

**Standardized pattern:**
```pine
// Long Stop Calculation
if cond_can_open_long
    calc_stop_long = calc_lowest_long - (calc_atr * stop_long_atr)
    calc_stop_distance_long = close - calc_stop_long
    calc_stop_pct_long = (calc_stop_distance_long / close) * 100

    // Check max stop %
    if stop_long_max_pct > 0 and calc_stop_pct_long > stop_long_max_pct
        // Skip trade
    else
        calc_target_long = close + (calc_stop_distance_long * stop_long_rr)
        // Enter trade
```

**Python translation:** Logic in `calculate_entry()`

---

## Reversal Strategy Template

**For strategies that are always in market:**
```pine
// @strategy-type: reversal
//@version=5
strategy("S_XX Reversal", overlay=true)

// ════════════════════════════════════════════════════
// INPUTS
// ════════════════════════════════════════════════════

input bool use_ma1 = true
input string ma1_type = "KAMA"
input int ma1_length = 15

input bool use_close_count = false
input int close_count_long = 3
input int close_count_short = 3

input bool trade_long = true
input bool trade_short = true

input float contract_size = 0.01

// ════════════════════════════════════════════════════
// CALCULATIONS
// ════════════════════════════════════════════════════

calc_ma1 = use_ma1 ? getMA(close, ma1_type, ma1_length) : na
calc_ma2 = use_ma2 ? getMA(close, ma2_type, ma2_length) : na
calc_ma3 = use_ma3 ? getMA(close, ma3_type, ma3_length) : na

// ════════════════════════════════════════════════════
// COUNTERS
// ════════════════════════════════════════════════════

var int counter_close_long = 0
var int counter_close_short = 0

if close > calc_ma3
    counter_close_long += 1
    counter_close_short := 0
else if close < calc_ma3
    counter_close_short += 1
    counter_close_long := 0
else
    counter_close_long := 0
    counter_close_short := 0

// ════════════════════════════════════════════════════
// CONDITIONS
// ════════════════════════════════════════════════════

cond_count_long = use_close_count ? counter_close_long >= close_count_long : true
cond_count_short = use_close_count ? counter_close_short >= close_count_short : true

cond_long = trade_long and cond_count_long
cond_short = trade_short and cond_count_short

// ════════════════════════════════════════════════════
// TRADE EXECUTION (Reversal Logic)
// ════════════════════════════════════════════════════

if cond_long
    if strategy.position_size < 0  // In short, reverse to long
        strategy.close("Short")
    if strategy.position_size == 0  // Flat, enter long
        strategy.entry("Long", strategy.long)

if cond_short
    if strategy.position_size > 0  // In long, reverse to short
        strategy.close("Long")
    if strategy.position_size == 0  // Flat, enter short
        strategy.entry("Short", strategy.short)
```

**Python translation hints:**
- Set `allows_reversal() = True`
- `should_exit()` returns `(False, None, '')` - no regular exits
- Reversal handled by `_run_simulation()` checking opposite signal

---

## Stops-Based Strategy Template

**For strategies with stops/targets:**
```pine
// @strategy-type: stops-based
//@version=5
strategy("S_XX StopsBased", overlay=true)

// ════════════════════════════════════════════════════
// INPUTS (use naming convention above)
// ════════════════════════════════════════════════════

// ... parameters

// ════════════════════════════════════════════════════
// CALCULATIONS
// ════════════════════════════════════════════════════

calc_ma_trend = getMA(close, ma_type, ma_length)
calc_atr = ta.atr(atr_period)
calc_trail_ma_long = getMA(close, trail_ma_long_type, trail_ma_long_length) * (1 + trail_ma_long_offset / 100)

calc_lowest_long = ta.lowest(low, stop_long_lp)
calc_highest_short = ta.highest(high, stop_short_lp)

// ════════════════════════════════════════════════════
// COUNTERS
// ════════════════════════════════════════════════════

var int counter_close_long = 0
var int counter_close_short = 0

// Update logic...

// ════════════════════════════════════════════════════
// CONDITIONS
// ════════════════════════════════════════════════════

cond_can_open_long = ...
cond_can_open_short = ...

// ════════════════════════════════════════════════════
// ENTRY LOGIC
// ════════════════════════════════════════════════════

if cond_can_open_long
    calc_stop_long = calc_lowest_long - (calc_atr * stop_long_atr)
    calc_stop_distance_long = close - calc_stop_long
    calc_target_long = close + (calc_stop_distance_long * stop_long_rr)

    // Check filters
    calc_stop_pct_long = (calc_stop_distance_long / close) * 100
    if stop_long_max_pct <= 0 or calc_stop_pct_long <= stop_long_max_pct
        strategy.entry("Long", strategy.long, stop=calc_stop_long, limit=calc_target_long)

// ════════════════════════════════════════════════════
// EXIT LOGIC (Trailing)
// ════════════════════════════════════════════════════

var float trail_price_long = na
var bool trail_activated_long = false

if strategy.position_size > 0
    // Check activation
    if not trail_activated_long
        calc_activation_long = strategy.position_avg_price + (strategy.position_avg_price - calc_stop_long) * trail_rr_long
        if high >= calc_activation_long
            trail_activated_long := true
            trail_price_long := calc_stop_long

    // Update trail
    if not na(calc_trail_ma_long)
        trail_price_long := na(trail_price_long) ? calc_trail_ma_long : math.max(trail_price_long, calc_trail_ma_long)

    // Exit on trail
    if trail_activated_long and low <= trail_price_long
        strategy.close("Long")
        trail_activated_long := false
        trail_price_long := na
```

---

## Testing Your Pine Code

### Before Translation

**1. Test in TradingView:**
- Verify strategy works as expected
- Check all conditions trigger correctly
- Verify stops/targets execute properly

**2. Document Expected Behavior:**
```pine
// @test-case: Uptrend with 7 closes above MA
// @expected: Enter long on bar 7
// @expected: Stop at lowest_low - (ATR * 2)
// @expected: Target at entry + (stop_distance * 3)
```

**3. Provide Sample Results:**
```pine
// @reference-test
// Symbol: BTCUSDT
// Timeframe: 15m
// Date Range: 2025-04-01 to 2025-09-01
// Expected Net Profit: +18.42%
// Expected Max DD: -5.67%
// Expected Trades: 23
```

This will be used to validate Python translation.

---

## Common Translation Issues

### Issue 1: `barstate.isconfirmed`

**Pine:**
```pine
if barstate.isconfirmed
    // Only execute on bar close
```

**Python:** Not needed! Python backtester already operates on closed bars.

**Solution:** Skip this check in translation. Mark with comment:
```pine
// @note: barstate.isconfirmed not needed in Python (bars already closed)
if barstate.isconfirmed  // @skip-in-python
    ...
```

---

### Issue 2: `strategy.position_size[1]`

**Pine:**
```pine
bool was_flat = strategy.position_size[1] == 0
```

**Python:** Use state management in strategy class.

**Solution:** Document requirement:
```pine
// @python-requirement: Track position state to check if just exited
// In Python: Use counter_trade_long/short to track this
```

---

### Issue 3: `ta.valuewhen()`

**Pine:**
```pine
entry_price = ta.valuewhen(entry_signal, close, 0)
```

**Python:** Store entry_price explicitly.

**Solution:** Use clear variables:
```pine
// @python-note: Store entry_price when entering trade
var float saved_entry_price = na
if entry_signal
    saved_entry_price := close
```

---

## Summary Checklist

Before requesting translation, ensure your Pine code has:

- [ ] Consistent naming convention (`snake_case` in Pine, `camelCase` in Python API)
- [ ] Clear variable names with prefixes (`calc_`, `counter_`, `cond_`)
- [ ] Annotated indicators (`@location: indicators.py` if needed)
- [ ] Strategy type annotation (`@strategy-type: ...`)
- [ ] Skip markers for Pine-specific code (`@skip-translation-start/end`)
- [ ] Test cases and expected results documented
- [ ] Works correctly in TradingView
- [ ] No undocumented "magic numbers"

---

## Example: Fully Annotated Strategy

See `/data/S_03 Reversal_v07 Light for PROJECT PLAN.pine` for reference implementation following all guidelines.

---

**End of pine_guidelines.md**
