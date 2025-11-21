# PineScript → Python Translation Guidelines

## Purpose

This document provides **guidelines for writing PineScript strategies** that are easy to translate into Python for the backtesting platform. Follow these conventions to help the coding agent convert your Pine code accurately and efficiently.

---

## Annotation Tags

Use these tags to guide the agent during translation:

| Tag | Usage | Description |
|-----|-------|-------------|
| `@python-note:` | Single-line comment | Explanation or instruction for agent |
| `@skip-translation-start` | Start marker | Begin block to skip during translation |
| `@skip-translation-end` | End marker | End block to skip during translation |
| `@skip-in-python` | Inline comment | Skip this specific line in Python |
| `@strategy-type:` | File header | Strategy type: `reversal` or `stops-based` |
| `@location:` | Above function | Where to place code (e.g., `indicators.py`) |
| `@reference-test` | Block header | Reference test configuration and expected results |
| `@future-enhancement:` | Comment | Note about potential future improvements |

**Example usage:**
```pine
// @python-note: Date filter already implemented in BaseStrategy
// @skip-translation-start
useDateFilter = input.bool(true, "Date Filter")
startDate = input.time(timestamp('2025-01-01'), "Start")
endDate = input.time(timestamp('2025-12-31'), "End")
// @skip-translation-end

longCondition = ma1 > ma2 and barstate.isconfirmed  // @skip-in-python
```

---

## Naming Conventions

### 1. Use camelCase (Default)

**For:** Input parameters, functions, most variables

```pine
// Inputs
maType          = input.string("EMA", "MA Type")
maLength        = input.int(45, "MA Length")
closeCountLong  = input.int(7, "Close Count")
stopLongAtr     = input.float(2.0, "Stop ATR Multiplier")
trailMaLongType = input.string("SMA", "Trail MA Type")

// Functions
getMA(source, maType, length) => ...
calcPositionSize(equity, price) => ...

// Calculated values
ma1             = getMA(close, maType, maLength)
longCondition   = close > ma1 and countLong >= closeCountLong
```

**Why camelCase:**
- Direct 1:1 mapping to Python API keys (`maLength` → `'maLength'`)
- No conversion needed during translation
- Consistent with project architecture

### 2. Use snake_case (Rare)

**Only for:** Persistent state variables with `var` keyword

```pine
// State variables that persist across bars
var t_entry           = 0.0      // Trade entry price
var t_stop            = 0.0      // Trade stop price
var t_target          = 0.0      // Trade target price
var trailMAPriceLong  = 0.0      // Trailing MA price (or camelCase)
```

**Why snake_case here:**
- Visually distinguishes persistent state from regular variables
- Optional: You can also use camelCase for state variables
- Consistency within your codebase is more important

---

## Code Structure

Organize Pine code into clear sections to help the agent understand the flow:

```pine
// ═══════════════════════════════════════════════════════════
// STRATEGY_NAME v.XX
// ═══════════════════════════════════════════════════════════
// @strategy-type: stops-based  (or reversal)
// @version: vXX
// @description: Brief strategy description
// ═══════════════════════════════════════════════════════════

//@version=5
strategy("Strategy Name", ...)


// ═══════════════════════════════════════════════════════════
// SECTION 1: INPUTS (Strategy Parameters)
// ═══════════════════════════════════════════════════════════

maType      = input.string("EMA", "MA Type", options=[...])
maLength    = input.int(45, "MA Length")
stopLongAtr = input.float(2.0, "Stop ATR Multiplier")
// ... all inputs


// ═══════════════════════════════════════════════════════════
// SECTION 2: INDICATOR CALCULATIONS
// ═══════════════════════════════════════════════════════════

// @python-note: getMA function already exists in indicators.py
// @skip-translation-start
getMA(source, maType, length) =>
    switch maType
        "SMA" => ta.sma(source, length)
        "EMA" => ta.ema(source, length)
        => ta.ema(source, length)
// @skip-translation-end

ma3 = getMA(close, maType, maLength)
atr = ta.atr(atrPeriod)


// ═══════════════════════════════════════════════════════════
// SECTION 3: COUNTERS (State Tracking)
// ═══════════════════════════════════════════════════════════

var int counterCloseLong  = 0
var int counterCloseShort = 0

if close > ma3
    counterCloseLong += 1
    counterCloseShort := 0
else if close < ma3
    counterCloseShort += 1
    counterCloseLong := 0


// ═══════════════════════════════════════════════════════════
// SECTION 4: ENTRY CONDITIONS
// ═══════════════════════════════════════════════════════════

upTrend        = counterCloseLong >= closeCountLong
canOpenLong    = upTrend and strategy.position_size == 0
longConditions = canOpenLong and barstate.isconfirmed  // @skip-in-python


// ═══════════════════════════════════════════════════════════
// SECTION 5: TRADE EXECUTION
// ═══════════════════════════════════════════════════════════

if longConditions
    lowestLong   = ta.lowest(low, stopLongLp)
    stopLong     = lowestLong - (atr * stopLongAtr)
    stopDistance = close - stopLong
    targetLong   = close + (stopDistance * stopLongRr)
    strategy.entry("Long", strategy.long, stop=stopLong, limit=targetLong)


// ═══════════════════════════════════════════════════════════
// SECTION 6: VISUALIZATION (TradingView Only)
// ═══════════════════════════════════════════════════════════
// @skip-translation-start
plot(ma3, "MA", color.blue)
plotshape(longConditions, "Long", shape.triangleup)
// @skip-translation-end
```

---

## Strategy Types

### Reversal Strategy

**Characteristics:**
- Always in market (long or short)
- No flat periods
- No stops/targets
- Exits only on opposite signal

**Annotations:**
```pine
// @strategy-type: reversal
// @python-note: Always-in-market - reverses on opposite signal
// @python-note: No stops, no targets, no trailing

if longCondition and strategy.position_size < 0
    strategy.close("Short", comment="Reverse: Short → Long")

if longCondition and strategy.position_size == 0
    strategy.entry("Long", strategy.long, qty=positionSize)
```

**Python implementation:**
- `allows_reversal()` returns `True`
- `calculate_entry()` returns `(entry_price, nan, nan)` (no stop/target)
- Position sizing: 100% of equity

### Stops-Based Strategy

**Characteristics:**
- Can be flat (no position)
- Uses stops and targets
- May have trailing logic

**Annotations:**
```pine
// @strategy-type: stops-based
// @python-note: Uses stops, targets, and trailing exits

if longConditions
    stopLong   = lowestLong - (atr * stopAtr)
    targetLong = close + ((close - stopLong) * stopRr)
    strategy.entry("Long", strategy.long, stop=stopLong, limit=targetLong)
```

**Python implementation:**
- `allows_reversal()` returns `False`
- `calculate_entry()` returns `(entry_price, stop_price, target_price)`
- Position sizing: Risk-based

---

## Common Patterns

### Pattern 1: Moving Averages

```pine
// @python-note: MA function already in indicators.py
// @location: indicators.py (already implemented)
getMA(source, maType, length) =>
    if length == 0
        na
    else
        switch maType
            "SMA" => ta.sma(source, length)
            "EMA" => ta.ema(source, length)
            "HMA" => ta.hma(source, length)
            => ta.ema(source, length)

// Usage
ma1 = getMA(close, maType1, maLength1)
ma2 = getMA(close, maType2, maLength2)
ma3 = getMA(close, maType3, maLength3)
```

### Pattern 2: Close Count

```pine
// @python-note: Track consecutive closes above/below MA
var int counterCloseLong  = 0
var int counterCloseShort = 0

if close > ma3
    counterCloseLong += 1
    counterCloseShort := 0
else if close < ma3
    counterCloseShort += 1
    counterCloseLong := 0
else
    counterCloseLong := 0
    counterCloseShort := 0

countLong  = counterCloseLong >= closeCountLong
countShort = counterCloseShort >= closeCountShort
```

### Pattern 3: Stop Loss Calculation

```pine
// @python-note: ATR-based stop with lookback period
atr         = ta.atr(atrPeriod)
lowestLong  = ta.lowest(low, stopLongLp)
highestShort = ta.highest(high, stopShortLp)

stopLong  = lowestLong - (atr * stopLongAtr)
stopShort = highestShort + (atr * stopShortAtr)

// Stop distance and target
stopDistanceLong = close - stopLong
targetLong       = close + (stopDistanceLong * stopLongRr)
```

### Pattern 4: Position Sizing

```pine
// Risk-based (stops-based strategy)
positionSize = math.floor(((strategy.equity * (riskPerTradePct/100)) / (close - stopLong)) / contractSize) * contractSize

// Equity-based (reversal strategy)
positionSize = math.floor((strategy.equity / close) / contractSize) * contractSize
```

### Pattern 5: Trailing Stop

```pine
// @python-note: Trailing MA activates at RR threshold
var trailMAPriceLong     = 0.0
var trailMAActivatedLong = 0

// Activation check
if high >= (t_entry + ((t_entry - t_stop) * trailRrLong)) and strategy.position_size > 0
    trailMAActivatedLong := 1

// Update trailing price
if trailMALong > trailMAPriceLong
    trailMAPriceLong := trailMALong

// Exit on trail breach
if strategy.position_size > 0 and trailMAActivatedLong == 1
    strategy.exit("Long Exit", from_entry="Long", stop=trailMAPriceLong)
```

---

## Reference Testing

Add reference test configuration to verify translation accuracy:

```pine
// ═══════════════════════════════════════════════════════════
// REFERENCE TEST CONFIGURATION
// ═══════════════════════════════════════════════════════════
// @reference-test
// Symbol: OKX_LINKUSDT.P
// Timeframe: 15m
// Date Range: 2025-04-01 to 2025-09-01
//
// Parameters (defaults set for reference test):
//   maType = "EMA", maLength = 45
//   closeCountLong = 7, closeCountShort = 5
//   stopLongAtr = 2.0, stopLongRr = 3.0
//   trailRrLong = 1.0, trailMaLongType = "SMA"
//   riskPerTradePct = 2.0, contractSize = 0.01
//
// Expected Results (TradingView):
//   Net Profit: +18.42%
//   Max Drawdown: -5.67%
//   Total Trades: 23
//   Win Rate: 52%
//
// @python-note: Python results must match within ±0.5% tolerance
// ═══════════════════════════════════════════════════════════
```

**Purpose:**
- Validate Python translation accuracy
- Catch calculation differences early
- Document expected behavior

---

## What to Remove/Skip

### Remove (Dead Code)

**Delete unused variables:**
```pine
// DELETE - never used
atr = ta.atr(14)  // Computed but not referenced

breakOut  = input.bool(false, "Breakout")   // Not used in logic
useClose  = input.bool(false, "Use Close")  // Not used in logic
comments  = input.string("", "Comments")    // UI-only field
```

### Skip (TradingView-Specific)

**Mark for skipping:**
```pine
// @skip-translation-start (UI parameters)
useMA1    = input.bool(true, "Show MA1")
maColor1  = input.color(color.blue, "MA1 Color")
maColor2  = input.color(color.red, "MA2 Color")
// @skip-translation-end

// @skip-translation-start (Plotting)
plot(ma1, "MA1", maColor1, linewidth=2)
plotshape(longCondition, "Long", shape.triangleup, location.belowbar, color.green)
table.new(...)  // Stats table
// @skip-translation-end

// @skip-translation-start (Days of week filter - not in Python)
tradeMonday   = input.bool(true, "Mon")
tradeTuesday  = input.bool(true, "Tue")
tradeDays = (tradeMonday and time(..., '2', 'GMT')) or (tradeTuesday and time(...))
// @skip-translation-end
```

### Skip (Already in Project)

**Date filter:**
```pine
// @python-note: Date filter already implemented in BaseStrategy
// @skip-translation-start
useDateFilter = input.bool(true, "Date Filter")
startDate     = input.time(timestamp('2025-01-01'), "Start Date")
endDate       = input.time(timestamp('2025-12-31'), "End Date")
timeInRange   = not useDateFilter or time >= startDate and time <= endDate
// @skip-translation-end

// @python-note: In Python - use self.params['dateFilter'], 'startDate', 'endDate'
```

**MA functions:**
```pine
// @python-note: These functions already exist in indicators.py
// @location: indicators.py (already implemented)
// @skip-translation-start
getMA(source, maType, length) => ...
gd(source, length) => ...  // T3 helper
// @skip-translation-end
```

### Skip (Pine-Specific)

**Bar state checks:**
```pine
longCondition = upTrend and strategy.position_size == 0 and barstate.isconfirmed  // @skip-in-python

// @python-note: barstate.isconfirmed not needed in Python (bars already closed)
```

---

## Pre-Translation Checklist

Before sending Pine code for translation:

- [ ] **Naming:** All variables use camelCase (snake_case only for `var` state variables like `t_entry`)
- [ ] **Annotations:** `@strategy-type` specified at top
- [ ] **Sections:** Code organized into clear sections (Inputs, Calculations, Counters, Conditions, Execution)
- [ ] **Skip markers:** UI/plotting code marked with `@skip-translation-start/end`
- [ ] **Dead code removed:** Unused variables/parameters deleted
- [ ] **Reference test:** Expected results documented with `@reference-test`
- [ ] **Functions annotated:** MA functions marked as `@location: indicators.py` or `@skip-translation`
- [ ] **Comments added:** Key logic explained with `@python-note`
- [ ] **Tested in TradingView:** Strategy works correctly with default parameters

---

## Example: Fully Annotated Strategy

See `/data/S_01 Movings_v26 TrailingMA Ultralight.pine` for a complete stops-based strategy example.

See `/data/S_03 Reversal_v07 Light for PROJECT PLAN.pine` for a complete reversal strategy example.

---

**End of pine_guidelines.md**
