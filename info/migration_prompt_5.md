# Migration Prompt 5: Add S_03 Reversal Strategy

**Phase:** 5 of 6
**Estimated Time:** 2-3 days
**Difficulty:** Medium
**Dependencies:** Phases 1-4 must be complete

---

## Overview

This phase adds the second strategy to the system: **S_03 Reversal v07 Light**. This is a reversal strategy that is always in the market (long or short), unlike S_01 which can be flat.

**Key Differences from S_01:**
- **Always in market**: No flat periods, switches directly from long to short
- **No stops or targets**: Only exits on reversal signals
- **Simpler parameters**: ~12 parameters vs S_01's 28
- **No trailing logic**: Pure MA crossover reversals
- **100% equity**: Always uses full account balance

This phase validates the multi-strategy architecture by proving a second strategy can be added with minimal effort.

---

## Objectives

1. Translate S_03 PineScript to Python following established patterns
2. Implement `S03Reversal` class extending `BaseStrategy`
3. Register strategy in `StrategyRegistry`
4. Establish reference test baseline
5. Verify reversal behavior (no flat gaps)
6. Ensure S_01 still works (no regression)

---

## Prerequisites

### Files That Must Exist

- `src/strategies/base_strategy.py` (from Phase 2)
- `src/strategies/s01_trailing_ma.py` (from Phase 3)
- `src/strategy_registry.py` (from Phase 2)
- `src/indicators.py` (from Phase 1)
- `src/backtest_engine.py` (refactored in Phase 4)
- `src/optimizer_engine.py` (refactored in Phase 4)

### Reference PineScript

**Location:** `data/S_03 Reversal_v07 Light for PROJECT PLAN.pine`

**Read this file carefully** to understand the strategy logic. Key sections:
- Lines 1-40: Parameter inputs
- Lines 50-120: Indicator calculations (3 MAs)
- Lines 130-200: Entry/exit conditions
- Lines 210-280: Counter logic for close counts

---

## Step 1: Create S03Reversal Class

### File: `src/strategies/s03_reversal.py`

Create a new file implementing the S_03 Reversal strategy.

#### 1.1 Class Structure

```python
"""
S_03 Reversal v07 Light Strategy

A reversal trading strategy that is always in the market (long or short).
Uses MA crossovers with close count confirmation to switch positions.

Key Characteristics:
- Always in market (no flat periods)
- No stop-loss or take-profit levels
- Position reverses on opposite signal
- Uses 100% equity for position sizing
"""

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy
from indicators import get_ma


class S03Reversal(BaseStrategy):
    """
    S_03 Reversal Strategy Implementation

    This strategy maintains a position at all times, switching between
    long and short based on MA crossovers confirmed by close counts.
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Initialize S_03 Reversal strategy with parameters.

        Args:
            params: Dictionary with all strategy parameters from get_param_definitions()
        """
        super().__init__(params)

        # Extract parameters for easy access
        self.ma_fast_type = params['maFastType']
        self.ma_fast_length = params['maFastLength']

        self.ma_slow_type = params['maSlowType']
        self.ma_slow_length = params['maSlowLength']

        self.ma_trend_type = params['maTrendType']
        self.ma_trend_length = params['maTrendLength']

        self.close_count_long = params['closeCountLong']
        self.close_count_short = params['closeCountShort']

        # Date filtering
        self.date_filter = params['dateFilter']
        self.start_date = pd.to_datetime(params['startDate']) if params['startDate'] else None
        self.end_date = pd.to_datetime(params['endDate']) if params['endDate'] else None

        # Position sizing
        self.equity_pct = params['equityPct']  # Default 100%

        # Internal state (set during _prepare_data)
        self.df = None
        self.ma_fast = None
        self.ma_slow = None
        self.ma_trend = None

        # Counters (reset during simulation)
        self.counter_close_long = 0
        self.counter_close_short = 0

    @staticmethod
    def get_param_definitions() -> Dict[str, Dict[str, Any]]:
        """
        Define all parameters for S_03 Reversal strategy.

        Returns complete parameter definitions with:
        - default: Default value for backtesting
        - type: Data type (int, float, str, bool, date)
        - min/max: Valid range (for numeric types)
        - options: Valid choices (for categorical types)
        - description: Human-readable explanation
        """
        return {
            # === Fast MA ===
            'maFastType': {
                'default': 'EMA',
                'type': 'str',
                'options': ['SMA', 'EMA', 'HMA', 'WMA', 'VWMA', 'ALMA', 'KAMA', 'T3', 'DEMA', 'TMA', 'VWAP'],
                'description': 'Fast MA type for entry signals'
            },
            'maFastLength': {
                'default': 21,
                'type': 'int',
                'min': 5,
                'max': 100,
                'description': 'Fast MA period length'
            },

            # === Slow MA ===
            'maSlowType': {
                'default': 'EMA',
                'type': 'str',
                'options': ['SMA', 'EMA', 'HMA', 'WMA', 'VWMA', 'ALMA', 'KAMA', 'T3', 'DEMA', 'TMA', 'VWAP'],
                'description': 'Slow MA type for entry signals'
            },
            'maSlowLength': {
                'default': 50,
                'type': 'int',
                'min': 10,
                'max': 200,
                'description': 'Slow MA period length'
            },

            # === Trend Filter MA ===
            'maTrendType': {
                'default': 'SMA',
                'type': 'str',
                'options': ['SMA', 'EMA', 'HMA', 'WMA', 'VWMA', 'ALMA', 'KAMA', 'T3', 'DEMA', 'TMA', 'VWAP'],
                'description': 'Trend filter MA type'
            },
            'maTrendLength': {
                'default': 100,
                'type': 'int',
                'min': 20,
                'max': 300,
                'description': 'Trend filter MA period'
            },

            # === Close Count Logic ===
            'closeCountLong': {
                'default': 3,
                'type': 'int',
                'min': 1,
                'max': 10,
                'description': 'Consecutive closes above fast MA required for long entry'
            },
            'closeCountShort': {
                'default': 3,
                'type': 'int',
                'min': 1,
                'max': 10,
                'description': 'Consecutive closes below fast MA required for short entry'
            },

            # === Date Filter ===
            'dateFilter': {
                'default': False,
                'type': 'bool',
                'description': 'Enable date range filtering'
            },
            'startDate': {
                'default': '2020-01-01',
                'type': 'date',
                'description': 'Start date for trading (if dateFilter enabled)'
            },
            'endDate': {
                'default': '2025-12-31',
                'type': 'date',
                'description': 'End date for trading (if dateFilter enabled)'
            },

            # === Position Sizing ===
            'equityPct': {
                'default': 100.0,
                'type': 'float',
                'min': 10.0,
                'max': 100.0,
                'description': 'Percentage of equity to use per trade'
            },

            # === Commission ===
            'commission': {
                'default': 0.06,
                'type': 'float',
                'min': 0.0,
                'max': 1.0,
                'description': 'Commission per trade (%)'
            }
        }

    @staticmethod
    def allows_reversal() -> bool:
        """
        Indicate this strategy supports reversal (always in market).

        Returns:
            True - this strategy can reverse from long to short directly
        """
        return True

    @staticmethod
    def get_cache_requirements(param_combinations: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Specify caching requirements for optimization.

        For S_03, we need:
        - MA values for fast/slow/trend MAs (all types and lengths in param grid)
        - No ATR needed (no stops)
        - No lookback caches needed (no trailing logic)

        Args:
            param_combinations: List of parameter dictionaries to optimize over

        Returns:
            Dictionary specifying what to pre-compute:
            {
                'ma_types_and_lengths': [(type, length), ...],
                'needs_atr': False,
                'needs_lookback': False
            }
        """
        if param_combinations is None:
            # Single backtest - just cache the defaults
            defaults = S03Reversal.get_param_definitions()
            ma_configs = [
                (defaults['maFastType']['default'], defaults['maFastLength']['default']),
                (defaults['maSlowType']['default'], defaults['maSlowLength']['default']),
                (defaults['maTrendType']['default'], defaults['maTrendLength']['default'])
            ]
            return {
                'ma_types_and_lengths': list(set(ma_configs)),  # Remove duplicates
                'needs_atr': False,
                'needs_lookback': False
            }

        # Optimization - collect all unique MA type/length combinations
        ma_configs = set()

        for params in param_combinations:
            ma_configs.add((params['maFastType'], params['maFastLength']))
            ma_configs.add((params['maSlowType'], params['maSlowLength']))
            ma_configs.add((params['maTrendType'], params['maTrendLength']))

        return {
            'ma_types_and_lengths': list(ma_configs),
            'needs_atr': False,
            'needs_lookback': False
        }

    def _prepare_data(self, df: pd.DataFrame, cached_data: Optional[Dict] = None):
        """
        Prepare data and indicators for simulation.

        Args:
            df: OHLCV DataFrame
            cached_data: Pre-computed indicator values (from optimizer)
        """
        self.df = df

        # Get MAs from cache or compute them
        if cached_data and 'ma_cache' in cached_data:
            ma_cache = cached_data['ma_cache']
            fast_key = (self.ma_fast_type, self.ma_fast_length)
            slow_key = (self.ma_slow_type, self.ma_slow_length)
            trend_key = (self.ma_trend_type, self.ma_trend_length)

            self.ma_fast = ma_cache.get(fast_key)
            self.ma_slow = ma_cache.get(slow_key)
            self.ma_trend = ma_cache.get(trend_key)

            if self.ma_fast is None or self.ma_slow is None or self.ma_trend is None:
                raise ValueError(f"Required MAs not found in cache: {fast_key}, {slow_key}, {trend_key}")
        else:
            # Compute MAs directly
            self.ma_fast = get_ma(df['Close'], self.ma_fast_type, self.ma_fast_length,
                                  volume=df['Volume'], high=df['High'], low=df['Low'])
            self.ma_slow = get_ma(df['Close'], self.ma_slow_type, self.ma_slow_length,
                                  volume=df['Volume'], high=df['High'], low=df['Low'])
            self.ma_trend = get_ma(df['Close'], self.ma_trend_type, self.ma_trend_length,
                                   volume=df['Volume'], high=df['High'], low=df['Low'])

    def _is_date_allowed(self, idx: int) -> bool:
        """
        Check if current bar is within allowed date range.

        Args:
            idx: Current bar index

        Returns:
            True if date filter disabled or date is in range
        """
        if not self.date_filter:
            return True

        current_date = self.df.index[idx]

        if self.start_date and current_date < self.start_date:
            return False
        if self.end_date and current_date > self.end_date:
            return False

        return True

    def should_long(self, idx: int) -> bool:
        """
        Check if conditions are met for long entry.

        Logic (from Pine):
        1. Fast MA > Slow MA (bullish crossover state)
        2. Close > Fast MA for closeCountLong consecutive bars
        3. Close > Trend MA (optional trend filter)
        4. Date in allowed range

        Args:
            idx: Current bar index

        Returns:
            True if should enter/maintain long position
        """
        if idx < 1 or not self._is_date_allowed(idx):
            return False

        # Check if we have valid MA values
        if (pd.isna(self.ma_fast[idx]) or pd.isna(self.ma_slow[idx]) or
            pd.isna(self.ma_trend[idx])):
            return False

        # Condition 1: Fast MA > Slow MA
        if self.ma_fast[idx] <= self.ma_slow[idx]:
            self.counter_close_long = 0  # Reset counter
            return False

        # Condition 2: Close > Fast MA (increment counter)
        if self.df['Close'].iloc[idx] > self.ma_fast[idx]:
            self.counter_close_long += 1
        else:
            self.counter_close_long = 0  # Reset if condition breaks

        # Condition 3: Counter reached threshold
        if self.counter_close_long < self.close_count_long:
            return False

        # Condition 4: Close > Trend MA (trend filter)
        if self.df['Close'].iloc[idx] <= self.ma_trend[idx]:
            return False

        return True

    def should_short(self, idx: int) -> bool:
        """
        Check if conditions are met for short entry.

        Logic (from Pine):
        1. Fast MA < Slow MA (bearish crossover state)
        2. Close < Fast MA for closeCountShort consecutive bars
        3. Close < Trend MA (optional trend filter)
        4. Date in allowed range

        Args:
            idx: Current bar index

        Returns:
            True if should enter/maintain short position
        """
        if idx < 1 or not self._is_date_allowed(idx):
            return False

        # Check if we have valid MA values
        if (pd.isna(self.ma_fast[idx]) or pd.isna(self.ma_slow[idx]) or
            pd.isna(self.ma_trend[idx])):
            return False

        # Condition 1: Fast MA < Slow MA
        if self.ma_fast[idx] >= self.ma_slow[idx]:
            self.counter_close_short = 0  # Reset counter
            return False

        # Condition 2: Close < Fast MA (increment counter)
        if self.df['Close'].iloc[idx] < self.ma_fast[idx]:
            self.counter_close_short += 1
        else:
            self.counter_close_short = 0  # Reset if condition breaks

        # Condition 3: Counter reached threshold
        if self.counter_close_short < self.close_count_short:
            return False

        # Condition 4: Close < Trend MA (trend filter)
        if self.df['Close'].iloc[idx] >= self.ma_trend[idx]:
            return False

        return True

    def calculate_entry(self, idx: int, direction: str) -> Tuple[float, float, float]:
        """
        Calculate entry price and levels.

        For reversal strategy:
        - Entry at next bar's open
        - NO stop-loss (returns NaN)
        - NO take-profit (returns NaN)

        Args:
            idx: Current bar index
            direction: 'long' or 'short'

        Returns:
            (entry_price, stop_price, target_price)
            For S_03: (next_open, NaN, NaN)
        """
        if idx + 1 >= len(self.df):
            return (np.nan, np.nan, np.nan)

        entry_price = self.df['Open'].iloc[idx + 1]

        # No stops or targets for reversal strategy
        return (entry_price, np.nan, np.nan)

    def calculate_position_size(self, idx: int, entry_price: float,
                                stop_price: float, direction: str,
                                current_equity: float) -> float:
        """
        Calculate position size based on equity percentage.

        For S_03:
        - Use fixed percentage of equity (default 100%)
        - No risk-based sizing (no stops)

        Args:
            idx: Current bar index
            entry_price: Entry price for position
            stop_price: Stop-loss price (unused for S_03)
            direction: 'long' or 'short'
            current_equity: Current account equity

        Returns:
            Position size in base currency units
        """
        if entry_price <= 0:
            return 0.0

        # Use percentage of equity
        position_value = current_equity * (self.equity_pct / 100.0)
        position_size = position_value / entry_price

        return position_size

    def should_exit(self, idx: int, position_info: Dict) -> Tuple[bool, Optional[float], str]:
        """
        Check if current position should be exited.

        For S_03 Reversal:
        - NEVER exits on stops/targets (they don't exist)
        - Only exits when opposite signal triggers (handled by BaseStrategy)
        - This method always returns False

        Args:
            idx: Current bar index
            position_info: Dictionary with position details:
                - direction: 'long' or 'short'
                - entry_idx: Bar index when entered
                - entry_price: Entry price
                - stop_price: Stop price (NaN for S_03)
                - target_price: Target price (NaN for S_03)
                - size: Position size

        Returns:
            (should_exit, exit_price, exit_reason)
            For S_03: Always (False, None, '')
        """
        # Reversal strategy never exits except on opposite signal
        # BaseStrategy will handle reversals automatically
        return (False, None, '')

    def _run_simulation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run the main trading simulation loop.

        This method is called by BaseStrategy.simulate() and handles
        the bar-by-bar logic specific to S_03 Reversal.

        IMPORTANT: For reversal strategies, we MUST always be in a position.
        The first valid signal determines initial direction.

        Returns:
            Dictionary with simulation results and metrics
        """
        # Reset counters at start of simulation
        self.counter_close_long = 0
        self.counter_close_short = 0

        # Use parent class simulation with reversal support
        # BaseStrategy._run_simulation() will handle:
        # - Checking allows_reversal() == True
        # - Switching positions on opposite signals
        # - Never going flat
        return super()._run_simulation(df)
```

**Key Implementation Notes:**

1. **Counter Management**: Counters are reset when MA crossover state changes (fast vs slow)
2. **Always In Market**: `allows_reversal() = True` signals to BaseStrategy to maintain positions
3. **No Exits**: `should_exit()` always returns False - only reversals close positions
4. **100% Equity**: Default position sizing uses full account balance
5. **Date Filter**: Optional date range restriction for backtesting specific periods

---

## Step 2: Register Strategy

### File: `src/strategy_registry.py`

Update the registry to include S_03:

```python
from strategies.s01_trailing_ma import S01TrailingMA
from strategies.s03_reversal import S03Reversal  # NEW


class StrategyRegistry:
    """Central registry of all available strategies."""

    _strategies = {
        's01_trailing_ma': S01TrailingMA,
        's03_reversal': S03Reversal  # NEW
    }

    @classmethod
    def get_strategy_class(cls, strategy_id: str):
        """Get strategy class by ID."""
        if strategy_id not in cls._strategies:
            raise ValueError(f"Unknown strategy: {strategy_id}. Available: {list(cls._strategies.keys())}")
        return cls._strategies[strategy_id]

    @classmethod
    def get_all_strategies(cls) -> Dict[str, Type[BaseStrategy]]:
        """Get all registered strategies."""
        return cls._strategies.copy()

    @classmethod
    def get_strategy_info(cls) -> List[Dict[str, str]]:
        """
        Get metadata about all strategies for API/UI.

        Returns:
            List of dicts with strategy_id, name, description
        """
        return [
            {
                'strategy_id': 's01_trailing_ma',
                'name': 'S_01 TrailingMA v26 Ultralight',
                'description': 'Trend-following with MA crossovers, trailing stops, and ATR-based exits',
                'type': 'trend'
            },
            {
                'strategy_id': 's03_reversal',
                'name': 'S_03 Reversal v07 Light',
                'description': 'Always-in-market reversal strategy using MA crossovers',
                'type': 'reversal'
            }
        ]
```

---

## Step 3: Update CLI

### File: `src/run_backtest.py`

Verify the CLI already supports `--strategy` flag (added in Phase 4):

```python
parser.add_argument('--strategy', type=str, default='s01_trailing_ma',
                    help='Strategy ID (s01_trailing_ma, s03_reversal)')
```

**Test the CLI:**

```bash
cd src
python run_backtest.py --csv ../data/OKX_LINKUSDT.P,\ 15...csv --strategy s03_reversal
```

Expected output:
```
Strategy: S_03 Reversal v07 Light
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Net Profit: XX.XX%
Max Drawdown: XX.XX%
Total Trades: XXX
...
```

---

## Step 4: Establish Reference Test Baseline

### Purpose

Create the baseline metrics that will be used to validate future changes don't break S_03.

### Procedure

1. **Run baseline test:**

```bash
cd src
python run_backtest.py --csv ../data/OKX_LINKUSDT.P,\ 15...csv --strategy s03_reversal
```

2. **Record ALL metrics** in `info/tests.md`:

Update the "S_03 Reversal v07 Light" section with actual values:

```markdown
### Expected Results

```
Strategy: S_03 Reversal v07 Light
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Performance Metrics:
├─ Net Profit:        XX.XX% ← FILL IN ACTUAL
├─ Max Drawdown:      XX.XX% ← FILL IN ACTUAL
├─ Total Trades:      XXX ← FILL IN ACTUAL
├─ Winning Trades:    XXX ← FILL IN ACTUAL
├─ Losing Trades:     XXX ← FILL IN ACTUAL
├─ Win Rate:          XX.XX% ← FILL IN ACTUAL
├─ Profit Factor:     X.XX ← FILL IN ACTUAL
└─ Sharpe Ratio:      X.XX ← FILL IN ACTUAL

Special Checks (Reversal Strategy):
├─ Always in position: YES / NO ← Should be YES
├─ Gaps (flat periods): XXX ← Should be 0
└─ Position changes: XXX ← Should be > 0
```
```

3. **Verify reversal behavior:**

Add debug logging to check for gaps:

```python
# In BaseStrategy._run_simulation() or in your test
position_history = []  # Track position state each bar

# After simulation
gaps = sum(1 for p in position_history if p is None)
print(f"Flat periods detected: {gaps}")  # Should be 0 for reversal strategy
```

---

## Step 5: Regression Testing

### Critical: S_01 Must Still Work

Run S_01 reference test to ensure adding S_03 didn't break anything:

```bash
cd src
python run_backtest.py --csv ../data/OKX_LINKUSDT.P,\ 15...csv --strategy s01_trailing_ma
```

Compare output with baseline from Phase 3. **All metrics must match within tolerance.**

### Run Small Optimization Test

Test that optimizer works with S_03:

```python
from optimizer_engine import run_optimization, OptimizationConfig

config = OptimizationConfig(
    csv_file=open('../data/OKX_LINKUSDT.P, 15...csv'),
    strategy_id='s03_reversal',  # NEW
    enabled_params={'maFastLength': True},
    param_ranges={'maFastLength': (15, 30, 5)},  # 4 values
    fixed_params={},  # Use defaults for all others
    worker_processes=2
)

results = run_optimization(config)
print(f"✅ Completed {len(results)} combinations")
assert len(results) == 4, "Should have 4 results"
```

Expected: No errors, 4 results, reasonable metrics.

---

## Step 6: Testing Checklist

Run through this checklist before committing:

### Functional Tests

- [ ] S_03 backtest runs without errors
- [ ] S_03 baseline metrics recorded in `tests.md`
- [ ] S_03 always in position (no flat gaps)
- [ ] S_03 has position changes (not stuck in one direction)
- [ ] S_01 reference test still passes (no regression)
- [ ] Small optimization test passes for S_03
- [ ] CLI `--strategy s03_reversal` works
- [ ] StrategyRegistry returns both strategies

### Code Quality

- [ ] Type hints on all methods
- [ ] Docstrings match BaseStrategy format
- [ ] Counter logic matches PineScript exactly
- [ ] Parameter definitions complete and accurate
- [ ] Cache requirements correct (MA only, no ATR)
- [ ] No hardcoded values (everything from params)

### Edge Cases

- [ ] Handles NaN in MA values gracefully
- [ ] Date filter works (test with startDate/endDate)
- [ ] Works with all 11 MA types (test at least SMA, EMA, HMA)
- [ ] Counter resets correctly on MA crossover changes
- [ ] Position sizing handles zero/negative equity

---

## Step 7: Commit

### Commit Message

```
feat: Add S_03 Reversal strategy (Phase 5)

Implements S_03 Reversal v07 Light as second strategy in multi-strategy system.

Changes:
- Add src/strategies/s03_reversal.py with S03Reversal class
- Register S_03 in StrategyRegistry
- Establish reference test baseline in info/tests.md
- Verify reversal behavior (always in market, no flat periods)

Strategy Characteristics:
- Always in market (long or short, never flat)
- Uses 3 MAs (fast/slow/trend) with close count confirmation
- No stop-loss or take-profit levels
- Reverses position on opposite signal
- ~12 parameters vs S_01's 28

Testing:
- S_03 baseline: [NET_PROFIT]%, [TRADES] trades, [SHARPE] sharpe
- No flat periods detected (reversal behavior confirmed)
- S_01 regression test passed (no impact to existing strategy)
- Optimization test passed with 4 parameter combinations

Phase 5 of 6 complete. Next: Update UI/API for strategy selection.
```

### Files Changed

```
M  src/strategy_registry.py
A  src/strategies/s03_reversal.py
M  info/tests.md
```

---

## Common Issues and Solutions

### Issue 1: S_03 Has Flat Periods

**Symptom:** Reference test shows gaps > 0

**Cause:** BaseStrategy not recognizing reversal capability

**Fix:**
- Verify `allows_reversal()` returns `True`
- Check BaseStrategy._run_simulation() has reversal logic
- Ensure first valid signal initializes position

**Debug:**
```python
# Add logging in BaseStrategy
print(f"Strategy allows reversal: {self.allows_reversal()}")
print(f"Position transitions: long→short={long_to_short}, short→long={short_to_long}")
```

### Issue 2: Counters Not Resetting

**Symptom:** No trades generated, counters stuck at 0

**Cause:** Counter reset logic in wrong place

**Fix:**
- Reset counters when MA crossover state CHANGES (fast vs slow relationship)
- Do NOT reset on every bar
- Increment only when close condition met, reset only when broken

**Correct Logic:**
```python
def should_long(self, idx):
    # Check crossover state FIRST
    if self.ma_fast[idx] <= self.ma_slow[idx]:
        self.counter_close_long = 0  # Reset because crossover state wrong
        return False

    # THEN check close condition
    if self.df['Close'].iloc[idx] > self.ma_fast[idx]:
        self.counter_close_long += 1
    else:
        self.counter_close_long = 0  # Reset because close condition broken

    # FINALLY check threshold
    return self.counter_close_long >= self.close_count_long
```

### Issue 3: Date Filter Not Working

**Symptom:** Trades outside specified date range

**Cause:** `_is_date_allowed()` called after signal check

**Fix:**
- Call `_is_date_allowed()` FIRST in `should_long()` and `should_short()`
- Return False immediately if date not in range

### Issue 4: Optimization Fails with Cache Error

**Symptom:** `KeyError` when accessing MA cache

**Cause:** `get_cache_requirements()` not returning all needed MAs

**Fix:**
- Ensure all 3 MAs (fast/slow/trend) are in cache requirements
- Check all unique combinations across parameter grid
- Verify cache keys match (type, length) tuples

---

## Acceptance Criteria

Phase 5 is complete when:

1. ✅ S_03 backtest runs successfully
2. ✅ S_03 baseline metrics recorded
3. ✅ S_03 has zero flat periods (always in position)
4. ✅ S_03 has > 0 position changes (not stuck)
5. ✅ S_01 regression test passes
6. ✅ Optimization works with S_03
7. ✅ CLI supports `--strategy s03_reversal`
8. ✅ Code follows established patterns
9. ✅ All tests in checklist pass
10. ✅ Changes committed and pushed

---

## Next Phase

**Phase 6:** Update UI and API to support strategy selection

This will add:
- Strategy dropdown in web UI
- Dynamic parameter forms based on `get_param_definitions()`
- API endpoint to list strategies
- Updated optimization interface

**Estimated Time:** 1-2 days

---

**End of migration_prompt_5.md**
