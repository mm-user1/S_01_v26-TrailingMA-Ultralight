# Migration Checklist - Multi-Strategy Architecture

## Overview

This document provides a **phase-by-phase migration plan** from the current single-strategy architecture to the multi-strategy system.

**Total Estimated Time:** 7-10 days
**Phases:** 6
**Safety:** Each phase is independently testable with rollback points

---

## Migration Principles

### 1. Safety First
- ✅ Each phase ends with working code
- ✅ Reference tests run after EVERY phase
- ✅ S_01 behavior must remain identical throughout
- ✅ Commit after each completed phase

### 2. Test-Driven Migration
- Run reference test BEFORE changes (baseline)
- Make changes
- Run reference test AFTER changes (must match baseline)
- If mismatch → rollback and debug

### 3. Backward Compatibility
- Keep old function names as aliases during migration
- Default `strategy_id='s01_trailing_ma'` in all endpoints
- Remove deprecated code only in final cleanup phase

---

## Pre-Migration Checklist

**Before starting Phase 1:**

- [ ] Backup entire project
  ```bash
  git tag pre-migration-backup
  git push --tags
  ```

- [ ] Create migration branch
  ```bash
  git checkout -b migration/multi-strategy
  ```

- [ ] Run baseline reference test for S_01
  ```bash
  cd src
  python run_backtest.py --csv ../data/OKX_LINKUSDT.P,\ 15...csv
  ```
  - [ ] Record Net Profit %: `_________`
  - [ ] Record Max DD %: `_________`
  - [ ] Record Total Trades: `_________`
  - [ ] Save output to `baseline_test_results.txt`

- [ ] Verify current system works
  - [ ] Single backtest via CLI works
  - [ ] Single backtest via UI works
  - [ ] Grid optimization works
  - [ ] Optuna optimization works
  - [ ] WFA works
  - [ ] CSV export/import works
  - [ ] Preset save/load works

---

## Phase 1: Extract Indicators Module

**Goal:** Move all indicator functions from `backtest_engine.py` to new `indicators.py` module.

**Duration:** 0.5-1 day

**Why First:** Establishes foundation for shared code. Low risk.

### Tasks

- [ ] Create `/src/indicators.py`

- [ ] Move indicator functions from `backtest_engine.py` to `indicators.py`:
  - [ ] `sma(series, length)`
  - [ ] `ema(series, length)`
  - [ ] `hma(series, length)`
  - [ ] `wma(series, length)`
  - [ ] `vwma(series, volume, length)`
  - [ ] `alma(series, length, offset, sigma)`
  - [ ] `kama(series, length)`
  - [ ] `tma(series, length)`
  - [ ] `t3(series, length, vfactor)`
  - [ ] `dema(series, length)`
  - [ ] `vwap(high, low, close, volume)`
  - [ ] `get_ma(series, ma_type, length, volume, high, low)` - unified interface
  - [ ] `atr(high, low, close, period)`

- [ ] Update imports in `backtest_engine.py`:
  ```python
  from indicators import (
      sma, ema, hma, wma, vwma, alma, kama,
      tma, t3, dema, vwap, get_ma, atr
  )
  ```

- [ ] Update imports in `optimizer_engine.py`:
  ```python
  from indicators import get_ma, atr
  ```

- [ ] Verify no broken imports:
  ```bash
  cd src
  python -c "import backtest_engine"
  python -c "import optimizer_engine"
  python -c "import optuna_engine"
  ```

### Testing

- [ ] **Reference Test:** Run S_01 baseline test
  - [ ] Net Profit % matches baseline: ✅ / ❌
  - [ ] Max DD % matches baseline: ✅ / ❌
  - [ ] Total Trades matches baseline: ✅ / ❌

- [ ] **Unit Test:** Test individual indicator functions
  ```python
  from indicators import sma, ema
  test_series = pd.Series([1, 2, 3, 4, 5])
  assert sma(test_series, 3).iloc[-1] == 4.0
  ```

### Commit

```bash
git add src/indicators.py src/backtest_engine.py src/optimizer_engine.py
git commit -m "Phase 1: Extract indicators module

- Create indicators.py with all MA functions and ATR
- Update imports in backtest_engine.py and optimizer_engine.py
- Reference test: S_01 results unchanged ✅"
```

**Detailed Prompt:** See `migration_prompt_1.md`

---

## Phase 2: Create Base Strategy Contract

**Goal:** Create `BaseStrategy` ABC class and `strategy_registry.py`.

**Duration:** 1-1.5 days

**Why:** Establishes the contract for all future strategies. No changes to existing code yet.

### Tasks

- [ ] Create `/src/strategies/` folder
  ```bash
  mkdir -p src/strategies
  ```

- [ ] Create `/src/strategies/__init__.py`
  ```python
  from .base_strategy import BaseStrategy
  ```

- [ ] Create `/src/strategies/base_strategy.py`
  - [ ] Import necessary modules (ABC, abstractmethod, pandas, numpy, math)
  - [ ] Define `BaseStrategy` class with `ABC` inheritance
  - [ ] Add class attributes: `STRATEGY_ID`, `STRATEGY_NAME`, `VERSION`
  - [ ] Define abstract methods:
    - [ ] `should_long(self, idx: int) -> bool`
    - [ ] `should_short(self, idx: int) -> bool`
    - [ ] `calculate_entry(self, idx: int, direction: str) -> Tuple[float, float, float]`
    - [ ] `calculate_position_size(self, idx: int, direction: str, entry_price: float, stop_price: float, equity: float) -> float`
    - [ ] `should_exit(self, idx: int, position_info: Dict) -> Tuple[bool, Optional[float], str]`
    - [ ] `get_param_definitions(cls) -> Dict[str, Dict]` - classmethod
    - [ ] `_prepare_data(self, df: pd.DataFrame, cached_data: Optional[Dict]) -> None`
  - [ ] Define concrete methods:
    - [ ] `__init__(self, params: Dict[str, Any])`
    - [ ] `simulate(self, df: pd.DataFrame, cached_data: Optional[Dict] = None) -> Dict[str, Any]`
    - [ ] `_run_simulation(self, df: pd.DataFrame) -> Dict[str, Any]`
    - [ ] `get_cache_requirements(cls, param_combinations: List[Dict]) -> Dict` - classmethod (default empty)
    - [ ] `allows_reversal(self) -> bool` (default False)
    - [ ] `_validate_params(self) -> None` (optional, empty default)
  - [ ] Add comprehensive docstrings

- [ ] Implement `_run_simulation()` with generic simulation loop:
  - [ ] Initialize position state (position=0, equity=100.0, trades=[])
  - [ ] Loop through all bars
  - [ ] Exit logic: call `should_exit()` if in position
  - [ ] Reversal logic: if `allows_reversal()` and has opposite signal
  - [ ] Entry logic: call `should_long()/should_short()` if flat
  - [ ] Track equity curve
  - [ ] Calculate metrics (net profit, max DD, sharpe, etc.)
  - [ ] Return standardized dict

- [ ] Create `/src/strategy_registry.py`
  - [ ] Define `StrategyRegistry` class
  - [ ] Add `_strategies` dict (empty for now)
  - [ ] Implement `get_strategy_class(strategy_id)` classmethod
  - [ ] Implement `get_strategy_instance(strategy_id, params)` classmethod
  - [ ] Implement `list_strategies()` classmethod
  - [ ] Add error handling for unknown strategy IDs

### Testing

- [ ] Test imports:
  ```bash
  python -c "from strategies.base_strategy import BaseStrategy"
  python -c "from strategy_registry import StrategyRegistry"
  ```

- [ ] Test that BaseStrategy cannot be instantiated:
  ```python
  from strategies.base_strategy import BaseStrategy
  try:
      strategy = BaseStrategy({})
      print("ERROR: Should not be able to instantiate ABC")
  except TypeError:
      print("OK: ABC enforcement works")
  ```

- [ ] **Reference Test:** S_01 unchanged (no changes to existing code yet)

### Commit

```bash
git add src/strategies/ src/strategy_registry.py
git commit -m "Phase 2: Create base strategy contract

- Create BaseStrategy ABC with all abstract methods
- Implement generic _run_simulation() loop
- Create StrategyRegistry for strategy management
- No changes to existing code yet"
```

**Detailed Prompt:** See `migration_prompt_2.md`

---

## Phase 3: Extract S_01 to Strategy Module

**Goal:** Move S_01 logic from `backtest_engine.py` into `strategies/s01_trailing_ma.py`.

**Duration:** 2-3 days

**Why:** Critical phase - must preserve exact behavior.

### Tasks

- [ ] Create `/src/strategies/s01_trailing_ma.py`

- [ ] Define `S01TrailingMA(BaseStrategy)` class:
  - [ ] Set class attributes:
    ```python
    STRATEGY_ID = "s01_trailing_ma"
    STRATEGY_NAME = "S_01 TrailingMA v26 Ultralight"
    VERSION = "26"
    ```

- [ ] Move S_01 parameters to `get_param_definitions()` (29 total):
  - [ ] Date filter: dateFilter, startDate, endDate
  - [ ] Trend MA: maType, maLength
  - [ ] Close counts: closeCountLong, closeCountShort
  - [ ] Long stops: stopLongAtr, stopLongRr, stopLongLp, stopLongMaxPct, stopLongMaxDays
  - [ ] Short stops: stopShortAtr, stopShortRr, stopShortLp, stopShortMaxPct, stopShortMaxDays
  - [ ] Trailing long: trailRrLong, trailMaLongType, trailMaLongLength, trailMaLongOffset
  - [ ] Trailing short: trailRrShort, trailMaShortType, trailMaShortLength, trailMaShortOffset
  - [ ] Risk: riskPerTradePct, contractSize, commissionRate, atrPeriod
  - [ ] All with correct types, defaults, min/max (camelCase keys)

- [ ] Implement `__init__`:
  - [ ] Call `super().__init__(params)`
  - [ ] Initialize caches: `_ma_trend`, `_atr`, `_trail_ma_long`, `_trail_ma_short`, `_lowest`, `_highest`
  - [ ] Initialize state: counters (close_trend_long/short, trade_long/short), trailing (activated, price)

- [ ] Implement `_validate_params`:
  - [ ] Check all required parameters present
  - [ ] Check types and ranges

- [ ] Implement `_prepare_data`:
  ```python
  def _prepare_data(self, df, cached_data):
      if cached_data:
          # Use pre-computed values from optimizer
          ma_key = (self.params['maType'], self.params['maLength'])
          self._ma_trend = cached_data['ma_cache'][ma_key]
          self._atr = cached_data['atr'][self.params['atrPeriod']]
          # ... etc
      else:
          # Compute on-the-fly for single backtest
          from indicators import get_ma, atr
          self._ma_trend = get_ma(df['Close'], ...).to_numpy()
          self._atr = atr(df['High'], df['Low'], df['Close'], ...).to_numpy()
          # ... etc
  ```

- [ ] Move logic from `backtest_engine.run_strategy()` to S_01 methods:

  - [ ] **should_long()**:
    - [ ] Update counters (counter_close_trend_long/short)
    - [ ] Update trade counters based on position
    - [ ] Check: counter_close_trend_long >= close_count_long
    - [ ] Check: counter_trade_long == 0 (not just exited long)
    - [ ] Check: ATR not NaN
    - [ ] Return True if all conditions met

  - [ ] **should_short()**:
    - [ ] Similar to should_long() but for short

  - [ ] **calculate_entry()**:
    - [ ] For long:
      - [ ] Get lowest low over lookback period (stopLongLp)
      - [ ] Calculate stop: lowest - (ATR * stopLongAtr)
      - [ ] Calculate entry: current close
      - [ ] Calculate target: entry + (entry - stop) * stopLongRr
      - [ ] Check max stop %: if (entry - stop) / entry > stopLongMaxPct → return (nan, nan, nan)
      - [ ] Return (entry, stop, target)
    - [ ] For short: similar with highest high

  - [ ] **calculate_position_size()**:
    - [ ] Risk-based sizing: risk_cash = equity * risk_per_trade_pct / 100
    - [ ] Quantity: risk_cash / abs(entry - stop)
    - [ ] Round to contract size
    - [ ] Return quantity

  - [ ] **should_exit()**:
    - [ ] If in long:
      - [ ] Check trailing activation: high >= entry + (entry - stop) * trail_rr_long
      - [ ] Update trailing price: max(trail_price, trail_ma_long[idx])
      - [ ] If trailing active: check low <= trail_price
      - [ ] If not trailing: check stop/target hit
      - [ ] Check max days in trade
      - [ ] Return (should_exit, exit_price, reason)
    - [ ] If in short: similar

- [ ] Implement `get_cache_requirements()`:
  ```python
  @classmethod
  def get_cache_requirements(cls, param_combinations):
      ma_specs = set()
      long_lp_values = set()
      short_lp_values = set()
      atr_periods = set()

      for combo in param_combinations:
          # Collect all MA specs
          ma_specs.add((combo['maType'], combo['maLength']))
          ma_specs.add((combo['trailMaLongType'], combo['trailMaLongLength']))
          ma_specs.add((combo['trailMaShortType'], combo['trailMaShortLength']))

          # Collect lookback periods
          long_lp_values.add(combo['stopLongLp'])
          short_lp_values.add(combo['stopShortLp'])

          # Collect ATR periods
          atr_periods.add(combo.get('atrPeriod', 14))

      return {
          'ma_types_and_lengths': list(ma_specs),
          'long_lp_values': list(long_lp_values),
          'short_lp_values': list(short_lp_values),
          'needs_atr': True
      }
  ```

- [ ] Register S_01 in `strategy_registry.py`:
  ```python
  from strategies.s01_trailing_ma import S01TrailingMA

  class StrategyRegistry:
      _strategies = {
          "s01_trailing_ma": S01TrailingMA,
      }
  ```

- [ ] Update `backtest_engine.py`:
  - [ ] Keep old `run_strategy()` function
  - [ ] Add new `run_strategy_v2()` function:
    ```python
    def run_strategy_v2(df, strategy, params):
        """New universal backtest engine"""
        result = strategy.simulate(df, cached_data=None)
        # Convert to StrategyResult format for compatibility
        return StrategyResult(...)
    ```
  - [ ] Add backward compatibility:
    ```python
    # Old function still works but uses new system internally
    def run_strategy(df, params, trade_start_idx=0):
        from strategy_registry import StrategyRegistry
        strategy = StrategyRegistry.get_strategy_instance('s01_trailing_ma', params.__dict__)
        return run_strategy_v2(df, strategy, params)
    ```

### Testing

- [ ] **Unit Test:** Test S_01 methods in isolation
  ```python
  from strategies.s01_trailing_ma import S01TrailingMA
  import pandas as pd

  # Create mock data
  df = pd.DataFrame({
      'Close': [100, 101, 102, 103, 104],
      'High': [101, 102, 103, 104, 105],
      'Low': [99, 100, 101, 102, 103],
      'Open': [100, 101, 102, 103, 104],
      'Volume': [1000, 1000, 1000, 1000, 1000]
  })

  # Test parameter definitions
  params = S01TrailingMA.get_param_definitions()
  assert 'maLength' in params
  assert params['maLength']['type'] == 'int'

  # Test strategy instantiation
  strategy = S01TrailingMA({
      'maType': 'EMA',
      'maLength': 45,
      # ... all required params
  })

  # Test should_long with controlled data
  # ... add specific tests
  ```

- [ ] **Integration Test:** Run S_01 through new system
  ```bash
  python -c "
  from strategies.s01_trailing_ma import S01TrailingMA
  from backtest_engine import load_data

  df = load_data('../data/OKX_LINKUSDT.P, 15...csv')
  strategy = S01TrailingMA({...})  # default params
  result = strategy.simulate(df)

  print(f'Net Profit: {result[\"net_profit_pct\"]:.2f}%')
  "
  ```

- [ ] **Reference Test:** CRITICAL - Must match baseline EXACTLY
  - [ ] Net Profit %: `Expected: _____ Got: _____` ✅ / ❌
  - [ ] Max DD %: `Expected: _____ Got: _____` ✅ / ❌
  - [ ] Total Trades: `Expected: _____ Got: _____` ✅ / ❌
  - [ ] **If ANY mismatch:** Debug before proceeding!

- [ ] Test via CLI (backward compatibility):
  ```bash
  python run_backtest.py --csv ../data/OKX_LINKUSDT.P,\ 15...csv
  ```

- [ ] Test via UI:
  - [ ] Open UI, run backtest with default S_01 params
  - [ ] Verify results match

### Debugging Checklist (if reference test fails)

- [ ] Check counter initialization (should start at 0)
- [ ] Check counter update logic (increment/reset conditions)
- [ ] Check stop/target calculation (sign errors, off-by-one)
- [ ] Check lowest/highest calculation (lookback period correct?)
- [ ] Check trailing activation logic (>= or >?)
- [ ] Check position sizing (rounding correct?)
- [ ] Check commission calculation
- [ ] Print intermediate values and compare with old run_strategy()

### Commit

```bash
git add src/strategies/s01_trailing_ma.py src/strategy_registry.py src/backtest_engine.py
git commit -m "Phase 3: Extract S_01 to strategy module

- Create S01TrailingMA class implementing BaseStrategy
- Move all S_01 logic from backtest_engine to strategy
- Add backward compatibility in run_strategy()
- Reference test: RESULTS IDENTICAL ✅"
```

**Detailed Prompt:** See `migration_prompt_3.md`

---

## Phase 4: Refactor Optimizer for Multi-Strategy

**Goal:** Update `optimizer_engine.py` to work with any strategy via registry.

**Duration:** 2-3 days

**Why:** Enable optimization for multiple strategies.

### Tasks

- [ ] Update `OptimizationConfig` dataclass in `optimizer_engine.py`:
  ```python
  @dataclass
  class OptimizationConfig:
      csv_file: IO[Any]
      strategy_id: str = "s01_trailing_ma"  # ⭐ NEW field
      enabled_params: Dict[str, bool]
      param_ranges: Dict[str, Tuple]
      # ... rest unchanged
  ```

- [ ] Refactor `run_optimization()`:
  ```python
  def run_optimization(config: OptimizationConfig) -> List[OptimizationResult]:
      # 1. Get strategy class from registry
      strategy_class = StrategyRegistry.get_strategy_class(config.strategy_id)

      # 2. Get parameter definitions from strategy
      param_defs = strategy_class.get_param_definitions()

      # 3. Generate parameter grid using strategy's definitions
      combinations = _generate_parameter_grid(config, param_defs)

      # 4. Get cache requirements from strategy
      cache_req = strategy_class.get_cache_requirements(combinations)

      # 5. Initialize worker pool with cache
      pool = mp.Pool(
          processes=config.worker_processes,
          initializer=_init_worker,
          initargs=(df, cache_req, strategy_class, config)
      )

      # 6. Run simulations
      results = []
      for batch in batches:
          batch_results = pool.map(_simulate_combination, batch)
          results.extend(batch_results)

      return results
  ```

- [ ] Refactor `_init_worker()`:
  ```python
  def _init_worker(df, cache_req, strategy_class, config):
      global _df, _strategy_class, _cached_data, _config

      _df = df
      _strategy_class = strategy_class
      _config = config

      # Pre-compute indicators based on strategy's cache requirements
      _cached_data = {}

      if 'ma_types_and_lengths' in cache_req:
          _cached_data['ma_cache'] = {}
          for ma_type, length in cache_req['ma_types_and_lengths']:
              from indicators import get_ma
              ma_values = get_ma(df['Close'], ma_type, length, ...).to_numpy()
              _cached_data['ma_cache'][(ma_type, length)] = ma_values

      if 'atr_periods' in cache_req:
          _cached_data['atr'] = {}
          for period in cache_req['atr_periods']:
              from indicators import atr
              atr_values = atr(df['High'], df['Low'], df['Close'], period).to_numpy()
              _cached_data['atr'][period] = atr_values

      if 'long_lp_values' in cache_req:
          _cached_data['lowest'] = {}
          for lp in cache_req['long_lp_values']:
              _cached_data['lowest'][lp] = df['Low'].rolling(window=lp).min().to_numpy()

      if 'short_lp_values' in cache_req:
          _cached_data['highest'] = {}
          for lp in cache_req['short_lp_values']:
              _cached_data['highest'][lp] = df['High'].rolling(window=lp).max().to_numpy()
  ```

- [ ] Refactor `_simulate_combination()`:
  ```python
  def _simulate_combination(params_dict: Dict[str, Any]) -> OptimizationResult:
      global _df, _strategy_class, _cached_data, _config

      # Create strategy instance
      strategy = _strategy_class(params_dict)

      # Run simulation with cached data
      result = strategy.simulate(_df, cached_data=_cached_data)

      # Convert to OptimizationResult
      return OptimizationResult(
          net_profit_pct=result['net_profit_pct'],
          max_drawdown_pct=result['max_drawdown_pct'],
          total_trades=result['total_trades'],
          # ... map all fields
          **params_dict  # Include parameters
      )
  ```

- [ ] Remove hardcoded `PARAMETER_MAP`:
  - [ ] Delete global `PARAMETER_MAP` constant
  - [ ] Generate mapping dynamically from `strategy.get_param_definitions()`

- [ ] Update `_generate_parameter_grid()`:
  ```python
  def _generate_parameter_grid(config, param_defs):
      combinations = []

      # Build ranges for enabled parameters
      enabled_params = {}
      for param_name, param_def in param_defs.items():
          # param_name is already in camelCase (e.g., 'maLength')

          if config.enabled_params.get(param_name):
              # This parameter varies
              start, stop, step = config.param_ranges[param_name]
              values = _generate_numeric_sequence(start, stop, step, param_def['type'] == 'int')
              enabled_params[param_name] = values
          else:
              # Fixed parameter
              enabled_params[param_name] = [config.fixed_params[param_name]]

      # Generate Cartesian product
      # ... existing grid generation logic

      return combinations
  ```

- [ ] Update `export_to_csv()`:
  - [ ] Get column specs from strategy dynamically
  - [ ] Include strategy_id in header

### Testing

- [ ] **Unit Test:** Test parameter grid generation
  ```python
  from optimizer_engine import _generate_parameter_grid
  from strategies.s01_trailing_ma import S01TrailingMA

  config = OptimizationConfig(
      strategy_id='s01_trailing_ma',
      enabled_params={'maLength': True},
      param_ranges={'maLength': (30, 60, 15)},
      # ...
  )

  param_defs = S01TrailingMA.get_param_definitions()
  combos = _generate_parameter_grid(config, param_defs)

  assert len(combos) == 3  # 30, 45, 60
  ```

- [ ] **Integration Test:** Run small optimization
  ```bash
  python -c "
  from optimizer_engine import run_optimization, OptimizationConfig

  config = OptimizationConfig(
      csv_file=open('../data/test.csv'),
      strategy_id='s01_trailing_ma',
      enabled_params={'maLength': True},
      param_ranges={'maLength': (30, 60, 15)},
      # ... minimal config
  )

  results = run_optimization(config)
  print(f'Tested {len(results)} combinations')
  "
  ```

- [ ] **Reference Test:** Full S_01 optimization
  - [ ] Run optimization with known parameter grid
  - [ ] Verify best result matches previous best
  - [ ] Verify number of combinations correct
  - [ ] Verify CSV export format

### Commit

```bash
git add src/optimizer_engine.py
git commit -m "Phase 4: Refactor optimizer for multi-strategy

- Add strategy_id to OptimizationConfig
- Use StrategyRegistry to get strategy class
- Generate parameter grid from strategy.get_param_definitions()
- Pre-compute cache from strategy.get_cache_requirements()
- Reference test: S_01 optimization works ✅"
```

**Detailed Prompt:** See `migration_prompt_4.md`

---

## Phase 5: Add S_03 Reversal Strategy

**Goal:** Implement S_03 Reversal strategy by translating Pine script.

**Duration:** 2-3 days

**Why:** Validate multi-strategy architecture with second strategy.

### Tasks

- [ ] Read and analyze Pine script:
  - [ ] Read `/data/S_03 Reversal_v07 Light for PROJECT PLAN.pine`
  - [ ] Identify all parameters
  - [ ] Identify entry conditions
  - [ ] Identify exit conditions
  - [ ] Note: NO stops/targets, only reversal

- [ ] Create `/src/strategies/s03_reversal.py`

- [ ] Define `S03Reversal(BaseStrategy)` class:
  - [ ] Set class attributes:
    ```python
    STRATEGY_ID = "s03_reversal"
    STRATEGY_NAME = "S_03 Reversal v07 Light"
    VERSION = "07"
    ```

- [ ] Implement `get_param_definitions()`:
  - [ ] MA1: use_ma1, ma1_type, ma1_length
  - [ ] MA2: use_ma2, ma2_type, ma2_length
  - [ ] MA3: use_ma3, ma3_type, ma3_length
  - [ ] Close count: use_close_count, close_count_long, close_count_short
  - [ ] Breakout: breakout_mode, use_close_price
  - [ ] Contract size
  - [ ] All with correct types, defaults (camelCase keys)

- [ ] Implement `__init__`:
  - [ ] Initialize caches: `_ma1`, `_ma2`, `_ma3`
  - [ ] Initialize state: `counter_close_long`, `counter_close_short`

- [ ] Implement `_prepare_data`:
  ```python
  def _prepare_data(self, df, cached_data):
      if cached_data:
          # Use cache
          self._ma1 = cached_data['ma_specs'][(self.params['ma1_type'], self.params['ma1_length'])]
          self._ma2 = cached_data['ma_specs'][(self.params['ma2_type'], self.params['ma2_length'])]
          self._ma3 = cached_data['ma_specs'][(self.params['ma3_type'], self.params['ma3_length'])]
      else:
          # Compute on-the-fly
          from indicators import get_ma
          self._ma1 = get_ma(df['Close'], self.params['ma1_type'], self.params['ma1_length']).to_numpy()
          self._ma2 = get_ma(df['Close'], self.params['ma2_type'], self.params['ma2_length']).to_numpy()
          self._ma3 = get_ma(df['Close'], self.params['ma3_type'], self.params['ma3_length']).to_numpy()
  ```

- [ ] Implement `should_long`:
  - [ ] Update close count: if close > ma3: increment counter_close_long, reset counter_close_short
  - [ ] Check close count condition (if enabled)
  - [ ] Check MA confirmation (if enabled): ma1 > ma2, close > ma3
  - [ ] Return True if all conditions met

- [ ] Implement `should_short`:
  - [ ] Similar to should_long but opposite direction

- [ ] Implement `calculate_entry`:
  ```python
  def calculate_entry(self, idx, direction):
      # Reversal strategy has no stops/targets
      entry_price = self.df['Close'].iloc[idx]
      return (entry_price, math.nan, math.nan)
  ```

- [ ] Implement `calculate_position_size`:
  ```python
  def calculate_position_size(self, idx, direction, entry_price, stop_price, equity):
      # 100% of equity
      contract_size = self.params['contract_size']
      qty = equity / entry_price

      if contract_size > 0:
          qty = math.floor(qty / contract_size) * contract_size

      return qty
  ```

- [ ] Implement `should_exit`:
  ```python
  def should_exit(self, idx, position_info):
      # Reversal strategy: exit only on reverse signal (handled in _run_simulation)
      # No stop/target exits
      return (False, None, "")
  ```

- [ ] Override `allows_reversal`:
  ```python
  def allows_reversal(self):
      return True  # This is a reversal strategy
  ```

- [ ] Implement `get_cache_requirements`:
  ```python
  @classmethod
  def get_cache_requirements(cls, param_combinations):
      ma_specs = set()

      for combo in param_combinations:
          if combo.get('use_ma1', True):
              ma_specs.add((combo['ma1_type'], combo['ma1_length']))
          if combo.get('use_ma2', True):
              ma_specs.add((combo['ma2_type'], combo['ma2_length']))
          if combo.get('use_ma3', True):
              ma_specs.add((combo['ma3_type'], combo['ma3_length']))

      return {'ma_specs': list(ma_specs)}
  ```

- [ ] Register S_03 in `strategy_registry.py`:
  ```python
  from strategies.s03_reversal import S03Reversal

  class StrategyRegistry:
      _strategies = {
          "s01_trailing_ma": S01TrailingMA,
          "s03_reversal": S03Reversal,  # ⭐ NEW
      }
  ```

- [ ] Update `BaseStrategy._run_simulation()` to handle reversal:
  ```python
  # In main loop, add reversal logic:
  if self.allows_reversal() and position != 0:
      if position > 0 and self.should_short(idx):
          # Close long
          # ... close logic
          # Open short immediately
          entry, stop, target = self.calculate_entry(idx, 'short')
          size = self.calculate_position_size(idx, 'short', entry, stop, equity)
          # ... open short logic

      elif position < 0 and self.should_long(idx):
          # Close short
          # ... close logic
          # Open long immediately
          entry, stop, target = self.calculate_entry(idx, 'long')
          size = self.calculate_position_size(idx, 'long', entry, stop, equity)
          # ... open long logic
  ```

### Testing

- [ ] **Manual Test:** Create test with known data
  ```python
  from strategies.s03_reversal import S03Reversal
  import pandas as pd

  # Create simple trending data
  df = pd.DataFrame({
      'Close': [100, 102, 104, 106, 108, 107, 105, 103],  # Up then down
      # ... other columns
  })

  strategy = S03Reversal({
      'ma1_type': 'SMA', 'ma1_length': 2,
      'ma2_type': 'SMA', 'ma2_length': 3,
      'ma3_type': 'SMA', 'ma3_length': 2,
      'use_close_count': False,
      'contract_size': 0.01,
  })

  result = strategy.simulate(df)

  # Should have trades due to trend changes
  print(f"Trades: {result['total_trades']}")
  print(f"Profit: {result['net_profit_pct']:.2f}%")
  ```

- [ ] **Reference Test:** Run S_03 with default params
  - [ ] Record baseline for S_03: Net Profit: `_____`, Max DD: `_____`, Trades: `_____`
  - [ ] Save to `tests.md`

- [ ] **Verify Reversal Logic:**
  - [ ] Position should ALWAYS be non-zero (long or short)
  - [ ] Should reverse from long → short when short signal appears
  - [ ] Should reverse from short → long when long signal appears
  - [ ] No gaps (flat periods) in position

- [ ] **Integration Test:** Run optimization for S_03
  ```bash
  # Small grid to verify it works
  python -c "
  from optimizer_engine import run_optimization, OptimizationConfig

  config = OptimizationConfig(
      csv_file=open('../data/test.csv'),
      strategy_id='s03_reversal',
      enabled_params={'ma1Length': True},
      param_ranges={'ma1Length': (10, 20, 5)},
      # ... minimal config
  )

  results = run_optimization(config)
  print(f'S_03 optimization: {len(results)} results')
  "
  ```

### Commit

```bash
git add src/strategies/s03_reversal.py src/strategy_registry.py src/strategies/base_strategy.py
git commit -m "Phase 5: Add S_03 Reversal strategy

- Translate S_03 Pine script to Python
- Implement reversal logic in BaseStrategy._run_simulation
- Register S_03 in StrategyRegistry
- Reference test: S_03 baseline established ✅"
```

**Detailed Prompt:** See `migration_prompt_5.md`

---

## Phase 6: Update API and UI

**Goal:** Add strategy selector to UI and update API endpoints.

**Duration:** 1-2 days

**Why:** Make multi-strategy accessible to users.

### Tasks

#### Backend (server.py)

- [ ] Add `/api/strategies` endpoint:
  ```python
  @app.route('/api/strategies', methods=['GET'])
  def list_strategies():
      from strategy_registry import StrategyRegistry
      strategies = StrategyRegistry.list_strategies()
      return jsonify({'strategies': strategies})
  ```

- [ ] Update `/api/backtest` endpoint:
  - [ ] Accept `strategy_id` parameter (default 's01_trailing_ma')
  - [ ] Get strategy from registry
  - [ ] Call `strategy.simulate()`

- [ ] Update `/api/optimize` endpoint:
  - [ ] Accept `strategy_id` parameter
  - [ ] Pass to `OptimizationConfig`

- [ ] Update `/api/walkforward` endpoint:
  - [ ] Accept `strategy_id` parameter
  - [ ] Pass to `WalkForwardEngine`

- [ ] Test all endpoints with curl:
  ```bash
  # List strategies
  curl http://localhost:8000/api/strategies

  # Backtest with S_01
  curl -X POST -F "strategy_id=s01_trailing_ma" -F "csv=@data/test.csv" http://localhost:8000/api/backtest

  # Backtest with S_03
  curl -X POST -F "strategy_id=s03_reversal" -F "csv=@data/test.csv" http://localhost:8000/api/backtest
  ```

#### Frontend (index.html)

- [ ] Add strategy selector dropdown:
  ```html
  <div class="strategy-selector">
      <label>Strategy:</label>
      <select id="strategySelector" onchange="onStrategyChange()">
          <option value="s01_trailing_ma">S_01 TrailingMA v26</option>
          <option value="s03_reversal">S_03 Reversal v07</option>
      </select>
  </div>
  ```

- [ ] Create parameter blocks for each strategy:
  ```html
  <!-- S_01 Parameters (existing UI) -->
  <div id="params_s01_trailing_ma" class="strategy-params">
      <!-- Current S_01 parameter form -->
  </div>

  <!-- S_03 Parameters (new UI) -->
  <div id="params_s03_reversal" class="strategy-params" style="display: none;">
      <div class="param-group">
          <h3>Moving Averages</h3>

          <label>MA1 Type:</label>
          <select name="ma1Type">
              <option value="SMA">SMA</option>
              <option value="EMA">EMA</option>
              <!-- ... all MA types -->
          </select>

          <label>MA1 Length:</label>
          <input type="number" name="ma1Length" value="15" min="1" max="500">

          <!-- Similar for MA2, MA3 -->
      </div>

      <div class="param-group">
          <h3>Close Count Filter</h3>

          <label>Use Close Count:</label>
          <input type="checkbox" name="useCloseCount">

          <label>Close Count Long:</label>
          <input type="number" name="closeCountLong" value="3" min="1" max="20">

          <label>Close Count Short:</label>
          <input type="number" name="closeCountShort" value="3" min="1" max="20">
      </div>

      <div class="param-group">
          <h3>Other Settings</h3>

          <label>Breakout Mode:</label>
          <input type="checkbox" name="breakoutMode">

          <label>Contract Size:</label>
          <input type="number" name="contractSize" value="0.01" step="0.01">
      </div>
  </div>
  ```

- [ ] Add JavaScript for strategy switching:
  ```javascript
  function onStrategyChange() {
      const strategyId = document.getElementById('strategySelector').value;

      // Hide all parameter blocks
      document.querySelectorAll('.strategy-params').forEach(el => {
          el.style.display = 'none';
      });

      // Show selected strategy's parameters
      document.getElementById(`params_${strategyId}`).style.display = 'block';
  }

  function collectParameters() {
      const strategyId = document.getElementById('strategySelector').value;
      const paramContainer = document.getElementById(`params_${strategyId}`);

      // Collect all inputs from visible parameter block
      const params = {};
      paramContainer.querySelectorAll('input, select').forEach(input => {
          if (input.type === 'checkbox') {
              params[input.name] = input.checked;
          } else if (input.type === 'number') {
              params[input.name] = parseFloat(input.value);
          } else {
              params[input.name] = input.value;
          }
      });

      return params;
  }

  function runOptimization() {
      const strategyId = document.getElementById('strategySelector').value;
      const params = collectParameters();

      const formData = new FormData();
      formData.append('strategy_id', strategyId);
      formData.append('params', JSON.stringify(params));
      // ... append other data

      fetch('/api/optimize', {
          method: 'POST',
          body: formData
      })
      .then(response => response.blob())
      .then(blob => {
          // Download CSV
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `${strategyId}_optimization_results.csv`;
          a.click();
      });
  }
  ```

- [ ] Update CSS for clean layout:
  ```css
  .strategy-params {
      border: 1px solid #ddd;
      padding: 20px;
      margin: 10px 0;
      border-radius: 5px;
  }

  .param-group {
      margin-bottom: 20px;
      padding-bottom: 20px;
      border-bottom: 1px solid #eee;
  }

  .param-group:last-child {
      border-bottom: none;
  }
  ```

#### CLI (run_backtest.py)

- [ ] Add `--strategy` argument:
  ```python
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--csv', required=True)
  parser.add_argument('--strategy', default='s01_trailing_ma',
                      choices=['s01_trailing_ma', 's03_reversal'])
  args = parser.parse_args()

  from strategy_registry import StrategyRegistry

  strategy_class = StrategyRegistry.get_strategy_class(args.strategy)
  # Load default params for strategy
  params = strategy_class.get_param_definitions()
  default_params = {k: v['default'] for k, v in params.items()}

  strategy = strategy_class(default_params)
  result = strategy.simulate(df)

  print(f"Strategy: {strategy.STRATEGY_NAME}")
  print(f"Net Profit: {result['net_profit_pct']:.2f}%")
  # ... etc
  ```

### Testing

- [ ] **UI Test:** Manual testing in browser
  - [ ] Start server: `python src/server.py`
  - [ ] Open http://localhost:8000
  - [ ] Select S_01 → verify S_01 parameters shown
  - [ ] Select S_03 → verify S_03 parameters shown
  - [ ] Run backtest with S_01 → verify results
  - [ ] Run backtest with S_03 → verify results
  - [ ] Run optimization with S_01 → verify CSV download
  - [ ] Run optimization with S_03 → verify CSV download

- [ ] **CLI Test:**
  ```bash
  # S_01
  python src/run_backtest.py --csv data/OKX_LINKUSDT.P,\ 15...csv --strategy s01_trailing_ma

  # S_03
  python src/run_backtest.py --csv data/OKX_LINKUSDT.P,\ 15...csv --strategy s03_reversal
  ```

- [ ] **API Test:** Use Postman or curl
  - [ ] GET /api/strategies → returns both strategies
  - [ ] POST /api/backtest with strategy_id=s01_trailing_ma → works
  - [ ] POST /api/backtest with strategy_id=s03_reversal → works
  - [ ] POST /api/optimize with strategy_id=s01_trailing_ma → works
  - [ ] POST /api/optimize with strategy_id=s03_reversal → works

### Commit

```bash
git add src/server.py src/index.html src/run_backtest.py
git commit -m "Phase 6: Update API and UI for multi-strategy

- Add /api/strategies endpoint
- Update all endpoints to accept strategy_id
- Add strategy selector dropdown in UI
- Add S_03 parameter form in UI
- Update CLI with --strategy argument
- UI test: Both strategies work ✅"
```

**Detailed Prompt:** See `migration_prompt_6.md`

---

## Post-Migration Tasks

### Cleanup

- [ ] Remove deprecated code:
  - [ ] Old `run_strategy()` implementation (keep wrapper for compatibility)
  - [ ] Commented-out code blocks
  - [ ] Unused imports

- [ ] Update documentation:
  - [ ] Update `README.md` with multi-strategy info
  - [ ] Update `CLAUDE.md` with new architecture
  - [ ] Add docstrings to all new classes/methods

- [ ] Code formatting:
  ```bash
  # If using black
  black src/
  ```

### Final Testing

- [ ] **Full System Test:**
  - [ ] S_01 single backtest ✅
  - [ ] S_03 single backtest ✅
  - [ ] S_01 grid optimization ✅
  - [ ] S_03 grid optimization ✅
  - [ ] S_01 Optuna optimization ✅
  - [ ] S_03 Optuna optimization ✅
  - [ ] S_01 WFA ✅
  - [ ] S_03 WFA ✅
  - [ ] CSV export/import ✅
  - [ ] Preset save/load (both strategies) ✅

- [ ] **Performance Test:**
  - [ ] Run 1000-combo optimization for S_01
  - [ ] Measure time, compare to baseline
  - [ ] Verify caching working (should be similar speed)

- [ ] **Reference Tests:**
  - [ ] S_01 reference test matches baseline ✅
  - [ ] S_03 reference test matches baseline ✅

### Merge to Main

- [ ] Create pull request:
  ```bash
  git push origin migration/multi-strategy
  # Create PR on GitHub
  ```

- [ ] Review checklist:
  - [ ] All tests passing
  - [ ] Reference tests match
  - [ ] Documentation updated
  - [ ] No breaking changes for existing users

- [ ] Merge:
  ```bash
  git checkout main
  git merge migration/multi-strategy
  git tag v2.0.0-multi-strategy
  git push origin main --tags
  ```

---

## Rollback Procedures

### If Phase Fails

1. **Identify which phase failed**
2. **Check last successful commit:**
   ```bash
   git log --oneline
   ```
3. **Rollback to last working phase:**
   ```bash
   git reset --hard <commit-hash-of-last-phase>
   ```
4. **Debug the issue offline**
5. **Restart failed phase**

### If Reference Test Fails

1. **Do NOT proceed to next phase**
2. **Compare outputs:**
   ```bash
   # Run old version (from tag)
   git checkout pre-migration-backup
   python src/run_backtest.py --csv data/test.csv > old_output.txt

   # Run new version
   git checkout migration/multi-strategy
   python src/run_backtest.py --csv data/test.csv > new_output.txt

   # Compare
   diff old_output.txt new_output.txt
   ```
3. **Debug differences** (see debugging checklist in Phase 3)
4. **Fix and re-test**

---

## Success Criteria

Migration is complete when:

- ✅ All 6 phases completed
- ✅ All reference tests pass (S_01 results unchanged)
- ✅ S_03 strategy works and tested
- ✅ UI allows switching between strategies
- ✅ API accepts strategy_id parameter
- ✅ CLI supports --strategy argument
- ✅ Optimization works for both strategies
- ✅ WFA works for both strategies
- ✅ Documentation updated
- ✅ Code merged to main branch

---

## Time Tracking

| Phase | Estimated | Actual | Notes |
|-------|-----------|--------|-------|
| Phase 1: Indicators | 0.5-1 day | ______ | ______ |
| Phase 2: Base Strategy | 1-1.5 days | ______ | ______ |
| Phase 3: Extract S_01 | 2-3 days | ______ | ______ |
| Phase 4: Refactor Optimizer | 2-3 days | ______ | ______ |
| Phase 5: Add S_03 | 2-3 days | ______ | ______ |
| Phase 6: Update UI/API | 1-2 days | ______ | ______ |
| **Total** | **9-14 days** | ______ | ______ |

---

## Notes and Issues Log

Use this section to track issues encountered during migration:

**Phase 1:**
- Issue: ___________________________
- Solution: _________________________

**Phase 2:**
- Issue: ___________________________
- Solution: _________________________

(Continue for all phases)

---

## Contact / Questions

If you encounter issues during migration:
1. Check detailed prompts in `migration_prompt_N.md`
2. Review `agents.md` for coding guidelines
3. Consult `project_structure.md` for architecture details
4. Check reference tests in `tests.md`
