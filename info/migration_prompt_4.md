# Migration Prompt 4: Refactor Optimizer for Multi-Strategy

## Context

**Current Phase:** Phase 4 of 6
**Previous Phases:**
- Phase 1 ✅ (indicators.py extracted)
- Phase 2 ✅ (BaseStrategy created)
- Phase 3 ✅ (S_01 extracted, reference test passed)

**Migration Checklist:** See `migration_checklist.md` - Phase 4
**Duration:** 2-3 days

**Goal:** Make optimizer_engine.py work with ANY strategy through StrategyRegistry. Remove hardcoded S_01-specific code.

---

## Current State

`optimizer_engine.py` currently has S_01-specific code:
1. **Hardcoded PARAMETER_MAP** - maps frontend names to Python names for S_01 only
2. **Fixed caching logic** - assumes MA/ATR/lowest/highest for S_01
3. **_simulate_combination()** - duplicates S_01 logic from backtest_engine

These need to become dynamic and strategy-agnostic.

---

## Task Overview

Refactor optimizer to:
1. Accept `strategy_id` parameter
2. Get strategy class from StrategyRegistry
3. Generate parameter grid from strategy's `get_param_definitions()`
4. Pre-compute cache based on strategy's `get_cache_requirements()`
5. Call `strategy.simulate()` instead of duplicating logic

---

## Detailed Implementation Steps

### Step 1: Update OptimizationConfig Dataclass

In `optimizer_engine.py`, modify `OptimizationConfig`:

```python
@dataclass
class OptimizationConfig:
    """Configuration for optimization run"""

    csv_file: IO[Any]
    strategy_id: str = "s01_trailing_ma"  # ⭐ NEW: Which strategy to optimize

    # Parameter control
    enabled_params: Dict[str, bool]  # Which params vary (frontend names)
    param_ranges: Dict[str, Tuple]   # Ranges for varying params
    fixed_params: Dict[str, Any]     # Values for fixed params

    # Optimization settings
    risk_per_trade_pct: float = 2.0
    contract_size: float = 0.01
    commission_rate: float = 0.0004
    atr_period: int = 14

    # Worker settings
    worker_processes: int = 6

    # Filtering
    filter_min_profit: bool = False
    min_profit_threshold: float = 0.0

    # Scoring
    score_config: Optional[Dict[str, Any]] = None

    # Output
    export_trades: bool = False
```

**Key change:** Added `strategy_id` field with default value for backward compatibility.

---

### Step 2: Refactor run_optimization() Main Function

Replace the beginning of `run_optimization()`:

```python
def run_optimization(config: OptimizationConfig) -> List[OptimizationResult]:
    """
    Run parameter optimization for specified strategy.

    Args:
        config: Optimization configuration including strategy_id

    Returns:
        List of OptimizationResult, one per parameter combination tested
    """

    # ════════════════════════════════════════════════════════════════
    # 1. GET STRATEGY CLASS FROM REGISTRY
    # ════════════════════════════════════════════════════════════════

    from strategy_registry import StrategyRegistry

    logger.info(f"Optimizing strategy: {config.strategy_id}")

    try:
        strategy_class = StrategyRegistry.get_strategy_class(config.strategy_id)
    except ValueError as e:
        raise ValueError(f"Cannot optimize unknown strategy: {e}")

    logger.info(f"Strategy: {strategy_class.STRATEGY_NAME} v{strategy_class.VERSION}")

    # ════════════════════════════════════════════════════════════════
    # 2. GET PARAMETER DEFINITIONS FROM STRATEGY
    # ════════════════════════════════════════════════════════════════

    param_definitions = strategy_class.get_param_definitions()

    logger.info(f"Strategy has {len(param_definitions)} parameters")

    # ════════════════════════════════════════════════════════════════
    # 3. GENERATE PARAMETER GRID
    # ════════════════════════════════════════════════════════════════

    combinations = _generate_parameter_grid(config, param_definitions)

    total_combinations = len(combinations)
    logger.info(f"Generated {total_combinations} parameter combinations")

    if total_combinations == 0:
        raise ValueError("No parameter combinations generated. Check enabled_params and param_ranges.")

    # ════════════════════════════════════════════════════════════════
    # 4. LOAD DATA
    # ════════════════════════════════════════════════════════════════

    df = load_data(config.csv_file)
    logger.info(f"Loaded data: {len(df)} bars")

    # ════════════════════════════════════════════════════════════════
    # 5. GET CACHE REQUIREMENTS FROM STRATEGY
    # ════════════════════════════════════════════════════════════════

    cache_requirements = strategy_class.get_cache_requirements(combinations)

    logger.info(f"Cache requirements: {list(cache_requirements.keys())}")

    # ════════════════════════════════════════════════════════════════
    # 6. INITIALIZE WORKER POOL WITH CACHE
    # ════════════════════════════════════════════════════════════════

    processes = min(32, max(1, int(config.worker_processes)))
    logger.info(f"Starting worker pool with {processes} processes")

    pool = mp.Pool(
        processes=processes,
        initializer=_init_worker,
        initargs=(df, cache_requirements, strategy_class, config)
    )

    # ════════════════════════════════════════════════════════════════
    # 7. RUN OPTIMIZATIONS
    # ════════════════════════════════════════════════════════════════

    # ... rest of existing optimization logic (batching, progress bar, etc.)
    # ... (keep existing code for running simulations)

    return results
```

---

### Step 3: Refactor _generate_parameter_grid()

Replace the function to work with dynamic parameter definitions:

```python
def _generate_parameter_grid(
    config: OptimizationConfig,
    param_definitions: Dict[str, Dict]
) -> List[Dict[str, Any]]:
    """
    Generate all parameter combinations from enabled ranges.

    Args:
        config: Optimization config with enabled_params and param_ranges
        param_definitions: Parameter definitions from strategy.get_param_definitions()

    Returns:
        List of dicts, each containing one parameter combination
    """

    # Build grid for each parameter
    grid_params = {}

    for param_name, param_def in param_definitions.items():
        # param_name is already in camelCase (e.g., 'maLength')

        # Check if this parameter varies
        if config.enabled_params.get(param_name, False):
            # This parameter varies - get range
            if param_name not in config.param_ranges:
                raise ValueError(f"Parameter '{param_name}' is enabled but has no range specified")

            start, stop, step = config.param_ranges[param_name]

            # Generate values based on type
            param_type = param_def['type']

            if param_type == 'int':
                values = _generate_numeric_sequence(start, stop, step, is_int=True)
                values = [int(round(v)) for v in values]

            elif param_type == 'float':
                values = _generate_numeric_sequence(start, stop, step, is_int=False)
                values = [float(v) for v in values]

            elif param_type == 'categorical':
                # For categorical, treat as list of choices
                # User should provide list in config.param_ranges[param_name]
                if isinstance(config.param_ranges[param_name], (list, tuple)):
                    values = list(config.param_ranges[param_name])
                else:
                    # Fall back to all choices if range not specified properly
                    values = param_def.get('choices', [param_def['default']])

            elif param_type == 'bool':
                # Boolean: if enabled, try both True and False
                values = [True, False]

            else:
                raise ValueError(f"Unsupported parameter type: {param_type}")

            grid_params[param_name] = values

        else:
            # This parameter is fixed - use value from config.fixed_params
            if param_name in config.fixed_params:
                value = config.fixed_params[param_name]
            else:
                value = param_def['default']

            grid_params[param_name] = [value]

    # Generate Cartesian product
    param_names = list(grid_params.keys())
    param_values_lists = [grid_params[name] for name in param_names]

    import itertools
    combinations = []

    for values_tuple in itertools.product(*param_values_lists):
        combo = dict(zip(param_names, values_tuple))
        combinations.append(combo)

    return combinations
```

---

### Step 4: Refactor _init_worker()

Make worker initialization dynamic based on cache requirements:

```python
# Global variables for worker processes
_df: Optional[pd.DataFrame] = None
_strategy_class: Optional[type] = None
_cached_data: Dict[str, Any] = {}
_config: Optional[OptimizationConfig] = None


def _init_worker(
    df: pd.DataFrame,
    cache_requirements: Dict[str, Any],
    strategy_class: type,
    config: OptimizationConfig
) -> None:
    """
    Initialize worker process with data and pre-computed indicators.

    Args:
        df: Market data DataFrame
        cache_requirements: Dict from strategy.get_cache_requirements()
        strategy_class: Strategy class to instantiate
        config: Optimization config
    """
    global _df, _strategy_class, _cached_data, _config

    _df = df
    _strategy_class = strategy_class
    _config = config
    _cached_data = {}

    logger.info(f"Initializing worker for {strategy_class.STRATEGY_ID}")

    # ════════════════════════════════════════════════════════════════
    # PRE-COMPUTE INDICATORS BASED ON CACHE REQUIREMENTS
    # ════════════════════════════════════════════════════════════════

    # Import indicators
    from indicators import get_ma, atr

    # 1. MA Types and Lengths
    if 'ma_types_and_lengths' in cache_requirements:
        _cached_data['ma_cache'] = {}

        ma_types_and_lengths = cache_requirements['ma_types_and_lengths']
        logger.info(f"Pre-computing {len(ma_types_and_lengths)} MA types/lengths")

        for ma_type, length in ma_types_and_lengths:
            # Compute MA
            ma_values = get_ma(
                df['Close'],
                ma_type,
                int(length),
                df['Volume'],
                df['High'],
                df['Low']
            ).to_numpy()

            _cached_data['ma_cache'][(ma_type, length)] = ma_values
            logger.debug(f"  Cached MA: {ma_type}({length})")

    # 2. ATR Values
    if 'atr_periods' in cache_requirements:
        _cached_data['atr'] = {}

        atr_periods = cache_requirements['atr_periods']
        logger.info(f"Pre-computing ATR for {len(atr_periods)} periods")

        for period in atr_periods:
            atr_values = atr(
                df['High'],
                df['Low'],
                df['Close'],
                int(period)
            ).to_numpy()

            _cached_data['atr'][period] = atr_values
            logger.debug(f"  Cached ATR: period={period}")

    # 3. Lowest Lows (for long stops with lookback)
    if 'long_lp_values' in cache_requirements:
        _cached_data['lowest'] = {}

        lp_values = cache_requirements['long_lp_values']
        logger.info(f"Pre-computing lowest lows for {len(lp_values)} lookback periods")

        for lp in lp_values:
            if lp > 0:
                lowest_values = df['Low'].rolling(window=int(lp)).min().to_numpy()
                _cached_data['lowest'][lp] = lowest_values
                logger.debug(f"  Cached lowest: lookback={lp}")

    # 4. Highest Highs (for short stops with lookback)
    if 'short_lp_values' in cache_requirements:
        _cached_data['highest'] = {}

        lp_values = cache_requirements['short_lp_values']
        logger.info(f"Pre-computing highest highs for {len(lp_values)} lookback periods")

        for lp in lp_values:
            if lp > 0:
                highest_values = df['High'].rolling(window=int(lp)).max().to_numpy()
                _cached_data['highest'][lp] = highest_values
                logger.debug(f"  Cached highest: lookback={lp}")

    # 5. Custom Cache Requirements
    # Future strategies might have custom requirements
    # They can define additional keys in get_cache_requirements()
    # and access them via cached_data in _prepare_data()

    logger.info("Worker initialization complete")
```

---

### Step 5: Refactor _simulate_combination()

Replace duplicated simulation logic with call to strategy.simulate():

```python
def _simulate_combination(params_dict: Dict[str, Any]) -> OptimizationResult:
    """
    Simulate one parameter combination.

    This function runs in worker process and has access to:
    - _df: Market data
    - _strategy_class: Strategy class
    - _cached_data: Pre-computed indicators
    - _config: Optimization config

    Args:
        params_dict: Parameter values for this combination

    Returns:
        OptimizationResult with performance metrics and parameters
    """
    global _df, _strategy_class, _cached_data, _config

    # ════════════════════════════════════════════════════════════════
    # 1. CREATE STRATEGY INSTANCE
    # ════════════════════════════════════════════════════════════════

    try:
        strategy = _strategy_class(params_dict)
    except Exception as e:
        # Invalid parameters - return zero result
        return OptimizationResult(
            net_profit_pct=0.0,
            max_drawdown_pct=100.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            sharpe_ratio=0.0,
            profit_factor=0.0,
            romad=0.0,
            recovery_factor=0.0,
            ulcer_index=0.0,
            consistency=0.0,
            score=0.0,
            **params_dict
        )

    # ════════════════════════════════════════════════════════════════
    # 2. RUN SIMULATION WITH CACHED DATA
    # ════════════════════════════════════════════════════════════════

    try:
        result = strategy.simulate(_df, cached_data=_cached_data)
    except Exception as e:
        # Simulation error - return zero result
        logger.warning(f"Simulation error: {e}")
        return OptimizationResult(
            net_profit_pct=0.0,
            max_drawdown_pct=100.0,
            total_trades=0,
            **params_dict
        )

    # ════════════════════════════════════════════════════════════════
    # 3. CALCULATE ADDITIONAL METRICS (if needed)
    # ════════════════════════════════════════════════════════════════

    # RoMaD
    if result['max_drawdown_pct'] > 0:
        romad = result['net_profit_pct'] / result['max_drawdown_pct']
    else:
        romad = 0.0

    # Recovery Factor
    if result['max_drawdown_pct'] > 0:
        recovery_factor = result['net_profit_pct'] / result['max_drawdown_pct']
    else:
        recovery_factor = 0.0

    # Ulcer Index (if not already in result)
    ulcer_index = result.get('ulcer_index', 0.0)

    # Consistency (if not already in result)
    consistency = result.get('consistency', 0.0)

    # ════════════════════════════════════════════════════════════════
    # 4. CREATE OPTIMIZATION RESULT
    # ════════════════════════════════════════════════════════════════

    optimization_result = OptimizationResult(
        # Performance metrics
        net_profit_pct=result['net_profit_pct'],
        max_drawdown_pct=result['max_drawdown_pct'],
        total_trades=result['total_trades'],
        winning_trades=result['winning_trades'],
        losing_trades=result['losing_trades'],
        sharpe_ratio=result['sharpe_ratio'],
        profit_factor=result['profit_factor'],

        # Risk metrics
        romad=romad,
        recovery_factor=recovery_factor,
        ulcer_index=ulcer_index,
        consistency=consistency,

        # Score (calculated later)
        score=0.0,

        # Parameters (include all for CSV export)
        **params_dict
    )

    return optimization_result
```

---

### Step 6: Remove Hardcoded PARAMETER_MAP

Delete the global `PARAMETER_MAP` constant:

```python
# DELETE THIS:
# PARAMETER_MAP = {
#     'maLength': ('ma_length', True),
#     'closeCountLong': ('close_count_long', True),
#     # ... etc
# }
```

Parameter mapping is now simple: `param_definitions` uses camelCase keys directly (e.g., 'maLength').

---

### Step 7: Update CSV Export

Modify `export_to_csv()` to use dynamic column specs:

```python
def export_to_csv(
    results: List[OptimizationResult],
    config: OptimizationConfig,
    strategy_class: type  # ⭐ NEW: Pass strategy class for dynamic columns
) -> io.BytesIO:
    """
    Export optimization results to CSV.

    Args:
        results: List of optimization results
        config: Optimization configuration
        strategy_class: Strategy class (for parameter definitions)

    Returns:
        BytesIO buffer with CSV data
    """

    # Get parameter definitions from strategy
    param_defs = strategy_class.get_param_definitions()

    # Build column specs dynamically
    csv_columns = []

    # Add result columns (fixed)
    csv_columns.extend([
        ('net_profit_pct', 'Net Profit %'),
        ('max_drawdown_pct', 'Max DD %'),
        ('total_trades', 'Trades'),
        ('winning_trades', 'Wins'),
        ('losing_trades', 'Losses'),
        ('sharpe_ratio', 'Sharpe'),
        ('profit_factor', 'PF'),
        ('romad', 'RoMaD'),
        ('score', 'Score')
    ])

    # Add parameter columns (dynamic based on strategy)
    for param_name, param_def in param_defs.items():
        # param_name is already in camelCase (e.g., 'maLength')
        display_name = param_def.get('description', param_name)

        # Add to columns (use param_name as is)
        csv_columns.append((param_name, display_name))

    # ... rest of CSV export logic using csv_columns
```

---

### Step 8: Update server.py to Pass strategy_id

In `server.py`, update `/api/optimize` endpoint:

```python
@app.route('/api/optimize', methods=['POST'])
def run_optimization_endpoint():
    """Run parameter optimization"""

    # Get strategy_id from request
    strategy_id = request.form.get('strategy_id', 's01_trailing_ma')

    # ... parse other form data ...

    # Create config with strategy_id
    optimization_config = OptimizationConfig(
        csv_file=data_source,
        strategy_id=strategy_id,  # ⭐ NEW
        enabled_params=enabled_params,
        param_ranges=param_ranges,
        fixed_params=fixed_params,
        # ... rest of config
    )

    # Run optimization
    results = run_optimization(optimization_config)

    # Export with strategy class for dynamic columns
    from strategy_registry import StrategyRegistry
    strategy_class = StrategyRegistry.get_strategy_class(strategy_id)

    csv_buffer = export_to_csv(results, optimization_config, strategy_class)

    # ... return CSV response
```

---

## Testing

### Test 1: Import Test

```bash
cd src
python -c "
from optimizer_engine import run_optimization, OptimizationConfig
from strategy_registry import StrategyRegistry

# Verify strategy_id field exists
config = OptimizationConfig(
    csv_file=None,
    strategy_id='s01_trailing_ma',
    enabled_params={},
    param_ranges={},
    fixed_params={}
)
print(f'✅ OptimizationConfig updated with strategy_id: {config.strategy_id}')
"
```

### Test 2: Small Optimization Test

```bash
cd src
python -c "
from optimizer_engine import run_optimization, OptimizationConfig
from backtest_engine import load_data

# Load small dataset
df = load_data(../data/"OKX_LINKUSDT.P, 15 2025.05.01-2025.11.20.csv")

# Create minimal config for S_01
config = OptimizationConfig(
    csv_file=df,
    strategy_id='s01_trailing_ma',
    enabled_params={'maLength': True},
    param_ranges={'maLength': (30, 60, 15)},  # 3 values: 30, 45, 60
    fixed_params={
        'maType': 'EMA',
        'closeCountLong': 7,
        # ... all other S_01 params at defaults
    },
    worker_processes=2
)

results = run_optimization(config)

print(f'✅ Optimization complete: {len(results)} results')
print(f'Best Net Profit: {max(r.net_profit_pct for r in results):.2f}%')
"
```

### Test 3: Reference Test

**Run full S_01 optimization with known parameters:**

```bash
cd src
python -c "
from optimizer_engine import run_optimization, OptimizationConfig

# Full optimization for S_01
config = OptimizationConfig(
    csv_file=open(../data/"OKX_LINKUSDT.P, 15 2025.05.01-2025.11.20.csv"),
    strategy_id='s01_trailing_ma',
    enabled_params={
        'maLength': True,
        'closeCountLong': True
    },
    param_ranges={
        'maLength': (30, 60, 15),      # 3 values
        'closeCountLong': (5, 9, 2)    # 3 values
    },
    fixed_params={...},  # All other params
    worker_processes=4
)

results = run_optimization(config)

# Total: 3 * 3 = 9 combinations
assert len(results) == 9, f'Expected 9 results, got {len(results)}'

print('✅ Optimization completed successfully')
print(f'Total combinations: {len(results)}')
print(f'Best result: {results[0].net_profit_pct:.2f}%')
"
```

### Test 4: CSV Export Test

```python
from optimizer_engine import export_to_csv
from strategy_registry import StrategyRegistry

strategy_class = StrategyRegistry.get_strategy_class('s01_trailing_ma')

csv_buffer = export_to_csv(results, config, strategy_class)

# Check CSV has correct columns
csv_text = csv_buffer.getvalue().decode('utf-8')
lines = csv_text.split('\n')

# Should have parameter header block
assert 'maType,EMA' in csv_text  # Fixed parameter in header

# Should have results table
assert 'Net Profit %' in csv_text
assert 'maLength' in csv_text  # Varied parameter in columns

print('✅ CSV export works with dynamic columns')
```

---

## Acceptance Criteria

Before committing:

1. **Code Changes:**
   - [ ] `OptimizationConfig` has `strategy_id` field
   - [ ] `run_optimization()` uses StrategyRegistry
   - [ ] `_generate_parameter_grid()` uses `param_definitions`
   - [ ] `_init_worker()` uses `cache_requirements`
   - [ ] `_simulate_combination()` calls `strategy.simulate()`
   - [ ] Hardcoded `PARAMETER_MAP` deleted
   - [ ] `export_to_csv()` uses dynamic columns
   - [ ] `server.py` passes `strategy_id` from request

2. **Testing:**
   - [ ] Import test passes
   - [ ] Small optimization (3 combos) works
   - [ ] Full optimization works
   - [ ] CSV export has correct structure
   - [ ] S_01 reference test still passes

3. **Performance:**
   - [ ] Optimization speed similar to before (caching works)
   - [ ] Memory usage reasonable

---

## Common Issues and Solutions

### Issue: "Missing parameter in param_ranges"

**Cause:** Parameter enabled but no range specified.

**Solution:**
```python
# Ensure all enabled params have ranges
enabled_params = {'maLength': True}
param_ranges = {'maLength': (30, 60, 15)}  # Must have range!
```

### Issue: Worker pool hangs

**Cause:** Error in _init_worker or _simulate_combination.

**Solution:**
- Check logs for initialization errors
- Test strategy instantiation manually
- Verify cache_requirements format

### Issue: CSV has wrong columns

**Cause:** Not passing strategy_class to export_to_csv.

**Solution:**
```python
strategy_class = StrategyRegistry.get_strategy_class(config.strategy_id)
csv_buffer = export_to_csv(results, config, strategy_class)
```

---

## Commit Message

```bash
git add src/optimizer_engine.py src/server.py
git commit -m "Phase 4: Refactor optimizer for multi-strategy

- Add strategy_id to OptimizationConfig
- Use StrategyRegistry to get strategy class
- Generate parameter grid from strategy.get_param_definitions()
- Pre-compute cache from strategy.get_cache_requirements()
- Call strategy.simulate() instead of duplicating logic
- Remove hardcoded PARAMETER_MAP
- Dynamic CSV columns from strategy params
- Update server.py to pass strategy_id

Testing:
- Small optimization (3 combos) works ✅
- Full S_01 optimization works ✅
- CSV export has correct structure ✅
- S_01 reference test UNCHANGED ✅

Performance: Optimization speed maintained (caching works)"
```

---

## Next Steps

After Phase 4 is complete and committed:

**Proceed to Phase 5: Add S_03 Reversal Strategy**
- See `migration_prompt_5.md`
- See `migration_checklist.md` - Phase 5

---

**End of Migration Prompt 4**
