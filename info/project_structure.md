# Project Structure - Multi-Strategy Architecture

## Overview

This document describes the **target architecture** after completing the multi-strategy migration. The system is designed to support multiple trading strategies with minimal code duplication, clean abstractions, and easy extensibility.

---

## Directory Structure

```
S_01_v26-TrailingMA-Ultralight/
├── data/                              # Market data and reference Pine scripts
│   ├── *.csv                          # OHLCV market data files
│   ├── S_01 Movings_v26 TrailingMA Ultralight.pine
│   └── S_03 Reversal_v07 Light for PROJECT PLAN.pine
│
├── info/                              # Documentation for migration and development
│   ├── project_structure.md           # This file
│   ├── migration_checklist.md         # Migration plan with checklist
│   ├── migration_prompt_*.md          # Step-by-step prompts for migration
│   ├── tests.md                       # Reference test specifications
│   ├── agents.md                      # Instructions for AI coding agents
│   └── pine_guidelines.md             # Pine → Python translation guidelines
│
├── src/                               # Main source code
│   ├── strategies/                    # ⭐ NEW: Strategy modules
│   │   ├── __init__.py
│   │   ├── base_strategy.py           # ABC BaseStrategy contract
│   │   ├── s01_trailing_ma.py         # S_01 TrailingMA strategy
│   │   └── s03_reversal.py            # S_03 Reversal strategy
│   │
│   ├── indicators.py                  # ⭐ NEW: Common indicators library
│   ├── strategy_registry.py           # ⭐ NEW: Strategy registration and routing
│   │
│   ├── backtest_engine.py             # 🔄 REFACTORED: Universal backtest engine
│   ├── optimizer_engine.py            # 🔄 REFACTORED: Grid search optimizer
│   ├── optuna_engine.py               # 🔄 REFACTORED: Bayesian optimizer
│   ├── walkforward_engine.py          # 🔄 REFACTORED: WFA engine
│   │
│   ├── server.py                      # 🔄 UPDATED: Flask API with strategy selector
│   ├── run_backtest.py                # 🔄 UPDATED: CLI with strategy parameter
│   │
│   ├── index.html                     # 🔄 UPDATED: UI with strategy dropdown
│   └── Presets/                       # Strategy presets (strategy-specific in future)
│
├── requirements.txt
├── CLAUDE.md                          # Project overview for Claude Code
└── README.md
```

**Legend:**
- ⭐ NEW - New files created during migration
- 🔄 REFACTORED - Existing files with major changes
- 🔄 UPDATED - Existing files with minor updates

---

## Core Modules

### 1. **strategies/** - Strategy Implementations

This folder contains all trading strategy implementations.

#### **base_strategy.py** - Abstract Base Class

**Purpose:** Defines the contract that ALL strategies must implement.

**Key Components:**

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Provides:
    - Contract enforcement through ABC
    - Single entry point (simulate) for both backtest and optimization
    - Parameter definition interface for UI autogeneration
    - Cache requirements interface for optimization
    """

    STRATEGY_ID: str = "base"
    STRATEGY_NAME: str = "Base Strategy"
    VERSION: str = "1.0"

    # ════════════════════════════════════════════════════
    # CORE ABSTRACT METHODS (must be implemented)
    # ════════════════════════════════════════════════════

    @abstractmethod
    def should_long(self, idx: int) -> bool:
        """Check if long entry conditions are met"""

    @abstractmethod
    def should_short(self, idx: int) -> bool:
        """Check if short entry conditions are met"""

    @abstractmethod
    def calculate_entry(self, idx: int, direction: str) -> Tuple[float, float, float]:
        """Calculate entry, stop, and target prices"""

    @abstractmethod
    def calculate_position_size(self, idx: int, direction: str,
                                entry_price: float, stop_price: float,
                                equity: float) -> float:
        """Calculate position size based on strategy rules"""

    @abstractmethod
    def should_exit(self, idx: int, position_info: Dict) -> Tuple[bool, Optional[float], str]:
        """Check if exit conditions are met"""

    # ════════════════════════════════════════════════════
    # CONFIGURATION METHODS
    # ════════════════════════════════════════════════════

    @classmethod
    @abstractmethod
    def get_param_definitions(cls) -> Dict[str, Dict]:
        """
        Define all strategy parameters for UI and optimization.

        Returns dict with structure:
        {
            'maLength': {  # camelCase key for API/frontend compatibility
                'type': 'int',
                'default': 45,
                'min': 10,
                'max': 200,
                'step': 5,
                'description': 'Moving average period'
            }
        }
        """

    @classmethod
    def get_cache_requirements(cls, param_combinations: List[Dict]) -> Dict:
        """
        Specify what indicators need to be cached for optimization.

        Returns dict with structure:
        {
            'ma_types_and_lengths': [('SMA', 50), ('EMA', 45), ...],
            'needs_atr': True,
            'long_lp_values': [2, 5, 10],
            'short_lp_values': [2, 5, 10]
        }
        """
        return {}  # Default: no caching needed

    # ════════════════════════════════════════════════════
    # SIMULATION (single entry point)
    # ════════════════════════════════════════════════════

    def simulate(self, df: pd.DataFrame, cached_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main simulation entry point.

        Used by:
        - Single backtest (cached_data=None, compute on-the-fly)
        - Optimization (cached_data={...}, use pre-computed values)

        Returns:
        {
            'net_profit_pct': float,
            'max_drawdown_pct': float,
            'total_trades': int,
            'winning_trades': int,
            'losing_trades': int,
            'sharpe_ratio': float,
            'profit_factor': float,
            'trades': List[Dict],
            'equity_curve': List[float]
        }
        """
        self._prepare_data(df, cached_data)
        return self._run_simulation(df)

    @abstractmethod
    def _prepare_data(self, df: pd.DataFrame, cached_data: Optional[Dict]) -> None:
        """Prepare indicators (use cache if available)"""

    def _run_simulation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Standard simulation loop (can be overridden if needed)"""
        # Implemented in base class with calls to abstract methods

    # ════════════════════════════════════════════════════
    # OPTIONAL METHODS
    # ════════════════════════════════════════════════════

    def allows_reversal(self) -> bool:
        """Can this strategy reverse positions? (for reversal strategies)"""
        return False

    def _validate_params(self) -> None:
        """Validate strategy parameters (optional)"""
        pass
```

**Why this design:**
- ✅ Single `simulate()` method eliminates duplication between backtest and optimizer
- ✅ ABC enforcement prevents forgotten method implementations
- ✅ `get_param_definitions()` enables UI autogeneration (DRY principle)
- ✅ `get_cache_requirements()` makes optimization strategy-agnostic
- ✅ Clear separation of concerns (entry, exit, sizing, data preparation)

---

#### **s01_trailing_ma.py** - S_01 TrailingMA Strategy

**Purpose:** Implements the original Trailing Moving Average strategy.

**Key Features:**
- Trend-following with MA crossover entry
- ATR-based stop loss with lookback period
- Risk/Reward based take profit
- Trailing MA exit with activation threshold
- Risk-based position sizing (2% of equity per trade)
- Max stop % and max days filters

**Parameters:** 29 total
- Date filter: enabled, start date, end date
- Trend MA: type, length
- Entry: close count long/short
- Stops: ATR multiplier, RR ratio, lookback period, max %, max days (separate for long/short)
- Trailing: activation RR, MA type/length/offset (separate for long/short)
- Risk: risk per trade %, contract size, commission rate, ATR period

**State Management:**
- Counters: close count, trade count
- Trailing: activation flag, current trail price
- Caches: MA values, ATR values, highest/lowest for lookback

---

#### **s03_reversal.py** - S_03 Reversal Strategy

**Purpose:** Implements the Reversal strategy (always in market).

**Key Features:**
- Reversal system (long ↔ short, never flat)
- Multiple MA confirmation (ma1, ma2, ma3)
- Close count filter (optional)
- Days of week filter (optional)
- NO stops/targets - only reversal exits
- 100% equity position sizing

**Parameters:** ~12 total
- MA1, MA2, MA3: type, length, enable flag
- Close count: enable flag, long/short counts
- Days filter: enable flag, allowed days
- Breakout mode: use close price for signals
- Contract size

**State Management:**
- Counters: close count long/short
- Caches: MA1, MA2, MA3 values

**Special:** Sets `allows_reversal() → True`

---

### 2. **indicators.py** - Common Indicators Library

**Purpose:** Centralized library of technical indicators used across strategies.

**Contains:**
- **11 Moving Average types:** SMA, EMA, HMA, WMA, ALMA, KAMA, TMA, T3, DEMA, VWMA, VWAP
- **ATR** (Average True Range)
- **Unified interface:** `get_ma(series, ma_type, length, volume=None, high=None, low=None)`

**Design Principles:**
- ✅ Pure functions (no state)
- ✅ Pandas Series input → Series/ndarray output
- ✅ Add new indicators here ONLY if used by 2+ strategies
- ✅ Single-strategy indicators stay in strategy file

**Example:**
```python
def sma(series: pd.Series, length: int) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(window=length).mean()

def ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=length, adjust=False).mean()

def get_ma(series: pd.Series, ma_type: str, length: int,
           volume: Optional[pd.Series] = None,
           high: Optional[pd.Series] = None,
           low: Optional[pd.Series] = None) -> pd.Series:
    """Unified MA interface - routes to appropriate function"""
    ma_type = ma_type.upper()
    if ma_type == "SMA": return sma(series, length)
    elif ma_type == "EMA": return ema(series, length)
    # ... etc
```

---

### 3. **strategy_registry.py** - Strategy Registry

**Purpose:** Central registry for strategy discovery and instantiation.

**Responsibilities:**
1. Map strategy IDs to strategy classes
2. Provide list of available strategies for UI
3. Instantiate strategy objects with parameters
4. Route API calls to correct strategy

**Design:**
```python
class StrategyRegistry:
    """Central registry for all trading strategies"""

    _strategies = {
        "s01_trailing_ma": S01TrailingMA,
        "s03_reversal": S03Reversal,
        # Future strategies added here
    }

    @classmethod
    def get_strategy_class(cls, strategy_id: str) -> type:
        """Get strategy class by ID"""
        if strategy_id not in cls._strategies:
            raise ValueError(f"Unknown strategy: {strategy_id}")
        return cls._strategies[strategy_id]

    @classmethod
    def get_strategy_instance(cls, strategy_id: str, params: Dict) -> BaseStrategy:
        """Create strategy instance with parameters"""
        strategy_class = cls.get_strategy_class(strategy_id)
        return strategy_class(params)

    @classmethod
    def list_strategies(cls) -> List[Dict]:
        """List all available strategies with metadata"""
        return [
            {
                'id': strategy_id,
                'name': strategy_class.STRATEGY_NAME,
                'version': strategy_class.VERSION,
                'param_definitions': strategy_class.get_param_definitions()
            }
            for strategy_id, strategy_class in cls._strategies.items()
        ]
```

**Why this design:**
- ✅ Single source of truth for available strategies
- ✅ Easy to add new strategies (one line in dict)
- ✅ No if/else chains scattered across codebase
- ✅ Supports dynamic strategy discovery for UI

---

### 4. **backtest_engine.py** - Universal Backtest Engine

**Purpose:** Execute backtests for ANY strategy implementing BaseStrategy.

**Key Changes from Original:**
- ❌ REMOVED: Strategy-specific logic (S_01 MA crossover, ATR stops, trailing exits)
- ✅ ADDED: Generic simulation loop that calls strategy methods
- ✅ ADDED: Support for reversal strategies via `allows_reversal()`

**New Architecture:**
```python
def run_strategy(
    df: pd.DataFrame,
    strategy: BaseStrategy,  # ← Strategy object, not params!
    params: StrategyParams   # ← Kept for backward compatibility
) -> StrategyResult:
    """
    Universal backtest engine.

    Works with ANY strategy implementing BaseStrategy.
    Calls strategy methods for all trading decisions.
    """

    # Delegate to strategy.simulate()
    result = strategy.simulate(df, cached_data=None)

    # Convert to StrategyResult format
    return StrategyResult(
        net_profit_pct=result['net_profit_pct'],
        max_drawdown_pct=result['max_drawdown_pct'],
        # ... etc
    )

# Backward compatibility alias
StrategyParams = StrategyParams  # Old name still works
```

**Simulation Loop (inside BaseStrategy._run_simulation):**
```python
for i in range(len(df)):
    # Exit logic (if in position)
    if position != 0:
        should_exit, exit_price, reason = strategy.should_exit(i, position_info)
        if should_exit:
            # Close position, record trade
            pass

    # Reversal logic (for reversal strategies)
    if strategy.allows_reversal() and position != 0:
        if position > 0 and strategy.should_short(i):
            # Close long + open short
            pass
        elif position < 0 and strategy.should_long(i):
            # Close short + open long
            pass

    # Entry logic (if flat)
    if position == 0:
        if strategy.should_long(i):
            entry, stop, target = strategy.calculate_entry(i, 'long')
            size = strategy.calculate_position_size(i, 'long', entry, stop, equity)
            # Open long position
            pass
        elif strategy.should_short(i):
            # Similar for short
            pass
```

**Responsibilities:**
- Load CSV data
- Prepare MarketData structure
- Execute main simulation loop
- Track equity curve
- Calculate performance metrics
- Return standardized results

**Does NOT contain:**
- MA calculations (→ indicators.py)
- Entry/exit logic (→ strategy.should_long/should_exit)
- Position sizing (→ strategy.calculate_position_size)

---

### 5. **optimizer_engine.py** - Grid Search Optimizer

**Purpose:** Optimize strategy parameters via grid search with multiprocessing.

**Key Changes:**
- ❌ REMOVED: Hardcoded PARAMETER_MAP for S_01
- ❌ REMOVED: Strategy-specific simulation code (_simulate_combination duplication)
- ✅ ADDED: Dynamic parameter mapping from strategy.get_param_definitions()
- ✅ ADDED: Dynamic caching from strategy.get_cache_requirements()

**New Flow:**
```python
class OptimizationConfig:
    csv_file: IO
    strategy_id: str         # ⭐ NEW: which strategy to optimize
    enabled_params: Dict
    param_ranges: Dict
    # ... rest unchanged

def run_optimization(config: OptimizationConfig) -> List[OptimizationResult]:
    # 1. Get strategy class
    strategy_class = StrategyRegistry.get_strategy_class(config.strategy_id)

    # 2. Generate parameter grid using strategy's param definitions
    param_defs = strategy_class.get_param_definitions()
    combinations = _generate_grid(config.param_ranges, param_defs)

    # 3. Get cache requirements from strategy
    cache_req = strategy_class.get_cache_requirements(combinations)

    # 4. Initialize worker pool with cache
    pool = mp.Pool(initializer=_init_worker, initargs=(df, cache_req, strategy_class))

    # 5. Run simulations
    results = pool.map(_simulate_combination, combinations)

    return results

def _simulate_combination(params_dict: Dict) -> OptimizationResult:
    """Worker function - creates strategy and calls simulate()"""
    global _strategy_class, _cached_data

    strategy = _strategy_class(params_dict)
    result = strategy.simulate(_df, cached_data=_cached_data)

    return OptimizationResult(**result)
```

**Caching:**
- Worker processes are initialized once with pre-computed indicators
- `_init_worker()` calls `strategy_class.get_cache_requirements()` to know what to cache
- Each worker has access to cached MA/ATR/etc values in global variables
- Strategies use cached data via `_prepare_data(df, cached_data)`

**No More Duplication:**
- ✅ Single simulation logic in `BaseStrategy.simulate()`
- ✅ Used by both single backtest and optimization
- ✅ Difference: cached_data parameter

---

### 6. **optuna_engine.py** - Bayesian Optimizer

**Purpose:** Optimize strategy parameters via Bayesian optimization (Optuna).

**Key Changes:**
- Similar to optimizer_engine.py
- Dynamic parameter space from `strategy.get_param_definitions()`
- Calls `strategy.simulate()` for each trial

**Integration:**
```python
class OptunaOptimizer:
    def __init__(self, base_config, optuna_config):
        self.strategy_id = base_config.strategy_id
        self.strategy_class = StrategyRegistry.get_strategy_class(self.strategy_id)

    def _build_search_space(self) -> Dict:
        """Build Optuna search space from strategy param definitions"""
        param_defs = self.strategy_class.get_param_definitions()
        # Convert to Optuna format
        return search_space

    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function for single trial"""
        params_dict = self._prepare_trial_parameters(trial, search_space)

        strategy = self.strategy_class(params_dict)
        result = strategy.simulate(df, cached_data=self._cached_data)

        return result['net_profit_pct']  # or other metric
```

---

### 7. **walkforward_engine.py** - Walk-Forward Analysis

**Purpose:** Perform walk-forward analysis (rolling optimization + out-of-sample testing).

**Key Changes:**
- Accepts `strategy_id` parameter
- Passes `strategy_id` to optimizer on each window

**Integration:**
```python
class WalkForwardEngine:
    def __init__(self, config: WFConfig, strategy_id: str, ...):
        self.strategy_id = strategy_id

    def _run_optuna_on_window(self, df_window):
        base_config = OptimizationConfig(
            csv_file=csv_buffer,
            strategy_id=self.strategy_id,  # ⭐ Pass strategy ID
            # ...
        )
        return run_optuna_optimization(base_config, optuna_settings)
```

---

### 8. **server.py** - Flask API Server

**Purpose:** REST API for web UI and external tools.

**New/Updated Endpoints:**

```python
@app.route('/api/strategies', methods=['GET'])
def list_strategies():
    """
    Get list of available strategies.

    Returns:
    {
        "strategies": [
            {
                "id": "s01_trailing_ma",
                "name": "S_01 TrailingMA v26",
                "version": "26",
                "param_definitions": {...}
            },
            {
                "id": "s03_reversal",
                "name": "S_03 Reversal v07",
                "version": "07",
                "param_definitions": {...}
            }
        ]
    }
    """
    strategies = StrategyRegistry.list_strategies()
    return jsonify({"strategies": strategies})

@app.route('/api/backtest', methods=['POST'])
def run_backtest_endpoint():
    """
    Run single backtest.

    Request:
    - strategy_id: str (e.g., "s01_trailing_ma")
    - csv_file: uploaded file or path
    - params: dict of strategy parameters

    Returns: StrategyResult as JSON
    """
    strategy_id = request.form.get('strategy_id', 's01_trailing_ma')
    params = request.form.get('params')  # JSON string

    strategy = StrategyRegistry.get_strategy_instance(strategy_id, params)
    df = load_data(csv_file)

    result = strategy.simulate(df)
    return jsonify(result)

@app.route('/api/optimize', methods=['POST'])
def run_optimization_endpoint():
    """
    Run parameter optimization.

    Request:
    - strategy_id: str
    - optimization_config: dict

    Returns: List[OptimizationResult] as CSV
    """
    strategy_id = request.form.get('strategy_id', 's01_trailing_ma')

    config = OptimizationConfig(
        csv_file=uploaded_file,
        strategy_id=strategy_id,
        # ... rest of config
    )

    results = run_optimization(config)
    return export_to_csv(results, config)
```

**Changes:**
- ✅ All endpoints accept `strategy_id` parameter
- ✅ Default to `s01_trailing_ma` for backward compatibility
- ✅ Use StrategyRegistry for strategy instantiation

---

### 9. **index.html** - Web UI

**Purpose:** Single-page application for backtesting and optimization.

**Key Changes:**

```html
<!-- Strategy Selector -->
<select id="strategySelector" onchange="loadStrategyParameters()">
    <option value="s01_trailing_ma" selected>S_01 TrailingMA v26</option>
    <option value="s03_reversal">S_03 Reversal v07</option>
</select>

<!-- Strategy Parameters Container (dynamic) -->
<div id="strategyParametersContainer">
    <!-- S_01 Parameters Block -->
    <div id="params_s01_trailing_ma" class="strategy-params">
        <!-- Current S_01 UI -->
    </div>

    <!-- S_03 Parameters Block -->
    <div id="params_s03_reversal" class="strategy-params" style="display: none;">
        <!-- New S_03 UI -->
    </div>
</div>

<script>
function loadStrategyParameters() {
    const strategyId = document.getElementById('strategySelector').value;

    // Hide all parameter blocks
    document.querySelectorAll('.strategy-params').forEach(el => {
        el.style.display = 'none';
    });

    // Show selected strategy's parameters
    document.getElementById(`params_${strategyId}`).style.display = 'block';
}

// On form submit
function runOptimization() {
    const strategyId = document.getElementById('strategySelector').value;
    const formData = new FormData();
    formData.append('strategy_id', strategyId);
    // ... append other parameters

    fetch('/api/optimize', {
        method: 'POST',
        body: formData
    });
}
</script>
```

**For MVP:**
- Static parameter blocks for each strategy (show/hide on selector change)
- Future enhancement: Dynamic form generation from `/api/strategies` endpoint

---

## Data Flow

### Single Backtest Flow

```
User (UI/CLI)
    ↓
    ↓ (strategy_id, params, csv_file)
    ↓
server.py /api/backtest
    ↓
    ↓ StrategyRegistry.get_strategy_instance(strategy_id, params)
    ↓
BaseStrategy instance (S01 or S03)
    ↓
    ↓ strategy.simulate(df, cached_data=None)
    ↓
BaseStrategy.simulate()
    ├─→ _prepare_data(df, None) → compute indicators on-the-fly
    └─→ _run_simulation(df)
         ├─→ should_long(i)
         ├─→ calculate_entry(i, 'long')
         ├─→ calculate_position_size(...)
         └─→ should_exit(i, position_info)
    ↓
    ↓ Returns: dict with results
    ↓
server.py
    ↓
    ↓ Returns: JSON response
    ↓
User receives results
```

---

### Optimization Flow

```
User (UI)
    ↓
    ↓ (strategy_id, param_ranges, csv_file)
    ↓
server.py /api/optimize
    ↓
optimizer_engine.run_optimization(config)
    ↓
    ↓ 1. Get strategy class from registry
    ↓ strategy_class = StrategyRegistry.get_strategy_class(strategy_id)
    ↓
    ↓ 2. Generate parameter grid
    ↓ param_defs = strategy_class.get_param_definitions()
    ↓ combinations = generate_grid(param_ranges, param_defs)
    ↓
    ↓ 3. Get cache requirements
    ↓ cache_req = strategy_class.get_cache_requirements(combinations)
    ↓
    ↓ 4. Pre-compute indicators
    ↓ cached_data = {
    ↓     'ma_SMA_50': compute_ma(df, 'SMA', 50),
    ↓     'ma_EMA_45': compute_ma(df, 'EMA', 45),
    ↓     'atr_14': compute_atr(df, 14),
    ↓     # ... all required indicators
    ↓ }
    ↓
    ↓ 5. Initialize worker pool
    ↓ pool = mp.Pool(initializer=_init_worker,
    ↓                 initargs=(df, cached_data, strategy_class))
    ↓
    ↓ 6. Map combinations to workers
    ↓ results = pool.map(_simulate_combination, combinations)
    ↓
Worker Process (each combination):
    ↓
    ↓ _simulate_combination(params_dict)
    ↓     strategy = strategy_class(params_dict)
    ↓     result = strategy.simulate(df, cached_data=cached_data)
    ↓     return OptimizationResult(**result)
    ↓
    ↓ strategy.simulate(df, cached_data)
    ↓     _prepare_data(df, cached_data)
    ↓         ← Uses pre-computed indicators from cached_data
    ↓     _run_simulation(df)
    ↓         ← Same logic as single backtest
    ↓
Results aggregated
    ↓
    ↓ Calculate scores, sort by performance
    ↓
server.py
    ↓
    ↓ Export to CSV
    ↓
User downloads results.csv
```

**Key Points:**
- ✅ **No duplication:** Same `simulate()` method for both flows
- ✅ **Performance:** Cached indicators in optimization
- ✅ **Flexibility:** Strategy controls what to cache via `get_cache_requirements()`

---

## Design Principles

### 1. Single Responsibility Principle

Each module has ONE clear purpose:
- `strategies/` - Trading logic
- `indicators.py` - Indicator calculations
- `backtest_engine.py` - Simulation execution
- `optimizer_engine.py` - Parameter optimization
- `strategy_registry.py` - Strategy management

### 2. Open/Closed Principle

**Open for extension (adding new strategies):**
```python
# To add S_04 strategy:
# 1. Create src/strategies/s04_my_strategy.py
# 2. Implement BaseStrategy abstract methods
# 3. Add to strategy_registry.py: {"s04_my_strategy": S04MyStrategy}
# 4. Add UI block in index.html
# Done!
```

**Closed for modification:**
- No changes needed in backtest_engine.py
- No changes needed in optimizer_engine.py
- No changes needed in server.py endpoints

### 3. Dependency Inversion

High-level modules (optimizer, backtest engine) depend on abstraction (BaseStrategy), not concrete implementations (S01, S03).

```
optimizer_engine.py
    ↓ depends on
BaseStrategy (interface)
    ↑ implemented by
S01TrailingMA, S03Reversal
```

### 4. DRY (Don't Repeat Yourself)

- Parameter definitions in ONE place: `strategy.get_param_definitions()`
- Simulation logic in ONE place: `BaseStrategy.simulate()`
- Indicator code in ONE place: `indicators.py`

### 5. Backward Compatibility

During migration:
- Old function names kept as aliases
- Default `strategy_id='s01_trailing_ma'` in API endpoints
- Old `StrategyParams` dataclass still works

---

## Extension Points

### Adding a New Strategy (S_04)

1. Create `/src/strategies/s04_breakout.py`
2. Inherit from `BaseStrategy`
3. Implement all abstract methods
4. Define parameters in `get_param_definitions()`
5. Optionally define cache requirements in `get_cache_requirements()`
6. Register in `strategy_registry.py`
7. Add UI parameter block in `index.html`
8. Create reference test in `tests.md`

**Estimated time:** 2-4 hours for experienced developer

### Adding a New Indicator

**If used by 2+ strategies:**
1. Add function to `indicators.py`
2. Update `get_ma()` unified interface if it's an MA variant
3. Document in docstring

**If used by 1 strategy:**
- Keep inside strategy file as private method

### Adding a New Optimizer

Example: Genetic Algorithm optimizer

1. Create `genetic_optimizer.py`
2. Accept `strategy_id` parameter
3. Use `StrategyRegistry.get_strategy_class(strategy_id)`
4. Call `strategy.simulate()` for fitness evaluation
5. Add API endpoint in `server.py`

---

## Testing Strategy

### Unit Tests
- `test_indicators.py` - Test each indicator function
- `test_s01_strategy.py` - Test S01 methods in isolation
- `test_s03_strategy.py` - Test S03 methods in isolation
- `test_registry.py` - Test strategy registration/instantiation

### Integration Tests
- `test_backtest_engine.py` - Test simulation with mock strategy
- `test_optimizer_engine.py` - Test optimization with simple test strategy

### Reference Tests
- Defined in `info/tests.md`
- Run after each migration phase
- Ensures S_01 results unchanged after refactoring

---

## Performance Considerations

### Caching Strategy

**Problem:** Computing MA/ATR for 10,000 parameter combinations is expensive.

**Solution:** Pre-compute all required indicators once before optimization.

**How:**
1. Optimizer calls `strategy.get_cache_requirements(combinations)`
2. Strategy returns: `{'ma_types_and_lengths': [('SMA', 50), ('EMA', 30), ...], 'needs_atr': True}`
3. Optimizer pre-computes all and stores in dict (keys: 'ma_cache', 'atr', 'lowest', 'highest')
4. Each worker process receives cached_data
5. Strategy uses cached values instead of recomputing

**Benefit:** 10-50x speedup for optimization

### Multiprocessing

- Grid optimizer uses `multiprocessing.Pool` with 6-16 workers (configurable)
- Optuna uses sequential trials but with worker pool for caching
- Future: Optuna parallel trials with shared memory cache

---

## Migration Safety

### Phase-by-Phase Approach

Each phase is independent and testable:
1. Extract indicators → test S_01 unchanged
2. Create base strategy → test S_01 unchanged
3. Extract S_01 to module → test results identical
4. Refactor optimizer → test optimization works
5. Add S_03 → test both strategies work
6. Update UI → test both strategies selectable

### Rollback Points

After each phase, code is in working state. If issues arise, can rollback to previous phase.

### Reference Tests

Defined in `info/tests.md`. Run after EVERY phase to ensure S_01 behavior unchanged.

---

## Future Enhancements (Post-MVP)

1. **Dynamic UI Generation**
   - Generate parameter forms from `/api/strategies` endpoint
   - No hardcoded HTML blocks

2. **Strategy-Specific Presets**
   - `Presets/s01_trailing_ma/` folder
   - `Presets/s03_reversal/` folder

3. **Strategy Composition**
   - Combine multiple strategies (e.g., S_01 for trending, S_03 for ranging)

4. **Strategy Performance Analytics**
   - Compare strategies side-by-side
   - Statistical tests for significance

5. **Numba JIT Compilation**
   - Compile simulation loops to machine code
   - 5-15x performance improvement

6. **Shared Memory Caching**
   - Optuna parallel trials with shared cache
   - 8x parallelization

---

## Questions & Answers

**Q: Why not use inheritance for S_01 and S_03 to share common code?**

A: Strategies are fundamentally different (stops vs reversal). Shared code (indicators, utils) goes in `indicators.py`. Strategy classes stay independent and self-contained.

**Q: Why ABC instead of simple classes?**

A: Type safety and contract enforcement. IDE will warn if method not implemented. Prevents runtime errors.

**Q: Why single `simulate()` method instead of separate backtest/optimize methods?**

A: Eliminates 400+ lines of code duplication. Uses `cached_data` parameter to differentiate. Same logic = no bugs from drift.

**Q: Can strategies share state between bars?**

A: Yes! Store in `self.counter_long`, `self.trail_price`, etc. Reset in `__init__` or when position closes.

**Q: What if a strategy needs a unique indicator not in `indicators.py`?**

A: Define as private method in strategy class (e.g., `_calculate_custom_oscillator()`). Only add to `indicators.py` if used by 2+ strategies.

---

## Summary

**This architecture provides:**
- ✅ Clean separation of concerns
- ✅ No code duplication
- ✅ Easy extensibility (new strategies in ~2 hours)
- ✅ Backward compatibility
- ✅ Type safety via ABC
- ✅ DRY principle throughout
- ✅ Performance through caching
- ✅ Testability at every layer

**Trade-offs accepted:**
- Slightly more boilerplate (ABC methods)
- One-time migration effort (~7-10 days)

**Long-term benefits:**
- Adding S_04, S_05, S_06... becomes trivial
- No regression bugs from code duplication
- Clear mental model of system
- New developers onboard quickly
