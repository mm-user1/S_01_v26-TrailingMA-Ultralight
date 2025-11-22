# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a cryptocurrency/forex trading strategy backtesting and optimization platform for the "Trailing Moving Average" strategy. It provides both a web interface (Flask SPA) and CLI tools to run single backtests or optimize across thousands of parameter combinations using either grid search or Bayesian optimization (Optuna).

## Running the Application

### Web Server
```bash
cd src
python server.py
```
Server runs at http://0.0.0.0:8000 and serves the embedded SPA from `index.html`.

### CLI Backtest
```bash
cd src
python run_backtest.py --csv ../data/OKX_LINKUSDT.P,\ 15...csv
```

### Dependencies
```bash
pip install -r requirements.txt
```
Key dependencies: Flask, pandas, numpy, matplotlib, backtesting, tqdm, optuna==4.4.0

## Directory Structure

- `./src/` - Main project folder containing all source code
- `./data/` - Example market data CSVs, reference PineScript strategies, and reference Python scripts
- `./src/Presets/` - Auto-created directory for saved trading configuration presets (JSON files)

## Architecture: Core Components

### 1. Entry Points
- **Web**: `server.py` - Flask REST API with 6 endpoints for backtesting, optimization, and preset management
- **CLI**: `run_backtest.py` - Command-line wrapper for single backtests

### 2. Optimization Modes

The system supports two distinct optimization approaches:

**Grid Search** (`optimizer_engine.py`):
- Cartesian product of all enabled parameter ranges
- Uses multiprocessing with pre-computed caches for performance
- Best for small-to-medium search spaces (<10,000 combinations)

**Bayesian Optimization** (`optuna_engine.py`):
- Optuna-based smart search that learns from previous trials
- 5 optimization targets: score, net_profit, romad, sharpe, max_drawdown
- 3 budget modes: n_trials, timeout, or patience (convergence)
- Includes pruning to eliminate unpromising trials early
- Best for large search spaces or expensive evaluations

### 3. Multi-Process Caching Architecture

**Critical for understanding performance**: `optimizer_engine.py` implements a sophisticated caching system in `_init_worker()`:

```python
# Each worker process pre-computes and caches:
_ma_cache = {}       # All MA values for all types/lengths
_lowest_cache = {}   # Lowest lows for trail calculations
_highest_cache = {}  # Highest highs for trail calculations
_atr_values = {}     # ATR values for stop-loss sizing
```

This architecture avoids recomputing expensive calculations across thousands of parameter combinations. The cache is initialized once per worker process and shared across all simulations in that worker. **When modifying optimization logic, preserve this caching pattern to maintain performance.**

### 4. Strategy Simulation Flow

`backtest_engine.py` → `run_strategy()` executes the core trading simulation:

1. **Indicator Calculation**: Computes MAs, ATR, trailing MAs (supports 11 MA types)
2. **Bar-by-Bar Simulation**: Iterates through OHLCV data generating entry/exit signals based on MA crossovers and close counts
3. **Position Management**: Handles stops (ATR-based, max %, max days) and trailing exits
4. **Metrics Calculation**: Returns net profit, max drawdown, trade count, and full trade history

### 5. Moving Average Types (11 Supported)

The system implements 11 MA types with different characteristics:
- **SMA**: Simple Moving Average (baseline)
- **EMA**: Exponential MA (responsive)
- **HMA**: Hull MA (fast trend detection, low lag)
- **ALMA**: Arnaud Legoux MA (optimized for low lag)
- **KAMA**: Kaufman Adaptive MA (adjusts to volatility)
- **WMA**: Weighted MA
- **TMA**: Triangular MA (double-smoothed)
- **T3**: T3 MA (advanced smoothing)
- **DEMA**: Double EMA (faster response than EMA)
- **VWMA**: Volume-Weighted MA
- **VWAP**: Volume-Weighted Average Price

Each is implemented in `backtest_engine.py` (lines ~188-331). These are used for both trend detection (`maType`) and trailing exit logic (`trailLongType`/`trailShortType`).

### 6. Scoring System (6 Metrics)

Optimization can use a composite score combining 6 risk-adjusted metrics (`optimizer_engine.py`, lines 27-34):

- **RoMaD**: Return Over Maximum Drawdown (25% default weight)
- **Sharpe Ratio**: Risk-adjusted return (20%)
- **Profit Factor**: Wins/Losses ratio (20%)
- **Ulcer Index**: Downside volatility measure (15%)
- **Recovery Factor**: Profit vs. max drawdown (10%)
- **Consistency**: Monthly return consistency (10%)

Users can adjust weights per their risk preferences. All 6 metrics are written to output CSV.

### 7. Parameter Flow and Preset System

**Configuration flow**:
1. User sets parameters in web UI or loads a preset
2. Frontend sends config to `/api/optimize` or `/api/backtest`
3. Server validates and converts to `OptimizationConfig` or `StrategyParams`
4. Optimizer generates combinations respecting enabled/disabled parameter flags
5. Results exported to CSV with parameter block header (fixed params that weren't varied)

**Preset system** (`src/Presets/`):
- Presets stored as JSON files
- Can import optimal parameters from output CSV using `/api/presets/import-csv`
- Only parameters in the "locked" block (not varied during optimization) are imported
- Default preset in `defaults.json`

### 8. API Endpoints

- `GET /` → Serves web UI
- `POST /api/backtest` → Single backtest with specific parameters
- `POST /api/optimize` → Grid or Optuna optimization
- `GET /api/presets` → List all saved presets
- `POST /api/presets` → Create new preset
- `PUT /api/presets/<name>` → Update existing preset
- `DELETE /api/presets/<name>` → Delete preset
- `POST /api/presets/import-csv` → Import parameters from optimization result CSV

## Multi-Strategy Support

The web UI supports multiple trading strategies:

1. **S_01 TrailingMA v26 Ultralight** — Trend-following strategy
2. **S_03 Reversal v07 Light** — Counter-trend reversal strategy

### Selecting a Strategy

1. Open the web UI at http://localhost:8000
2. In the Backtester or Optimizer panel, select a strategy from the dropdown
3. Strategy metadata (name, type, description) appears in the info panel
4. Parameters and requests are routed using the selected `strategy_id`

### Adding New Strategies

1. Create a new strategy class inheriting from `BaseStrategy`
2. Implement required methods (`should_long`, `should_short`, `calculate_entry`, etc.)
3. Call `StrategyRegistry.register_strategy()` in `src/strategies/__init__.py`
4. The strategy will automatically appear in the web UI dropdown via `/api/strategies`

## Dynamic Parameter Forms (Phase 8)

- Backtester and Optimizer parameter forms are generated dynamically from the strategy metadata returned by `/api/strategies`.
- Optimizer range controls (enable + from/to/step) are built at runtime for numeric parameters, so adding new strategies or parameters no longer requires HTML edits.
- MA type collections for S_01 remain supported; ensure at least one trend and trailing type is selected before running grid optimizations.

## Key Design Constraints

From `agents.md`, critical development guidelines:

1. **Strict specification adherence**: Any deviations from requirements require explicit user consent
2. **Performance is critical**: The script must be maximally efficient and fast. Respect the caching architecture.
3. **Light theme for GUI**: UI must use light theme (already implemented in `index.html`)

## Output CSV Format

Optimization results are exported as CSV with:
1. **Parameter block header**: Fixed parameters that weren't varied (key-value pairs)
2. **Results table**: One row per combination tested with columns for each varied parameter plus all metrics (net profit %, trades, sharpe, score, etc.)

This format allows users to:
- Import winning parameters back as presets
- Analyze which parameter ranges performed best
- Filter results by minimum score or profit thresholds

## Performance Considerations

- **Multiprocessing**: Default 6 worker processes (user-configurable)
- **Vectorized operations**: Uses numpy/pandas for bulk calculations
- **Pre-computation**: All MA values cached before worker pool starts
- **Memory sharing**: Workers inherit cached data via fork (Linux) or explicit initialization
- **Progress tracking**: `tqdm` progress bars in terminal show ETA based on worker throughput

When modifying optimizer or backtest logic, run performance tests with realistic parameter grids (1000+ combinations) to ensure changes don't break the caching benefits.

## Common Workflow

1. User uploads OHLCV CSV data (or selects multiple files)
2. Configures strategy parameters and selects which to vary
3. Chooses optimization mode (grid vs optuna) and scoring preferences
4. Clicks "Optimize" → Server spawns worker pool
5. Results stream back as CSV download
6. User analyzes results, identifies best parameters
7. Imports winning parameters as new preset via `/api/presets/import-csv`
8. Runs additional focused optimizations or tests on different date ranges

## Testing Notes

No formal test suite exists. When making changes:
- Test with small parameter grids first (e.g., 2-3 values per parameter)
- Verify optimization completes without errors
- Check output CSV format matches expected structure (parameter block + results table)
- Test both grid and optuna modes if modifying optimization logic
- Verify preset save/load/import functionality if touching preset endpoints
