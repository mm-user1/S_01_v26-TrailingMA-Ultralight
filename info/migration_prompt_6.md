# Migration Prompt 6: Update UI and API for Multi-Strategy Support

**Phase:** 6 of 6 (Final)
**Estimated Time:** 1-2 days
**Difficulty:** Medium
**Dependencies:** Phases 1-5 must be complete

---

## Overview

This final phase updates the web interface and REST API to support multiple strategies. Currently, the UI is hardcoded for S_01 parameters. We need to make it dynamic to work with any strategy registered in the system.

**Key Changes:**
- Add strategy selection dropdown
- Make parameter forms dynamic based on `get_param_definitions()`
- Add API endpoint to list available strategies
- Update optimization/backtest endpoints to accept `strategy_id`
- Validate S_01 and S_03 work through the UI

**Scope:** MVP implementation with static UI (no advanced features)

---

## Objectives

1. Add `/api/strategies` endpoint to list available strategies
2. Update `/api/backtest` and `/api/optimize` to accept `strategy_id` parameter
3. Add strategy selector to web UI
4. Make parameter sections dynamic (show/hide based on strategy)
5. Validate both S_01 and S_03 work end-to-end
6. Run final regression tests

---

## Prerequisites

### Files That Must Exist

- `src/server.py` (Flask REST API)
- `src/index.html` (Web UI SPA)
- `src/strategy_registry.py` (with both strategies registered)
- All strategy implementations from Phases 1-5

### Reference Tests Must Pass

Before starting Phase 6:
- S_01 reference test passes
- S_03 reference test passes
- Both strategies work via CLI

---

## Step 1: Add Strategy List API Endpoint

### File: `src/server.py`

Add a new endpoint to return available strategies:

#### 1.1 Import StrategyRegistry

At the top of `server.py`, add:

```python
from strategy_registry import StrategyRegistry
```

#### 1.2 Add `/api/strategies` Endpoint

Add this new route before the other API endpoints:

```python
@app.route('/api/strategies', methods=['GET'])
def get_strategies():
    """
    Get list of all available strategies with metadata.

    Returns:
        JSON array of strategy objects:
        [
            {
                "strategy_id": "s01_trailing_ma",
                "name": "S_01 TrailingMA v26 Ultralight",
                "description": "Trend-following with MA crossovers...",
                "type": "trend",
                "parameter_count": 28,
                "parameters": {
                    "maType": {"default": "HMA", "type": "str", ...},
                    ...
                }
            },
            ...
        ]
    """
    try:
        strategies = []

        for strategy_id, strategy_class in StrategyRegistry.get_all_strategies().items():
            param_defs = strategy_class.get_param_definitions()

            # Get metadata from registry
            info = next((s for s in StrategyRegistry.get_strategy_info()
                        if s['strategy_id'] == strategy_id), None)

            strategies.append({
                'strategy_id': strategy_id,
                'name': info['name'] if info else strategy_id,
                'description': info['description'] if info else '',
                'type': info.get('type', 'unknown') if info else 'unknown',
                'allows_reversal': strategy_class.allows_reversal(),
                'parameter_count': len(param_defs),
                'parameters': param_defs
            })

        return jsonify(strategies)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

**Test the endpoint:**

```bash
# Start server
cd src
python server.py

# In another terminal
curl http://localhost:8000/api/strategies | jq
```

Expected output:
```json
[
  {
    "strategy_id": "s01_trailing_ma",
    "name": "S_01 TrailingMA v26 Ultralight",
    "type": "trend",
    "allows_reversal": false,
    "parameter_count": 28,
    "parameters": { ... }
  },
  {
    "strategy_id": "s03_reversal",
    "name": "S_03 Reversal v07 Light",
    "type": "reversal",
    "allows_reversal": true,
    "parameter_count": 12,
    "parameters": { ... }
  }
]
```

---

## Step 2: Update Backtest and Optimization Endpoints

### File: `src/server.py`

Update existing endpoints to handle `strategy_id`.

#### 2.1 Update `/api/backtest`

Modify the backtest endpoint to accept and use `strategy_id`:

```python
@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    """
    Run a single backtest with specified parameters.

    Request body:
    {
        "strategy_id": "s01_trailing_ma",  // NEW - defaults to s01
        "csv_file": "data/...",
        "parameters": {
            "maType": "HMA",
            "maLength": 45,
            ...
        }
    }
    """
    try:
        data = request.json

        # Get strategy ID (default to S_01 for backward compatibility)
        strategy_id = data.get('strategy_id', 's01_trailing_ma')

        # Get strategy class
        strategy_class = StrategyRegistry.get_strategy_class(strategy_id)

        # Load data
        csv_path = data['csv_file']
        df = load_data(csv_path)

        # Get parameters (merge with defaults)
        param_defs = strategy_class.get_param_definitions()
        params = {k: v['default'] for k, v in param_defs.items()}
        params.update(data.get('parameters', {}))

        # Run simulation
        strategy = strategy_class(params)
        result = strategy.simulate(df)

        # Return results
        return jsonify({
            'strategy_id': strategy_id,
            'strategy_name': strategy_class.__name__,
            'results': result
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

#### 2.2 Update `/api/optimize`

Modify the optimization endpoint:

```python
@app.route('/api/optimize', methods=['POST'])
def run_optimization_api():
    """
    Run optimization with parameter grid or Optuna.

    Request body:
    {
        "strategy_id": "s01_trailing_ma",  // NEW
        "mode": "grid",  // or "optuna"
        "csv_files": ["data/..."],
        "enabled_params": {"maLength": true, ...},
        "param_ranges": {"maLength": [30, 60, 5], ...},
        "fixed_params": {"maType": "HMA", ...},
        ...
    }
    """
    try:
        data = request.json

        # Get strategy ID
        strategy_id = data.get('strategy_id', 's01_trailing_ma')

        # Verify strategy exists
        strategy_class = StrategyRegistry.get_strategy_class(strategy_id)

        # Create optimization config
        config = OptimizationConfig(
            csv_files=data['csv_files'],
            strategy_id=strategy_id,  # Pass to config
            enabled_params=data['enabled_params'],
            param_ranges=data['param_ranges'],
            fixed_params=data.get('fixed_params', {}),
            mode=data.get('mode', 'grid'),
            worker_processes=data.get('worker_processes', 6),
            # ... other config fields
        )

        # Run optimization (optimizer_engine handles strategy_id)
        if config.mode == 'grid':
            from optimizer_engine import run_optimization
            results = run_optimization(config)
        else:
            from optuna_engine import run_optuna_optimization
            results = run_optuna_optimization(config)

        # Convert results to CSV
        csv_output = export_results_to_csv(results, config)

        # Return CSV as download
        return Response(
            csv_output,
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename=optimization_{strategy_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            }
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

#### 2.3 Verify OptimizationConfig Has strategy_id Field

Check `optimizer_engine.py` and `optuna_engine.py` to ensure `OptimizationConfig` includes:

```python
@dataclass
class OptimizationConfig:
    csv_files: List[str]
    strategy_id: str = 's01_trailing_ma'  # Should be added in Phase 4
    enabled_params: Dict[str, bool] = field(default_factory=dict)
    param_ranges: Dict[str, Tuple] = field(default_factory=dict)
    # ... rest of fields
```

If not present, add it now.

---

## Step 3: Update Web UI

### File: `src/index.html`

Update the web interface to support strategy selection and dynamic parameters.

#### 3.1 Add Strategy Selector Section

Add this section after the `<h1>` header and before the file upload section:

```html
<!-- Strategy Selection Section -->
<div class="section">
    <h2>1. Select Strategy</h2>
    <div class="form-group">
        <label for="strategySelect">Trading Strategy:</label>
        <select id="strategySelect" class="form-control">
            <option value="">Loading strategies...</option>
        </select>
        <small class="form-text">Choose the trading strategy to backtest or optimize</small>
    </div>

    <div id="strategyInfo" class="alert alert-info" style="display: none; margin-top: 10px;">
        <strong id="strategyName"></strong>
        <p id="strategyDescription"></p>
        <p>
            <strong>Type:</strong> <span id="strategyType"></span> |
            <strong>Parameters:</strong> <span id="strategyParamCount"></span> |
            <strong>Reversal Support:</strong> <span id="strategyReversal"></span>
        </p>
    </div>
</div>
```

#### 3.2 Update Section Numbers

Change all subsequent sections from "1. Upload Data" → "2. Upload Data", etc.

#### 3.3 Add JavaScript to Load Strategies

Add this JavaScript code at the end of the existing `<script>` section:

```javascript
// ========================================
// Strategy Management
// ========================================

let availableStrategies = [];
let currentStrategy = null;

/**
 * Load available strategies from API
 */
async function loadStrategies() {
    try {
        const response = await fetch('/api/strategies');
        if (!response.ok) throw new Error('Failed to load strategies');

        availableStrategies = await response.json();

        // Populate dropdown
        const select = document.getElementById('strategySelect');
        select.innerHTML = availableStrategies.map(s =>
            `<option value="${s.strategy_id}">${s.name}</option>`
        ).join('');

        // Set default to S_01
        select.value = 's01_trailing_ma';
        onStrategyChanged();

    } catch (error) {
        console.error('Error loading strategies:', error);
        alert('Failed to load strategies: ' + error.message);
    }
}

/**
 * Handle strategy selection change
 */
function onStrategyChanged() {
    const strategyId = document.getElementById('strategySelect').value;
    currentStrategy = availableStrategies.find(s => s.strategy_id === strategyId);

    if (!currentStrategy) return;

    // Update strategy info display
    document.getElementById('strategyName').textContent = currentStrategy.name;
    document.getElementById('strategyDescription').textContent = currentStrategy.description;
    document.getElementById('strategyType').textContent = currentStrategy.type;
    document.getElementById('strategyParamCount').textContent = currentStrategy.parameter_count;
    document.getElementById('strategyReversal').textContent = currentStrategy.allows_reversal ? 'Yes' : 'No';
    document.getElementById('strategyInfo').style.display = 'block';

    // Update parameter sections
    updateParameterVisibility();
}

/**
 * Show/hide parameter sections based on current strategy
 */
function updateParameterVisibility() {
    if (!currentStrategy) return;

    const params = currentStrategy.parameters;

    // Example: Hide ATR section if strategy doesn't use ATR
    const hasAtrParams = 'atrPeriod' in params || 'stopLongX' in params;
    const atrSection = document.getElementById('atrSection');
    if (atrSection) {
        atrSection.style.display = hasAtrParams ? 'block' : 'none';
    }

    // Example: Hide trailing section if no trailing params
    const hasTrailingParams = 'trailLongType' in params || 'trailShortType' in params;
    const trailingSection = document.getElementById('trailingSection');
    if (trailingSection) {
        trailingSection.style.display = hasTrailingParams ? 'block' : 'none';
    }

    // Update parameter min/max/default values from param definitions
    for (const [paramName, paramDef] of Object.entries(params)) {
        const input = document.getElementById(paramName);
        if (!input) continue;

        // Update default value
        if (paramDef.type === 'bool') {
            input.checked = paramDef.default;
        } else {
            input.value = paramDef.default;
        }

        // Update min/max for numeric inputs
        if (paramDef.type === 'int' || paramDef.type === 'float') {
            if (paramDef.min !== undefined) input.min = paramDef.min;
            if (paramDef.max !== undefined) input.max = paramDef.max;
        }

        // Update options for select inputs
        if (paramDef.type === 'str' && paramDef.options) {
            if (input.tagName === 'SELECT') {
                input.innerHTML = paramDef.options.map(opt =>
                    `<option value="${opt}" ${opt === paramDef.default ? 'selected' : ''}>${opt}</option>`
                ).join('');
            }
        }
    }
}

// Load strategies on page load
document.addEventListener('DOMContentLoaded', () => {
    loadStrategies();

    // Add event listener to strategy selector
    document.getElementById('strategySelect').addEventListener('change', onStrategyChanged);
});
```

#### 3.4 Update Backtest/Optimization Functions

Modify existing JavaScript functions to include `strategy_id`:

```javascript
/**
 * Run single backtest
 */
async function runBacktest() {
    if (!currentStrategy) {
        alert('Please select a strategy first');
        return;
    }

    const files = document.getElementById('csvFiles').files;
    if (files.length === 0) {
        alert('Please select at least one CSV file');
        return;
    }

    try {
        // Gather all parameter values from form
        const parameters = {};
        for (const paramName in currentStrategy.parameters) {
            const input = document.getElementById(paramName);
            if (!input) continue;

            if (input.type === 'checkbox') {
                parameters[paramName] = input.checked;
            } else if (input.type === 'number') {
                parameters[paramName] = parseFloat(input.value);
            } else {
                parameters[paramName] = input.value;
            }
        }

        // Send backtest request
        const response = await fetch('/api/backtest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                strategy_id: currentStrategy.strategy_id,  // NEW
                csv_file: files[0].name,  // Use first file
                parameters: parameters
            })
        });

        if (!response.ok) throw new Error('Backtest failed');

        const result = await response.json();

        // Display results
        displayBacktestResults(result);

    } catch (error) {
        console.error('Backtest error:', error);
        alert('Backtest failed: ' + error.message);
    }
}

/**
 * Run optimization
 */
async function runOptimization() {
    if (!currentStrategy) {
        alert('Please select a strategy first');
        return;
    }

    // ... existing validation code ...

    try {
        // Gather enabled params and ranges
        const enabledParams = {};
        const paramRanges = {};
        const fixedParams = {};

        for (const paramName in currentStrategy.parameters) {
            const enabledCheckbox = document.getElementById(`enable_${paramName}`);
            const input = document.getElementById(paramName);

            if (!input) continue;

            if (enabledCheckbox && enabledCheckbox.checked) {
                // Parameter is being optimized
                enabledParams[paramName] = true;

                // Get range from UI (min, max, step)
                const minInput = document.getElementById(`${paramName}_min`);
                const maxInput = document.getElementById(`${paramName}_max`);
                const stepInput = document.getElementById(`${paramName}_step`);

                if (minInput && maxInput && stepInput) {
                    paramRanges[paramName] = [
                        parseFloat(minInput.value),
                        parseFloat(maxInput.value),
                        parseFloat(stepInput.value)
                    ];
                }
            } else {
                // Parameter is fixed
                if (input.type === 'checkbox') {
                    fixedParams[paramName] = input.checked;
                } else if (input.type === 'number') {
                    fixedParams[paramName] = parseFloat(input.value);
                } else {
                    fixedParams[paramName] = input.value;
                }
            }
        }

        // Send optimization request
        const response = await fetch('/api/optimize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                strategy_id: currentStrategy.strategy_id,  // NEW
                mode: document.getElementById('optimizationMode').value,
                csv_files: Array.from(files).map(f => f.name),
                enabled_params: enabledParams,
                param_ranges: paramRanges,
                fixed_params: fixedParams,
                worker_processes: parseInt(document.getElementById('workerCount').value)
            })
        });

        if (!response.ok) throw new Error('Optimization failed');

        // Download CSV result
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `optimization_${currentStrategy.strategy_id}_${Date.now()}.csv`;
        a.click();

    } catch (error) {
        console.error('Optimization error:', error);
        alert('Optimization failed: ' + error.message);
    }
}
```

#### 3.5 Add Strategy-Specific Section IDs

Add IDs to parameter sections for visibility control:

```html
<!-- Example: ATR section -->
<div id="atrSection" class="param-section">
    <h3>ATR & Stop Loss</h3>
    <!-- ATR parameters -->
</div>

<!-- Example: Trailing section -->
<div id="trailingSection" class="param-section">
    <h3>Trailing Exits</h3>
    <!-- Trailing parameters -->
</div>
```

---

## Step 4: Testing Through UI

### 4.1 Test S_01 Backtest

1. Start server: `cd src && python server.py`
2. Open browser: `http://localhost:8000`
3. Select "S_01 TrailingMA v26 Ultralight" from dropdown
4. Upload CSV file
5. Click "Run Backtest"
6. **Verify:** Results match CLI baseline

### 4.2 Test S_03 Backtest

1. Select "S_03 Reversal v07 Light" from dropdown
2. **Verify:** ATR section hidden (S_03 doesn't use ATR)
3. **Verify:** Trailing section hidden
4. Upload CSV file
5. Click "Run Backtest"
6. **Verify:** Results match S_03 baseline
7. **Verify:** No flat periods in position history

### 4.3 Test S_01 Optimization

1. Select "S_01 TrailingMA"
2. Enable 2-3 parameters (e.g., maLength, stopLongX)
3. Set small ranges (e.g., maLength: 40-50 step 5)
4. Click "Run Optimization"
5. **Verify:** CSV downloads with results
6. **Verify:** CSV has correct parameter columns
7. **Verify:** Best result matches expectations

### 4.4 Test S_03 Optimization

1. Select "S_03 Reversal"
2. Enable maFastLength (15-30 step 5)
3. Click "Run Optimization"
4. **Verify:** CSV downloads
5. **Verify:** All results have zero flat periods (reversal behavior)
6. **Verify:** Performance metrics reasonable

---

## Step 5: Final Regression Tests

Run all reference tests to ensure nothing broke:

### Test Matrix

| Test | Strategy | Mode | Expected | Status |
|------|----------|------|----------|--------|
| CLI Backtest S_01 | s01_trailing_ma | Single | Baseline matches | [ ] |
| CLI Backtest S_03 | s03_reversal | Single | Baseline matches | [ ] |
| CLI Optimize S_01 | s01_trailing_ma | Grid | Completes | [ ] |
| CLI Optimize S_03 | s03_reversal | Grid | Completes | [ ] |
| UI Backtest S_01 | s01_trailing_ma | Single | Baseline matches | [ ] |
| UI Backtest S_03 | s03_reversal | Single | Baseline matches | [ ] |
| UI Optimize S_01 | s01_trailing_ma | Grid | CSV downloads | [ ] |
| UI Optimize S_03 | s03_reversal | Grid | CSV downloads | [ ] |
| API /strategies | - | - | Returns 2 strategies | [ ] |
| Preset Import S_01 | s01_trailing_ma | - | Imports correctly | [ ] |
| Preset Import S_03 | s03_reversal | - | Imports correctly | [ ] |

### Commands

```bash
# CLI Tests
cd src

# S_01 baseline
python run_backtest.py --csv ../data/"OKX_LINKUSDT.P, 15 2025.05.01-2025.11.20.csv" --strategy s01_trailing_ma

# S_03 baseline
python run_backtest.py --csv ../data/"OKX_LINKUSDT.P, 15 2025.05.01-2025.11.20.csv" --strategy s03_reversal

# S_01 small optimization
python -c "
from optimizer_engine import run_optimization, OptimizationConfig
config = OptimizationConfig(
    csv_files=[../data/"OKX_LINKUSDT.P, 15 2025.05.01-2025.11.20.csv"],
    strategy_id='s01_trailing_ma',
    enabled_params={'maLength': True},
    param_ranges={'maLength': (40, 50, 5)},
    worker_processes=2
)
results = run_optimization(config)
print(f'✅ {len(results)} results')
"

# S_03 small optimization
python -c "
from optimizer_engine import run_optimization, OptimizationConfig
config = OptimizationConfig(
    csv_files=[../data/"OKX_LINKUSDT.P, 15 2025.05.01-2025.11.20.csv"],
    strategy_id='s03_reversal',
    enabled_params={'maFastLength': True},
    param_ranges={'maFastLength': (15, 25, 5)},
    worker_processes=2
)
results = run_optimization(config)
print(f'✅ {len(results)} results')
"
```

All tests must pass before proceeding to commit.

---

## Step 6: Update Documentation

### File: `CLAUDE.md`

Update the project documentation to reflect multi-strategy support:

#### Add to "Project Overview" section:

```markdown
## Multi-Strategy Architecture (v2.0)

As of Phase 6, the platform supports multiple trading strategies:

1. **S_01 TrailingMA v26 Ultralight**: Trend-following strategy with MA crossovers, trailing stops, and ATR-based risk management (28 parameters)

2. **S_03 Reversal v07 Light**: Reversal strategy that is always in the market, switching between long and short on MA crossovers (12 parameters)

### Adding New Strategies

To add a new strategy:

1. Create `src/strategies/your_strategy.py` extending `BaseStrategy`
2. Implement abstract methods: `should_long()`, `should_short()`, `calculate_entry()`, `calculate_position_size()`, `should_exit()`
3. Define `get_param_definitions()` with all parameters
4. Define `get_cache_requirements()` for optimization caching
5. Register in `src/strategy_registry.py`
6. Establish reference test baseline in `info/tests.md`

See `info/migration_prompt_5.md` for detailed example.
```

#### Update "API Endpoints" section:

```markdown
### API Endpoints

- `GET /` → Serves web UI
- `GET /api/strategies` → List all available strategies with parameters
- `POST /api/backtest` → Single backtest (requires `strategy_id` field)
- `POST /api/optimize` → Optimization (requires `strategy_id` field)
- `GET /api/presets` → List saved presets for all strategies
- `POST /api/presets` → Create new preset (strategy-specific)
- `PUT /api/presets/<name>` → Update existing preset
- `DELETE /api/presets/<name>` → Delete preset
- `POST /api/presets/import-csv` → Import parameters from optimization CSV
```

---

## Step 7: Commit

### Commit Message

```
feat: Add multi-strategy UI and API support (Phase 6)

Completes multi-strategy migration with full UI/API integration.

Changes:
- Add GET /api/strategies endpoint (returns strategy list with params)
- Update POST /api/backtest to accept strategy_id parameter
- Update POST /api/optimize to accept strategy_id parameter
- Add strategy selector dropdown to web UI
- Make parameter forms dynamic based on get_param_definitions()
- Hide/show parameter sections based on strategy (e.g., ATR only for S_01)
- Update CLAUDE.md with multi-strategy documentation

Testing:
- S_01 CLI backtest: ✅ matches baseline
- S_03 CLI backtest: ✅ matches baseline
- S_01 UI backtest: ✅ matches baseline
- S_03 UI backtest: ✅ matches baseline
- S_01 optimization: ✅ CSV exports correctly
- S_03 optimization: ✅ CSV exports correctly
- /api/strategies: ✅ returns 2 strategies
- Parameter visibility: ✅ S_03 hides ATR/trailing sections

Phase 6 of 6 complete. Multi-strategy migration finished.

Migration Statistics:
- Duration: [X days]
- Strategies added: 2 (S_01, S_03)
- Lines of code refactored: ~1500
- New files created: 8
- All reference tests passing: ✅
```

### Files Changed

```
M  src/server.py
M  src/index.html
M  CLAUDE.md
```

---

## Common Issues and Solutions

### Issue 1: /api/strategies Returns Empty Array

**Cause:** StrategyRegistry not importing strategy classes

**Fix:**
```python
# In strategy_registry.py
from strategies.s01_trailing_ma import S01TrailingMA
from strategies.s03_reversal import S03Reversal  # Make sure this is imported

_strategies = {
    's01_trailing_ma': S01TrailingMA,
    's03_reversal': S03Reversal  # Make sure this is registered
}
```

### Issue 2: Parameters Not Updating When Switching Strategies

**Cause:** JavaScript not reading parameter definitions correctly

**Fix:**
- Check `currentStrategy.parameters` contains all params
- Verify input IDs match parameter names exactly (case-sensitive)
- Add console.log() to debug which params are found

### Issue 3: Optimization Fails with "Unknown strategy"

**Cause:** Frontend sending wrong strategy_id format

**Fix:**
- Verify `strategy_id` field in POST body matches registry key
- Check network tab in browser devtools for actual request
- Ensure no typos in strategy IDs

### Issue 4: S_03 Shows ATR Parameters

**Cause:** Section visibility logic not working

**Fix:**
- Ensure sections have correct IDs (`atrSection`, `trailingSection`)
- Check `updateParameterVisibility()` is called on strategy change
- Verify `currentStrategy.parameters` doesn't include ATR params for S_03

---

## Acceptance Criteria

Phase 6 is complete when:

1. ✅ `/api/strategies` endpoint returns both strategies
2. ✅ `/api/backtest` accepts and uses `strategy_id`
3. ✅ `/api/optimize` accepts and uses `strategy_id`
4. ✅ UI has strategy selector dropdown
5. ✅ UI shows/hides parameters based on strategy
6. ✅ S_01 works through UI (backtest + optimize)
7. ✅ S_03 works through UI (backtest + optimize)
8. ✅ All CLI tests still pass
9. ✅ Documentation updated
10. ✅ Changes committed and pushed

---

## Post-Migration Tasks

After Phase 6 is complete, consider:

1. **Add more strategies**: Use Phase 5 as template for S_04, S_05, etc.
2. **Improve UI**: Add charts, real-time progress, parameter presets per strategy
3. **Add tests**: Unit tests for each strategy, integration tests for API
4. **Performance tuning**: Profile optimization with multiple strategies
5. **Documentation**: Write user guide for adding new strategies

---

## Migration Complete! 🎉

All 6 phases are now finished. The codebase has been successfully migrated from a single hardcoded strategy to a flexible multi-strategy architecture.

**Key Achievements:**

- ✅ Eliminated code duplication (single `simulate()` entry point)
- ✅ Dynamic parameter system (auto-generated UI)
- ✅ Dynamic caching (strategy-specific optimization)
- ✅ Clean architecture (ABC contracts, registry pattern)
- ✅ Backward compatible (existing code still works)
- ✅ Reference tests passing (no behavior changes)
- ✅ Two working strategies (S_01, S_03)

**Time to add new strategies:** ~2-4 hours per strategy

**Maintenance effort:** Minimal (well-structured, documented)

---

**End of migration_prompt_6.md**
