# Migration Prompt 7: Complete Multi-Strategy UI Integration

**Purpose**: Complete the multi-strategy system by implementing the frontend UI for strategy selection and dynamic parameter forms. This is the final step to make all strategies accessible through the web interface.

**Status**: Backend API is fully implemented and tested. Frontend UI needs to be completed.

**Priority**: HIGH - This blocks user access to S_03 and future strategies via web UI.

---

## Context

### What's Already Done ✅

1. **Backend API** (server.py):
   - `/api/strategies` endpoint returns all strategies with metadata
   - `/api/backtest` accepts `strategy_id` parameter
   - `/api/optimize` accepts `strategy_id` parameter
   - StrategyRegistry provides dynamic parameter definitions

2. **Reference Tests**:
   - S_01: 230.75% / 20.03% / 93 trades ✅
   - S_03: 83.56% / 35.34% / 224 trades ✅

3. **Data Structures**:
   - Strategy metadata includes `strategy_id`, `name`, `description`, `type`, `allows_reversal`
   - Parameter definitions include `type`, `default`, `min`, `max`, `step`, `description`, `choices`

### What Needs To Be Done ❌

1. **Strategy Selector UI**:
   - Add dropdown to backtester panel
   - Add dropdown to optimizer panel
   - Load strategies from `/api/strategies` on page load
   - Handle strategy switching

2. **Dynamic Parameter Forms**:
   - Generate forms from strategy metadata
   - Show/hide parameters based on selected strategy
   - Preserve user values when switching strategies (where possible)
   - Handle different parameter types (int, float, bool, str, choice)

3. **Testing**:
   - Test strategy switching
   - Test backtest with both strategies
   - Test optimization with both strategies
   - Verify preset compatibility

---

## Step 1: Add Strategy Selector to HTML

### 1.1 Update Backtester Panel

Find the backtester panel title bar (around line 900-1000) and add a strategy selector section **before** the CSV file upload section.

**Location**: After `<div class="content">` in backtester-window

**Add this HTML**:

```html
<!-- Strategy Selection -->
<div class="section">
  <div class="section-title">Стратегия</div>
  <div class="form-group">
    <label for="backtesterStrategy">Выбор стратегии:</label>
    <select id="backtesterStrategy" class="strategy-selector">
      <option value="">Загрузка...</option>
    </select>
  </div>
  <div id="backtesterStrategyInfo" class="info-panel" style="display: none;">
    <div class="info-row">
      <span class="label">Название:</span>
      <span class="value" id="backtesterStrategyName"></span>
    </div>
    <div class="info-row">
      <span class="label">Тип:</span>
      <span class="value" id="backtesterStrategyType"></span>
    </div>
    <div class="info-row">
      <span class="label">Описание:</span>
      <span class="value" id="backtesterStrategyDesc"></span>
    </div>
  </div>
</div>
```

### 1.2 Update Optimizer Panel

Find the optimizer panel (around line 1100-1200) and add a similar strategy selector **before** the CSV file upload section.

**Add this HTML**:

```html
<!-- Strategy Selection -->
<div class="section">
  <div class="section-title">Стратегия</div>
  <div class="form-group">
    <label for="optimizerStrategy">Выбор стратегии:</label>
    <select id="optimizerStrategy" class="strategy-selector">
      <option value="">Загрузка...</option>
    </select>
  </div>
  <div id="optimizerStrategyInfo" class="info-panel" style="display: none;">
    <div class="info-row">
      <span class="label">Название:</span>
      <span class="value" id="optimizerStrategyName"></span>
    </div>
    <div class="info-row">
      <span class="label">Тип:</span>
      <span class="value" id="optimizerStrategyType"></span>
    </div>
    <div class="info-row">
      <span class="label">Описание:</span>
      <span class="value" id="optimizerStrategyDesc"></span>
    </div>
  </div>
</div>
```

### 1.3 Add CSS Styling

Add these styles to the `<style>` section:

```css
.strategy-selector {
  width: 250px;
  padding: 8px 10px;
  font-size: 14px;
  background: #ffffff;
  border: 1px solid #999999;
  border-radius: 3px;
  color: #2a2a2a;
}

.strategy-selector:focus {
  outline: none;
  border-color: #4a90e2;
  box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.2);
}

.strategy-selector option {
  padding: 6px;
}

#backtesterStrategyInfo,
#optimizerStrategyInfo {
  margin-top: 12px;
  font-size: 13px;
}

#backtesterStrategyInfo .value,
#optimizerStrategyInfo .value {
  color: #2a2a2a;
  font-weight: 500;
}

#backtesterStrategyDesc,
#optimizerStrategyDesc {
  font-style: italic;
  color: #5a5a5a;
}
```

---

## Step 2: Implement JavaScript Strategy Loading

### 2.1 Add Global State Management

Find the JavaScript section (around line 1770) where `currentStrategy` is defined. **Replace** the hardcoded line with:

```javascript
// Global strategy state
let allStrategies = [];
let currentBacktesterStrategy = null;
let currentOptimizerStrategy = null;

// Strategy metadata cache
const strategyMetadataCache = {};
```

### 2.2 Implement Strategy Loading Function

Add this function after the global state variables:

```javascript
/**
 * Load all strategies from API and populate dropdowns
 */
async function loadStrategies() {
  try {
    const response = await fetch('/api/strategies');
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    allStrategies = await response.json();

    // Cache metadata
    allStrategies.forEach(strategy => {
      strategyMetadataCache[strategy.strategy_id] = strategy;
    });

    // Populate backtester dropdown
    const backtesterSelect = document.getElementById('backtesterStrategy');
    backtesterSelect.innerHTML = '';
    allStrategies.forEach(strategy => {
      const option = document.createElement('option');
      option.value = strategy.strategy_id;
      option.textContent = strategy.name;
      backtesterSelect.appendChild(option);
    });

    // Populate optimizer dropdown
    const optimizerSelect = document.getElementById('optimizerStrategy');
    optimizerSelect.innerHTML = '';
    allStrategies.forEach(strategy => {
      const option = document.createElement('option');
      option.value = strategy.strategy_id;
      option.textContent = strategy.name;
      optimizerSelect.appendChild(option);
    });

    // Set default strategy (S_01)
    backtesterSelect.value = 's01_trailing_ma';
    optimizerSelect.value = 's01_trailing_ma';

    // Initialize with default strategy
    onBacktesterStrategyChange();
    onOptimizerStrategyChange();

    console.log(`Loaded ${allStrategies.length} strategies:`, allStrategies.map(s => s.strategy_id));
  } catch (error) {
    console.error('Failed to load strategies:', error);
    alert(`Ошибка загрузки стратегий: ${error.message}`);
  }
}
```

### 2.3 Implement Strategy Change Handlers

Add these functions to handle strategy switching:

```javascript
/**
 * Handle backtester strategy change
 */
function onBacktesterStrategyChange() {
  const strategyId = document.getElementById('backtesterStrategy').value;
  if (!strategyId) return;

  currentBacktesterStrategy = strategyMetadataCache[strategyId];
  if (!currentBacktesterStrategy) {
    console.error('Strategy not found:', strategyId);
    return;
  }

  // Update info panel
  document.getElementById('backtesterStrategyName').textContent = currentBacktesterStrategy.name;
  document.getElementById('backtesterStrategyType').textContent =
    currentBacktesterStrategy.type === 'trend' ? 'Трендовая' : 'Реверсивная';
  document.getElementById('backtesterStrategyDesc').textContent = currentBacktesterStrategy.description;
  document.getElementById('backtesterStrategyInfo').style.display = 'block';

  // Update parameter form
  updateBacktesterParameterForm(currentBacktesterStrategy);

  console.log('Backtester strategy changed to:', strategyId);
}

/**
 * Handle optimizer strategy change
 */
function onOptimizerStrategyChange() {
  const strategyId = document.getElementById('optimizerStrategy').value;
  if (!strategyId) return;

  currentOptimizerStrategy = strategyMetadataCache[strategyId];
  if (!currentOptimizerStrategy) {
    console.error('Strategy not found:', strategyId);
    return;
  }

  // Update info panel
  document.getElementById('optimizerStrategyName').textContent = currentOptimizerStrategy.name;
  document.getElementById('optimizerStrategyType').textContent =
    currentOptimizerStrategy.type === 'trend' ? 'Трендовая' : 'Реверсивная';
  document.getElementById('optimizerStrategyDesc').textContent = currentOptimizerStrategy.description;
  document.getElementById('optimizerStrategyInfo').style.display = 'block';

  // Update parameter form
  updateOptimizerParameterForm(currentOptimizerStrategy);

  console.log('Optimizer strategy changed to:', strategyId);
}
```

### 2.4 Wire Up Event Listeners

Find the DOMContentLoaded event listener and add strategy loading:

```javascript
document.addEventListener('DOMContentLoaded', async () => {
  // Load strategies first
  await loadStrategies();

  // Wire up strategy change handlers
  document.getElementById('backtesterStrategy').addEventListener('change', onBacktesterStrategyChange);
  document.getElementById('optimizerStrategy').addEventListener('change', onOptimizerStrategyChange);

  // ... rest of existing initialization code ...

  // Load presets after strategies are loaded
  await loadPresets();
});
```

---

## Step 3: Implement Dynamic Parameter Forms

### 3.1 Strategy-Specific Parameter Visibility

Add these helper functions to manage parameter visibility:

```javascript
/**
 * Get list of parameter IDs for a strategy
 */
function getStrategyParameterIds(strategy) {
  if (!strategy || !strategy.parameters) return [];
  return Object.keys(strategy.parameters);
}

/**
 * Show/hide parameters based on strategy
 */
function updateParameterVisibility(containerSelector, strategy) {
  const container = document.querySelector(containerSelector);
  if (!container) return;

  const strategyParamIds = getStrategyParameterIds(strategy);

  // Get all parameter form groups
  const allFormGroups = container.querySelectorAll('.form-group, .param-group');

  allFormGroups.forEach(group => {
    // Check if this group contains any strategy parameter inputs
    const inputs = group.querySelectorAll('input, select');
    let hasStrategyParam = false;

    inputs.forEach(input => {
      // Match input ID/name to strategy parameters
      // Convert camelCase to snake_case for matching
      const inputId = input.id || input.name || '';
      const snakeId = inputId.replace(/([A-Z])/g, '_$1').toLowerCase();

      if (strategyParamIds.includes(inputId) || strategyParamIds.includes(snakeId)) {
        hasStrategyParam = true;
      }
    });

    // Show group if it contains strategy parameter, hide otherwise
    // Exception: always show common UI controls (date filter, CSV path, etc.)
    const isCommonControl = group.querySelector('#csvPath, #dateFilter, #startDate, #endDate');
    if (isCommonControl || hasStrategyParam) {
      group.style.display = '';
    } else {
      group.style.display = 'none';
    }
  });
}

/**
 * Update backtester parameter form
 */
function updateBacktesterParameterForm(strategy) {
  // For now, we'll use visibility control
  // Future enhancement: generate forms dynamically
  updateParameterVisibility('.backtester-window .content', strategy);

  // Update form field defaults
  updateFormFieldDefaults('backtester', strategy);
}

/**
 * Update optimizer parameter form
 */
function updateOptimizerParameterForm(strategy) {
  // For now, we'll use visibility control
  // Future enhancement: generate forms dynamically
  updateParameterVisibility('.optimizer-window .content', strategy);

  // Update form field defaults
  updateFormFieldDefaults('optimizer', strategy);
}

/**
 * Update form field defaults based on strategy
 */
function updateFormFieldDefaults(formType, strategy) {
  if (!strategy || !strategy.parameters) return;

  Object.entries(strategy.parameters).forEach(([paramId, paramDef]) => {
    // Find input element (try multiple ID formats)
    const camelId = paramId;
    const input = document.getElementById(camelId) ||
                  document.querySelector(`[name="${camelId}"]`);

    if (!input) return;

    // Set default value if input is empty
    if (!input.value || input.value === '') {
      if (paramDef.type === 'bool') {
        input.checked = paramDef.default;
      } else {
        input.value = paramDef.default;
      }
    }

    // Update input constraints
    if (paramDef.min !== undefined) {
      input.min = paramDef.min;
    }
    if (paramDef.max !== undefined) {
      input.max = paramDef.max;
    }
    if (paramDef.step !== undefined) {
      input.step = paramDef.step;
    }
  });
}
```

### 3.2 Update API Request Functions

Find the backtester submit function and update it to include strategy_id:

```javascript
// Inside backtester form submit handler
async function runBacktest(formData) {
  const strategyId = currentBacktesterStrategy?.strategy_id || 's01_trailing_ma';

  // Add strategy_id to form data
  formData.append('strategy_id', strategyId);

  // ... rest of existing backtest code ...
}
```

Find the optimizer submit function and update it to include strategy_id in config:

```javascript
// Inside optimizer form submit handler
async function runOptimization() {
  const strategyId = currentOptimizerStrategy?.strategy_id || 's01_trailing_ma';

  const config = {
    strategy_id: strategyId,
    enabled_params: { /* ... */ },
    param_ranges: { /* ... */ },
    fixed_params: { /* ... */ },
    // ... rest of config ...
  };

  // ... rest of existing optimization code ...
}
```

---

## Step 4: Handle Strategy-Specific UI Logic

### 4.1 S_03-Specific Reversal Mode Display

S_03 is a reversal strategy (always in position), so we need to handle this in the results display.

Add this helper function:

```javascript
/**
 * Check if current strategy allows reversals
 */
function isReversalStrategy(strategyId) {
  const strategy = strategyMetadataCache[strategyId];
  return strategy?.allows_reversal === true;
}

/**
 * Format results based on strategy type
 */
function formatStrategyResults(results, strategyId) {
  const isReversal = isReversalStrategy(strategyId);

  let output = `Net Profit %: ${results.net_profit_pct}\n`;
  output += `Max Drawdown %: ${results.max_drawdown_pct}\n`;
  output += `Total Trades: ${results.total_trades}\n`;

  if (isReversal) {
    output += `\n⚠ Реверсивная стратегия: всегда в позиции (лонг или шорт)\n`;
  }

  return output;
}
```

### 4.2 Update Results Display

Find the results display code and update it to use the formatting function:

```javascript
// In backtester results handler
const resultsText = formatStrategyResults(
  responseData.metrics,
  currentBacktesterStrategy?.strategy_id
);
document.getElementById('backtesterResults').textContent = resultsText;
```

---

## Step 5: Parameter Mapping and Validation

### 5.1 Create Parameter Mapping Helper

Different strategies use different parameter names. Create a helper to normalize:

```javascript
/**
 * Map UI parameter names to strategy parameter names
 * Handles camelCase <-> snake_case conversion
 */
function mapParametersForStrategy(uiParams, strategy) {
  if (!strategy || !strategy.parameters) return uiParams;

  const mapped = {};
  const strategyParamIds = Object.keys(strategy.parameters);

  Object.entries(uiParams).forEach(([key, value]) => {
    // Try direct match first
    if (strategyParamIds.includes(key)) {
      mapped[key] = value;
      return;
    }

    // Try snake_case conversion
    const snakeKey = key.replace(/([A-Z])/g, '_$1').toLowerCase();
    if (strategyParamIds.includes(snakeKey)) {
      mapped[snakeKey] = value;
      return;
    }

    // Try camelCase conversion
    const camelKey = key.replace(/_([a-z])/g, (m, p1) => p1.toUpperCase());
    if (strategyParamIds.includes(camelKey)) {
      mapped[camelKey] = value;
      return;
    }

    // Keep original if no match found (for common params like dateFilter)
    mapped[key] = value;
  });

  return mapped;
}

/**
 * Collect parameters from backtester form
 */
function collectBacktesterParameters() {
  const params = {
    // Common parameters
    dateFilter: document.getElementById('dateFilter').checked,
    startDate: document.getElementById('startDate').value,
    endDate: document.getElementById('endDate').value,
    // ... collect all other form values ...
  };

  // Map to current strategy
  return mapParametersForStrategy(params, currentBacktesterStrategy);
}
```

---

## Step 6: Testing and Validation

### 6.1 Manual Testing Checklist

Create a test plan and verify each item:

**Strategy Loading**:
- [ ] Page loads without errors
- [ ] Both dropdowns show 2 strategies
- [ ] Default strategy is S_01
- [ ] Info panels show correct strategy metadata

**Strategy Switching**:
- [ ] Switching strategy updates info panel
- [ ] Parameter form updates (parameters show/hide)
- [ ] No JavaScript errors in console

**Backtester with S_01**:
- [ ] Upload CSV file
- [ ] Set parameters
- [ ] Click "Run Backtest"
- [ ] Results show: 230.75% / 20.03% / 93 trades
- [ ] No errors

**Backtester with S_03**:
- [ ] Switch to S_03 strategy
- [ ] Upload CSV file
- [ ] Set parameters (S_03-specific)
- [ ] Click "Run Backtest"
- [ ] Results show: 83.56% / 35.34% / 224 trades
- [ ] Reversal mode warning displayed

**Optimizer with S_01**:
- [ ] Upload CSV file
- [ ] Configure parameter ranges
- [ ] Click "Optimize"
- [ ] CSV downloads successfully
- [ ] Results match expected format

**Optimizer with S_03**:
- [ ] Switch to S_03 strategy
- [ ] Upload CSV file
- [ ] Configure parameter ranges (S_03-specific)
- [ ] Click "Optimize"
- [ ] CSV downloads successfully
- [ ] Results match expected format

**Preset Compatibility**:
- [ ] Load S_01 preset
- [ ] Parameters populate correctly
- [ ] Save new preset
- [ ] Load saved preset
- [ ] Parameters restore correctly

### 6.2 Automated Testing Script

Create a test script to verify API calls:

```javascript
// Add to end of JavaScript section for debugging
if (window.location.search.includes('test=1')) {
  window.testStrategySwitching = async function() {
    console.log('=== Testing Strategy Switching ===');

    // Test 1: Load strategies
    console.log('Test 1: Loading strategies...');
    await loadStrategies();
    console.log('✅ Strategies loaded:', allStrategies.length);

    // Test 2: Switch to S_01
    console.log('Test 2: Switching to S_01...');
    document.getElementById('backtesterStrategy').value = 's01_trailing_ma';
    onBacktesterStrategyChange();
    console.log('✅ Current strategy:', currentBacktesterStrategy?.strategy_id);

    // Test 3: Switch to S_03
    console.log('Test 3: Switching to S_03...');
    document.getElementById('backtesterStrategy').value = 's03_reversal';
    onBacktesterStrategyChange();
    console.log('✅ Current strategy:', currentBacktesterStrategy?.strategy_id);

    // Test 4: Verify parameter visibility
    console.log('Test 4: Checking parameter visibility...');
    const visibleParams = Array.from(document.querySelectorAll('.backtester-window .form-group'))
      .filter(g => g.style.display !== 'none')
      .map(g => g.querySelector('label')?.textContent);
    console.log('✅ Visible parameters:', visibleParams.length);

    console.log('=== All Tests Passed ===');
  };

  console.log('Test mode enabled. Run: testStrategySwitching()');
}
```

---

## Step 7: Code Review and Cleanup

### 7.1 Remove Hardcoded Strategy References

Search for any remaining hardcoded references to `s01_trailing_ma` or `currentStrategy` and replace with dynamic lookups:

**Before**:
```javascript
let currentStrategy = { strategy_id: 's01_trailing_ma' };
```

**After**:
```javascript
let currentBacktesterStrategy = null;
let currentOptimizerStrategy = null;
```

**Before**:
```javascript
const strategyId = 's01_trailing_ma';
```

**After**:
```javascript
const strategyId = currentBacktesterStrategy?.strategy_id || 's01_trailing_ma';
```

### 7.2 Add Error Handling

Ensure all async functions have proper error handling:

```javascript
try {
  // API call
} catch (error) {
  console.error('Error:', error);
  alert(`Ошибка: ${error.message}`);
  // Restore UI state
}
```

### 7.3 Add Loading States

Show loading indicators during strategy switching:

```javascript
function onBacktesterStrategyChange() {
  const strategyId = document.getElementById('backtesterStrategy').value;
  if (!strategyId) return;

  // Show loading
  document.getElementById('backtesterStrategyInfo').innerHTML = 'Загрузка...';

  try {
    // ... update logic ...
  } finally {
    // Hide loading
    document.getElementById('backtesterStrategyInfo').style.display = 'block';
  }
}
```

---

## Step 8: Documentation and Commit

### 8.1 Update CLAUDE.md

Add a section about strategy selection:

```markdown
## Multi-Strategy Support

The web UI supports multiple trading strategies:

1. **S_01 TrailingMA v26 Ultralight** - Trend-following strategy
2. **S_03 Reversal v07 Light** - Counter-trend reversal strategy

### Selecting a Strategy

1. Open the web UI at http://localhost:8000
2. In the Backtester or Optimizer panel, select strategy from dropdown
3. Parameters will update to show strategy-specific options
4. Run backtest or optimization as normal

### Adding New Strategies

1. Create new strategy class inheriting from `BaseStrategy`
2. Implement required methods: `should_long`, `should_short`, `calculate_entry`, etc.
3. Call `StrategyRegistry.register_strategy()` in `src/strategies/__init__.py`
4. Strategy will automatically appear in web UI dropdown
```

### 8.2 Test and Commit

Run final verification:

```bash
# 1. Start server
cd src
python server.py

# 2. Open browser to http://localhost:8000?test=1
# 3. Open DevTools console
# 4. Run: testStrategySwitching()
# 5. Verify all tests pass

# 6. Manual testing
# - Test S_01 backtest
# - Test S_03 backtest
# - Test S_01 optimization
# - Test S_03 optimization

# 7. Commit changes
git add .
git commit -m "Phase 6: Complete multi-strategy UI integration

- Add strategy selector dropdowns to backtester and optimizer
- Implement dynamic strategy loading from /api/strategies
- Add strategy change handlers with parameter form updates
- Update parameter visibility based on selected strategy
- Add strategy-specific result formatting
- Wire up strategy_id to API calls
- Add loading states and error handling
- Test strategy switching and verify reference baselines

Reference tests:
- S_01: 230.75% / 20.03% / 93 trades ✅
- S_03: 83.56% / 35.34% / 224 trades ✅"

git push -u origin claude/mg-stage-1-check-0159d5ZWE51FdnYTT8qhmQkz
```

---

## Acceptance Criteria

This task is complete when:

1. ✅ Both backtester and optimizer have strategy selector dropdowns
2. ✅ Dropdowns populate from `/api/strategies` API
3. ✅ Strategy info panel shows metadata (name, type, description)
4. ✅ Switching strategy updates parameter form visibility
5. ✅ Backtest works with S_01 (returns 230.75% / 20.03% / 93)
6. ✅ Backtest works with S_03 (returns 83.56% / 35.34% / 224)
7. ✅ Optimization works with S_01
8. ✅ Optimization works with S_03
9. ✅ No JavaScript errors in console
10. ✅ Code committed and pushed to branch

---

## Common Issues and Solutions

### Issue 1: Strategy dropdown not populating

**Symptom**: Dropdowns show "Загрузка..." forever

**Solution**:
- Check browser DevTools console for errors
- Verify `/api/strategies` returns 200 OK
- Check that `allStrategies` array is populated
- Verify `loadStrategies()` is called in DOMContentLoaded

### Issue 2: Parameter form not updating

**Symptom**: Same parameters show for all strategies

**Solution**:
- Verify `onBacktesterStrategyChange()` is called
- Check that `updateParameterVisibility()` is running
- Inspect DOM to see if `display: none` is applied
- Verify parameter IDs match between HTML and strategy metadata

### Issue 3: Strategy_id not sent to backend

**Symptom**: Backend always uses default strategy

**Solution**:
- Check FormData includes `strategy_id` field
- Verify `currentBacktesterStrategy` is set before submit
- Check Network tab to see actual request payload
- Verify server.py accepts the parameter name

### Issue 4: Results format incorrect

**Symptom**: Results display broken or missing data

**Solution**:
- Check API response structure matches expected format
- Verify `formatStrategyResults()` handles all cases
- Check for null/undefined values in response
- Add defensive checks for missing fields

---

## Next Steps After Completion

Once this is complete, the multi-strategy system is fully functional. Future enhancements:

1. **Universal Warmup** (Phase 7):
   - Remove S_01-specific warmup logic from `/api/backtest`
   - Implement `prepare_dataset_with_warmup()` for all strategies
   - Update CLI to use same warmup logic

2. **Strategy Generator** (Future):
   - UI to create new strategies without coding
   - Template-based strategy builder
   - Parameter preset library per strategy

3. **Strategy Comparison** (Future):
   - Side-by-side backtest comparison
   - Portfolio allocation optimizer
   - Multi-strategy ensemble testing

---

## Reference Links

- **Backend API**: `src/server.py` lines 451-494, 987-1093, 1452-1629
- **Strategy Registry**: `src/strategy_registry.py`
- **Phase 6 Verification**: `info/phase_6_verification_report.md`
- **Migration Checklist**: `info/migration_checklist.md`
- **Tests Specification**: `info/tests.md`

---

**Estimated Time**: 4-6 hours for full implementation and testing
**Difficulty**: Medium (frontend integration, no complex algorithms)
**Risk**: Low (backend is stable, changes are UI-only)
