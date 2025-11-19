# Migration Prompts 4, 5, 6 (Combined)

---

## PROMPT 4: Refactor Optimizer for Multi-Strategy

**Phase:** 4 of 6
**Duration:** 2-3 days
**Goal:** Make optimizer work with any strategy via StrategyRegistry

### Key Changes

1. **Update OptimizationConfig:**
   ```python
   @dataclass
   class OptimizationConfig:
       csv_file: IO[Any]
       strategy_id: str = "s01_trailing_ma"  # NEW
       # ... rest unchanged
   ```

2. **Refactor run_optimization():**
   ```python
   def run_optimization(config):
       strategy_class = StrategyRegistry.get_strategy_class(config.strategy_id)
       param_defs = strategy_class.get_param_definitions()
       combinations = _generate_grid(config, param_defs)
       cache_req = strategy_class.get_cache_requirements(combinations)
       
       pool = mp.Pool(initializer=_init_worker,
                      initargs=(df, cache_req, strategy_class, config))
       results = pool.map(_simulate_combination, combinations)
       return results
   ```

3. **Refactor _init_worker():**
   ```python
   def _init_worker(df, cache_req, strategy_class, config):
       global _df, _strategy_class, _cached_data
       
       _df = df
       _strategy_class = strategy_class
       _cached_data = {}
       
       # Pre-compute based on cache_req
       if 'ma_specs' in cache_req:
           _cached_data['ma_specs'] = {}
           for ma_type, length in cache_req['ma_specs']:
               ma = get_ma(df['Close'], ma_type, length, ...).to_numpy()
               _cached_data['ma_specs'][(ma_type, length)] = ma
       
       if 'atr_periods' in cache_req:
           _cached_data['atr'] = {}
           for period in cache_req['atr_periods']:
               _cached_data['atr'][period] = atr(...).to_numpy()
       
       # Similar for lowest/highest
   ```

4. **Refactor _simulate_combination():**
   ```python
   def _simulate_combination(params_dict):
       global _df, _strategy_class, _cached_data
       
       strategy = _strategy_class(params_dict)
       result = strategy.simulate(_df, cached_data=_cached_data)
       
       return OptimizationResult(
           net_profit_pct=result['net_profit_pct'],
           **params_dict
       )
   ```

### Testing
- Small optimization with S_01
- Verify CSV export
- Reference test

### Commit
```
Phase 4: Refactor optimizer for multi-strategy
- Add strategy_id to OptimizationConfig
- Dynamic parameter grid from strategy.get_param_definitions()
- Dynamic caching from strategy.get_cache_requirements()
- S_01 optimization works ✅
```

---

## PROMPT 5: Add S_03 Reversal Strategy

**Phase:** 5 of 6
**Duration:** 2-3 days
**Goal:** Translate S_03 Pine script and validate multi-strategy architecture

### Key Tasks

1. **Read Pine Script:**
   - Study `/data/S_03 Reversal_v07 Light for PROJECT PLAN.pine`
   - Note: NO stops/targets, always in market, reversal on opposite signal

2. **Create s03_reversal.py:**
   ```python
   class S03Reversal(BaseStrategy):
       STRATEGY_ID = "s03_reversal"
       STRATEGY_NAME = "S_03 Reversal v07 Light"
       VERSION = "07"
       
       def allows_reversal(self):
           return True  # Key difference!
   ```

3. **Implement Parameters (~12 total):**
   - MA1, MA2, MA3: type, length, enable flag
   - Close count: enable, count_long, count_short
   - Breakout mode, use_close_price
   - Contract size

4. **Implement Methods:**
   
   **should_long:**
   ```python
   def should_long(self, idx):
       c = self.close[idx]
       ma3 = self._ma3[idx]
       
       # Update counters
       if c > ma3:
           self.counter_close_long += 1
           self.counter_close_short = 0
       elif c < ma3:
           self.counter_close_short += 1
           self.counter_close_long = 0
       
       # Check conditions
       count_ok = True
       if self.params['use_close_count']:
           count_ok = self.counter_close_long >= self.params['close_count_long']
       
       # MA confirmation (if enabled)
       ma_ok = True
       if self.params['use_ma1'] and self.params['use_ma2']:
           ma_ok = self._ma1[idx] > self._ma2[idx]
       
       return count_ok and ma_ok
   ```
   
   **calculate_entry:**
   ```python
   def calculate_entry(self, idx, direction):
       # Reversal strategy: no stops/targets
       return (self.close[idx], math.nan, math.nan)
   ```
   
   **calculate_position_size:**
   ```python
   def calculate_position_size(self, idx, direction, entry_price, stop_price, equity):
       # 100% of equity
       qty = equity / entry_price
       contract_size = self.params['contract_size']
       if contract_size > 0:
           qty = math.floor(qty / contract_size) * contract_size
       return qty
   ```
   
   **should_exit:**
   ```python
   def should_exit(self, idx, position_info):
       # No regular exits - only reversal (handled by _run_simulation)
       return (False, None, '')
   ```

5. **Register:**
   ```python
   # strategy_registry.py
   from strategies.s03_reversal import S03Reversal
   
   _strategies = {
       "s01_trailing_ma": S01TrailingMA,
       "s03_reversal": S03Reversal,
   }
   ```

### Testing
- Manual test with known data
- Verify always in position (no gaps)
- Verify reversal logic works
- Establish S_03 baseline in tests.md
- Run optimization

### Commit
```
Phase 5: Add S_03 Reversal strategy
- Translate S_03 Pine to Python
- Implement reversal logic (allows_reversal=True)
- Register in StrategyRegistry
- Reference test: S_03 baseline established ✅
```

---

## PROMPT 6: Update API and UI

**Phase:** 6 of 6
**Duration:** 1-2 days
**Goal:** Make multi-strategy accessible via UI and API

### Backend (server.py)

1. **New endpoint:**
   ```python
   @app.route('/api/strategies', methods=['GET'])
   def list_strategies():
       strategies = StrategyRegistry.list_strategies()
       return jsonify({'strategies': strategies})
   ```

2. **Update /api/backtest:**
   ```python
   @app.route('/api/backtest', methods=['POST'])
   def run_backtest():
       strategy_id = request.form.get('strategy_id', 's01_trailing_ma')
       params = json.loads(request.form.get('params'))
       
       strategy = StrategyRegistry.get_strategy_instance(strategy_id, params)
       result = strategy.simulate(df)
       return jsonify(result)
   ```

3. **Update /api/optimize:**
   ```python
   strategy_id = request.form.get('strategy_id', 's01_trailing_ma')
   config = OptimizationConfig(
       csv_file=file,
       strategy_id=strategy_id,
       # ...
   )
   ```

### Frontend (index.html)

1. **Add selector:**
   ```html
   <select id="strategySelector" onchange="onStrategyChange()">
       <option value="s01_trailing_ma">S_01 TrailingMA v26</option>
       <option value="s03_reversal">S_03 Reversal v07</option>
   </select>
   ```

2. **Parameter blocks:**
   ```html
   <div id="params_s01_trailing_ma" class="strategy-params">
       <!-- Current S_01 UI -->
   </div>
   
   <div id="params_s03_reversal" class="strategy-params" style="display:none">
       <!-- S_03 parameters: 3 MAs, close count, breakout mode -->
   </div>
   ```

3. **JavaScript:**
   ```javascript
   function onStrategyChange() {
       const strategyId = document.getElementById('strategySelector').value;
       document.querySelectorAll('.strategy-params').forEach(el => {
           el.style.display = 'none';
       });
       document.getElementById(`params_${strategyId}`).style.display = 'block';
   }
   ```

### CLI (run_backtest.py)

```python
parser.add_argument('--strategy', default='s01_trailing_ma',
                    choices=['s01_trailing_ma', 's03_reversal'])

strategy_class = StrategyRegistry.get_strategy_class(args.strategy)
params = {k: v['default'] for k, v in strategy_class.get_param_definitions().items()}
strategy = strategy_class(params)
result = strategy.simulate(df)
```

### Testing
- UI: Switch strategies, run backtests
- API: Test all endpoints with both strategies
- CLI: Test both strategies

### Commit
```
Phase 6: Update API and UI for multi-strategy
- Add /api/strategies endpoint
- Update all endpoints to accept strategy_id
- Add strategy selector dropdown
- Add S_03 parameter form
- Update CLI with --strategy flag
- UI test: Both strategies work ✅
```

---

**End of Prompts 4, 5, 6**
