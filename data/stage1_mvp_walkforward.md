# Walk-Forward Analysis - Stage 1: MVP Implementation

## Project Context

You are working on a crypto trading strategy backtesting system. The current implementation optimizes parameters on the entire dataset, leading to overfitting. We need to implement Walk-Forward Analysis to find robust parameters.

**Existing modules:**

- `backtest_engine.py` - Strategy execution engine
- `optimizer_engine.py` - Grid search optimization
- `optuna_engine.py` - Bayesian optimization using Optuna
- `server.py` - Flask API backend
- `index.html` - Web UI
- `run_backtest.py` - CLI tool

**Goal:** Implement minimal but complete Walk-Forward system that works end-to-end.

---

## Stage 1 Scope - What We're Building

### Core Features (Keep It Simple!)

1. **Data Splitting:**
   - Rolling window mode ONLY
   - Fixed allocation: 80% WF Zone + 20% Forward Reserve
   - Each window: 70% IS + 30% OOS
   - Fixed gap between IS and OOS
   - Simple warmup (1000 bars before first IS)

2. **Optimization Loop:**
   - For each window:
     - Run existing Optuna on IS data (no changes to optuna!)
     - Take Top-K results
     - Test them on OOS
     - Save metrics

3. **Simple Filtering:**
   - Remove params with OOS profit ≤ 0
   - That's it. No complex filters.

4. **Aggregation:**
   - Count appearances in Top-K across windows
   - Calculate average OOS profit
   - Calculate OOS win rate (% profitable windows)
   - Simple ranking by average OOS profit

5. **Forward Test:**
   - Take Top-10 aggregated params
   - Test on Forward Reserve
   - Rank by Forward profit

6. **UI - Minimal:**
   - Checkbox: "Enable Walk-Forward"
   - Three input fields (when enabled)
   - Run button
   - Results table showing Top-10
   - Automatic CSV download

7. **CSV Export:**
   - Clear structure with understandable data
   - Unique IDs for param combinations: "MA_TYPE MA_LENGTH_hash"
   - Full parameter details after metrics

---

## Implementation Plan

### Step 1: Create `walkforward_engine.py`

This is a NEW file. Create it with the following structure:

```python
"""
Walk-Forward Analysis Engine - Stage 1 MVP
Simple implementation focusing on core functionality.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import hashlib
import json

@dataclass
class WFConfig:
    """Simple configuration for Walk-Forward"""
    num_windows: int = 5
    gap_bars: int = 100
    topk_per_window: int = 20
    
    # Fixed percentages (not configurable in Stage 1)
    wf_zone_pct: float = 80.0
    forward_pct: float = 20.0
    is_pct: float = 70.0
    oos_pct: float = 30.0
    warmup_bars: int = 1000

@dataclass
class WindowSplit:
    """One IS/OOS window"""
    window_id: int
    is_start: int
    is_end: int
    gap_start: int
    gap_end: int
    oos_start: int
    oos_end: int

@dataclass
class WindowResult:
    """Results from one window"""
    window_id: int
    top_params: List[Dict[str, Any]]  # Top-K from IS optimization
    oos_profits: List[float]  # OOS profit for each param
    oos_drawdowns: List[float]
    oos_trades: List[int]

@dataclass
class AggregatedResult:
    """Aggregated results for one param set"""
    param_id: str  # "EMA 45_abc123"
    params: Dict[str, Any]
    appearances: int  # How many windows
    avg_oos_profit: float
    oos_win_rate: float  # % windows profitable
    oos_profits: List[float]  # All OOS profits

@dataclass
class WFResult:
    """Complete Walk-Forward results"""
    config: WFConfig
    windows: List[WindowSplit]
    window_results: List[WindowResult]
    aggregated: List[AggregatedResult]
    forward_profits: List[float]  # Forward test results for Top-10
    forward_params: List[Dict[str, Any]]

class WalkForwardEngine:
    """Main engine for Walk-Forward Analysis"""
    
    def __init__(self, config: WFConfig):
        self.config = config
    
    def split_data(self, df: pd.DataFrame) -> Tuple[List[WindowSplit], int, int]:
        """
        Split data into WF windows + Forward Reserve
        
        Returns:
            windows: List of WindowSplit objects
            forward_start: Start index of forward period
            forward_end: End index of forward period
        """
        total_bars = len(df)
        warmup = self.config.warmup_bars
        
        # Calculate zones
        wf_zone_end = int(total_bars * (self.config.wf_zone_pct / 100))
        forward_start = wf_zone_end
        forward_end = total_bars
        
        # Available bars for WF (after warmup)
        wf_available_start = warmup
        wf_available_end = wf_zone_end
        wf_available_bars = wf_available_end - wf_available_start
        
        # Calculate window size
        window_total_bars = wf_available_bars // self.config.num_windows
        is_bars = int(window_total_bars * (self.config.is_pct / 100))
        oos_bars = int(window_total_bars * (self.config.oos_pct / 100))
        
        # Create windows
        windows = []
        for i in range(self.config.num_windows):
            # Calculate indices
            is_start = wf_available_start + (i * window_total_bars)
            is_end = is_start + is_bars
            gap_start = is_end
            gap_end = gap_start + self.config.gap_bars
            oos_start = gap_end
            oos_end = oos_start + oos_bars
            
            # Check if window fits in WF zone
            if oos_end > wf_available_end:
                break
            
            windows.append(WindowSplit(
                window_id=i + 1,
                is_start=is_start,
                is_end=is_end,
                gap_start=gap_start,
                gap_end=gap_end,
                oos_start=oos_start,
                oos_end=oos_end
            ))
        
        return windows, forward_start, forward_end
    
    def run_wf_optimization(
        self,
        df: pd.DataFrame,
        optuna_config: Dict[str, Any]
    ) -> WFResult:
        """
        Main function - runs complete Walk-Forward Analysis
        
        Steps:
        1. Split data
        2. For each window: optimize IS, test OOS
        3. Aggregate results
        4. Forward test
        5. Return results
        """
        print("Starting Walk-Forward Analysis...")
        
        # Step 1: Split data
        windows, fwd_start, fwd_end = self.split_data(df)
        print(f"Created {len(windows)} windows")
        print(f"Forward Reserve: bars {fwd_start} to {fwd_end}")
        
        # Step 2: Process each window
        window_results = []
        for window in windows:
            print(f"\n--- Window {window.window_id}/{len(windows)} ---")
            
            # Get IS data (includes warmup for indicators)
            # Warmup starts 1000 bars before IS
            warmup_start = max(0, window.is_start - self.config.warmup_bars)
            is_df_with_warmup = df.iloc[warmup_start:window.is_end].copy()
            
            print(f"IS optimization: bars {window.is_start} to {window.is_end}")
            
            # Run Optuna on IS
            from optuna_engine import run_optuna_optimization
            
            # Prepare config for Optuna
            optuna_params = {
                **optuna_config,
                'n_trials': optuna_config.get('n_trials', 100)
            }
            
            # Run optimization (returns list of results sorted by score)
            # Note: This uses the EXISTING optuna_engine without modifications
            optimization_results = run_optuna_optimization(is_df_with_warmup, optuna_params)
            
            # Take Top-K
            topk = min(self.config.topk_per_window, len(optimization_results))
            top_params = optimization_results[:topk]
            
            print(f"Got {len(top_params)} top parameter sets")
            
            # Test on OOS
            print(f"OOS validation: bars {window.oos_start} to {window.oos_end}")
            
            # Get OOS data (also needs warmup for indicators)
            # Use all data up to OOS end for indicator calculation
            oos_df_with_history = df.iloc[warmup_start:window.oos_end].copy()
            
            oos_profits = []
            oos_drawdowns = []
            oos_trades = []
            
            from backtest_engine import run_strategy, StrategyParams
            
            for params in top_params:
                # Run backtest on full history (for indicators)
                strategy_params = StrategyParams.from_dict(params)
                result = run_strategy(oos_df_with_history, strategy_params)
                
                # But we only care about trades in OOS period
                oos_start_time = df.index[window.oos_start]
                oos_end_time = df.index[window.oos_end - 1]
                
                # Filter trades to OOS period only
                oos_period_trades = [
                    t for t in result.trades
                    if oos_start_time <= t.entry_time <= oos_end_time
                ]
                
                # Calculate OOS metrics from OOS trades only
                if len(oos_period_trades) > 0:
                    # Calculate equity curve from OOS trades
                    oos_pnl = sum(t.net_pnl for t in oos_period_trades)
                    initial_equity = 10000  # Standard starting equity
                    oos_profit_pct = (oos_pnl / initial_equity) * 100
                    
                    # Simple max drawdown calculation
                    equity_curve = [initial_equity]
                    running_equity = initial_equity
                    for t in oos_period_trades:
                        running_equity += t.net_pnl
                        equity_curve.append(running_equity)
                    
                    peak = equity_curve[0]
                    max_dd = 0
                    for eq in equity_curve:
                        if eq > peak:
                            peak = eq
                        dd = ((eq - peak) / peak) * 100
                        if dd < max_dd:
                            max_dd = dd
                    
                    oos_profits.append(oos_profit_pct)
                    oos_drawdowns.append(max_dd)
                    oos_trades.append(len(oos_period_trades))
                else:
                    # No trades in OOS
                    oos_profits.append(0.0)
                    oos_drawdowns.append(0.0)
                    oos_trades.append(0)
            
            window_results.append(WindowResult(
                window_id=window.window_id,
                top_params=top_params,
                oos_profits=oos_profits,
                oos_drawdowns=oos_drawdowns,
                oos_trades=oos_trades
            ))
            
            print(f"OOS results: {len([p for p in oos_profits if p > 0])}/{len(oos_profits)} profitable")
        
        # Step 3: Aggregate results
        print("\n--- Aggregating Results ---")
        aggregated = self._aggregate_results(window_results)
        print(f"Found {len(aggregated)} unique parameter sets")
        
        # Step 4: Forward Test
        print("\n--- Forward Test ---")
        forward_df = df.iloc[fwd_start:fwd_end].copy()
        
        # Take Top-10 for forward test
        top10 = aggregated[:10]
        forward_profits = []
        forward_params = []
        
        from backtest_engine import run_strategy, StrategyParams
        
        for agg in top10:
            strategy_params = StrategyParams.from_dict(agg.params)
            result = run_strategy(forward_df, strategy_params)
            forward_profits.append(result.net_profit_pct)
            forward_params.append(agg.params)
        
        print(f"Forward Test complete: {len([p for p in forward_profits if p > 0])}/{len(forward_profits)} profitable")
        
        # Create final result
        wf_result = WFResult(
            config=self.config,
            windows=windows,
            window_results=window_results,
            aggregated=aggregated,
            forward_profits=forward_profits,
            forward_params=forward_params
        )
        
        return wf_result
    
    def _aggregate_results(self, window_results: List[WindowResult]) -> List[AggregatedResult]:
        """
        Aggregate results across windows
        
        Simple approach:
        - Group params that appear in multiple windows
        - Calculate average OOS profit
        - Calculate win rate
        - Sort by average OOS profit
        """
        # Collect all params
        param_map = {}  # param_id -> data
        
        for window_result in window_results:
            for i, params in enumerate(window_result.top_params):
                oos_profit = window_result.oos_profits[i]
                
                # Simple filter: skip if OOS profit <= 0
                if oos_profit <= 0:
                    continue
                
                # Create param ID
                param_id = self._create_param_id(params)
                
                if param_id not in param_map:
                    param_map[param_id] = {
                        'params': params,
                        'appearances': 0,
                        'oos_profits': []
                    }
                
                param_map[param_id]['appearances'] += 1
                param_map[param_id]['oos_profits'].append(oos_profit)
        
        # Convert to AggregatedResult objects
        aggregated = []
        for param_id, data in param_map.items():
            oos_profits = data['oos_profits']
            avg_profit = np.mean(oos_profits)
            win_rate = len([p for p in oos_profits if p > 0]) / len(oos_profits)
            
            aggregated.append(AggregatedResult(
                param_id=param_id,
                params=data['params'],
                appearances=data['appearances'],
                avg_oos_profit=avg_profit,
                oos_win_rate=win_rate,
                oos_profits=oos_profits
            ))
        
        # Sort by average OOS profit (descending)
        aggregated.sort(key=lambda x: x.avg_oos_profit, reverse=True)
        
        return aggregated
    
    def _create_param_id(self, params: Dict[str, Any]) -> str:
        """
        Create unique ID for param set
        Format: "MA_TYPE MA_LENGTH_hash"
        Example: "EMA 45_6d4ad0df"
        """
        # Get MA type and length
        ma_type = params.get('maType', 'UNKNOWN')
        ma_length = params.get('maLength', 0)
        
        # Create hash of all parameters
        param_str = json.dumps(params, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        
        return f"{ma_type} {ma_length}_{param_hash}"


def export_wf_results_csv(result: WFResult, output_path: str) -> None:
    """Export Walk-Forward results to CSV"""
    import csv
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # === HEADER ===
        writer.writerow(['=== WALK-FORWARD ANALYSIS - RESULTS ==='])
        writer.writerow([])
        
        # === SUMMARY ===
        writer.writerow(['=== SUMMARY ==='])
        writer.writerow(['Total Windows', len(result.windows)])
        writer.writerow(['WF Zone', f"{result.config.wf_zone_pct}%"])
        writer.writerow(['Forward Reserve', f"{result.config.forward_pct}%"])
        writer.writerow(['Gap Between IS/OOS', f"{result.config.gap_bars} bars"])
        writer.writerow(['Top-K Per Window', result.config.topk_per_window])
        writer.writerow([])
        
        # === TOP 10 RANKING ===
        writer.writerow(['=== TOP 10 PARAMETER SETS (by Avg OOS Profit) ==='])
        writer.writerow([
            'Rank',
            'Param ID',
            'Appearances',
            'Avg OOS Profit %',
            'OOS Win Rate',
            'Forward Profit %'
        ])
        
        for rank, agg in enumerate(result.aggregated[:10], 1):
            forward_profit = result.forward_profits[rank - 1] if rank <= len(result.forward_profits) else 'N/A'
            
            writer.writerow([
                rank,
                agg.param_id,
                f"{agg.appearances}/{len(result.windows)}",
                f"{agg.avg_oos_profit:.2f}%",
                f"{agg.oos_win_rate * 100:.1f}%",
                f"{forward_profit:.2f}%" if isinstance(forward_profit, float) else forward_profit
            ])
        
        writer.writerow([])
        
        # === WINDOW DETAILS ===
        writer.writerow(['=== WINDOW DETAILS ==='])
        writer.writerow([
            'Window',
            'IS Start',
            'IS End',
            'Gap Start',
            'Gap End',
            'OOS Start',
            'OOS End',
            'Top Param ID',
            'OOS Profit %'
        ])
        
        for window_result in result.window_results:
            window = result.windows[window_result.window_id - 1]
            
            # Find best OOS profit in this window
            if len(window_result.oos_profits) > 0:
                best_idx = np.argmax(window_result.oos_profits)
                best_param = window_result.top_params[best_idx]
                best_param_id = WalkForwardEngine(result.config)._create_param_id(best_param)
                best_oos_profit = window_result.oos_profits[best_idx]
            else:
                best_param_id = 'N/A'
                best_oos_profit = 0.0
            
            writer.writerow([
                window.window_id,
                window.is_start,
                window.is_end,
                window.gap_start,
                window.gap_end,
                window.oos_start,
                window.oos_end,
                best_param_id,
                f"{best_oos_profit:.2f}%"
            ])
        
        writer.writerow([])
        
        # === FORWARD TEST ===
        writer.writerow(['=== FORWARD TEST RESULTS ==='])
        writer.writerow([
            'Rank',
            'Param ID',
            'Forward Profit %'
        ])
        
        for rank, agg in enumerate(result.aggregated[:10], 1):
            if rank <= len(result.forward_profits):
                forward_profit = result.forward_profits[rank - 1]
                writer.writerow([
                    rank,
                    agg.param_id,
                    f"{forward_profit:.2f}%"
                ])
        
        writer.writerow([])
        
        # === DETAILED PARAMETERS ===
        writer.writerow(['=== DETAILED PARAMETERS FOR TOP 10 ==='])
        writer.writerow([])
        
        for rank, agg in enumerate(result.aggregated[:10], 1):
            writer.writerow([f"--- Rank #{rank}: {agg.param_id} ---"])
            
            # Write all parameters
            params = agg.params
            writer.writerow(['Parameter', 'Value'])
            writer.writerow(['MA Type', params.get('maType', 'N/A')])
            writer.writerow(['MA Length', params.get('maLength', 'N/A')])
            writer.writerow(['Close Count Long', params.get('closeCountLong', 'N/A')])
            writer.writerow(['Close Count Short', params.get('closeCountShort', 'N/A')])
            writer.writerow(['Stop Long ATR', params.get('stopLongX', 'N/A')])
            writer.writerow(['Stop Long RR', params.get('stopLongRR', 'N/A')])
            writer.writerow(['Stop Long LP', params.get('stopLongLP', 'N/A')])
            writer.writerow(['Stop Short ATR', params.get('stopShortX', 'N/A')])
            writer.writerow(['Stop Short RR', params.get('stopShortRR', 'N/A')])
            writer.writerow(['Stop Short LP', params.get('stopShortLP', 'N/A')])
            writer.writerow(['Stop Long Max %', params.get('stopLongMaxPct', 'N/A')])
            writer.writerow(['Stop Short Max %', params.get('stopShortMaxPct', 'N/A')])
            writer.writerow(['Stop Long Max Days', params.get('stopLongMaxDays', 'N/A')])
            writer.writerow(['Stop Short Max Days', params.get('stopShortMaxDays', 'N/A')])
            writer.writerow(['Trail RR Long', params.get('trailRRLong', 'N/A')])
            writer.writerow(['Trail RR Short', params.get('trailRRShort', 'N/A')])
            writer.writerow(['Trail MA Long Type', params.get('trailLongType', 'N/A')])
            writer.writerow(['Trail MA Long Length', params.get('trailLongLength', 'N/A')])
            writer.writerow(['Trail MA Long Offset', params.get('trailLongOffset', 'N/A')])
            writer.writerow(['Trail MA Short Type', params.get('trailShortType', 'N/A')])
            writer.writerow(['Trail MA Short Length', params.get('trailShortLength', 'N/A')])
            writer.writerow(['Trail MA Short Offset', params.get('trailShortOffset', 'N/A')])
            
            # Performance metrics
            writer.writerow([])
            writer.writerow(['Performance Metrics', ''])
            writer.writerow(['Appearances', f"{agg.appearances}/{len(result.windows)}"])
            writer.writerow(['Avg OOS Profit %', f"{agg.avg_oos_profit:.2f}%"])
            writer.writerow(['OOS Win Rate', f"{agg.oos_win_rate * 100:.1f}%"])
            
            # OOS profits per window
            writer.writerow(['OOS Profits by Window', ', '.join([f"{p:.2f}%" for p in agg.oos_profits])])
            
            # Forward profit
            if rank <= len(result.forward_profits):
                writer.writerow(['Forward Test Profit %', f"{result.forward_profits[rank - 1]:.2f}%"])
            
            writer.writerow([])
            writer.writerow([])
```

---

### Step 2: Add Endpoint to `server.py`

Add this new endpoint to handle Walk-Forward requests:

```python
@app.route('/api/walkforward', methods=['POST'])
def run_walkforward_optimization():
    """Run Walk-Forward Analysis"""
    try:
        # Get form data
        data = request.form
        csv_file = request.files.get('csv')
        
        if not csv_file:
            return jsonify({'error': 'No CSV file provided'}), 400
        
        # Parse WF config
        from walkforward_engine import WFConfig, WalkForwardEngine, export_wf_results_csv
        
        wf_config = WFConfig(
            num_windows=int(data.get('wf_num_windows', 5)),
            gap_bars=int(data.get('wf_gap_bars', 100)),
            topk_per_window=int(data.get('wf_topk', 20))
        )
        
        # Load data
        from backtest_engine import load_data
        df = load_data(csv_file)
        
        # Parse Optuna config (same as existing optimization)
        optuna_config = {
            'n_trials': int(data.get('optunaTrials', 100)),
            # ... other optuna params from form
            # Copy the existing optuna config parsing from your current code
        }
        
        # Run Walk-Forward
        engine = WalkForwardEngine(wf_config)
        result = engine.run_wf_optimization(df, optuna_config)
        
        # Export to CSV
        import uuid
        import os
        output_filename = f"wf_results_{uuid.uuid4().hex[:8]}.csv"
        output_path = os.path.join('static', 'results', output_filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        export_wf_results_csv(result, output_path)
        
        # Prepare response
        top10 = []
        for rank, agg in enumerate(result.aggregated[:10], 1):
            forward_profit = result.forward_profits[rank - 1] if rank <= len(result.forward_profits) else None
            top10.append({
                'rank': rank,
                'param_id': agg.param_id,
                'appearances': f"{agg.appearances}/{len(result.windows)}",
                'avg_oos_profit': round(agg.avg_oos_profit, 2),
                'oos_win_rate': round(agg.oos_win_rate * 100, 1),
                'forward_profit': round(forward_profit, 2) if forward_profit else None
            })
        
        return jsonify({
            'status': 'success',
            'summary': {
                'total_windows': len(result.windows),
                'top_param_id': result.aggregated[0].param_id if result.aggregated else 'N/A',
                'top_avg_oos_profit': round(result.aggregated[0].avg_oos_profit, 2) if result.aggregated else 0
            },
            'top10': top10,
            'csv_url': f"/static/results/{output_filename}"
        })
        
    except Exception as e:
        import traceback
        print("Error in Walk-Forward:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500
```

---

### Step 3: Add UI Section to `index.html`

Find the optimization form section and add this Walk-Forward section:

```html
<!-- Walk-Forward Analysis Section -->
<div class="wf-section" style="margin-top: 30px; padding: 20px; border: 2px solid #3498db; border-radius: 8px; background-color: #f8f9fa;">
    <h3 style="color: #3498db;">⚡ Walk-Forward Analysis</h3>
    
    <div class="form-group">
        <label>
            <input type="checkbox" id="enableWF" onchange="toggleWFSettings()">
            <strong>Enable Walk-Forward Optimization</strong>
        </label>
        <p style="font-size: 12px; color: #666; margin-top: 5px;">
            Splits data into multiple training/testing windows to find robust parameters.
        </p>
    </div>
    
    <!-- WF Settings (hidden by default) -->
    <div id="wfSettings" style="display: none; margin-top: 20px;">
        <div class="form-group">
            <label>Number of Windows:</label>
            <input type="number" id="wfNumWindows" value="5" min="3" max="10" style="width: 100px;">
            <span style="font-size: 12px; color: #666; margin-left: 10px;">
                (Recommended: 5-6 for 1-2 years of data)
            </span>
        </div>
        
        <div class="form-group">
            <label>Gap (bars):</label>
            <input type="number" id="wfGapBars" value="100" min="0" max="500" style="width: 100px;">
            <span style="font-size: 12px; color: #666; margin-left: 10px;">
                (Bars between training and testing to prevent look-ahead bias)
            </span>
        </div>
        
        <div class="form-group">
            <label>Top-K per Window:</label>
            <input type="number" id="wfTopK" value="20" min="5" max="50" style="width: 100px;">
            <span style="font-size: 12px; color: #666; margin-left: 10px;">
                (Best parameter sets to test on each window)
            </span>
        </div>
        
        <div style="background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-top: 15px;">
            <strong>Fixed Settings (Stage 1):</strong>
            <ul style="margin-top: 10px; font-size: 13px; color: #555;">
                <li>WF Zone: 80% of data</li>
                <li>Forward Reserve: 20% of data</li>
                <li>Each window: 70% training (IS), 30% testing (OOS)</li>
                <li>Rolling window mode</li>
            </ul>
        </div>
    </div>
</div>

<!-- Results Section -->
<div id="wfResults" style="display: none; margin-top: 30px;">
    <h3>Walk-Forward Results</h3>
    
    <div id="wfSummary" style="background-color: #d4edda; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
        <!-- Summary will be inserted here -->
    </div>
    
    <h4>Top 10 Parameter Sets</h4>
    <table id="wfTable" style="width: 100%; border-collapse: collapse;">
        <thead>
            <tr style="background-color: #3498db; color: white;">
                <th style="padding: 10px; border: 1px solid #ddd;">Rank</th>
                <th style="padding: 10px; border: 1px solid #ddd;">Param ID</th>
                <th style="padding: 10px; border: 1px solid #ddd;">Appearances</th>
                <th style="padding: 10px; border: 1px solid #ddd;">Avg OOS Profit %</th>
                <th style="padding: 10px; border: 1px solid #ddd;">OOS Win Rate</th>
                <th style="padding: 10px; border: 1px solid #ddd;">Forward Profit %</th>
            </tr>
        </thead>
        <tbody id="wfTableBody">
            <!-- Results will be inserted here -->
        </tbody>
    </table>
    
    <div style="margin-top: 20px;">
        <p style="font-size: 14px; color: #666;">
            Full detailed results have been automatically downloaded as CSV.
        </p>
    </div>
</div>

<script>
function toggleWFSettings() {
    const enabled = document.getElementById('enableWF').checked;
    document.getElementById('wfSettings').style.display = enabled ? 'block' : 'none';
}

function runOptimization() {
    // Check if WF is enabled
    const wfEnabled = document.getElementById('enableWF').checked;
    
    if (wfEnabled) {
        runWalkForward();
    } else {
        // Run normal optimization (existing code)
        runNormalOptimization();
    }
}

function runWalkForward() {
    // Get form data
    const formData = new FormData(document.getElementById('optimizationForm'));
    
    // Add WF params
    formData.append('wf_num_windows', document.getElementById('wfNumWindows').value);
    formData.append('wf_gap_bars', document.getElementById('wfGapBars').value);
    formData.append('wf_topk', document.getElementById('wfTopK').value);
    
    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('wfResults').style.display = 'none';
    
    // Send request
    fetch('/api/walkforward', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('loading').style.display = 'none';
        
        if (data.status === 'success') {
            displayWFResults(data);
            
            // Automatic download
            window.location.href = data.csv_url;
        } else {
            alert('Error: ' + data.error);
        }
    })
    .catch(error => {
        document.getElementById('loading').style.display = 'none';
        alert('Error: ' + error);
    });
}

function displayWFResults(data) {
    // Show results section
    document.getElementById('wfResults').style.display = 'block';
    
    // Display summary
    const summaryHtml = `
        <strong>Summary:</strong><br>
        Total Windows: ${data.summary.total_windows}<br>
        Best Parameter Set: ${data.summary.top_param_id}<br>
        Best Avg OOS Profit: ${data.summary.top_avg_oos_profit}%
    `;
    document.getElementById('wfSummary').innerHTML = summaryHtml;
    
    // Display top 10 table
    const tbody = document.getElementById('wfTableBody');
    tbody.innerHTML = '';
    
    data.top10.forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">${row.rank}</td>
            <td style="padding: 10px; border: 1px solid #ddd;">${row.param_id}</td>
            <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">${row.appearances}</td>
            <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">${row.avg_oos_profit}%</td>
            <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">${row.oos_win_rate}%</td>
            <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">${row.forward_profit !== null ? row.forward_profit + '%' : 'N/A'}</td>
        `;
        tbody.appendChild(tr);
    });
    
    // Scroll to results
    document.getElementById('wfResults').scrollIntoView({ behavior: 'smooth' });
}

function runNormalOptimization() {
    // Your existing optimization code
    // ...
}
</script>
```

---

## Testing Checklist

After implementation, test the following:

### Basic Functionality
- [ ] WF checkbox appears in UI
- [ ] Settings show/hide when checkbox toggled
- [ ] Input fields have correct default values
- [ ] Run button triggers WF optimization
- [ ] Console shows progress ("Window 1/5...", etc.)
- [ ] Results table displays after completion
- [ ] CSV downloads automatically
- [ ] CSV has all expected sections

### Data Validation
- [ ] Works with 1 year of data (15min TF)
- [ ] Works with 2 years of data
- [ ] Handles small datasets gracefully (error message if too small)
- [ ] Warmup period is correctly applied
- [ ] Gap is correctly applied between IS and OOS

### Results Validation
- [ ] Top 10 ranking makes sense (sorted by avg OOS profit)
- [ ] Param IDs are unique and readable (e.g., "EMA 45_abc123")
- [ ] OOS win rates are calculated correctly
- [ ] Forward test shows results for Top 10
- [ ] CSV contains detailed parameters for each rank
- [ ] All metrics are present in CSV

### Edge Cases
- [ ] Handles case where all params fail OOS (all negative)
- [ ] Handles case where < 10 params pass filters
- [ ] Handles case where window doesn't fit exactly (truncates gracefully)
- [ ] Error message if dataset too small for requested windows

---

## Expected Output

### Console Output During Execution:

```
Starting Walk-Forward Analysis...
Created 5 windows
Forward Reserve: bars 28000 to 35000

--- Window 1/5 ---
IS optimization: bars 1000 to 16800
Got 20 top parameter sets
OOS validation: bars 16900 to 22400
OOS results: 14/20 profitable

--- Window 2/5 ---
IS optimization: bars 5600 to 22400
Got 20 top parameter sets
OOS validation: bars 22500 to 28000
OOS results: 16/20 profitable

[... windows 3-5 ...]

--- Aggregating Results ---
Found 47 unique parameter sets

--- Forward Test ---
Forward Test complete: 8/10 profitable
```

### CSV Structure:

```csv
=== WALK-FORWARD ANALYSIS - RESULTS ===

=== SUMMARY ===
Total Windows,5
WF Zone,80%
Forward Reserve,20%
Gap Between IS/OOS,100 bars
Top-K Per Window,20

=== TOP 10 PARAMETER SETS (by Avg OOS Profit) ===
Rank,Param ID,Appearances,Avg OOS Profit %,OOS Win Rate,Forward Profit %
1,EMA 45_6d4ad0df,4/5,14.52%,100.0%,12.80%
2,SMA 50_0868588e,5/5,13.21%,100.0%,11.10%
3,EMA 48_a1b2c3d4,3/5,12.87%,100.0%,10.50%
...

=== WINDOW DETAILS ===
Window,IS Start,IS End,Gap Start,Gap End,OOS Start,OOS End,Top Param ID,OOS Profit %
1,1000,16800,16800,16900,16900,22400,EMA 45_6d4ad0df,16.30%
2,5600,22400,22400,22500,22500,28000,EMA 45_6d4ad0df,14.80%
...

=== FORWARD TEST RESULTS ===
Rank,Param ID,Forward Profit %
1,EMA 45_6d4ad0df,12.80%
2,SMA 50_0868588e,11.10%
...

=== DETAILED PARAMETERS FOR TOP 10 ===

--- Rank #1: EMA 45_6d4ad0df ---
Parameter,Value
MA Type,EMA
MA Length,45
Close Count Long,7
Close Count Short,5
Stop Long ATR,2.0
Stop Long RR,3.0
...
[all parameters]

Performance Metrics,
Appearances,4/5
Avg OOS Profit %,14.52%
OOS Win Rate,100.0%
OOS Profits by Window,"16.30%, 14.80%, 13.20%, 15.10%"
Forward Test Profit %,12.80%


--- Rank #2: SMA 50_0868588e ---
...
```

---

## Key Implementation Notes

### Simplicity First
- Use existing `optuna_engine.py` without modifications
- Use existing `backtest_engine.py` without modifications
- No complex abstractions or class hierarchies
- Straightforward procedural flow

### Warmup Handling
- Simple: always use 1000 bars before IS start
- Enough for most MA periods (up to 200-300)
- If user's MA range goes higher, they'll need more data

### OOS Metrics Calculation
- Run backtest on full history (for indicator calculation)
- Filter trades to OOS period only
- Calculate metrics from OOS trades only
- This ensures indicators are properly initialized

### Parameter Identification
- Use MA type and length in ID for readability
- Add hash for uniqueness
- Format: "EMA 45_abc123"
- Easy to scan in results

### CSV Design
- Multiple sections for different views
- Top-level summary
- Ranking table
- Window details
- Forward test
- Full parameter details
- Easy to open in Excel and navigate

---

## Success Criteria

Implementation is complete when:

1. ✅ User can enable WF in UI
2. ✅ User can configure 3 basic parameters
3. ✅ Optimization runs without errors
4. ✅ Progress is visible in console
5. ✅ Results table shows Top 10
6. ✅ CSV downloads automatically
7. ✅ CSV has all required sections
8. ✅ Param IDs are readable (MA type + length + hash)
9. ✅ Forward test shows real results
10. ✅ Full parameters are in CSV for each rank

---

## Common Issues & Solutions

### Issue: "Not enough data"
**Solution:** User needs to load CSV with more bars. Need at least ~10,000 bars for 5 windows with 80/20 split.

### Issue: "All params filtered out"
**Solution:** This means all parameter sets had negative OOS profit. Either:
- Market conditions changed drastically
- Strategy is not robust
- Try different parameter ranges

### Issue: "Forward test all negative"
**Solution:** This is actually valuable information! It means even the "best" params don't work on unseen data. Strategy needs rework.

### Issue: CSV download doesn't start
**Solution:** Check browser console for errors. Ensure `/static/results/` directory exists and is writable.

### Issue: Optimization takes very long
**Solution:** Reduce `n_trials` in Optuna config, or reduce `num_windows`, or use smaller dataset for testing.

---

## Next Steps (Future Stages)

After Stage 1 is working:

**Stage 2 will add:**
- Anchored mode
- Configurable WF/Forward split percentages
- Advanced filters (degradation, min trades)
- Better UI with sliders

**Stage 3 will add:**
- Cross-validation inside IS
- Statistical tests
- Parameter ensembles
- Progress bar in UI

But for now, focus on getting Stage 1 working perfectly!

---

## Final Implementation Instructions

1. Create `walkforward_engine.py` with all the code above
2. Add the endpoint to `server.py`
3. Add the UI section to `index.html`
4. Test with sample data (1-2 years, 15min TF)
5. Verify CSV output is clean and understandable
6. Test edge cases (small dataset, all negative OOS, etc.)

**Keep it simple. Make it work. Stage 1 is MVP only.**
