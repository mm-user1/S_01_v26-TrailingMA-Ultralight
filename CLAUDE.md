# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working with this repository.

## Shared Agent Rules

These rules apply to every coding agent working in this repository:

- Act as an experienced Python and Pine Script developer with trading and
  crypto algorithmic-trading expertise.
- Work strictly to the given specification. Do not make unapproved deviations.
- Keep implementation code efficient, fast, clear, concise, and logically
  organized.
- Keep the GUI on its established light theme unless the user explicitly
  requests a different design.
- **Windows only:** always use
  `C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe` for Python
  commands and tests, for example
  `C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe -m pytest -q`.
  This absolute interpreter requirement does not apply on Linux/VPS hosts;
  there, use the Python environment configured for that host and repository.

Repository directory roles:

- `./data` contains examples, market data, and related inputs.
- `./docs` contains documentation, plans, and reference material.
- `./src` is the main application source directory.
- `./tools/strategy_lab` contains the Backtester-V2-only Strategy Lab Phase 0
  run-spec, plan-identity, and read-only inventory foundation. Its tracked
  `runspecs/` are cross-host identity; `output/`, `cache/`, and `tmp/` are local
  ignored artifacts.
- `./tests/strategy_lab` contains focused Phase 0 tests and deliberately has no
  package `__init__.py` or local `conftest.py`.

Strategy Lab external market data is read-only. Resolve its root from an
explicit `--data-root`, then `MERLIN_STRATEGY_LAB_DATA_ROOT`; never search for
substitute files or track an absolute root. The confirmed Windows pack may be
absent on Linux/VPS, where Stage A remains valid but Stage B must be reported as
incomplete. Run Phase 0 tests from the repository root through the configured
interpreter: `python -m pytest tests/strategy_lab -q`.

## Project: Merlin

Cryptocurrency trading strategy backtesting and Optuna optimization platform with a Flask SPA frontend.

## Running the Application

### Web Server
```bash
cd src/ui
python server.py
```
Server runs at http://127.0.0.1:5000

### CLI Backtest
```bash
cd src
python run_backtest.py --csv ../data/raw/OKX_LINKUSDT.P,\ 15\ 2025.05.01-2025.11.20.csv
```

### Tests
```bash
pytest tests/ -v
```

### Dependencies
```bash
pip install -r requirements.txt
```
Key: Flask, pandas, numpy, matplotlib, optuna==4.6.0

## Architecture

### Core Principles

1. **Config-driven design** - Parameter schemas in `config.json`, UI renders dynamically
2. **camelCase naming** - End-to-end: Pine Script -> config.json -> Python -> CSV
3. **Dual optimizer modes** - Optuna (Bayesian/evolutionary) and Grid (deterministic; per-strategy backend chooses generation: S03 = LHS/full by mode, S06 = complete enumeration); both share constraints/objectives/storage. Optuna offers optional Initial Search Coverage for systematic parameter exploration.
4. **Strategy isolation** - Each strategy owns its params dataclass; optional Numba-accelerated fast Grid backend per strategy (`s03_reversal_v10/fast_grid.py`, `s03_reversal_v11/fast_grid.py`, `s06_r_trend_v02/fast_grid.py`)
5. **Rolling WFA** - Fixed WFA supports unchanged day windows (default) or complete calendar-month windows; Adaptive remains day-only. Both use stitched OOS equity, annualized WFE, optional cooldown after adaptive triggers, and per-module top-N trial retention.
6. **In-memory backend** - RAM-based Optuna journal storage for faster multiprocess optimization
7. **Trial deduplication** - Automatic detection/skipping of duplicate parameter sets with search space exhaustion early stopping
8. **Database persistence** - All optimization results automatically saved to SQLite, browsable through web UI; analytics group caches keep aggregated WFA equity computations warm
9. **Multi-database support** - Multiple `.db` files with active DB switching
10. **Three-page UI** - Start (configuration), Results (studies browser), Analytics (WFA research)
11. **Bundle export** - `s03_reversal_v10` Optuna/Grid trials and WFA windows can be exported as a Lancelot-compatible partial bundle for downstream execution

### Directory Structure
```
src/
|-- core/                     # Engines + utilities
|   |-- backtest_engine.py    # Trade simulation, TradeRecord, StrategyResult
|   |-- optuna_engine.py      # Optimization, OptimizationResult, OptunaConfig, InMemoryJournalBackend, coverage, dedup
|   |-- grid_engine.py        # Deterministic Grid optimizer: backend dispatch/metadata, mode allocation, LHS/full generation, fast/slow refinement, ranking, validation
|   |-- walkforward_engine.py # WFA orchestration (fixed + adaptive + cooldown)
|   |-- metrics.py            # BasicMetrics, AdvancedMetrics, WFAMetrics (incl. Consistency R²)
|   |-- analytics.py          # Portfolio equity aggregation for Analytics page
|   |-- storage.py            # SQLite database operations (studies/trials/wfa_windows/wfa_window_trials/study_sets/analytics_group_cache)
|   |-- export.py             # Trade CSV export functions
|   |-- bundle_export.py      # Lancelot partial-bundle export
|   |-- param_identity.py     # Display identity helpers (canonical params, hashed IDs)
|   |-- post_process.py       # Forward Test, DSR, Stress Test validation
|   `-- testing.py            # OOS selection and test utilities
|-- indicators/               # Technical indicators
|   |-- ma.py                 # 11 MA types via get_ma()
|   |-- volatility.py         # ATR, NATR
|   `-- oscillators.py        # RSI, StochRSI
|-- strategies/               # Trading strategies
|   |-- base.py               # BaseStrategy class
|   |-- s01_trailing_ma/
|   |-- s03_reversal_v10/     # Includes fast_grid.py (LHS/full by-mode Numba Grid backend)
|   |-- s03_reversal_v11/     # V1 v10 clone with optional Emergency SL + fast_grid.py
|   |-- s04_stochrsi/
|   `-- s06_r_trend_v02/      # Includes fast_grid.py (full-enumeration Numba Grid backend)
|-- storage/                  # Database storage (gitignored)
|   |-- *.db                  # SQLite database files (WAL mode, multiple supported)
|   |-- journals/             # SQLite journal files
|   `-- queue.json            # Scheduled run queue state
`-- ui/                       # Web interface
    |-- server.py                 # Thin entrypoint + app creation + route registration
    |-- server_services.py        # Helpers/shared logic (no route decorators)
    |-- server_routes_data.py     # Pages + studies/tests/trades + presets + strategies + DB/CSV/queue endpoints
    |-- server_routes_run.py      # Optimization status/cancel + optimize/walkforward/backtest
    |-- server_routes_analytics.py # Analytics page + WFA summary API
    |-- templates/
    |   |-- index.html        # Start page (configuration)
    |   |-- results.html      # Results page (studies browser)
    |   `-- analytics.html    # Analytics page (WFA research)
    `-- static/
        |-- js/
        |   |-- main.js                  # Start page logic (Optuna + Grid + WFA launch)
        |   |-- results-state.js         # Results state + localStorage/sessionStorage + URL helpers
        |   |-- results-format.js        # Results formatters + labels + MD5
        |   |-- results-tables.js        # Results table/chart renderers + row selection
        |   |-- results-controller.js    # Results orchestration + API calls + event binding
        |   |-- api.js                   # API client
        |   |-- strategy-config.js       # Dynamic form generation from config.json
        |   |-- ui-handlers.js           # Shared UI event handlers (incl. Grid settings panel)
        |   |-- optuna-ui.js             # Optuna Start-page UI helpers + coverage analysis
        |   |-- optuna-results-ui.js     # Optuna/Grid Results-page UI helpers
        |   |-- post-process-ui.js       # Post process UI helpers
        |   |-- oos-test-ui.js           # OOS test UI helpers
        |   |-- wfa-results-ui.js        # WFA Results-page UI helpers
        |   |-- presets.js               # Preset management
        |   |-- results.js               # Results page initialization
        |   |-- queue.js                 # Scheduled run queue management
        |   |-- dataset-preview.js       # WFA window layout preview
        |   |-- analytics.js             # Analytics page logic + state
        |   |-- analytics-equity.js      # Analytics equity curve rendering
        |   |-- analytics-filters.js     # Analytics filter panel management
        |   |-- analytics-table.js       # Analytics study table rendering
        |   |-- analytics-sets.js        # Analytics study sets management (CRUD, members)
        |   |-- analytics-sets-view.js   # Analytics study sets view (sort/filter/bulk-color/bulk-delete)
        |   `-- utils.js                 # Shared utility functions
        `-- css/
```

### Data Structure Ownership

| Structure | Module |
|-----------|--------|
| `TradeRecord`, `StrategyResult` | `backtest_engine.py` (`TradeRecord.exit_reason` is optional; S03 v11 Emergency SL exits use `"Emergency SL"`) |
| `BasicMetrics`, `AdvancedMetrics` | `metrics.py` |
| `OptimizationResult`, `OptunaConfig`, `OptimizationConfig`, `InMemoryJournalBackend` | `optuna_engine.py` |
| `GridSelectionConfig`, `GridAllocation`, grid preview/dispatch, backend metadata normalization | `grid_engine.py` |
| `GridParameterSpace`, `GridCandidate`, `FastGridData` (S03 LHS/full fast backend) | `strategies/s03_reversal_v10/fast_grid.py`, `strategies/s03_reversal_v11/fast_grid.py` |
| `GridParameterSpace`, `GridCandidate`, `CandidateSequence`, `FastGridData` (S06 full-enumeration fast backend) | `strategies/s06_r_trend_v02/fast_grid.py` |
| `WFConfig`, `WFResult`, `WindowResult`, `WindowSplit`, `StitchWindow`, `TriggerResult`, `ISPipelineResult`, `WindowExecutionPlan`, `OOSStitchedResult` | `walkforward_engine.py` |
| Lancelot partial bundle builder | `bundle_export.py` |
| Display-identity canonicalization / param hashing | `param_identity.py` |
| `aggregate_equity_curves` | `analytics.py` |
| Strategy params dataclass | Each strategy's `strategy.py` |

## Parameter Naming Rules

**CRITICAL: Use camelCase everywhere**

- Correct: `maType`, `closeCountLong`, `rsiLen`, `stopLongMaxPct`
- Avoid: `ma_type`, `close_count_long`, `rsi_len`, `stop_long_max_pct`

Internal control fields (`use_backtester`, `start`, `end`) may use snake_case but are excluded from UI/config.

**Do NOT add:**
- `to_dict()` methods - use `dataclasses.asdict(params)` instead
- Snake<->camel conversion helpers
- Feature flags

## Adding New Strategies

See `docs/ADDING_NEW_STRATEGY.md` for complete guide.

Quick checklist:
1. Create `src/strategies/<strategy_id>/` directory
2. Create `config.json` with parameter schema (camelCase)
3. Create `strategy.py` with params dataclass and strategy class
4. Ensure `STRATEGY_ID`, `STRATEGY_NAME`, `STRATEGY_VERSION` class attributes
5. Implement `run(df, params, trade_start_idx) -> StrategyResult` static method
6. Strategy auto-discovered - no manual registration needed

## Database Operations

### Accessing Studies
```python
from core.storage import list_studies, load_study_from_db

# List all saved studies
studies = list_studies()
for study in studies:
    print(f"{study['study_name']}: {study['saved_trials']} trials")

# Load complete study with trials/windows
study_data = load_study_from_db(study_id)
print(study_data['study'])      # Study metadata
print(study_data['trials'])     # Optuna trials (if mode='optuna')
print(study_data['windows'])    # WFA windows (if mode='wfa')
print(study_data['csv_exists']) # Whether CSV file still exists

### Understanding Study Storage

**Optuna studies:**
- Saved to `studies` table (metadata) + `trials` table (parameter sets)
- Trials include: params (JSON), metrics, composite score
- Multi-objective studies store objective vectors and Pareto/feasibility flags (constraints)
- Study summaries may include completed/failed/pruned counts; results lists include COMPLETE trials (failed trials are retained only if explicitly stored for debugging)
- Optional filters (by score/profit threshold) may reduce stored trials for UI browsing

**WFA studies:**
- Saved to `studies` table (metadata) + `wfa_windows` table (per-window results)
- Each window includes: best params, IS/OOS metrics, equity curves (JSON arrays)
- WFE (Walk-Forward Efficiency) stored as `best_value`

### Database Location

**Note:** Database files are gitignored. Only `.gitkeep` files are tracked.

## Common Tasks

### Running Single Backtest
```python
from core.backtest_engine import load_data, prepare_dataset_with_warmup
from strategies.s01_trailing_ma.strategy import S01TrailingMA

df = load_data("data/raw/OKX_LINKUSDT.P, 15 2025.05.01-2025.11.20.csv")
df_prepared, trade_start_idx = prepare_dataset_with_warmup(df, start, end, warmup_bars=1000)
result = S01TrailingMA.run(df_prepared, params, trade_start_idx)

### Calculating Metrics
```python
from core import metrics
basic = metrics.calculate_basic(result, initial_capital=100.0)
advanced = metrics.calculate_advanced(result)  # includes consistency_score (R²)

### Walk-Forward Analysis (Rolling)
```python
from core.walkforward_engine import WFConfig, WalkForwardEngine

wf_config = WFConfig(
    strategy_id="s01_trailing_ma",
    is_period_days=180,
    oos_period_days=60,
    warmup_bars=1000,
)
engine = WalkForwardEngine(wf_config, base_config_template, optuna_settings)
wf_result = engine.run_wf_optimization(df)

### Walk-Forward Analysis (Adaptive)
```python
wf_config = WFConfig(
    strategy_id="s01_trailing_ma",
    is_period_days=180,
    oos_period_days=60,
    warmup_bars=1000,
    adaptive_mode=True,
    max_oos_period_days=90,
    min_oos_trades=5,
    cusum_threshold=5.0,
    dd_threshold_multiplier=1.5,
    inactivity_multiplier=5.0,
    cooldown_enabled=True,   # Skip re-optimization for cooldown_days after a trigger
    cooldown_days=15,
    store_top_n_trials=50,   # Per-module top-N trial retention for storage
)

### Walk-Forward Analysis (Fixed Calendar Months)

Fixed WFA can use `period_unit="months"` with authoritative month counts and
`is_period_days=None` / `oos_period_days=None`. Calendar windows are anchored
to the requested UTC Start, which requires Date Filter and calendar day 1
through 28. The anchor day is preserved across every month boundary. Actual CSV
bars are aligned inside half-open logical month
boundaries; only complete OOS months covered by the inclusive requested End
and last available CSV date are emitted, so an incomplete tail is ignored.
Adaptive WFA remains day-only. Month WFE uses nominal `12 / months`
annualization factors; day WFE remains `365 / days`. Queue, Results, and
Analytics display month settings as compact values such as `2m/1m`.

### Using Indicators
```python
from indicators.ma import get_ma
from indicators.volatility import atr
from indicators.oscillators import rsi, stoch_rsi

ma_values = get_ma(df["Close"], "HMA", 50)
atr_values = atr(df["High"], df["Low"], df["Close"], 14)
rsi_values = rsi(df["Close"], 14)

## Testing

### Run All Tests
```bash
pytest tests/ -v

### Key Test Files
- `conftest.py` - Shared fixtures (storage isolated under pytest temp roots outside the repo, Flask test client)
- `test_sanity.py` - Infrastructure checks
- `test_regression_s01.py` - S01 baseline regression
- `test_s03_reversal_v10.py` - S03 strategy tests
- `test_s04_stochrsi.py` - S04 strategy tests
- `test_s06_r_trend_v02.py` - S06 strategy tests (slow execution contract, baselines)
- `test_s06_fast_grid.py` - S06 fast Numba Grid tests (full enumeration, fast-vs-slow execution parity, determinism)
- `test_naming_consistency.py` - camelCase guardrails
- `test_storage.py` - Database storage tests
- `test_server.py` - HTTP API endpoint tests (incl. Grid Settings sidebar Constraints row)
- `test_post_process.py` - Post-process module tests
- `test_dsr.py` - Deflated Sharpe Ratio tests
- `test_oos_selection.py` - OOS selection tests
- `test_stress_test.py` - Stress test tests
- `test_analytics.py` - Analytics equity aggregation tests
- `test_adaptive_wfa.py` - Adaptive WFA trigger detection tests
- `test_db_management.py` - Multi-database management tests
- `test_coverage_startup.py` - Initial Search Coverage mode tests
- `test_strategy_loop_regression.py` - Strategy loop performance regression tests
- `test_grid_engine.py` - Grid optimizer tests (allocation, LHS/full, validation, fast/slow refinement, backend metadata + default modes, diversity-field shape)
- `test_multiprocess_score.py` - Multi-process composite-score optimization tests

Tests never write generated files under `tests/`: storage/journal/CSV/queue/export
artifacts use pytest `tmp_path` / `tmp_path_factory` (or a session temp root)
outside the repository, and CWD changes are restored via `monkeypatch`.

### Regenerate S01 Baseline
```bash
python tools/generate_baseline_s01.py

## Optuna: Multi-objective & constraints

**Key behavioral rules (keep these consistent across backend + UI):**

- **Single objective vs multi-objective**
  - 1 objective: create study with `direction=...`
  - 2+ objectives: create study with `directions=[...]` and return a tuple of objective values
  - Multi-objective results are a **Pareto front**; UI sorts Pareto-first then by **primary objective**

- **Pruning**
  - Pruning is supported for **single-objective** only.
  - Optuna `Trial.should_prune()` does **not** support multi-objective optimization.

- **Invalid objectives / missing metrics**
  - If an objective value is missing/NaN, return `float("nan")` (or a NaN tuple for multi-objective).
  - Optuna treats NaN returns as **FAILED trials** (study continues).
  - Failed trials are ignored by Optuna samplers (they do not affect future suggestions).

- **Constraints**
- Constraints are **soft**: infeasible trials are retained but deprioritized in UI and "best" selection.
  - `constraints_func` is evaluated only after **successful** trials; it is not called for failed/pruned trials.
- Sorting/labeling should follow: feasible Pareto -> feasible non-Pareto -> infeasible (then by total violation, then primary objective).

- **Initial Search Coverage**
  - Optional coverage mode (`coverage_mode: true`) for systematic parameter space exploration during startup.
  - Generates structured coverage trials from categorical combinations and numeric quantiles.
  - UI provides coverage analysis with block size hints (multipliers: 1, 3, 5, 9, 17) and auto-fill for warmup trials.
  - Bool group rules (e.g., `at_least_one_true`) reduce coverage block size by excluding invalid combinations.

- **Trial deduplication**
  - Duplicate parameter sets are detected via deterministic JSON key comparison.
  - Duplicates are marked FAIL with `merlin.duplicate_skipped` attribute and skipped.
  - Soft duplicate cycle limit (`dispatcher_duplicate_cycle_limit`, default 18) prevents infinite loops.
  - Search space exhaustion triggers early stopping.

- **Trial logging**
  - `trials_log` flag (default false) controls Optuna trial-level INFO logging.
  - Togglable from UI via "Trials Log" checkbox.

- **In-memory backend**
  - `InMemoryJournalBackend` replaces file-based journal storage for multiprocess optimization.
  - Uses `mp.Manager().list()` for process-shared storage.

- **Concurrency**
- Keep Merlin's existing multi-process optimization architecture. Do not replace it with `study.optimize(..., n_jobs=...)` threading.

## Grid: Deterministic optimizer (parallel to Optuna)

Grid is generic in `grid_engine.py` (discovery, ranking, validation, storage,
DSR, diversity) and dispatches to a strategy-owned `fast_grid.py` backend. Each
backend advertises capability via `get_backend_metadata()` (normalized by
`get_fast_grid_backend_metadata`): `profile`, ordered `modes` (with
`default_enabled`), seed/allocation support, and `diversity_group_fields`.

**Key behavioral rules:**

- **Generic V2 full/budgeted planning**
  - V2 retains backend profile `full_enumeration_v2`, but planning is selected
    explicitly by `grid_v2_planning_policy`: missing/`full` preserves exact
    legacy enumeration; `sampled` requests a deterministic budgeted plan.
  - The sampled path is core-owned and generic across ordered logical blocks.
    It uses versioned ordered allocation plus balanced discrete LHS driven by
    named raw-PCG64 substreams, deterministic mixed-radix dedup/top-up, and
    canonical per-block ordering. It builds O(K) rows, not the full N rows.
  - Sampled plans reject inactive-axis dedup, varied dependency parents, and
    block layouts whose semantic disjointness cannot be proved. `K >= N` uses
    the unchanged full builder; seed/allocation are then non-operative.
  - Preview/run/WFA/Queue/storage report full, requested, planned/delivered,
    effective policy, versions, per-block facts, and plan fingerprint. V1 Grid,
    Optuna, semantic-key payloads, and selected reference reruns are unchanged.
  - `full_enumeration_v2` is a static backend profile, not proof that a specific
    plan is full. Use `effective_planning_policy` and the effective allocation
    facts to distinguish full from sampled execution.
  - For automatic sampled allocation, the budget must cover every enabled
    non-empty block and each such block receives at least one candidate. Manual
    allocation may use zero-percent blocks and may use a budget smaller than the
    block count. Capacity reflow can still place candidates into a zero-percent
    block after positively weighted blocks are exhausted.
  - Current plans publish `grid_v2_plan_identity_v3`,
    `grid_v2_semantic_identity_v2`, and `v2_runtime_contract_v1`. Plan identity
    includes normalized execution modes and feeds the WFA reuse identity. The
    plan fingerprint streams ordered semantic keys with constant additional
    memory instead of materializing a second key collection.
  - The reserved V2 runtime fields are, in order, `dateFilter`, `start`, `end`,
    and `warmupBars`. They are normalized by the core, excluded from candidate
    domains and semantic/plan identity, and cannot be Grid axes or option
    overrides. Only the three date fields are rebased for WFA windows.
  - New V2 Optuna, Grid, and WFA studies persist one exact request-level
    `v2_runtime_metadata_v1` envelope in `config_json.v2_runtime`; V1 omits the
    key and no migration is performed. Stored execution resolves the current
    registry/profile first, uses the shared current/legacy/defaulted runtime
    reader, ignores candidate runtime authority, applies operation dates last,
    and keeps Warmup separate behind the `dateFilter` preparation gate.
  - V2 profiles are validated when parsed: certified execution modes,
    variant-selector mappings, declared consumers, and dependency topology must
    be valid before runner or compiled packing. An optimized truly unbound
    execution parameter is fatal; a fixed truly unbound parameter is a warning.
    A family-compatible certified-but-unselected fixed mode parameter is
    informational, while an optimized one is fatal. Certified consumers from
    incompatible execution families follow the truthful unbound warning/fatal
    policy. Bool-group block discriminators count as covered
    axes; only axes inactive in every selected block warn. Plan diagnostics live
    at top-level `metadata["diagnostics"]`, not inside planning metadata.
    Stage 2 blocking surfaces must render structured `error` diagnostics, the
    normal warning panel must filter structured `warning` diagnostics, and
    `info` must remain secondary/non-nagging. UI rendering must not treat the
    string-only `validation_warnings` projection as its authority.

- **Legacy V1 strategy-owned generation profiles**
  - S03 (`sampled_by_mode`): splits the search into `cc_only` / `tbands_only` /
    `both` and allocates a budget across them; default generation is "LHS by
    mode" (seeded by `gridSeed`), falling back to full enumeration when a mode
    fits its budget.
  - S06 (`full_enumeration`): deterministically enumerates every selected
    `bracket` / `trail` combination — 48,480 by default, up to 436,320 when both
    optional Threshold OS/OB axes (`20, 30, 40`) are enabled. No seed/budget
    sampling in V1; an explicitly empty mode selection is an error. Imported V2
    strategies using equivalent execution variants remain eligible for generic
    V2 budgeted planning.
  - Missing `grid_enabled_modes` defaults to the backend's `default_enabled`
    modes (`default_grid_enabled_modes`) — no per-strategy server hardcode.
  - Allocation, mode budgets, and coverage % are surfaced through the Start page
    Grid preview (`POST /api/grid/preview`).

- **Fast / slow refinement**
  - Fast pass uses the strategy's Numba backend to screen candidates against a
    restricted "fast" objective set (`net_profit_pct`, `max_drawdown_pct`,
    `romad`, `profit_factor`, `win_rate`, `sharpe_ratio`, `sqn`). Sharpe and SQN
    are calculated only when requested.
  - Optional slow refinement re-runs the top-N fast candidates through the full
    Python strategy with the broader slow objective set (adds `sortino_ratio`,
    `ulcer_index`, `consistency_score`). Slow Objectives operate only on the
    selected slow-validated top candidates.
  - Selected fast candidates are always slow-validated against the real strategy
    (`validate_selected_candidates`); WFA OOS is also slow-authoritative.
  - Final objectives, primary objective, and constraint feasibility are stored on
    each trial; UI sorts feasible Pareto → feasible non-Pareto → infeasible. The
    shared Grid V1/V2 path computes exact two-objective Pareto membership in
    `O(n log n)` time. Three or more Grid objectives remain exact through the
    historical quadratic fallback; Optuna Pareto behavior is unchanged.
  - Direct Grid V1 and Grid V2 summaries time full fast ranking plus diversity
    selection as `ranking_seconds`. V1 WFA does not currently project this as a
    per-window timing diagnostic.

- **Constraints & diversity**
  - Constraints are soft feasibility/ranking rules (not candidate pruning),
    shared with Optuna, and are surfaced in the Grid Settings sidebar on both
    Results and Analytics (`build_grid_settings_view`).
  - Diversity capping groups candidates by the backend's `diversity_group_fields`,
    whose JSON-safe shape (S03 `list[str]`, S06 `dict[str, list[str]]`) is
    preserved end-to-end by `normalize_diversity_group_fields`.

- **Validation & ranking**
  - Constraints and score formula use shared `optuna_engine` helpers. Exact
    Grid Pareto membership is owned by `grid_pareto`; Grid results remain
    interoperable with Optuna results (same `trials` table, same display schema).
    DSR for Grid uses `build_grid_dsr_results`.
  - Budget is parsed from compact strings (e.g. `200k`) on the UI side; backends
    use canonical integer counts.

- **Strategy coverage**
  - Grid is enabled for `s03_reversal_v10`, `s03_reversal_v11`, and
    `s06_r_trend_v02`. A new strategy participates by adding its own
    `fast_grid.py` backend (and metadata); no shared-code or server changes are
    required.
  - `s03_reversal_v11` is a Backtester V1 strategy based on v10 with optional
    Emergency SL. Emergency SL exits set `TradeRecord.exit_reason` to
    `"Emergency SL"`. Same-bar re-entry after Emergency SL is intentionally
    Pine-compatible; normal signal exits preserve existing S03 V1 reversal
    timing. Do not restore older delayed Emergency SL re-entry text because that
    variant failed the TradingView parity gate.

## Lancelot Bundle Export

- `POST /api/studies/<id>/export/lancelot` is a narrow legacy integration certified only for `s03_reversal_v10`; a read-only two-column identity lookup rejects every other strategy before full study/candidate loading, stitched-OOS backfill, stored-runtime resolution, CSV access, hashing, or bundle work.
- Supported sources:
  - **Optuna / Grid studies** — `trialNumber` in payload selects the trial/candidate.
  - **WFA studies** — `windowNumber` in payload selects the window; the source trial is resolved via `is_best_trial_number` (or the selected per-module trial in `wfa_window_trials`).
- Bundle is built by `core.bundle_export.build_lancelot_partial_bundle`, which stamps the Merlin version, strategy version, CSV-derived symbol/timeframe, and canonical params.
- Lancelot export is not part of the generic Backtester V2 import or certification contract. New V2 strategies require neither Lancelot aliases nor export certification. Supporting another strategy requires a separate reviewed Merlin/Lancelot contract and implementation task after its live-trading contract is known.
- The accepted S03 v10 bundle remains live and candidate-only: `dateFilter=false`, `start=null`, and `end=null` are applied last, while Warmup remains a top-level bundle field. Historical Merlin trade/equity exports continue to use study/window dates.


## UI Notes

### Three-Page Architecture

**Start Page (`/` - index.html):**
- Strategy selection and parameter configuration
- Optimizer mode selector: **Optuna** or **Grid**
- Optuna settings (objectives + primary objective, budget, sampler, pruner, constraints)
- Grid settings (V2 Full/Budgeted planning, candidate budget, seed, top candidates, fast + optional slow objective sets, mode allocation, diversity, advanced)
- Grid preview panel calls `POST /api/grid/preview` to show parameter space size, mode allocation and coverage
- Initial Search Coverage mode toggle (Optuna) with coverage analysis and warmup auto-fill
- Trials Log toggle for Optuna trial-level logging control
- Walk-Forward Analysis settings (fixed day/calendar-month IS/OOS periods, day-only adaptive mode, adaptive cooldown, store top-N trials)
- Scheduled run queue management
- CSV file browser
- Dataset preview (WFA window layout)
- Run Optuna / Run Grid / Run WFA buttons
- Results automatically saved to database
- Light theme UI with dynamic forms from `config.json`

**Results Page (`/results` - results.html):**
- Studies Manager: List all saved optimization studies
- Database switching (multi-database support)
- Study details: View trials (Optuna) or windows (WFA)
- Pareto badge + constraint feasibility indicators for Optuna trials
- Grid Settings sidebar (shared with Analytics) including an enabled-Constraints row (`None` when none enabled)
- Equity curve visualization
- Parameter comparison tables
- Download trades CSV for IS/FT/OOS/Manual/WFA results (on-demand generation)
- Export an S03 v10 selected trial or WFA window as a Lancelot partial bundle
- Delete studies or update CSV file paths

**Analytics Page (`/analytics` - analytics.html):**
- WFA-focused research and analysis
- Multi-study equity curve comparison
- Aggregated (portfolio) equity curve with annualized profit and max drawdown
- Focused study mode with WFA window boundary overlays on equity chart
- Same shared Grid Settings sidebar as Results (identical enabled-Constraints row for Grid/WFA-Grid studies)
- Study sets: save/load/reorder named collections of studies (persisted in DB)
- Study summary table with sorting and filtering
- Filter by strategy, symbol, timeframe, WFA mode, IS/OOS periods
- Aggregated metrics: profit %, max DD %, win rate, WFE %, profitable windows %

### Frontend Architecture

- **main.js**: Start page logic, form handling, optimization launch
- **results-state.js**: Results page state management, localStorage/sessionStorage, URL query helpers
- **results-format.js**: Results page formatters, labels, stableStringify, MD5 hashing
- **results-tables.js**: Results page table/chart renderers, row selection, parameter details
- **results-controller.js**: Results page orchestration, API calls, event binding, modals
- **api.js**: Centralized API calls for all pages
- **strategy-config.js**: Dynamic form generation from `config.json`
- **ui-handlers.js**: Shared UI event handlers
- **optuna-ui.js**: Optuna Start-page UI helpers (objectives/constraints/sampler panels, coverage analysis)
- **optuna-results-ui.js**: Optuna Results-page UI helpers (dynamic columns/badges)
- **post-process-ui.js**: Post process UI helpers (Forward Test, DSR panels)
- **oos-test-ui.js**: OOS test UI helpers
- **wfa-results-ui.js**: WFA Results-page UI helpers
- **presets.js**: Preset management (load/save/import)
- **results.js**: Results page initialization
- **queue.js**: Scheduled run queue management (add/remove/execute items)
- **dataset-preview.js**: WFA window layout preview and validation
- **analytics.js**: Analytics page logic, state management, study selection
- **analytics-equity.js**: Analytics equity curve SVG rendering
- **analytics-filters.js**: Analytics filter panel (strategy/symbol/TF/WFA/IS-OOS)
- **analytics-table.js**: Analytics sortable study table with checkbox selection
- **analytics-sets.js**: Analytics study sets management (CRUD + membership)
- **analytics-sets-view.js**: Study sets view — sorting, filtering, bulk color/delete actions
- **utils.js**: Shared utility functions
- Forms generated dynamically from `config.json`
- Strategy dropdown auto-populated from discovered strategies
- No hardcoded parameters in frontend
- Direct Start-page Backtest, trade download, Optuna/Grid/WFA, Queue-item
  creation, and automatic Grid Preview require a successfully rendered config
  whose accepted request ID matches the selected strategy. A new or failed
  config load clears only strategy-generated fields, strategy information, and
  Preview state; CSV selection, date/Warmup, database, budget, WFA, Queue, and
  Preset controls are preserved. Obsolete config successes and failures are
  ignored. Persisted Queue execution remains independent of editable-form
  readiness.
- Config-load failures reuse the backend's concise `error` text. Warning-only
  configs remain ready and may log `validation_warnings` to the console;
  informational diagnostics are not promoted into user-facing warnings. TZ64C
  intentionally does not add diagnostic panels, contract-default seeding,
  Preset/runtime precedence, a centralized JavaScript runtime-name fallback,
  or blank-End-Time semantics.

### Backend Architecture (server split)

- **server.py**: Thin entrypoint, Flask app creation, route registration, test re-exports
- **server_services.py**: All helper/utility functions (no route decorators), safe logging via `_get_logger()`
- **server_routes_data.py**: Pages + studies/tests/trades + presets + strategies + DB management + CSV browse + queue endpoints
- **server_routes_run.py**: Optimization status/cancel + optimize/walkforward/backtest (run endpoints)
- **server_routes_analytics.py**: Analytics page + WFA summary API endpoint

## API Endpoints Reference

### Page Routes
- `GET /` - Serve Start page
- `GET /results` - Serve Results page
- `GET /analytics` - Serve Analytics page

### Optimization
- `POST /api/optimize` - Run Optuna or Grid optimization (chosen via `optimization_mode`/`optimizer_mode`), returns study_id
- `POST /api/grid/preview` - Preview Grid parameter space, mode allocation and coverage without running
- `POST /api/walkforward` - Run WFA (fixed or adaptive mode), returns study_id
- `POST /api/backtest` - Run single backtest (no database storage)
- `POST /api/backtest/trades` - Download trades CSV for single backtest
- `GET /api/optimization/status` - Get current optimization state
- `POST /api/optimization/cancel` - Cancel running optimization

### Studies Management

- `GET /api/studies` - List all saved studies
- `GET /api/studies/<study_id>` - Load study with trials/windows
- `DELETE /api/studies/<study_id>` - Delete study
- `POST /api/studies/<study_id>/update-csv-path` - Update CSV path
- `POST /api/studies/<study_id>/test` - Run manual test on selected trials
- `GET /api/studies/<study_id>/tests` - List manual tests
- `GET /api/studies/<study_id>/tests/<test_id>` - Load manual test results
- `DELETE /api/studies/<study_id>/tests/<test_id>` - Delete manual test
- `POST /api/studies/<study_id>/trials/<trial_number>/trades` - Download IS trades CSV
- `POST /api/studies/<study_id>/trials/<trial_number>/ft-trades` - Download Forward Test trades CSV
- `POST /api/studies/<study_id>/trials/<trial_number>/oos-trades` - Download OOS Test trades CSV
- `POST /api/studies/<study_id>/tests/<test_id>/trials/<trial_number>/mt-trades` - Download Manual Test trades CSV
- `GET /api/studies/<study_id>/wfa/windows/<window_number>` - Get WFA window details with module trials
- `POST /api/studies/<study_id>/wfa/windows/<window_number>/equity` - Generate WFA window equity curve on-demand
- `POST /api/studies/<study_id>/wfa/windows/<window_number>/trades` - Download WFA window trades CSV
- `POST /api/studies/<study_id>/wfa/trades` - Download stitched WFA OOS trades CSV
- `POST /api/studies/<study_id>/export/lancelot` - Build an S03 v10 Lancelot partial bundle from a trial (Optuna/Grid) or window (WFA); other strategies return HTTP 400

### Database Management
- `GET /api/databases` - List all `.db` files with active marker
- `POST /api/databases/active` - Switch active database
- `POST /api/databases` - Create new timestamped database

### CSV Browse
- `GET /api/csv/browse` - Browse CSV directory (files + subdirectories)

### Run Queue
- `GET /api/queue` - Load scheduled run queue state
- `PUT /api/queue` - Save/update queue state
- `DELETE /api/queue` - Clear queue state

Queue is a generic V1/V2 transport; normal Optimize/WFA launch boundaries own
runtime validation. Queue reads never rewrite or delete `src/storage/queue.json`.
Unreadable encoding/JSON or invalid top-level Queue shape is preserved and
reported to the UI, while legacy item/source normalization remains lenient.
A legacy item without `warmupBars` omits that launch field so the core default
`1000` applies; an explicitly present malformed value still reaches strict V2
validation. Legacy WFA items without `periodUnit` remain day items; fixed
calendar-month items retain authoritative month counts and use labels such as
`WFA-F 2m/1m`. Preset runtime integration remains deferred and uncertified.

### Analytics
- `GET /api/analytics/summary` - WFA studies summary with filters and aggregated metrics
- `POST /api/analytics/equity` - Aggregate equity curves for selected study IDs
- `POST /api/analytics/equity/batch` - Batch aggregate equity curves for multiple groups
- `GET /api/analytics/studies/<study_id>/equity` - Stitched OOS equity for a single WFA study (cached)
- `GET /api/analytics/studies/<study_id>/window-boundaries` - Get WFA window boundary timestamps
- `GET /api/analytics/all-studies/equity` - Aggregated equity across all WFA studies (cached)

### Study Sets
- `GET /api/analytics/sets` - List all study sets (optionally hydrated with cached analytics summaries)
- `POST /api/analytics/sets` - Create a new study set
- `PUT /api/analytics/sets/<set_id>` - Update study set (name, color, study_ids, sort_order)
- `DELETE /api/analytics/sets/<set_id>` - Delete a study set
- `GET /api/analytics/sets/<set_id>/equity` - Aggregated equity curve for a study set (cached)
- `PUT /api/analytics/sets/bulk-color` - Bulk-assign a color token to multiple sets
- `POST /api/analytics/sets/bulk-delete` - Bulk-delete study sets
- `PUT /api/analytics/sets/reorder` - Reorder study sets

### Strategy & Presets
- `GET /api/strategies` - List available strategies
- `GET /api/strategies/<strategy_id>` - Get strategy metadata
- `GET /api/strategy/<strategy_id>/config` - Get strategy parameter schema
- `GET /api/presets` - List presets
- `POST /api/presets` - Create preset
- `GET/PUT/DELETE /api/presets/<name>` - Load/update/delete preset
- `PUT /api/presets/defaults` - Update default preset values
- `POST /api/presets/import-csv` - Import preset from CSV parameter block

## Performance Considerations

- Use vectorized pandas/numpy operations
- Pre-extract NumPy arrays from DataFrame columns before strategy loops (`.to_numpy()`)
- Reuse indicator calculations where possible
- Avoid expensive logging in hot paths (optimization loops)
- `trade_start_idx` skips warmup bars in simulation
- Database uses WAL mode for concurrent read access
- Bulk inserts used for saving trials (executemany, not loop)
- In-memory Optuna backend eliminates file I/O for trial communication between processes
- Trial deduplication prevents wasted evaluations of already-seen parameter sets

## Current Strategies

| ID | Name | Description |
|----|------|-------------|
| `s01_trailing_ma` | S01 Trailing MA | Complex trailing MA with 11 MA types, close counts, ATR stops |
| `s03_reversal_v10` | S03 Reversal | Reversal strategy using close-count confirmation and T-Bands hysteresis (LHS/full Grid backend) |
| `s03_reversal_v11` | S03 Reversal | Backtester V1 v10 variant with optional Emergency SL and LHS/full Grid backend |
| `s04_stochrsi` | S04 StochRSI | StochRSI swing strategy with swing-based stops |
| `s06_r_trend_v02` | S06 R-Trend | Williams %R trend/reversal entries with Bracket or Trail exits (full-enumeration Grid backend) |

## Key Files for Reference

| Purpose | File |
|---------|------|
| Full architecture | `docs/PROJECT_OVERVIEW.md` |
| Adding strategies | `docs/ADDING_NEW_STRATEGY.md` |
| Database operations | `src/core/storage.py` |
| WFA engine (fixed + adaptive) | `src/core/walkforward_engine.py` |
| Start page logic | `src/ui/static/js/main.js` |
| Results page logic | `src/ui/static/js/results-controller.js` (orchestration) |
| Analytics page logic | `src/ui/static/js/analytics.js` |
| Analytics study sets | `src/ui/static/js/analytics-sets.js` |
| Equity aggregation | `src/core/analytics.py` |
| Queue management | `src/ui/static/js/queue.js` |
| Flask API entrypoint | `src/ui/server.py` |
| Flask services/helpers | `src/ui/server_services.py` |
| Flask data routes | `src/ui/server_routes_data.py` |
| Flask run routes | `src/ui/server_routes_run.py` |
| Flask analytics routes | `src/ui/server_routes_analytics.py` |
| S03 example | `src/strategies/s03_reversal_v10/strategy.py` |
| S03 v11 Emergency SL | `src/strategies/s03_reversal_v11/strategy.py` |
| S04 example | `src/strategies/s04_stochrsi/strategy.py` |
| S03 Grid backend (LHS/full) | `src/strategies/s03_reversal_v10/fast_grid.py` |
| S03 v11 Grid backend (LHS/full) | `src/strategies/s03_reversal_v11/fast_grid.py` |
| S06 Grid backend (full enumeration) | `src/strategies/s06_r_trend_v02/fast_grid.py` |
| config.json example | `src/strategies/s04_stochrsi/config.json` |
| Test baseline | `data/baseline/` |

### Conditional Fast Grid Sharpe and SQN

V1 and V2 Grid expose eight common Fast Objective controls: Net Profit, Max
Drawdown, RoMaD, Profit Factor, Win Rate, Monthly Sharpe, Daily Sharpe, and SQN.
At most six may be selected in one request; Optuna independently retains its
existing six-objective cap. Monthly Sharpe, Daily Sharpe, and SQN are computed
only when requested. Daily Sharpe is explicitly Fast-only; Sortino, Ulcer
Index, and Consistency remain Slow-only, and Fast Constraints are unchanged.

Grid drops a candidate when any selected objective is non-finite. SQN is
undefined below 30 completed trades, so short direct-Grid or WFA windows can
lose much of their rankable population; a WFA window with no usable candidate
fails with its window number and selected objective context.
Daily Sharpe can similarly shrink the rankable population when its ratio is
unavailable because of zero trades, too few observations, zero variance, or an
invalid series.

### Advanced Metric Boundary Contract

`StrategyResult.metric_start_idx` identifies the first evaluation observation
inside its curves; `metric_initial_equity` is the equity immediately before
that observation. Strategy producers must populate both when prepared data
contains technical warmup. Advanced Sharpe, Sortino, Ulcer Index, and
Consistency use this boundary; realized basic metrics and their drawdown/RoMaD
contract do not. Calendar returns assign a transition bar to its new month and
close the old month with the preceding bar. Fast Grid implementations must
match this contract and must not count technical warmup months.

Realized Max DD is balance-based, not mark-to-market. It scans the complete
realized `balance_curve`, including the final observation and flat technical
warmup prefix; the flat prefix cannot change the maximum. At each finite
observation it uses the running realized-balance peak. Percentage DD is the
maximum `(peak - balance) / peak * 100` for positive peaks, while absolute DD
is the independent maximum `peak - balance`. `metric_start_idx`,
`metric_initial_equity`, and the optional Net Profit starting balance do not
seed or truncate this scan. RoMaD uses the corrected percentage DD. Slow, all
three V1 Fast backends, and both V2 compiled families share this contract.
Future Strategy Lab mark-to-market DD must use a distinct metric name.

Monthly Sortino requires real downside months and is often `None` on two- to
four-month windows; selecting it as a Slow objective may remove many or all
candidates.

Delayed WFA OOS transformations remove live technical warmup observations,
retain only the sparse scheduled flat prefix, produce strictly increasing
unique timestamps, and rebase the metric boundary to zero with an explicit
anchor. Non-delayed stitched output remains unchanged.

### Request-gated Daily Sharpe metric

`AdvancedMetrics.sharpe_daily` is an explicit opt-in metric and
is distinct from the existing unannualized Monthly Sharpe. It groups the
evaluation-only mark-to-market curve by UTC days since the Unix epoch, uses the
canonical pre-evaluation equity anchor, emits fractional simple returns for
observed days (including partial edge days), subtracts `rf / 365`, uses
population variance, and annualizes with `sqrt(365)`. Any non-finite evaluation
equity observation or non-positive opening denominator invalidates the complete
Daily Sharpe series.

When the request is disabled or the enabled daily series is structurally
invalid, both optional diagnostics are `None`. With a valid constructed series,
both are integers, including a genuine `0`; the ratio may still be `None` for
no completed trades, insufficient observations, or zero variance. Active means
`abs(raw return) > 1e-12` and is descriptive, not a hard gate. The three V1
Fast backends and both V2 compiled execution families can calculate the same
ratio and diagnostics when their internal `compute_sharpe_daily` request is
true. They build one shared contiguous `int32` UTC day array per dataset/window,
stream constant per-candidate state, and validate selected rows against the
canonical reference. The canonical compounded-return self-check stays
reference-only; Fast parity supplies the same boundaries, Welford statistics,
eligibility, and strict invalidation without a product accumulator. Sparse WFA
timestamps never imply synthesized dates. It is a maximize Optuna objective and
an explicit Fast-only Grid objective; generic Queue transport preserves it.
Final fixed and Adaptive WFA windows always report Daily Sharpe for real IS and
real undelayed dense OOS series, even when another objective selected the
candidate. Delayed/no-trade OOS fields remain absent because a sparse flat
prefix would invent omitted calendar gaps. SQLite storage is nullable and does
not backfill historical rows. DSR, Monthly Sharpe, exports, stitched portfolio
metrics, and active-day constraints remain unchanged/deferred.
