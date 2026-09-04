# Maintaining or Importing a Backtester V1 Strategy

This filename is retained for compatibility, but this is the legacy V1 guide.
New strategy development should normally target Backtester V2 and follow the
[V2 import procedure](ADDING_NEW_STRATEGY_V2.md). Use this document when a task
explicitly maintains or imports a V1 strategy.

V1 strategies own their Python execution loop and may optionally own a Fast
Grid backend. That is not the V2 design: V2 strategies use generic core
execution and Grid planning.

## 1. Define the Pine/external contract

Before translating code, record:

- strategy name/version and the exact source used;
- inputs, types, defaults, groups, bounds, and option order;
- indicator and signal rules, execution timing, sizing, fees, and exits;
- date filtering, warmup requirements, chart timezone, and bar assumptions;
- expected metrics and trades from a fixed dataset.

The tracked [S03 Reversal v10 Pine source](S_03-Reversal_v10_for-import.pine)
is useful provenance and a concrete V1 translation example. Preserve external
reference files byte-for-byte and normalize only separate test fixtures.

## 2. Create the package

```text
src/strategies/<strategy_id>/
  __init__.py
  config.json
  strategy.py
  fast_grid.py       # optional V1 Fast Grid backend
```

Packages containing `config.json` and an importable strategy class are
discovered through `src/strategies/__init__.py`; do not add a second registry.

## 3. Define `config.json`

V1 is the default when `engine` is absent. Keep all public parameter names
camelCase end to end.

```json
{
  "name": "S05 Example",
  "version": "v01",
  "description": "Example legacy strategy",
  "parameters": {
    "maLength": {
      "type": "int",
      "label": "MA Length",
      "default": 50,
      "min": 2,
      "max": 500,
      "group": "Signal",
      "optimize": {
        "enabled": true,
        "min": 10,
        "max": 100,
        "step": 5
      }
    }
  }
}
```

Supported parameter types include `bool`, `int`, `float`, and `select`. Keep
select option spelling/order stable when it participates in identities or
baselines. Optimization metadata controls the UI search domain; disabled
parameters remain fixed at their configured/requested values.

Use `depends_on` for a child whose optimization relevance depends on a parent
boolean:

```json
"trailDistance": {
  "type": "float",
  "default": 2.0,
  "depends_on": "useTrail",
  "optimize": {"enabled": true, "min": 0.5, "max": 5.0, "step": 0.5}
}
```

Small invalid boolean combinations can use the established
`optimization_rules.bool_groups` declaration, such as
`mode="at_least_one_true"`. Keep rule semantics declarative; do not hardcode
the same restriction independently in UI, Optuna, and Grid.

## 4. Define the params dataclass

Use a dataclass whose field names and defaults match `config.json` exactly:

```python
from dataclasses import dataclass


@dataclass
class S05Params:
    maLength: int = 50
    useTrail: bool = True
    trailDistance: float = 2.0
```

Follow the repository-wide serialization and public-name conventions in
[`CLAUDE.md`](../CLAUDE.md).

## 5. Implement the V1 strategy

Subclass the established base where appropriate and expose stable identity:

```python
class S05Example:
    STRATEGY_ID = "s05_example"
    STRATEGY_NAME = "S05 Example"
    STRATEGY_VERSION = "v01"

    @staticmethod
    def run(df, params, trade_start_idx=0):
        ...
```

Implementation rules:

- pre-extract numeric NumPy arrays and normalize floating working arrays;
- calculate indicators once, outside the bar loop;
- preserve Pine bar timing and avoid lookahead/repainting;
- use `trade_start_idx` to exclude technical warmup from trading;
- keep position sizing, commission, stop/target/trail behavior, and date
  boundaries explicit;
- return the standard `StrategyResult` with aligned balance/equity/timestamps
  and complete trades;
- set `metric_start_idx` and `metric_initial_equity` when warmup precedes the
  evaluation interval, per [Metrics](METRICS.md).

Do not change shared engines merely to hide a strategy translation mismatch.
First establish whether the Pine contract, input data, or V1 implementation is
responsible.

## 6. Preserve V1 optimizer behavior

V1 supports Optuna and Grid. Optuna objectives, constraints, coverage,
deduplication, multiprocessing, storage, and result semantics are shared
infrastructure; strategy code supplies only correct parameters and execution.
The live contracts are in [V1 optimizers](OPTIMIZERS.md).

If no Fast Grid backend exists, Grid is unavailable for that strategy and the
UI retains its V1 fallback to Optuna. Do not claim generic V2 Grid capability
for a V1 package.

### Optional V1 Fast Grid backend

Add `fast_grid.py` only when the task explicitly requires a certified V1 Fast
path. Follow an existing V1 backend with the appropriate profile:

- `sampled_by_mode` for ordered logical modes with seeded allocation/LHS;
- `full_enumeration` for deterministic complete enumeration.

The backend exposes metadata, parameter-space construction, candidate
generation, data preparation, evaluation, and selected-candidate validation
through the interfaces consumed by `src/core/grid_engine.py`. It must:

- preserve config domain and option order;
- produce deterministic candidate/identity order;
- evaluate supported Fast objectives and guardrail facts correctly;
- match Slow strategy execution on selected candidates;
- declare diversity fields in the established JSON-safe shape;
- avoid file I/O and strategy-global mutable caches in candidate loops.

Fast objective and metric rules are centralized in [Metrics](METRICS.md).
Adding a V1 Fast backend does not create a V2 profile or satisfy the V2 import
contract.

### S03 v11 preservation guard

For the existing `s03_reversal_v11` strategy, Emergency SL exits retain the
established `Emergency SL` reason and intentionally permit Pine-compatible
same-bar re-entry. Normal signal exits retain the current V1 reversal timing.
Do not restore the older delayed Emergency-SL re-entry variant; it failed
TradingView parity.

## 7. Test and certify

Add focused coverage for:

- config discovery, defaults, types, and camelCase names;
- deterministic execution on a bounded fixture;
- signal/indicator edge cases and warmup behavior;
- external/Pine baseline metrics and trade signatures where parity is claimed;
- date filtering, fills, stops, targets, trails, fees, and exit reasons;
- Optuna configuration/execution when the strategy supports V1 Optuna;
- Fast Grid count/order/determinism and Fast-vs-Slow parity when a backend is
  present;
- storage/UI transport affected by strategy-specific fields.

Use isolated pytest paths and never overwrite a baseline to make a failure
pass. Baseline regeneration requires an explicit reviewed task. See the
[test guide](../tests/README.md) for suite selection.

## Common V1 pitfalls

- mismatched config/dataclass defaults or snake_case public names;
- centered rolling windows, negative shifts, or other lookahead;
- applying date filters before required technical warmup;
- off-by-one Pine fill/exit timing;
- converting option labels differently in UI and Python;
- calculating indicators or constructing objects inside the hot bar loop;
- treating a strategy-owned Fast approximation as the Slow execution
  authority;
- copying V2 profile concepts into a V1 package without a migration contract.
