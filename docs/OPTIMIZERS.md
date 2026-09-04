# V1 Optimizer Contracts

This document owns the compact current contracts for Backtester V1 Optuna and
strategy-owned Fast Grid. Metric definitions remain in [Metrics](METRICS.md).

## Policy boundary

V1 supports Optuna and strategy-owned Fast Grid where a backend exists. New V2
Optimize and WFA requests are explicitly Grid-only; historical V2 Optuna
compatibility remains supported as described in
[V2 architecture](engine_v2/ARCHITECTURE.md). Grid V2 planning details also
belong there.

## V1 Optuna

- Pruning is supported only for single-objective studies because Optuna does
  not support `Trial.should_prune()` for multi-objective trials.
- A non-finite or NaN objective follows the failed-trial contract and is never
  presented as a valid sampler result.
- Constraints are soft feasibility/ranking information and are evaluated only
  for successful trials. Multi-objective ordering is feasible Pareto, feasible
  non-Pareto, infeasible, total violation, primary objective, then trial number.
- UI Initial Search Coverage hints use block multipliers `1`, `3`, `5`, `9`,
  and `17` and reduce invalid boolean-group combinations. Engine coverage-trial
  generation is separate: it uses the full Cartesian product of the prepared
  categorical axes and quantized numeric anchors.
- Duplicate detection uses deterministic parameter identity, marks skips with
  `merlin.duplicate_skipped`, applies the default soft duplicate-cycle limit
  `18`, and stops when a finite discrete search space is exhausted.
- `trials_log` defaults to false.
- Multiprocess Optuna remains process-based; its journal-backed path uses
  `InMemoryJournalBackend` over process-shared `mp.Manager().list()` state.
  Preserve that architecture; do not replace it with threaded
  `study.optimize(..., n_jobs=...)` execution.

## V1 Fast Grid

The current strategy-owned backends are `s03_reversal_v10`,
`s03_reversal_v11`, and `s06_r_trend_v02`.

- S03 uses `sampled_by_mode` with ordered `cc_only`, `tbands_only`, and `both`
  logical modes and shared mode-budget allocation. Per mode, it uses full
  enumeration when the budget covers the mode space, a deterministic seeded
  subset at coverage of at least 50% but below full, and seeded LHS below 50%.
  The sampling seed is transported as `grid_seed`.
- S06 uses deterministic `full_enumeration`. Its verified default count is
  currently `48,480`, derived from the current config optimize ranges. The
  current maximum is `436,320` when both optional Threshold OS/OB axes use
  `20`, `30`, and `40`.
- When `grid_enabled_modes` is missing, `default_grid_enabled_modes` uses
  explicit backend mode metadata, including S06's. S03 declares no explicit
  enabled-mode list, so the helper returns an empty list and preserves its
  existing S03-specific defaulting; neither path uses a per-strategy server
  hardcode.
- Preserve the accepted JSON-safe diversity shapes: S03 uses `list[str]` and
  S06 uses `dict[str, list[str]]`; shared Grid normalizes both.
- Every selected Fast candidate is Slow-rerun before authoritative final use.
