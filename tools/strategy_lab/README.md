# Strategy Lab — Phase 0 through Phase 1-B Stage 1

Strategy Lab is a local, research-only tool for certified Backtester V2
strategies. Phase 0 ships configuration, pre-registration, deterministic V2
plan identity, and a read-only market-data inventory. Phase 1-A adds strict
source/segment validation and a bounded, resumable candidate-level dataset
generator. It does not calculate selection rules or inspect
development/holdout performance.

Phase 1-B Stage 1 adds opt-in real-pack structural certification, compiled to
reference and selected typed-Slow parity, direct one-thread/two-thread
determinism evidence, deterministic real smoke generation, and an isolated
production HTTP WFA tie-back. These are pass/fail certification surfaces, not
strategy analysis.

The real-pack parity/thread/smoke command is verified. The production WFA
tie-back is currently an opt-in blocker reproducer: production does not yet
propagate the requested worker count or persist the Grid V2 plan fingerprint
and selected candidate identity. A separate reviewed production patch must fix
those three surfaces before this tie-back can pass.

The initial frozen run is `runspecs/s06_bracket_mvp.json`: S06 B2 bracket-only,
480 full-plan candidates, 30-minute data, 2-month IS / 1-month OOS, eight
calendar windows, and a 24/94 development/holdout ticker split. Normalized
economics are `initialCapital=1000.0`, `riskPerTrade=2.0`,
`contractSize=0.0001`, and `commissionPct=0.05`. Slippage and funding are
intentionally not modelled. `tickSize=0.0001` is provenance only because the
execution profile uses `priceRounding=none`.

## Commands

Run every command from the repository root with the configured project Python.
The module commands bootstrap the repository `src/` path themselves and do not
require `PYTHONPATH` or prior imports.
On this Windows repository, use:

```powershell
C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe -m tools.strategy_lab.config tools\strategy_lab\runspecs\s06_bracket_mvp.json
C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe -m pytest tests\strategy_lab -q
```

Run the explicit real-pack certification into a fresh task-owned directory:

```powershell
C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe -m tools.strategy_lab.certify `
  tools\strategy_lab\runspecs\s06_bracket_mvp.json `
  --data-root "<read-only-data-root>" `
  --work-dir tools\strategy_lab\tmp\phase1b-certification
```

The isolated WFA tie-back is deliberately outside normal pytest discovery and
reuses `smoke_one` from that command. It currently must stop on the three known
production transport/resource blockers; do not treat it as a passing
certification until the separate production patch lands:

```powershell
$env:MERLIN_STRATEGY_LAB_DATA_ROOT = "<read-only-data-root>"
$env:MERLIN_STRATEGY_LAB_CERT_WORK_DIR = "<absolute-phase1b-certification-dir>"
C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe -m pytest `
  tests\strategy_lab\phase1b_real_wfa_certification.py -q `
  --basetemp=tools\strategy_lab\tmp\phase1b-wfa-pytest
```

Generate an explicitly scoped smoke dataset (replace the example root and
inventory member with local values):

```powershell
C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe -m tools.strategy_lab.generate `
  tools\strategy_lab\runspecs\s06_bracket_mvp.json `
  --data-root "<read-only-data-root>" `
  --output-dir tools\strategy_lab\output\phase1a-smoke `
  --ticker <canonical-symbol> --window 1
```

Repeat `--ticker` and `--window` for a larger subset. Use `--resume` only for
an identity-compatible partial output. The normal data-root precedence is an
explicit `--data-root`, then `MERLIN_STRATEGY_LAB_DATA_ROOT`; absence is an
error. CLI progress reports validation, ticker/window group reuse or
regeneration, completion state, output path, and duration, never performance
summaries.

Rebuild the current tracked inventory only during an explicitly reviewed
pre-registration update:

```powershell
C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe -m tools.strategy_lab.inventory `
  --data-root "C:\Users\mt\Desktop\Strategy\S_Python\Market Data_PY\test_pack_all-nw-2_0801\30m" `
  --output tools\strategy_lab\runspecs\tickers_current.json `
  --provenance-output tools\strategy_lab\output\phase0_current_pack_provenance.json `
  --expected-ticker-count 118 --development-ticker-count 24 `
  --timeframe-minutes 30 --initial-capital 1000 --risk-per-trade-pct 2 `
  --contract-size 0.0001 --max-stop-pct 8 --minimum-size-steps 100
```

Linux/VPS hosts use their configured project environment and Linux path syntax.
The tool never searches for alternate data or stores the absolute root in
tracked identity.

## Read-only and output contract

Source CSV files are opened only for reading. Before Merlin's `load_data()` is
called, Phase 1-A verifies the accepted filename/file identity, exact raw
header and column count, integer Unix-second timestamps, strict ordering,
duplicates, exact 30-minute cadence, first/last boundaries, row count, finite
positive OHLC values and price relationships, and finite non-negative Volume.
Every selected source is checked before any candidate executes, and failures
produce deterministic `data_quality.csv` diagnostics without a complete
manifest. The inventory builder rejects
subdirectories, source symlinks, malformed filenames/schema, duplicate
canonical symbols, unexpected counts, and sizing headroom below 100 contract
steps. Filename dates are inclusive UTC days; each file must have the derived
full-day row count and exact first/last bar boundaries. Numeric timestamps are
Unix seconds, with no millisecond inference. The sizing gate is an
initial-capital bound and does not claim unchanged headroom after an extreme
equity drawdown. The builder never sorts or rewrites source rows and never
writes beside market data.

Tracked, cross-host facts live in `runspecs/`. Host/platform, resolved absolute
root, source mtimes, verification time, and tool Git revision are local
provenance under ignored `output/`. Future local caches and temporary files use
the narrowly ignored `cache/` and `tmp/` directories.

## Dataset and resume contract

The generator constructs one validated V2 plan and directly executes every
candidate for independent IS and OOS prepared segments. It requires effective
`compiled_numba` execution for every returned row and rejects silent reference
fallback before group publication. A live availability precheck reports the
compiled-unavailability reason before any candidate execution or publication;
post-execution row and backend checks remain in force. Candidate error rows
retain their candidate ID and original error text. Group records bind IS/OOS backend facts for
resume, and the final manifest summarizes execution modes and config packing.
It pins one sequential outer worker, one separate compiled worker, one runtime
Numba thread, the run-spec-owned 512 MB signal-cache limit, and
`slow_enrich_selected=False`; process-global Numba thread state is restored even
after failure. Cleanup failures are attached to an existing primary exception
instead of replacing it. Calendar boundaries come from Merlin's
authoritative calendar-month builder, and each segment must contain exactly
1,000 warmup bars followed by its complete evaluation range.

`candidates.json` records all candidate identities and geometry in matrix row
order. Each `groups/<symbol>/window_<NN>.npy` is an atomic float64 array with
shape `(candidate_count, 2, 20)`. The segment axis is `[is, oos]`; the metric
axis is declared in the manifest and includes canonical performance metrics,
Daily Sharpe diagnostics, SQN, and five guardrail fields. Natural NaN and
infinity remain unavailable facts, diagnostic `None` becomes NaN, genuine
zero remains `0.0`, and zero-trade candidates are retained. SQN can naturally
remain unavailable on short segments.

JSON, CSV, and group files publish through unique same-directory temporary
files and atomic replacement. `manifest.partial.json` claims only groups that
have been published and checksummed. Resume requires exact schema, run,
inventory, plan, axes, scope, version, and resource identity; claimed files are
rechecked for SHA-256, size, shape, and dtype. Missing or corrupt claimed
groups regenerate, while unlisted files are never reused. `manifest.json` is
the sole complete marker, and an exact verified completed run is a no-op. A
completed output is immutable during every re-run: source, identity, or artifact
failure is reported without writing diagnostics or otherwise changing that
directory. Output paths equal to or beneath the resolved market-data root are
rejected before any write or candidate execution.

The scope label is `full` only when the exact complete accepted inventory and
all declared windows are selected; every strict subset is `smoke`. Phase 1-A
verification uses only synthetic temporary sources and bounded smoke scopes.

## Stage boundary

- Stage A is host-independent: package/config/inventory code, synthetic tests,
  canonical serialization, V2 plan construction, bounded smoke generation,
  deterministic projection, and resume verification.
- Stage B requires the exact configured source pack and produces/verifies the
  tracked inventory, 118-file and 24/94 facts, sizing headroom, deterministic
  digests, and before/after source preservation.

The current primary holdout is cross-sectional confirmation on new tickers over
the same six months as development, not independent temporal confirmation.
Windows 7-8 are mandatory descriptive temporal checks in the eventual MVP and
have no Phase 0 result, bootstrap interval, or pass/fail threshold.

Execution resources intentionally remain inside the current generation
identity, so changing worker, Numba-thread, or cache limits requires a reviewed
re-freeze for this MVP.

## Phase boundary and explicit non-goals

Phase 1-B Stage 1 may validate the complete source pack but executes only the
outcome-independent CRVUSDT representative, creates temporary smoke datasets,
and runs one WFA study under pytest's isolated storage. It does not publish the
canonical dataset. Stage 2 alone may generate the canonical 944-group output at
`tools/strategy_lab/output/s06_bracket_mvp_pre_mtm_v1`, and only after Stage 1
is independently reviewed, the production WFA fix lands, and the tie-back is
rerun successfully from an approved commit. Canonical generation remains
prohibited while that gate is blocked. No command here calculates selection
rules, inspects development or holdout outcomes, changes metrics, or draws a
profitability conclusion. Backtester V1 is permanently unsupported.

The historical prototype archive under
`docs/_work/dev_01_strategy-lab/prototype_01_grid-selection-research/` is
research evidence only and is not imported by production tools.
