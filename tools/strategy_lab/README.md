# Strategy Lab — Phase 0 through Phase 3L-B

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

The real-pack parity/thread/smoke command and complete eight-window production
HTTP WFA tie-back are verified. The transport path honours the frozen requested
worker count, persists the exact Grid V2 plan fingerprint, and retains selected
Fast rank plus semantic/candidate identity. Phase 1-B is complete: the immutable
944-group canonical pre-MTM dataset was generated from clean commit `1bdfda76`
at `output/s06_bracket_mvp_pre_mtm_v1`, and its second invocation was a verified
no-op. Phase 2 must use a new schema and output directory.

Phase 2 Stage 1 adds the research-only, request-gated bar-close metric
`max_drawdown_mtm_pct` for generic V2 position-family execution. Dataset schema
`strategy_lab_dataset_v2` appends it after the 20 pre-MTM metrics, producing
`float64` groups shaped `[480, 2, 21]`. The first 20 columns and all frozen
candidate, plan, source, and window identities remain unchanged. MTM is not a
public objective or persisted Merlin result; requested signal-reversal
execution is unsupported and fails explicitly. The immutable canonical
944-group v2 dataset is the read-only input to Phase 3L-A. Phase 3L-A adds a
compact, version-dispatched analysis core for the frozen development rules and
evidence contract. Phase 3L-B adds a separate fixed-capacity development
allocation and exact-calendar dataset-comparison path. Neither phase nominates
a policy or concludes strategy quality.

The initial frozen run is `runspecs/s06_bracket_mvp.json`: S06 B2 bracket-only,
480 full-plan candidates, 30-minute data, 2-month IS / 1-month OOS, eight
calendar windows, and a 24/94 development/holdout ticker split. Normalized
economics are `initialCapital=1000.0`, `riskPerTrade=2.0`,
`contractSize=0.0001`, and `commissionPct=0.05`. Slippage and funding are
intentionally not modelled. `tickSize=0.0001` is provenance only because the
execution profile uses `priceRounding=none`.

TZ-17.2 adds six tracked full-policy run specs named
`s06_v064a2_{reversal,trend}_{r_trail,chandelier,fixed_af_sar}.json`.
They retain the canonical inventory, windows, resources, economics, and
preregistration while fixing one entry/trail pair. Full dataset generation is
deferred. The reusable smoke certifier derives count and group shape from the
loaded plan; only the 480-candidate RR Bracket profile is compared with the
immutable legacy matrix when strategy ID/version, candidate count, plan
fingerprint, semantic-key digest, and RR/no-trail topology all match the frozen
`s06_bracket_mvp` identity.

## Commands

Run every command from the repository root with the configured project Python.
The module commands bootstrap the repository `src/` path themselves and do not
require `PYTHONPATH` or prior imports.
On this Windows repository, use:

```powershell
C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe -m tools.strategy_lab.config tools\strategy_lab\runspecs\s06_bracket_mvp.json
C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe -m pytest tests\strategy_lab -q
```

Run the safe development analysis. The output must be outside the input
dataset and contains exactly five deterministic files:

```powershell
C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe -m tools.strategy_lab.analysis.cli analyze `
  --dataset tools\strategy_lab\output\s06_bracket_mvp_mtm_v2 `
  --scope development `
  --output tools\strategy_lab\tmp\phase3la-development
```

The reader supports `strategy_lab_dataset_v2`, `strategy_lab_rules_v1`, and
`strategy_lab_evidence_v1`. The manifest owns axes, actual windows, group
records, and identities; the normalized run spec owns scopes, aggregation,
the trade gate, rule membership, tie-breaking, evidence thresholds, outlier
handling, and bootstrap parameters. Missing dataset columns make only their
dependent rules `unsupported_for_dataset`; non-finite candidate values make
that candidate unavailable; finite zero remains a real observation. Rules see
an IS-only view, and OOS labels are loaded only after ordered candidate IDs are
frozen.

Headline means equal-weight valid tickers inside each UTC OOS block and then
equal-weight blocks. Medians and profitable shares remain pooled according to
the observation contract. Every official selectable rule reports deterministic
descriptive month-block intervals for Top-1 absolute return, Top-1 paired lift,
Top-5 paired lift, and the recomputed outlier-robust Top-1 paired lift. These
intervals are weak with only five or six independent blocks and are never a
pass/fail criterion. `report.md` includes the complete scope and UTC-block
identity, all 11 evaluable evidence rows per supported selectable rule,
bootstrap facts, unsupported/unavailable reasons, flag-bit counts, execution
fault observations, rejected-fill diagnostics, and material metric
unavailability.

Analysis Git provenance is anchored to the Merlin code repository containing
the analysis module, never to the dataset or output location. If Git cannot be
executed or inspected, publication continues conservatively with
`code_commit: unavailable` and `dirty_worktree: true`; holdout unlock evidence
uses the same code-root facts. Locked scopes require both `--unlock-scope` and
a frozen `--policy` file; development is the default. Real holdout and temporal
outcomes are not used by Phase 3L-A certification.

Analysis and allocation use an analysis-local canonical JSON serializer so
package imports, CLI help, command JSON, and certifier JSON remain free of
production strategy-discovery output.

The explicit non-default real certification first runs a bounded development
subset, then reproduces only frozen point-estimate oracles on the full 24 by 6
development cell:

```powershell
C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe -m tools.strategy_lab.analysis.certify `
  --dataset tools\strategy_lab\output\s06_bracket_mvp_mtm_v2
```

Run fixed-capacity allocation with explicit stable dataset labels. N=1 and N>1
use the same path and align only exact `(canonical ticker, OOS start UTC, OOS
end UTC)` cells; labels are never inferred in comparison mode:

```powershell
C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe -m tools.strategy_lab.analysis.cli allocate `
  --dataset canonical=tools\strategy_lab\output\s06_bracket_mvp_mtm_v2 `
  --scope development `
  --rule primary_profit `
  --primary-k 6 --sensitivity-k 8 `
  --output tools\strategy_lab\tmp\phase3lb-canonical
```

The candidate rule first freezes one candidate per ticker from current IS.
The official second-level ticker score is that candidate's finite current-IS
`net_profit_pct`; it ranks descending, then finite selected IS trades
descending, then canonical symbol ascending. A Python-only custom
`TickerScorer` receives an immutable scalar `SelectedISTickerView`, explicit
name/version/JSON configuration, and no arrays, data accessors, OOS, or future
facts. Candidate decisions and ordered ticker sets freeze before OOS is loaded.
For each aligned multi-dataset calendar block, allocation first intersects the
tickers having a valid IS candidate decision and finite IS ticker score in every
input dataset. Each dataset ranks that same pool independently; OOS values are
revealed only after the common pool and ranks are frozen.

For each concrete primary, sensitivity, and matched-fraction K, slot weight is
always `1/K`. Underfill leaves cash and never renormalizes selected tickers.
`requested_capacity_fraction` is K divided by the available pool and may exceed
one; `realized_selected_fraction` and canonical `selectivity` are selected
count divided by available and remain within zero to one. Matched fraction is a
diagnostic K derived from the declared development pool and never replaces the
operational fixed K.

Every K reports all-available breadth, Bottom-K, Random-K, oracle, and
anti-oracle controls plus named spreads. Random-K uses the persisted uncertainty
seed/draw count and a canonical SHA-256 seed payload; publication retains only
one random summary per K/block, never individual draws. Oracle controls are
hindsight and non-deployable. Turnover counts common cash slots, while a
matched-fraction K change makes that transition unavailable. Compounding and
full-tail drawdown use calendar-block capacity returns and are explicitly not
bar-level portfolio equity.

Allocation publishes exactly six deterministic files, adding
`ticker_allocations.csv` to its own output set. The existing `analyze` command
remains five-file-only and byte-compatible. Both writers reject output beneath
an input dataset, preserve incompatible existing output, and return
`verified_noop` for a byte-identical rerun. Locked holdout and temporal scopes
still require the accepted explicit policy/unlock path and are not accessed by
Phase 3L-B development certification. Universe sizes, tickers, and windows come
from accepted dataset contracts rather than hardcoded counts.

The opt-in allocation certifier runs bounded, underfilled, and full-development
checks against a direct independent oracle. Repeat `--dataset label=path` for
an aligned N=2 check:

```powershell
C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe -m tools.strategy_lab.analysis.allocation_certify `
  --dataset canonical=tools\strategy_lab\output\s06_bracket_mvp_mtm_v2 `
  --rule primary_profit
```

Run the explicit real-pack certification into a fresh task-owned directory:

```powershell
C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe -m tools.strategy_lab.certify `
  tools\strategy_lab\runspecs\s06_bracket_mvp.json `
  --data-root "<read-only-data-root>" `
  --work-dir tools\strategy_lab\tmp\phase1b-certification
```

The isolated WFA tie-back is deliberately outside normal pytest discovery and
reuses `smoke_one` from that command. Run it only after a successful fresh
real-pack certification and point it at the same work directory:

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
  --output-dir tools\strategy_lab\tmp\phase2-mtm-smoke `
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

Phase 1-B certification uses the outcome-independent CRVUSDT representative,
temporary smoke datasets, and one WFA study under pytest's isolated storage.
Its accepted Stage 2 published the canonical 944-group pre-MTM output at
`tools/strategy_lab/output/s06_bracket_mvp_pre_mtm_v1`. That dataset is immutable
and is not extended in place. No current command calculates selection rules,
inspects development or holdout outcomes, changes metrics, or draws a
profitability conclusion. Backtester V1 is permanently unsupported.

The v2 generator requests the appended MTM metric but does not rank or
interpret it. `max_drawdown_mtm_pct` uses every finite bar-close equity
observation from `trade_start_idx` through the final bar with initial capital
as its positive pre-evaluation peak anchor. Empty evaluation intervals and any
non-finite observation return `NaN`; a present flat interval returns `0.0`;
negative equity may produce values above 100%. It is distinct from
balance-based realized Max DD and TradingView's intrabar High/Low drawdown.
Future `romad_mtm` will return `NaN` when this denominator is zero,
intentionally different from legacy realized RoMaD.

The historical prototype archive under
`docs/_work/dev_01_strategy-lab/prototype_01_grid-selection-research/` is
research evidence only and is not imported by production tools.
