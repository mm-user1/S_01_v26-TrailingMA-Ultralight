# Strategy Lab — Phase 0

Strategy Lab is a local, research-only tool for certified Backtester V2
strategies. Phase 0 ships configuration, pre-registration, deterministic V2
plan identity, and a read-only market-data inventory. It does not execute
candidates, construct WFA windows, generate outcome datasets, calculate rules,
or inspect development/holdout performance.

The initial frozen run is `runspecs/s06_bracket_mvp.json`: S06 B2 bracket-only,
480 full-plan candidates, 30-minute data, 2-month IS / 1-month OOS, eight
calendar windows, and a 24/94 development/holdout ticker split. Normalized
economics are `initialCapital=1000.0`, `riskPerTrade=2.0`,
`contractSize=0.0001`, and `commissionPct=0.05`. Slippage and funding are
intentionally not modelled. `tickSize=0.0001` is provenance only because the
execution profile uses `priceRounding=none`.

## Commands

Run every command from the repository root with the configured project Python.
On this Windows repository, use:

```powershell
C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe -m tools.strategy_lab.config tools\strategy_lab\runspecs\s06_bracket_mvp.json
C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe -m pytest tests\strategy_lab -q
```

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
The data root is resolved only from an explicit `--data-root`, then from
`MERLIN_STRATEGY_LAB_DATA_ROOT`; absence is an error. The tool never searches
for alternate data or stores the absolute root in tracked identity.

## Read-only and output contract

Source CSV files are opened only for reading. The inventory builder rejects
subdirectories, source symlinks, malformed filenames/schema, duplicate
canonical symbols, unexpected counts, and sizing headroom below 100 contract
steps. It never sorts or rewrites source rows and never writes beside market
data.

Tracked, cross-host facts live in `runspecs/`. Host/platform, resolved absolute
root, source mtimes, verification time, and tool Git revision are local
provenance under ignored `output/`. Future local caches and temporary files use
the narrowly ignored `cache/` and `tmp/` directories.

## Stage boundary

- Stage A is host-independent: package/config/inventory code, synthetic tests,
  canonical serialization, V2 plan construction, and documentation.
- Stage B requires the exact configured source pack and produces/verifies the
  tracked inventory, 118-file and 24/94 facts, sizing headroom, deterministic
  digests, and before/after source preservation.

The current primary holdout is cross-sectional confirmation on new tickers over
the same six months as development, not independent temporal confirmation.
Windows 7-8 are mandatory descriptive temporal checks in the eventual MVP and
have no Phase 0 result, bootstrap interval, or pass/fail threshold.

## Explicit non-goals

Phase 0 contains no candidate execution, Fast/Slow metric change, Grid ranking,
WFA execution, data-quality/window gate, outcome matrix, selection-rule code,
holdout analyzer, MTM drawdown, UI/API/storage integration, database, Queue, or
Preset work. Backtester V1 is permanently unsupported.

The historical prototype archive under
`docs/_work/dev_01_strategy-lab/prototype_01_grid-selection-research/` is
research evidence only and is not imported by production tools.
