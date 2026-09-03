# Tools

Run repository tools from the project root. On Windows, use the configured
project interpreter:

```powershell
C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe <command>
```

Linux/VPS environments use their configured project Python and native paths.

## Baselines and checks

`generate_baseline_s01.py` regenerates the S01 regression evidence after an
explicitly reviewed behavioral change:

```powershell
C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe tools\generate_baseline_s01.py
```

It writes `data/baseline/s01_metrics.json` and
`data/baseline/s01_trades.csv`. Do not refresh these files merely to make a
regression failure pass.

`test_all_ma_types.py` exercises all supported S01 moving-average types.
`benchmark_indicators.py` and `benchmark_metrics.py` provide focused timing
checks for their respective subsystems.

## Grid V2 diagnostics

`benchmark_grid_v2.py` measures direct Grid V2 runs and inspects saved WFA
diagnostics:

```powershell
C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe tools\benchmark_grid_v2.py --help
C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe tools\benchmark_grid_v2.py inspect-wfa-db --db <snapshot.db>
C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe tools\benchmark_grid_v2.py direct-grid --config tools\benchmark_configs\s06_b2_sui_baseline_grid.json --workers 1,6 --warmup-runs 1 --runs 2
```

Candidate domains come from strategy `config.json` optimization metadata,
`enabled_params`, and selected `{param}_options`. Numeric `param_ranges` in a
benchmark payload do not independently redefine V2 grid granularity.
`inspect-wfa-db` uses SQLite read-only immutable mode and is for frozen
snapshots, not a live database with possible WAL frames.

## Pytest wrapper

`run_pytest.ps1` selects the required Merlin interpreter and places pytest
temporary files in an isolated `.pytest_tmp/run_<pid>` directory:

```powershell
.\tools\run_pytest.ps1 -q tests\test_benchmark_grid_v2.py
.\tools\run_pytest.ps1 -q tests\v2
```

Set `MERLIN_PYTHON` only when an alternate project interpreter is intentional.
Use `-KeepTemp` before pytest arguments to retain a run directory for debugging.

## Strategy Lab

Strategy Lab is a local, read-only-input research pipeline for certified V2
strategies. Its current commands cover run-spec validation, inventory,
resumable dataset generation, real-pack certification, deterministic analysis,
and fixed-capacity allocation:

```powershell
C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe -m tools.strategy_lab.config tools\strategy_lab\runspecs\s06_bracket_mvp.json
C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe -m tools.strategy_lab.generate --help
C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe -m tools.strategy_lab.certify --help
C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe -m tools.strategy_lab.analysis.cli --help
C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe -m tools.strategy_lab.analysis.allocation_certify --help
```

See the [Strategy Lab manual](strategy_lab/README.md) for identity, isolation,
schema, resume, scope-unlock, analysis, allocation, and certification contracts.

## Related documentation

- [Test workflow](../tests/README.md)
- [V2 architecture](../docs/engine_v2/ARCHITECTURE.md)
- [Performance evidence](../docs/engine_v2/PERFORMANCE.md)
