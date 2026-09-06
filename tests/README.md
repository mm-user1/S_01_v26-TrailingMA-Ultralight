# Tests

Run from the repository root using the configured project interpreter. On Windows:

```powershell
$py = 'C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe'
# Required when the default sibling directory is inside an enclosing Git checkout:
$env:MERLIN_TEST_ROOT = Join-Path $env:LOCALAPPDATA 'Temp\merlin-tests'
& $py tools/run_tests.py fast
& $py tools/run_tests.py full -- --durations=30
& $py tools/run_tests.py -- tests/test_metrics.py
& $py tools/run_tests.py -- tests/test_metrics.py::TestMetricsEdgeCases
& $py tools/run_tests.py fast -- -k calendar
& $py tools/run_tests.py full --keep-temp -- --durations=20
& $py tools/run_tests.py -- --collect-only
```

On Linux, activate the configured environment and use the same arguments:

```bash
export MERLIN_TEST_ROOT="${TMPDIR:-/tmp}/merlin-tests"
python tools/run_tests.py fast
python tools/run_tests.py full -- --durations=30
python tools/run_tests.py -- tests/v2
```

## Selection and dependencies

Fast adds `-m "not slow"`. Full includes every normally discovered test with no
cost filter. Focused mode (no mode before `--`) adds no selector. Pytest owns
discovery through `testpaths = tests`; new ordinary tests enter full automatically.
The launcher prints its command. A full command with file targets, `-k`,
`--deselect`, or other filters is useful but is not an unfiltered acceptance gate.
Reserve `-m` for named modes; use focused mode for custom markers:

```powershell
& $py tools/run_tests.py -- -m regression
& $py tools/run_tests.py -- tests/strategy_lab
$jsTests = @(Get-ChildItem tests -Filter 'test_js_*.py' -File |
    Sort-Object Name | Select-Object -ExpandProperty FullName)
if ($jsTests.Count -eq 0) { throw 'No JavaScript pytest wrappers found.' }
& $py tools/run_tests.py -- @jsTests
if ($LASTEXITCODE -ne 0) { throw 'JavaScript wrapper tests failed.' }
```

All six JS wrappers belong in fast. They invoke Node with script-specific inputs;
they skip when Node is absent, which leaves JS acceptance pending. Compiled gates
require Numba and actual compiled backend execution, not reference fallback.
Explicit module-level JIT-off and missing-dependency guards remain supported for
focused/raw use; their skips cannot certify compiled execution.

Use static `@pytest.mark.slow` on expensive parity, full-population identity,
numerical oracle, or worker checks based on warm cost and purpose. Keep cheap
baseline/contract checks in fast. A subprocess alone does not justify `slow`.
Full always includes slow tests. Keep the existing `regression` marker for evidence
selection; collection output and in-code markers own the current inventory.

## Isolation, preflight, and retention

`tools/run_tests.py` uses only the standard library and launches pytest with
`sys.executable`, the repository cwd, and its tracked pytest configuration.
Everything after `--` is pytest input. Every mode rejects nonblank `PYTEST_ADDOPTS`:
unset it and pass intentional options explicitly after `--`. Named fast/full modes
also reject nonempty `NUMBA_DISABLE_JIT` other than `0`; unset it or use `0`.
The launcher never silently changes JIT or thread counts.

The launcher owns `--basetemp`, `cache_dir`, and configuration selection. Alternate
configs, `addopts` overrides, and argument files are rejected because they can
hide selectors or path overrides. Advanced configuration can use prepared raw
pytest below. Disabling the cache provider (`-p no:cacheprovider`) is allowed.

Default root: `repository_root.parent / "merlin-tests"`. Override with
`MERLIN_TEST_ROOT` when needed. The resolved root must be outside every Git
worktree, including `.git` files for linked worktrees/submodules and enclosing
checkouts. No implicit alternate root is chosen on rejection.

```text
merlin-tests/
  cache/numba/<python-and-numba-version>/
  cache/pycache/<python-version>/
  runs/<unique-run>/
    pytest/
    pytest-cache/
    tmp/
```

Before the child starts, the launcher sets `NUMBA_CACHE_DIR`,
`PYTHONPYCACHEPREFIX`, `TMPDIR`, `TMP`, and `TEMP`, plus pytest paths. Subprocesses
inherit this isolation. Existing storage/journal/Queue/CSV-root fixtures remain
responsible for application state. The Lab two-process determinism test reuses
the configured Numba cache; with an unset/empty value it uses one temporary
`tmp_path / "numba_cache"` shared by its two children, never an in-tree default.

Successful runs delete only their unique run directory unless `--keep-temp` is
set. Failures and handled interruptions retain it and print its location. Cleanup
failure preserves a passing exit code and reports the retained directory. Shared
caches and other runs are never cleaned. The PowerShell `tools/run_pytest.ps1`
shim forwards ordinary pytest arguments in focused mode and supports `-KeepTemp`;
it retains its configured interpreter selection (`MERLIN_PYTHON` override).

Numba cache reuse does not guarantee invalidation of dependencies in other files
or compile-time globals; see [Numba caching limitations](https://numba.readthedocs.io/en/stable/developer/caching.html#caching-limitations).
When kernels, relevant dependencies, or compiler versions change, use a fresh
external root for a targeted cold check. Do not delete existing shared caches:

```powershell
$env:MERLIN_TEST_ROOT = Join-Path $env:LOCALAPPDATA ("Temp\merlin-cold-" + [guid]::NewGuid())
& $py tools/run_tests.py -- tests/test_s06_fast_grid.py tests/v2/test_v2_grid_s06_gate.py::test_s06_t1_reference_subset_metrics_match_v1_fast_grid --deselect=tests/test_s06_fast_grid.py
# Use another fresh root for cold fast evidence, then repeat for warm evidence.
$env:MERLIN_TEST_ROOT = Join-Path $env:LOCALAPPDATA ("Temp\merlin-fast-" + [guid]::NewGuid())
& $py tools/run_tests.py fast -- --durations=20
& $py tools/run_tests.py fast -- --durations=20
```

## Prepared raw pytest and coverage

Raw collection, the environment's pytest entry point, and pytest-cov remain
supported. An unprepared bare pytest invocation is not fully isolated. Configure
all paths before Python starts, including the parent for subprocess tests.
Choose a new task-owned directory outside every checkout for each raw run:

```powershell
$raw = Join-Path $env:LOCALAPPDATA ("Temp\merlin-raw-" + [guid]::NewGuid())
New-Item -ItemType Directory -Force -Path $raw, "$raw\tmp" | Out-Null
$env:PYTHONPYCACHEPREFIX = "$raw\pycache"
$env:NUMBA_CACHE_DIR = "$raw\numba"
$env:TMPDIR = "$raw\tmp"; $env:TMP = $env:TMPDIR; $env:TEMP = $env:TMPDIR
$env:COVERAGE_FILE = "$raw\.coverage"
& $py -m pytest --basetemp "$raw\pytest" -o "cache_dir=$raw\pytest-cache" --collect-only
# Or use the configured environment's pytest.exe with the same arguments.
# Coverage: replace --collect-only with --cov=src --cov-report=term-missing.
```

```bash
raw=$(mktemp -d "${TMPDIR:-/tmp}/merlin-raw-XXXXXXXX")
mkdir "$raw/tmp"
export PYTHONPYCACHEPREFIX="$raw/pycache" NUMBA_CACHE_DIR="$raw/numba"
export TMPDIR="$raw/tmp" TMP="$raw/tmp" TEMP="$raw/tmp"
export COVERAGE_FILE="$raw/.coverage"
python -m pytest --basetemp "$raw/pytest" -o "cache_dir=$raw/pytest-cache" --collect-only
# pytest --basetemp "$raw/pytest" -o "cache_dir=$raw/pytest-cache" --collect-only
```

Raw runs own their retention; remove only exact directories you created. Keep
coverage reports external too if requesting HTML/XML output.

## Evidence and external certification

Tracked baselines under `data/baseline/` and `data/baseline_v2/` are immutable
oracles. A mismatch requires review, not baseline regeneration. Preserve numerical
tolerances, candidate order, fingerprints, trades, and meaningful negative cases.
Performance comparisons need the same dataset, plan, workers, warmup, and cache
conditions; see [performance evidence](../docs/engine_v2/PERFORMANCE.md).

Real WFA certification remains explicitly outside normal discovery. It requires
the exact external read-only pack and existing `smoke_one` output; missing
prerequisites fail instead of producing an all-skipped success:

```powershell
$env:MERLIN_STRATEGY_LAB_DATA_ROOT = '<read-only-data-root>'
$env:MERLIN_STRATEGY_LAB_CERT_WORK_DIR = '<absolute-certification-dir>'
& $py tools/run_tests.py -- tests/strategy_lab/phase1b_real_wfa_certification.py
```

Follow the [Strategy Lab manual](../tools/strategy_lab/README.md) for preparation
and authorization. Normalization cases are ordinarily collected without importing
this opt-in module. Ordinary full runs use synthetic SQLite data and require no
operational database. See the [V1](../docs/ADDING_NEW_STRATEGY.md) and
[V2](../docs/ADDING_NEW_STRATEGY_V2.md) guides for strategy-specific test obligations.
