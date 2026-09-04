# Tests

Merlin uses pytest for Python tests and Python wrappers that invoke Node for
browser-side JavaScript tests. Tests must use isolated temporary storage and
synthetic or explicitly authorized read-only inputs; they must not modify
protected market data, baselines, or a live database.

On Windows, run Python gates with the configured interpreter:

```powershell
C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe -m pytest <arguments>
```

Linux/VPS environments use their configured project Python and native paths.
The wrapper `tools/run_pytest.ps1` additionally creates a per-run pytest temp
directory under `.pytest_tmp/`.

## Verification tiers

Start with the narrowest test that owns the changed behavior, then expand in
proportion to risk:

```powershell
# Focused file or test
C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe -m pytest tests\test_metrics.py -q

# Root V1 and shared tests
C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe -m pytest tests -q --ignore=tests\v2 --ignore=tests\strategy_lab

# V2 engine and HTTP contracts
C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe -m pytest tests\v2 -q

# Strategy Lab
C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe -m pytest tests\strategy_lab -q
```

Run every browser-side JavaScript wrapper in deterministic order:

```powershell
$jsTests = @(Get-ChildItem -LiteralPath tests -Filter 'test_js_*.py' -File |
    Sort-Object Name | Select-Object -ExpandProperty FullName)
if ($jsTests.Count -eq 0) { throw 'No JavaScript pytest wrappers found.' }
& C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe -m pytest @jsTests -q
if ($LASTEXITCODE -ne 0) {
    throw "JavaScript wrapper tests failed with exit code $LASTEXITCODE."
}
```

These wrappers invoke Node, provide script-specific inputs where required, and
skip explicitly when Node is unavailable. The scripts are not all safely
runnable through one uniform direct-Node command.

Run the complete Python suite only when the change warrants it:

```powershell
C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe -m pytest tests -q
```

Use collection as a fast structural gate when full execution is not required:

```powershell
C:\Users\mt\Desktop\Strategy\S_Python\.venv\Scripts\python.exe -m pytest --collect-only -q
```

Use that collection output for the exact current test inventory; this guide
does not maintain a brittle file-by-file table.

## Evidence-bearing tests

Regression and certification tests may compare against tracked evidence under
`data/baseline/` and `data/baseline_v2/`. A mismatch is a product or evidence
review event, not permission to rewrite the baseline. Real-data Strategy Lab
certification is opt-in, requires the exact external read-only pack, and is
documented in the [Strategy Lab manual](../tools/strategy_lab/README.md).

Performance changes should also run the relevant benchmark and compare the
same dataset, candidate plan, worker settings, warmup count, and measurement
protocol. See [performance evidence](../docs/engine_v2/PERFORMANCE.md).

## Test design rules

- Assert public behavior and stable identities rather than implementation
  accidents.
- Cover success, validation failures, and boundary cases.
- Keep storage, temp files, environment variables, and process-global thread
  state isolated and restored.
- Add parity evidence when a V2 compiled path or execution mode is changed.
- Keep V1 and V2 optimizer expectations explicit: V1 supports Optuna and Grid;
  V2 is Grid-only.

For strategy integration requirements, use the [V1 guide](../docs/ADDING_NEW_STRATEGY.md)
or the [V2 guide](../docs/ADDING_NEW_STRATEGY_V2.md).
