"""Fresh interpreters expose backend poisoning hidden by collection warmup."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(os.environ.get("NUMBA_DISABLE_JIT", "") not in ("", "0"),
                    reason="backend identity requires Numba JIT")
@pytest.mark.parametrize("v1_first", [False, True])
def test_v1_oracle_imports_preserve_dispatchers_in_fresh_process(v1_first):
    pytest.importorskip("numba")
    repo = Path(__file__).resolve().parents[2]
    script = """
import os, runpy, sys
import numba
from numba.core.registry import CPUDispatcher
original = os.environ.get('NUMBA_DISABLE_JIT')
if sys.argv[1] == 'True':
    from strategies.s06_r_trend_v02 import fast_grid
for filename in ('test_v2_grid_identity.py', 'test_v2_grid_s06_gate.py'):
    helper = runpy.run_path('tests/v2/' + filename)['_fast_grid']
    backend = helper()
    assert os.environ.get('NUMBA_DISABLE_JIT') == original
    assert not numba.config.DISABLE_JIT
    assert isinstance(backend._S06_FAST_LOOP, CPUDispatcher)
    assert isinstance(backend._S06_FAST_BATCH_LOOP, CPUDispatcher)
print('V1 scalar and batch CPUDispatchers preserved')
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(repo), str(repo / "src"), env.get("PYTHONPATH", "")])
    result = subprocess.run([sys.executable, "-c", script, str(v1_first)],
                            cwd=repo, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CPUDispatchers preserved" in result.stdout
