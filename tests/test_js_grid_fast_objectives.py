from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
JAVASCRIPT_TEST = REPOSITORY_ROOT / "tests" / "js" / "test_grid_fast_objectives.js"


def test_grid_fast_objective_javascript_behavior():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable; skipping Grid Fast objective JavaScript test.")

    completed = subprocess.run(
        [node, str(JAVASCRIPT_TEST)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, (
        "Grid Fast objective JavaScript test failed.\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
