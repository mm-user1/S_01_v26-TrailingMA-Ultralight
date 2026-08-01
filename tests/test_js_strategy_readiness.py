from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
JAVASCRIPT_TEST = REPOSITORY_ROOT / "tests" / "js" / "test_strategy_readiness.js"


def test_strategy_readiness_javascript_behavior():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable; skipping strategy-readiness JavaScript test.")

    completed = subprocess.run(
        [node, str(JAVASCRIPT_TEST)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, (
        "Strategy-readiness JavaScript test failed.\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
