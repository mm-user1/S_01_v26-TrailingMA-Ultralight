from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ui.server import app


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
JAVASCRIPT_TEST = (
    REPOSITORY_ROOT / "tests" / "js" / "test_s06_v064a2_variant_selector.js"
)
STRATEGY_ID = "s06_r_trend_v06_4_a2_b2"


def test_s06_v064a2_variant_selector_javascript_behavior():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable; skipping variant-selector JavaScript test.")

    completed = subprocess.run(
        [node, str(JAVASCRIPT_TEST)],
        cwd=REPOSITORY_ROOT,
        input=json.dumps(
            app.test_client().get(f"/api/strategy/{STRATEGY_ID}/config").get_json()
        ),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, (
        "S06 variant-selector JavaScript test failed.\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
