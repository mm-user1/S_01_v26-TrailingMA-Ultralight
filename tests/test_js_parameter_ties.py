import shutil
import subprocess
from pathlib import Path


def test_parameter_tie_javascript_behavior():
    node = shutil.which("node")
    assert node, "Node is required to certify parameter tie UI behavior."
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([node, str(root / "tests/js/test_parameter_ties.js")], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
