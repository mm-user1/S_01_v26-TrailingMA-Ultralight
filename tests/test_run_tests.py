"""Behavioral contracts for the external pytest launcher."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools import run_tests as runner


@pytest.mark.parametrize("mode,options,environment", [
    ("full", [], {"PYTEST_ADDOPTS": "--lf"}),
    (None, [], {"PYTEST_ADDOPTS": "-k hidden"}),
    ("fast", [], {"NUMBA_DISABLE_JIT": "1"}),
    ("full", [], {"NUMBA_DISABLE_JIT": "false"}),
    ("full", ["-m", "slow"], {}),
    ("fast", ["--markexpr=slow"], {}),
    (None, ["--basetemp=/tmp/unsafe"], {}),
    (None, ["--base", "/tmp/unsafe"], {}),
    (None, ["-o", "cache_dir=/tmp/unsafe"], {}),
    (None, ["-ocache_dir=/tmp/unsafe"], {}),
    (None, ["-o=cache_dir=/tmp/unsafe"], {}),
    (None, ["--override-ini=addopts=-k hidden"], {}),
    (None, ["-c", "other.ini"], {}),
    (None, ["--config-file=other.ini"], {}),
    (None, ["@hidden-args"], {}),
])
def test_runner_rejects_hidden_selection_and_path_overrides(mode, options, environment):
    args = runner.parse_args(([mode] if mode else []) + ["--"] + options)
    with pytest.raises(ValueError):
        runner.validate_options(args, environment)


@pytest.mark.parametrize("git_file", [False, True])
def test_runner_rejects_git_ancestors_before_creating_root(tmp_path, git_file):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    marker = checkout / ".git"
    if git_file:
        marker.write_text("gitdir: elsewhere", encoding="utf-8")
    else:
        marker.mkdir()
    root = checkout / "absent" / "runs"
    with pytest.raises(ValueError, match="MERLIN_TEST_ROOT"):
        runner.external_root(root)
    assert not root.exists()


def test_runner_selection_environment_and_cleanup_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MERLIN_TEST_ROOT", str(tmp_path / "runner"))
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    monkeypatch.delenv("NUMBA_DISABLE_JIT", raising=False)
    calls = []

    def child(command, env):
        calls.append(command)
        assert command[:3] == [sys.executable, "-m", "pytest"]
        assert Path(env["NUMBA_CACHE_DIR"]).is_relative_to(tmp_path)
        assert Path(env["PYTHONPYCACHEPREFIX"]).is_relative_to(tmp_path)
        assert env["TEMP"] == env["TMP"] == env["TMPDIR"]
        return 0

    monkeypatch.setattr(runner, "run_child", child)
    assert runner.main(["fast", "--", "-k", "calendar"]) == 0
    assert calls[-1][-4:] == ["-m", "not slow", "-k", "calendar"]
    assert runner.main(["full"]) == 0
    assert "-m" not in calls[-1][3:] and "tests" not in calls[-1]

    def fail_cleanup(*args):
        raise PermissionError("controlled cleanup failure")

    monkeypatch.setattr(runner, "cleanup_run", fail_cleanup)
    assert runner.main(["--", "tests/test_metrics.py::TestMetricsEdgeCases"]) == 0
    assert calls[-1][-1] == "tests/test_metrics.py::TestMetricsEdgeCases"
    assert "Retained test run" in capsys.readouterr().err
    assert len({command[command.index("--basetemp") + 1] for command in calls}) == 3


def test_runner_real_child_exit_retention_and_focused_selection(tmp_path):
    root = tmp_path / "external root Кириллица"
    probe = tmp_path / "test_probe.py"
    probe.write_text("def test_probe():\n    assert True\n", encoding="utf-8")
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    env["MERLIN_TEST_ROOT"] = str(root)
    command = [sys.executable, str(runner.REPO_ROOT / "tools/run_tests.py")]
    first = subprocess.run(command + ["--", str(probe)], env=env, capture_output=True, text=True, timeout=60)
    assert first.returncode == 0, first.stdout + first.stderr
    assert "1 passed" in first.stdout
    assert list((root / "runs").iterdir()) == []
    kept = subprocess.run(command + ["--keep-temp", "--", str(probe)], env=env, capture_output=True, text=True, timeout=60)
    assert kept.returncode == 0, kept.stdout + kept.stderr
    kept_runs = set((root / "runs").iterdir())
    assert len(kept_runs) == 1
    probe.write_text("def test_probe():\n    assert False\n", encoding="utf-8")
    failed = subprocess.run(command + ["--", str(probe)], env=env, capture_output=True, text=True, timeout=60)
    assert failed.returncode == 1, failed.stdout + failed.stderr
    assert "Retained test run" in failed.stdout
    assert len(set((root / "runs").iterdir()) - kept_runs) == 1
    assert all(path.exists() for path in kept_runs)
    assert any((root / "cache/pycache").iterdir())


def test_runner_interrupt_reaps_child_before_return(monkeypatch):
    events = []

    class Child:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            events.append("reaped")

        def wait(self, timeout=None):
            events.append(("wait", timeout))
            if timeout is None:
                raise KeyboardInterrupt
            if "terminated" not in events:
                raise subprocess.TimeoutExpired("probe", timeout)
            return 1

        def terminate(self):
            events.append("terminated")

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: Child())
    assert runner.run_child(["owned-probe"], {}) == 130
    assert events == [("wait", None), ("wait", 5), "terminated", ("wait", 5), "reaped"]
