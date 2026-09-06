"""Launch pytest with external artifacts and reusable interpreter/compiler caches."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Isolated pytest: fast excludes slow; full includes normal discovery; "
        "omit mode for focused pytest arguments after --.",
        epilog="Artifacts: MERLIN_TEST_ROOT or ../merlin-tests, outside any Git worktree. "
        "Caches persist. Successful run directories are removed unless --keep-temp; "
        "failures and interruptions are retained. Unset PYTEST_ADDOPTS and pass options after --.",
    )
    parser.add_argument("mode", nargs="?", choices=("fast", "full"))
    parser.add_argument("--keep-temp", action="store_true")
    divider = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:divider])
    args.pytest_args = argv[divider + 1:]
    return args


def validate_options(args, env):
    if env.get("PYTEST_ADDOPTS", "").strip():
        raise ValueError("Unset PYTEST_ADDOPTS and pass intentional pytest options explicitly after --.")
    if args.mode and env.get("NUMBA_DISABLE_JIT", "") not in ("", "0"):
        raise ValueError("fast/full require JIT: unset NUMBA_DISABLE_JIT or set it to 0.")
    tokens = args.pytest_args
    for index, token in enumerate(tokens):
        option = token.split("=", 1)[0]
        if token.startswith("@"):
            raise ValueError("Pass pytest arguments directly after --; argument files can hide overrides.")
        if (option.startswith("--") and any(name.startswith(option) for name in (
                "--basetemp", "--config-file", "--rootdir"))) or token.startswith("-c") and not token.startswith("--"):
            raise ValueError(f"Runner owns isolation/configuration; forbidden pytest option: {token}")
        if args.mode and (token.startswith("-m") and not token.startswith("--") or
                          option.startswith("--") and "--markexpr".startswith(option)):
            raise ValueError("-m is reserved for fast/full; use focused mode for custom markers.")
        ini = None
        if token == "-o" or option.startswith("--") and "--override-ini".startswith(option):
            ini = token.split("=", 1)[1] if "=" in token else (tokens[index + 1] if index + 1 < len(tokens) else "")
        elif token.startswith("-o") and not token.startswith("--"):
            ini = token[2:].removeprefix("=")
        if ini is not None and ini.split("=", 1)[0].strip() in {"cache_dir", "addopts"}:
            raise ValueError(f"Runner owns cache_dir/addopts; forbidden override: {ini}")


def external_root(value):
    root = Path(value).expanduser().resolve()
    for ancestor in (root, *root.parents):
        if (ancestor / ".git").exists():
            raise ValueError(
                f"Test root {root} is inside Git worktree {ancestor}. "
                "Set MERLIN_TEST_ROOT to an external location, e.g. "
                f"{Path(tempfile.gettempdir()) / 'merlin-tests'}."
            )
    return root


def prepare_run(root, env):
    python_key = f"{sys.implementation.name}-{platform.python_version()}"
    try:
        numba_version = importlib.metadata.version("numba")
    except importlib.metadata.PackageNotFoundError:
        numba_version = "unavailable"
    numba_cache = root / "cache" / "numba" / f"{python_key}-numba-{numba_version}"
    pycache = root / "cache" / "pycache" / python_key
    runs = root / "runs"
    for path in (numba_cache, pycache, runs):
        path.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="run-", dir=runs)).resolve()
    for name in ("pytest", "pytest-cache", "tmp"):
        (run / name).mkdir()
    env.update(NUMBA_CACHE_DIR=str(numba_cache), PYTHONPYCACHEPREFIX=str(pycache),
               TMPDIR=str(run / "tmp"), TMP=str(run / "tmp"), TEMP=str(run / "tmp"))
    return run


def run_child(command, env):
    with subprocess.Popen(command, cwd=REPO_ROOT, env=env) as child:
        try:
            return child.wait()
        except KeyboardInterrupt:
            # The foreground child receives the terminal interrupt too. Reap it
            # before returning so retained artifacts are no longer in use by it.
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            return 130


def cleanup_run(run, root):
    target = run.resolve()
    if target != run or target.parent != (root / "runs").resolve():
        raise ValueError(f"Cleanup target changed: {run}")
    shutil.rmtree(target)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    env = os.environ.copy()
    try:
        validate_options(args, env)
        root = external_root(env.get("MERLIN_TEST_ROOT") or REPO_ROOT.parent / "merlin-tests")
        run = prepare_run(root, env)
    except (ValueError, OSError) as exc:
        print(f"run_tests: {exc}", file=sys.stderr)
        return 2
    command = [sys.executable, "-m", "pytest", "-c", str(REPO_ROOT / "pytest.ini"),
               "--basetemp", str(run / "pytest"),
               "-o", f"cache_dir={run / 'pytest-cache'}"]
    if args.mode == "fast":
        command += ["-m", "not slow"]
    command += args.pytest_args
    print(f"Test run: {run}", flush=True)
    print(f"Command (cwd={REPO_ROOT}): {subprocess.list2cmdline(command)}", flush=True)
    try:
        code = run_child(command, env)
    except OSError as exc:
        print(f"Could not start pytest: {exc}", file=sys.stderr)
        code = 1
    if code == 0 and not args.keep_temp:
        try:
            cleanup_run(run, root)
        except (OSError, ValueError) as exc:
            print(f"Cleanup failed: {exc}. Retained test run: {run}", file=sys.stderr)
    else:
        print(f"Retained test run: {run}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
