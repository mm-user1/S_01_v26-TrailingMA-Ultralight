from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numba
import numpy as np
import pytest

from tools.strategy_lab.config import load_run_spec, semantic_key_digest
from tools.strategy_lab.dataset import DatasetError, METRIC_AXIS
from tools.strategy_lab.generate import GenerationResult, generate_dataset, main, runtime_thread_report

from tests.strategy_lab.phase1a_helpers import (
    REPO_ROOT,
    fake_execute,
    write_full_synthetic_pack,
)


@pytest.fixture(scope="module")
def synthetic_pack(tmp_path_factory):
    root = tmp_path_factory.mktemp("strategy_lab_phase1a_pack")
    run_spec, inventory = write_full_synthetic_pack(root)
    before = {
        path.name: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size, path.stat().st_mtime_ns)
        for path in (root / "market").iterdir()
    }
    return root, run_spec, inventory, before


def _generate_fake(synthetic_pack, output: Path, monkeypatch, **kwargs):
    root, run_spec, _, _ = synthetic_pack
    monkeypatch.setattr("tools.strategy_lab.generate.execute_grid_v2_candidates", fake_execute)
    return generate_dataset(
        run_spec,
        data_root=root / "market",
        output_dir=output,
        ticker_selectors=["AAAUSDT"],
        window_selectors=[1],
        repo_root=root,
        **kwargs,
    )


def _artifact_outcome(output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    return {
        "candidates": hashlib.sha256((output / "candidates.json").read_bytes()).hexdigest(),
        "quality": hashlib.sha256((output / "data_quality.csv").read_bytes()).hexdigest(),
        "groups": [(record["path"], record["sha256"]) for record in manifest["groups"]],
        "identity": manifest["identity"],
        "scope": manifest["scope"],
    }


def test_synthetic_plan_preserves_frozen_480_candidate_identity_and_settings(synthetic_pack):
    root, run_spec, _, _ = synthetic_pack
    spec = load_run_spec(run_spec, repo_root=root)
    assert spec.plan is not None
    assert spec.plan.deduped_candidate_count == 480
    assert spec.plan.plan_fingerprint == "c0e40ede6521a1cc02063ef2c9245f58c0093ca97aeb4bd858b75b5d09c7f434"
    assert semantic_key_digest(spec.plan) == "60e563c74876258e52de4c4ff3b598ed3a3a12d55d640f52ce262cd6b543fb55"
    assert spec.plan.settings.prefer_compiled is True
    assert spec.plan.settings.slow_enrich_selected is False
    assert spec.plan.settings.compiled_workers == 1
    assert spec.plan.settings.max_signal_cache_mb == 512.0
    assert spec.plan.settings.planning_policy == "full"
    assert all(
        "start" not in spec.plan.candidate_table.params_for_index(index)
        and "end" not in spec.plan.candidate_table.params_for_index(index)
        for index in range(480)
    )


def test_plan_is_built_once_and_unranked_execution_contract_is_explicit(
    synthetic_pack, tmp_path, monkeypatch
):
    root, run_spec, _, _ = synthetic_pack
    import tools.strategy_lab.config as config_module

    original = config_module.build_grid_v2_plan
    build_calls = 0
    execution_calls = 0

    def counted_build(*args, **kwargs):
        nonlocal build_calls
        build_calls += 1
        return original(*args, **kwargs)

    def checked_execute(plan, *_args, **kwargs):
        nonlocal execution_calls
        execution_calls += 1
        assert plan.settings.slow_enrich_selected is False
        assert kwargs == {
            "compute_sharpe": False,
            "compute_sharpe_daily": True,
            "compute_sqn": True,
        }
        return fake_execute(plan)

    monkeypatch.setattr(config_module, "build_grid_v2_plan", counted_build)
    monkeypatch.setattr("tools.strategy_lab.generate.execute_grid_v2_candidates", checked_execute)
    result = generate_dataset(
        run_spec,
        data_root=root / "market",
        output_dir=tmp_path / "output",
        ticker_selectors=["AAAUSDT"],
        window_selectors=[1],
        repo_root=root,
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert build_calls == manifest["provenance"]["plan_build_count"] == 1
    assert execution_calls == 2
    assert manifest["provenance"]["selected_slow_row_count"] == 0
    assert result.scope == "smoke"


def test_selector_order_cannot_change_inventory_or_window_order(synthetic_pack, tmp_path, monkeypatch):
    root, run_spec, inventory, _ = synthetic_pack
    monkeypatch.setattr("tools.strategy_lab.generate.execute_grid_v2_candidates", fake_execute)
    requested = [entry["canonical_symbol"] for entry in reversed(inventory["entries"])]
    result = generate_dataset(
        run_spec,
        data_root=root / "market",
        output_dir=tmp_path / "output",
        ticker_selectors=requested,
        window_selectors=[2, 1],
        repo_root=root,
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert [item["canonical_symbol"] for item in manifest["identity"]["included_tickers"]] == [
        entry["canonical_symbol"] for entry in inventory["entries"]
    ]
    assert [item["window_id"] for item in manifest["identity"]["windows"]] == [1, 2]
    assert manifest["scope"] == "smoke"


def test_raw_failure_publishes_deterministic_quality_before_candidate_execution(
    synthetic_pack, tmp_path, monkeypatch
):
    source_root, run_spec_source, _, _ = synthetic_pack
    copied = tmp_path / "copied"
    shutil.copytree(source_root, copied)
    target = next((copied / "market").glob("OKX_AAAUSDT*.csv"))
    target.write_bytes(target.read_bytes() + b"\n")
    calls = 0

    def forbidden(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("candidate execution must not start")

    monkeypatch.setattr("tools.strategy_lab.generate.execute_grid_v2_candidates", forbidden)
    quality_bytes = []
    for name in ("one", "two"):
        output = tmp_path / name
        with pytest.raises(ValueError, match="source validation failed"):
            generate_dataset(
                copied / run_spec_source.name,
                data_root=copied / "market",
                output_dir=output,
                ticker_selectors=["AAAUSDT"],
                window_selectors=[1],
                repo_root=copied,
            )
        quality_bytes.append((output / "data_quality.csv").read_bytes())
        assert not (output / "manifest.json").exists()
    assert calls == 0
    assert quality_bytes[0] == quality_bytes[1]
    assert b"file size does not match" in quality_bytes[0]


def test_runtime_thread_report_never_mislabels_clamped_or_single_thread_evidence():
    assert runtime_thread_report(2, 1, 1)["multi_thread_verified"] is False
    assert runtime_thread_report(2, 1, 8)["multi_thread_verified"] is False
    assert runtime_thread_report(1, 1, 8)["multi_thread_verified"] is False
    assert runtime_thread_report(2, 2, 8)["multi_thread_verified"] is True
    with pytest.raises(DatasetError):
        runtime_thread_report(0, 1, 1)


def test_numba_thread_state_and_sources_are_restored_and_preserved_on_failure(
    synthetic_pack, tmp_path, monkeypatch
):
    root, run_spec, _, before = synthetic_pack
    prior_threads = numba.get_num_threads()
    observed = []

    def failing(*_args, **_kwargs):
        observed.append(numba.get_num_threads())
        raise RuntimeError("controlled execution failure")

    monkeypatch.setattr("tools.strategy_lab.generate.execute_grid_v2_candidates", failing)
    with pytest.raises(RuntimeError, match="controlled execution failure"):
        generate_dataset(
            run_spec,
            data_root=root / "market",
            output_dir=tmp_path / "output",
            ticker_selectors=["AAAUSDT"],
            window_selectors=[1],
            repo_root=root,
        )

    assert observed == [1]
    assert numba.get_num_threads() == prior_threads
    after = {
        path.name: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size, path.stat().st_mtime_ns)
        for path in (root / "market").iterdir()
    }
    assert after == before


def test_interruption_resume_reuse_and_uninterrupted_outcome_equality(
    synthetic_pack, tmp_path, monkeypatch
):
    root, run_spec, _, _ = synthetic_pack
    monkeypatch.setattr("tools.strategy_lab.generate.execute_grid_v2_candidates", fake_execute)
    interrupted = tmp_path / "interrupted"

    def stop_after_first(group_index, _path):
        if group_index == 1:
            raise RuntimeError("controlled interruption")

    with pytest.raises(RuntimeError, match="controlled interruption"):
        generate_dataset(
            run_spec,
            data_root=root / "market",
            output_dir=interrupted,
            ticker_selectors=["AAAUSDT"],
            window_selectors=[1, 2],
            repo_root=root,
            _after_group=stop_after_first,
        )
    partial = json.loads((interrupted / "manifest.partial.json").read_text(encoding="utf-8"))
    assert len(partial["groups"]) == 1
    resumed = generate_dataset(
        run_spec,
        data_root=root / "market",
        output_dir=interrupted,
        ticker_selectors=["AAAUSDT"],
        window_selectors=[1, 2],
        repo_root=root,
        resume=True,
    )
    uninterrupted = tmp_path / "uninterrupted"
    generate_dataset(
        run_spec,
        data_root=root / "market",
        output_dir=uninterrupted,
        ticker_selectors=["AAAUSDT"],
        window_selectors=[1, 2],
        repo_root=root,
    )

    assert resumed.reused_groups == 1
    assert resumed.regenerated_groups == 1
    assert not (interrupted / "manifest.partial.json").exists()
    assert _artifact_outcome(interrupted) == _artifact_outcome(uninterrupted)
    for relative, _ in _artifact_outcome(interrupted)["groups"]:
        assert np.array_equal(
            np.load(interrupted / relative, allow_pickle=False),
            np.load(uninterrupted / relative, allow_pickle=False),
            equal_nan=True,
        )


@pytest.mark.parametrize("corruption", ["checksum", "shape", "dtype"])
def test_resume_regenerates_invalid_claimed_groups(
    synthetic_pack, tmp_path, monkeypatch, corruption
):
    root, run_spec, _, _ = synthetic_pack
    monkeypatch.setattr("tools.strategy_lab.generate.execute_grid_v2_candidates", fake_execute)
    output = tmp_path / corruption

    with pytest.raises(RuntimeError):
        generate_dataset(
            run_spec,
            data_root=root / "market",
            output_dir=output,
            ticker_selectors=["AAAUSDT"],
            window_selectors=[1],
            repo_root=root,
            _after_group=lambda *_args: (_ for _ in ()).throw(RuntimeError("stop")),
        )
    group = next((output / "groups").rglob("*.npy"))
    if corruption == "checksum":
        group.write_bytes(group.read_bytes() + b"corrupt")
    elif corruption == "shape":
        np.save(group, np.zeros((1, 2, 20), dtype=np.float64), allow_pickle=False)
    else:
        np.save(group, np.zeros((480, 2, 20), dtype=np.float32), allow_pickle=False)

    result = generate_dataset(
        run_spec,
        data_root=root / "market",
        output_dir=output,
        ticker_selectors=["AAAUSDT"],
        window_selectors=[1],
        repo_root=root,
        resume=True,
    )
    assert result.reused_groups == 0
    assert result.regenerated_groups == 1
    assert np.load(group, mmap_mode="r", allow_pickle=False).shape == (480, 2, 20)
    assert np.load(group, mmap_mode="r", allow_pickle=False).dtype == np.float64


@pytest.mark.parametrize("field", ["schema", "axes", "resources", "scope"])
def test_resume_rejects_incompatible_identity_and_scope(
    synthetic_pack, tmp_path, monkeypatch, field
):
    root, run_spec, _, _ = synthetic_pack
    monkeypatch.setattr("tools.strategy_lab.generate.execute_grid_v2_candidates", fake_execute)
    output = tmp_path / field
    with pytest.raises(RuntimeError):
        generate_dataset(
            run_spec,
            data_root=root / "market",
            output_dir=output,
            ticker_selectors=["AAAUSDT"],
            window_selectors=[1],
            repo_root=root,
            _after_group=lambda *_args: (_ for _ in ()).throw(RuntimeError("stop")),
        )
    partial_path = output / "manifest.partial.json"
    partial = json.loads(partial_path.read_text(encoding="utf-8"))
    if field == "schema":
        partial["identity"]["dataset_schema"] = "wrong"
    elif field == "axes":
        partial["identity"]["metric_axis"] = list(reversed(METRIC_AXIS))
    elif field == "resources":
        partial["identity"]["resources"]["numba_threads"] = 2
    else:
        partial["scope"] = "full"
    partial_path.write_text(json.dumps(partial), encoding="utf-8")

    with pytest.raises(DatasetError, match="incompatible schema, identity, resources, axes, or scope"):
        generate_dataset(
            run_spec,
            data_root=root / "market",
            output_dir=output,
            ticker_selectors=["AAAUSDT"],
            window_selectors=[1],
            repo_root=root,
            resume=True,
        )


def test_unlisted_group_and_temp_file_are_not_reused(synthetic_pack, tmp_path, monkeypatch):
    root, run_spec, _, _ = synthetic_pack
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return fake_execute(*args, **kwargs)

    monkeypatch.setattr("tools.strategy_lab.generate.execute_grid_v2_candidates", counted)
    output = tmp_path / "output"
    unlisted = output / "groups" / "AAAUSDT" / "window_01.npy"
    unlisted.parent.mkdir(parents=True)
    np.save(unlisted, np.zeros((480, 2, 20), dtype=np.float64), allow_pickle=False)
    temporary = unlisted.with_name(".window_01.npy.unowned.tmp")
    temporary.write_bytes(b"unowned")

    result = generate_dataset(
        run_spec,
        data_root=root / "market",
        output_dir=output,
        ticker_selectors=["AAAUSDT"],
        window_selectors=[1],
        repo_root=root,
        resume=True,
    )
    assert calls == 2
    assert result.reused_groups == 0 and result.regenerated_groups == 1
    assert temporary.read_bytes() == b"unowned"


def test_completed_run_is_an_exact_verified_no_op(synthetic_pack, tmp_path, monkeypatch):
    output = tmp_path / "output"
    _generate_fake(synthetic_pack, output, monkeypatch)
    before = {
        path.relative_to(output).as_posix(): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in output.rglob("*")
        if path.is_file()
    }
    result = _generate_fake(synthetic_pack, output, monkeypatch)
    after = {
        path.relative_to(output).as_posix(): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in output.rglob("*")
        if path.is_file()
    }
    assert result.no_op is True
    assert result.reused_groups == 1 and result.regenerated_groups == 0
    assert before == after


def test_completed_output_checksum_corruption_is_not_silently_overwritten(
    synthetic_pack, tmp_path, monkeypatch
):
    output = tmp_path / "output"
    _generate_fake(synthetic_pack, output, monkeypatch)
    candidates = output / "candidates.json"
    candidates.write_bytes(candidates.read_bytes() + b" ")
    with pytest.raises(DatasetError, match="checksum/shape/dtype verification"):
        _generate_fake(synthetic_pack, output, monkeypatch)


def test_cli_forwards_repeatable_smoke_selectors_and_reports_completion(tmp_path, monkeypatch, capsys):
    captured = {}

    def fake_generate(run_spec, **kwargs):
        captured.update({"run_spec": run_spec, **kwargs})
        return GenerationResult(
            output_dir=tmp_path / "output",
            manifest_path=tmp_path / "output" / "manifest.json",
            scope="smoke",
            completed_groups=4,
            reused_groups=1,
            regenerated_groups=3,
            no_op=False,
            timings={"total_seconds": 1.25},
        )

    monkeypatch.setattr("tools.strategy_lab.generate.generate_dataset", fake_generate)
    assert main(
        [
            "run.json",
            "--data-root",
            str(tmp_path / "market"),
            "--output-dir",
            str(tmp_path / "output"),
            "--resume",
            "--ticker",
            "AAAUSDT",
            "--ticker",
            "ZZZUSDT",
            "--window",
            "1",
            "--window",
            "2",
        ]
    ) == 0
    assert captured["ticker_selectors"] == ["AAAUSDT", "ZZZUSDT"]
    assert captured["window_selectors"] == [1, 2]
    assert captured["resume"] is True
    assert captured["_progress"] is not None
    assert "smoke complete" in capsys.readouterr().out


def test_two_fresh_processes_produce_identical_real_smoke_outcomes(synthetic_pack, tmp_path):
    root, run_spec, _, _ = synthetic_pack
    script = (
        "from tools.strategy_lab.generate import generate_dataset; import sys; "
        "generate_dataset(sys.argv[1], data_root=sys.argv[2], output_dir=sys.argv[3], "
        "ticker_selectors=['AAAUSDT'], window_selectors=[1], repo_root=sys.argv[4])"
    )
    outputs = [tmp_path / "process_one", tmp_path / "process_two"]
    environment = os.environ.copy()
    environment["NUMBA_CACHE_DIR"] = str(tmp_path / "numba_cache")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), str(REPO_ROOT)]
        + ([existing_pythonpath] if existing_pythonpath else [])
    )
    for output in outputs:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(run_spec),
                str(root / "market"),
                str(output),
                str(root),
            ],
            cwd=REPO_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr

    assert _artifact_outcome(outputs[0]) == _artifact_outcome(outputs[1])
    first_manifest = json.loads((outputs[0] / "manifest.json").read_text(encoding="utf-8"))
    assert first_manifest["scope"] == "smoke"
    assert first_manifest["provenance"]["selected_slow_row_count"] == 0
    assert first_manifest["provenance"]["plan_build_count"] == 1
    assert first_manifest["identity"]["expected_group_shape"] == [480, 2, 20]
    relative = first_manifest["groups"][0]["path"]
    array = np.load(outputs[0] / relative, allow_pickle=False)
    assert array.shape == (480, 2, 20) and array.dtype == np.float64
    assert array.shape[0] == 480
    assert np.isnan(array[:, :, METRIC_AXIS.index("sqn")]).any()
