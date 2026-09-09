from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numba
import numpy as np
import pytest

from tools.strategy_lab import generate as generate_module
from tools.strategy_lab.config import load_run_spec, semantic_key_digest
from tools.strategy_lab.dataset import DatasetError, METRIC_AXIS
from tools.strategy_lab.generate import (
    GenerationResult,
    _require_compiled_backend_available,
    generate_dataset,
    main,
    runtime_thread_report,
)

from tests.strategy_lab.phase1a_helpers import (
    REPO_ROOT,
    fake_execute,
    fake_rows,
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


def _file_snapshot(root: Path) -> dict[str, tuple[str, int, int]]:
    return {
        path.relative_to(root).as_posix(): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("partial", [False, True])
def test_s03_changed_ties_reject_resume_without_mutating_equal_fixed_candidates(
    synthetic_pack, tmp_path, monkeypatch, partial,
):
    from tests.strategy_lab.s03_helpers import small_raw, pin_small_plan, write_spec
    root, original_path, _, _ = synthetic_pack
    original = json.loads(original_path.read_text(encoding="utf-8"))
    raw = small_raw()
    raw["generation"]["inventory"] = original["generation"]["inventory"]
    raw["generation"]["market_data"] = original["generation"]["market_data"]
    raw["preregistration"]["split"] = original["preregistration"]["split"]
    path = write_spec(tmp_path / "s03.json", raw)
    on = load_run_spec(path, repo_root=root)
    monkeypatch.setattr(generate_module, "execute_grid_v2_candidates", fake_execute)
    output = tmp_path / "output"
    kwargs = dict(data_root=root / "market", output_dir=output, repo_root=root,
                  ticker_selectors=["AAAUSDT"], window_selectors=[1, 2])
    def interrupt(*args):
        raise InterruptedError("test partial publication")
    if partial:
        with pytest.raises(InterruptedError):
            generate_dataset(path, **kwargs, _after_group=interrupt)
    else:
        generate_dataset(path, **kwargs)
    before = _file_snapshot(output)
    raw["generation"]["planning"]["enabled_tie_groups"] = []
    pin_small_plan(raw)
    changed_path = write_spec(tmp_path / "changed.json", raw)
    off = load_run_spec(changed_path, repo_root=root)
    assert on.plan.deduped_candidate_count == off.plan.deduped_candidate_count == 1
    assert on.plan.candidate_table.params_for_index(0) == off.plan.candidate_table.params_for_index(0)
    assert semantic_key_digest(on.plan) == semantic_key_digest(off.plan)
    assert on.plan.plan_fingerprint != off.plan.plan_fingerprint
    with pytest.raises(DatasetError, match="identity|incompatible"):
        generate_dataset(changed_path, **kwargs, resume=True)
    assert before == _file_snapshot(output)


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


def test_plan_is_built_once_across_complete_eight_window_smoke(
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
            "compute_max_drawdown_mtm": generate_module.COMPUTE_MAX_DRAWDOWN_MTM,
        }
        return fake_execute(plan)

    monkeypatch.setattr(config_module, "build_grid_v2_plan", counted_build)
    monkeypatch.setattr("tools.strategy_lab.generate.execute_grid_v2_candidates", checked_execute)
    result = generate_dataset(
        run_spec,
        data_root=root / "market",
        output_dir=tmp_path / "output",
        ticker_selectors=["AAAUSDT"],
        repo_root=root,
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert build_calls == manifest["provenance"]["plan_build_count"] == 1
    assert len(manifest["groups"]) == 8
    assert execution_calls == 8 * 2
    assert manifest["provenance"]["selected_slow_row_count"] == 0
    assert generate_module.COMPUTE_MAX_DRAWDOWN_MTM is True
    assert manifest["identity"]["metric_requests"][
        "compute_max_drawdown_mtm"
    ] is generate_module.COMPUTE_MAX_DRAWDOWN_MTM
    assert manifest["provenance"]["execution_backend"] == {
        "backend_kind": "compiled_numba",
        "compiled_batch_used": True,
        "execution_modes": ["stacked"],
        "config_packings": ["table"],
        "unavailable_reasons": [],
        "segment_execution_count": 16,
    }
    assert manifest["groups"][0]["execution_backend"]["backend_kind"] == "compiled_numba"
    assert manifest["groups"][0]["execution_backend"]["compiled_batch_used"] is True
    assert manifest["identity"]["resources"] == {
        "numba_threads": 1,
        "compiled_workers": 1,
        "outer_workers": 1,
        "max_signal_cache_mb": 512.0,
        "prefer_compiled": True,
        "slow_enrich_selected": False,
    }


def test_manifest_and_execution_share_mtm_request_authority(
    synthetic_pack, tmp_path, monkeypatch
):
    root, run_spec, _, _ = synthetic_pack
    observed_requests = []

    def checked_execute(plan, *_args, **kwargs):
        observed_requests.append(kwargs["compute_max_drawdown_mtm"])
        return fake_execute(plan)

    monkeypatch.setattr(generate_module, "COMPUTE_MAX_DRAWDOWN_MTM", False)
    monkeypatch.setattr(generate_module, "execute_grid_v2_candidates", checked_execute)
    result = generate_dataset(
        run_spec,
        data_root=root / "market",
        output_dir=tmp_path / "output",
        ticker_selectors=["AAAUSDT"],
        window_selectors=[1],
        repo_root=root,
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert observed_requests == [False, False]
    assert manifest["identity"]["metric_requests"] == {
        "compute_max_drawdown_mtm": False,
    }
    assert manifest["provenance"]["timings"]["run_spec_load_seconds"] >= 0.0
    assert manifest["provenance"]["timings"]["plan_build_seconds"] >= 0.0
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


@pytest.mark.parametrize("case", ["fallback", "mixed_rows", "missing_metadata"])
def test_effective_compiled_backend_is_required_before_group_publication(
    synthetic_pack, tmp_path, monkeypatch, case
):
    root, run_spec, _, _ = synthetic_pack

    def invalid_execute(plan, *_args, **_kwargs):
        rows = list(fake_rows(plan))
        metadata = {
            "backend_kind": "compiled_numba",
            "compiled_batch_used": True,
            "compiled_execution_mode": "stacked",
            "compiled_config_packing": "table",
            "compiled_unavailable_reason": None,
        }
        if case == "fallback":
            metadata["backend_kind"] = "reference"
            metadata["compiled_batch_used"] = False
            rows = [replace(row, backend_kind="reference") for row in rows]
        elif case == "mixed_rows":
            rows[0] = replace(rows[0], backend_kind="reference")
        else:
            metadata = {}
        return SimpleNamespace(rows=tuple(rows), selected=(), metadata=metadata)

    monkeypatch.setattr(
        "tools.strategy_lab.generate.execute_grid_v2_candidates", invalid_execute
    )
    output = tmp_path / case
    with pytest.raises(DatasetError, match="backend|compiled_batch_used"):
        generate_dataset(
            run_spec,
            data_root=root / "market",
            output_dir=output,
            ticker_selectors=["AAAUSDT"],
            window_selectors=[1],
            repo_root=root,
        )
    assert not (output / "manifest.json").exists()
    assert not list(output.glob("groups/**/*.npy"))


def test_candidate_error_precedes_backend_homogeneity_diagnostic(
    synthetic_pack, tmp_path, monkeypatch
):
    root, run_spec, _, _ = synthetic_pack

    def error_execute(plan, *_args, **_kwargs):
        rows = list(fake_rows(plan))
        rows[7] = replace(
            rows[7],
            status="error",
            error="build_execution_data blew up",
            backend_kind="reference",
        )
        return SimpleNamespace(
            rows=tuple(rows),
            selected=(),
            metadata={
                "backend_kind": "compiled_numba",
                "compiled_batch_used": True,
                "compiled_execution_mode": "stacked",
                "compiled_config_packing": "table",
                "compiled_unavailable_reason": None,
            },
        )

    monkeypatch.setattr(
        "tools.strategy_lab.generate.execute_grid_v2_candidates", error_execute
    )
    output = tmp_path / "candidate_error"
    with pytest.raises(
        DatasetError, match="candidate 8: build_execution_data blew up"
    ):
        generate_dataset(
            run_spec,
            data_root=root / "market",
            output_dir=output,
            ticker_selectors=["AAAUSDT"],
            window_selectors=[1],
            repo_root=root,
        )
    assert not (output / "manifest.json").exists()
    assert not list(output.glob("groups/**/*.npy"))


def test_compiled_unavailability_fails_before_execution_or_publication(
    synthetic_pack, tmp_path, monkeypatch
):
    root, run_spec, _, _ = synthetic_pack
    output = tmp_path / "compiled_unavailable"
    execution_calls = 0

    def forbidden_execute(*_args, **_kwargs):
        nonlocal execution_calls
        execution_calls += 1
        raise AssertionError("candidate execution must not start")

    monkeypatch.setattr(
        "tools.strategy_lab.generate.compiled_batch_available", lambda: False
    )
    monkeypatch.setattr(
        "tools.strategy_lab.generate.compiled_unavailable_reason",
        lambda: "NUMBA_DISABLE_JIT is set.",
    )
    monkeypatch.setattr(
        "tools.strategy_lab.generate.execute_grid_v2_candidates", forbidden_execute
    )

    with pytest.raises(
        DatasetError,
        match="compiled backend is unavailable: NUMBA_DISABLE_JIT is set",
    ):
        generate_dataset(
            run_spec,
            data_root=root / "market",
            output_dir=output,
            ticker_selectors=["AAAUSDT"],
            window_selectors=[1],
            repo_root=root,
        )

    assert execution_calls == 0
    assert not output.exists()


def test_compiled_availability_precheck_accepts_live_available_backend(monkeypatch):
    reason_calls = 0

    def forbidden_reason():
        nonlocal reason_calls
        reason_calls += 1
        raise AssertionError("unavailable reason must not be requested")

    monkeypatch.setattr(
        "tools.strategy_lab.generate.compiled_batch_available", lambda: True
    )
    monkeypatch.setattr(
        "tools.strategy_lab.generate.compiled_unavailable_reason", forbidden_reason
    )

    _require_compiled_backend_available()
    assert reason_calls == 0


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


def test_truncated_oos_is_rejected_before_candidate_execution(
    synthetic_pack, tmp_path, monkeypatch
):
    root, run_spec, _, _ = synthetic_pack
    from tools.strategy_lab import generate as generate_module

    original_builder = generate_module.build_authoritative_windows
    calls = 0

    def truncated_builder(spec, sources):
        windows = list(original_builder(spec, sources))
        windows[0] = replace(
            windows[0], oos_end=windows[0].oos_end - timedelta(minutes=30)
        )
        return tuple(windows)

    def forbidden(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("candidate execution must not start")

    monkeypatch.setattr(generate_module, "build_authoritative_windows", truncated_builder)
    monkeypatch.setattr(generate_module, "execute_grid_v2_candidates", forbidden)
    with pytest.raises(ValueError, match="OOS is truncated"):
        generate_dataset(
            run_spec,
            data_root=root / "market",
            output_dir=tmp_path / "output",
            ticker_selectors=["AAAUSDT"],
            window_selectors=[1],
            repo_root=root,
        )
    assert calls == 0
    assert not (tmp_path / "output" / "manifest.json").exists()


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


def test_primary_execution_error_remains_primary_when_preservation_also_fails(
    synthetic_pack, tmp_path, monkeypatch
):
    root, run_spec, _, _ = synthetic_pack

    def execution_failure(*_args, **_kwargs):
        raise RuntimeError("primary execution failure")

    def preservation_failure(_sources):
        raise ValueError("secondary preservation failure")

    monkeypatch.setattr(
        "tools.strategy_lab.generate.execute_grid_v2_candidates", execution_failure
    )
    monkeypatch.setattr(
        "tools.strategy_lab.generate.verify_source_preservation", preservation_failure
    )
    with pytest.raises(RuntimeError, match="primary execution failure") as raised:
        generate_dataset(
            run_spec,
            data_root=root / "market",
            output_dir=tmp_path / "output",
            ticker_selectors=["AAAUSDT"],
            window_selectors=[1],
            repo_root=root,
        )
    assert any("secondary preservation failure" in note for note in raised.value.__notes__)


def test_preservation_failure_is_raised_when_generation_has_no_primary_error(
    synthetic_pack, tmp_path, monkeypatch
):
    root, run_spec, _, _ = synthetic_pack
    prior_threads = numba.get_num_threads()
    monkeypatch.setattr("tools.strategy_lab.generate.execute_grid_v2_candidates", fake_execute)
    monkeypatch.setattr(
        "tools.strategy_lab.generate.verify_source_preservation",
        lambda _sources: (_ for _ in ()).throw(ValueError("preservation-only failure")),
    )
    with pytest.raises(ValueError, match="preservation-only failure"):
        generate_dataset(
            run_spec,
            data_root=root / "market",
            output_dir=tmp_path / "output",
            ticker_selectors=["AAAUSDT"],
            window_selectors=[1],
            repo_root=root,
        )
    assert numba.get_num_threads() == prior_threads
    assert not (tmp_path / "output" / "manifest.json").exists()


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
        np.save(group, np.zeros((480, 2, 21), dtype=np.float32), allow_pickle=False)

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
    assert np.load(group, mmap_mode="r", allow_pickle=False).shape == (480, 2, 21)
    assert np.load(group, mmap_mode="r", allow_pickle=False).dtype == np.float64


def test_resume_rejects_group_without_compiled_backend_provenance(
    synthetic_pack, tmp_path, monkeypatch
):
    root, run_spec, _, _ = synthetic_pack
    monkeypatch.setattr("tools.strategy_lab.generate.execute_grid_v2_candidates", fake_execute)
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="stop"):
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
    partial["groups"][0].pop("execution_backend")
    partial_path.write_text(json.dumps(partial), encoding="utf-8")

    with pytest.raises(DatasetError, match="does not prove compiled_numba"):
        generate_dataset(
            run_spec,
            data_root=root / "market",
            output_dir=output,
            ticker_selectors=["AAAUSDT"],
            window_selectors=[1],
            repo_root=root,
            resume=True,
        )


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
    np.save(unlisted, np.zeros((480, 2, 21), dtype=np.float64), allow_pickle=False)
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


def test_completed_output_is_immutable_across_source_failure_restore_and_no_op(
    synthetic_pack, tmp_path, monkeypatch
):
    root, run_spec, _, _ = synthetic_pack
    output = tmp_path / "output"
    _generate_fake(synthetic_pack, output, monkeypatch)
    before = _file_snapshot(output)
    source = next((root / "market").glob("OKX_AAAUSDT*.csv"))
    original_bytes = source.read_bytes()
    original_stat = source.stat()
    try:
        source.write_bytes(original_bytes + b"\n")
        with pytest.raises(ValueError, match="source validation failed"):
            _generate_fake(synthetic_pack, output, monkeypatch)
        assert _file_snapshot(output) == before
    finally:
        source.write_bytes(original_bytes)
        os.utime(
            source,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
        )

    result = _generate_fake(synthetic_pack, output, monkeypatch)
    assert result.no_op is True
    assert _file_snapshot(output) == before


def test_completed_output_checksum_corruption_is_not_silently_overwritten(
    synthetic_pack, tmp_path, monkeypatch
):
    output = tmp_path / "output"
    _generate_fake(synthetic_pack, output, monkeypatch)
    candidates = output / "candidates.json"
    candidates.write_bytes(candidates.read_bytes() + b" ")
    before = _file_snapshot(output)
    with pytest.raises(DatasetError, match="checksum/shape/dtype verification"):
        _generate_fake(synthetic_pack, output, monkeypatch)
    assert _file_snapshot(output) == before


def test_incompatible_completed_manifest_is_not_overwritten(
    synthetic_pack, tmp_path, monkeypatch
):
    output = tmp_path / "output"
    _generate_fake(synthetic_pack, output, monkeypatch)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["dataset_schema"] = "incompatible"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    before = _file_snapshot(output)
    with pytest.raises(DatasetError, match="incompatible run identity"):
        _generate_fake(synthetic_pack, output, monkeypatch)
    assert _file_snapshot(output) == before


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


@pytest.mark.slow  # Hashes the complete local Lab tree twice to prove preservation.
def test_module_cli_help_bootstraps_without_pythonpath_and_writes_nothing():
    package_root = REPO_ROOT / "tools" / "strategy_lab"
    before = _file_snapshot(package_root)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-m", "tools.strategy_lab.generate", "--help"],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "--data-root" in completed.stdout
    assert "--output-dir" in completed.stdout
    assert "--resume" in completed.stdout
    assert _file_snapshot(package_root) == before


@pytest.mark.parametrize("location", ["equal", "beneath"])
def test_output_inside_market_data_root_is_rejected_before_writes_or_execution(
    synthetic_pack, monkeypatch, location
):
    root, run_spec, _, _ = synthetic_pack
    data_root = root / "market"
    output = data_root if location == "equal" else data_root / "generated" / "output"
    before = _file_snapshot(data_root)
    calls = 0

    def forbidden(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("candidate execution must not start")

    monkeypatch.setattr("tools.strategy_lab.generate.execute_grid_v2_candidates", forbidden)
    with pytest.raises(DatasetError, match="must not equal or be beneath"):
        generate_dataset(
            run_spec,
            data_root=data_root,
            output_dir=output,
            ticker_selectors=["AAAUSDT"],
            window_selectors=[1],
            repo_root=root,
        )
    assert calls == 0
    assert _file_snapshot(data_root) == before
    if location == "beneath":
        assert not output.exists()


def test_resource_owners_are_independent_and_manifest_uses_effective_values(
    synthetic_pack, tmp_path, monkeypatch
):
    root, run_spec, _, _ = synthetic_pack
    payload = json.loads(run_spec.read_text(encoding="utf-8"))
    payload["generation"]["resources"]["grid_v2_max_cache_mb"] = 256.0
    changed_spec = tmp_path / "cache_run.json"
    changed_spec.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr("tools.strategy_lab.generate.execute_grid_v2_candidates", fake_execute)
    result = generate_dataset(
        changed_spec,
        data_root=root / "market",
        output_dir=tmp_path / "output",
        ticker_selectors=["AAAUSDT"],
        window_selectors=[1],
        repo_root=root,
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    resources = manifest["identity"]["resources"]
    assert resources["outer_workers"] == 1
    assert resources["compiled_workers"] == 1
    assert resources["numba_threads"] == 1
    assert resources["max_signal_cache_mb"] == 256.0


@pytest.mark.parametrize(("field", "value"), [("outer_workers", 2), ("numba_threads", 2)])
def test_nonsequential_resource_settings_fail_clearly(
    synthetic_pack, tmp_path, field, value
):
    root, run_spec, _, _ = synthetic_pack
    payload = json.loads(run_spec.read_text(encoding="utf-8"))
    payload["generation"]["resources"][field] = value
    changed_spec = tmp_path / f"{field}.json"
    changed_spec.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=rf"{field}.*requires exactly 1"):
        load_run_spec(changed_spec, repo_root=root)


def test_generation_avoids_ranking_objective_diversity_and_external_writes(
    synthetic_pack, tmp_path, monkeypatch
):
    root, run_spec, _, _ = synthetic_pack
    output = tmp_path / "output"
    outside_before = _file_snapshot(root)
    calls: list[str] = []

    def forbidden(name):
        def fail(*_args, **_kwargs):
            calls.append(name)
            raise AssertionError(f"{name} must not be called")

        return fail

    monkeypatch.setattr("core.grid_engine.rank_grid_results", forbidden("ranking"))
    monkeypatch.setattr("core.grid_engine._validate_objective_set", forbidden("objectives"))
    monkeypatch.setattr("core.grid_engine.apply_diversity_cap", forbidden("diversity"))
    monkeypatch.setattr("tools.strategy_lab.generate.execute_grid_v2_candidates", fake_execute)

    from tools.strategy_lab import dataset as dataset_module

    original_atomic_write = dataset_module.atomic_write
    write_targets: list[Path] = []

    def checked_atomic_write(path, writer):
        resolved = Path(path).resolve()
        resolved.relative_to(output.resolve())
        write_targets.append(resolved)
        return original_atomic_write(path, writer)

    monkeypatch.setattr(dataset_module, "atomic_write", checked_atomic_write)
    generate_dataset(
        run_spec,
        data_root=root / "market",
        output_dir=output,
        ticker_selectors=["AAAUSDT"],
        window_selectors=[1],
        repo_root=root,
    )
    assert calls == []
    assert write_targets
    outside_after = {
        path: facts for path, facts in _file_snapshot(root).items() if not path.startswith("output/")
    }
    assert outside_after == outside_before


def test_two_fresh_processes_produce_identical_real_smoke_outcomes(synthetic_pack, tmp_path):
    root, run_spec, _, _ = synthetic_pack
    script = (
        "from tools.strategy_lab.generate import generate_dataset; import sys; "
        "generate_dataset(sys.argv[1], data_root=sys.argv[2], output_dir=sys.argv[3], "
        "ticker_selectors=['AAAUSDT'], window_selectors=[1], repo_root=sys.argv[4])"
    )
    outputs = [tmp_path / "process_one", tmp_path / "process_two"]
    environment = os.environ.copy()
    environment["NUMBA_CACHE_DIR"] = environment.get("NUMBA_CACHE_DIR") or str(tmp_path / "numba_cache")
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
    assert first_manifest["identity"]["expected_group_shape"] == [480, 2, 21]
    relative = first_manifest["groups"][0]["path"]
    array = np.load(outputs[0] / relative, allow_pickle=False)
    assert array.shape == (480, 2, 21) and array.dtype == np.float64
    assert array.shape[0] == 480
    assert np.isnan(array[:, :, METRIC_AXIS.index("sqn")]).any()
    assert np.isfinite(
        array[:, :, METRIC_AXIS.index("max_drawdown_mtm_pct")]
    ).any()
