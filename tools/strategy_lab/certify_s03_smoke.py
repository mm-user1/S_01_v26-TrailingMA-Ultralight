"""Opt-in real S03 gate: CRV, windows 1..3; never part of test discovery."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np

from core.grid_v2 import (GridV2StrategyHooks, deterministic_candidate_subset_indices,
                          execute_grid_v2_candidates)
from strategies.s03_reversal_v16_4_a_adaptive_ma_b2 import strategy
from tools.strategy_lab.certify import (
    _compare_grid_runs, _deterministic_manifest, _snapshot_tree,
    assert_certification_work_dir_allowed, candidate_identity_mappings,
    finite_mtm_group_facts, geometry_candidate_ids,
)
from tools.strategy_lab.config import load_run_spec
from tools.strategy_lab.inventory import resolve_data_root
from tools.strategy_lab.data_quality import (
    build_authoritative_windows, prepare_segment, validate_selected_sources,
    verify_source_preservation,
)
from tools.strategy_lab.dataset import (
    DATASET_SCHEMA_VERSION, METRIC_AXIS, SEGMENT_AXIS, atomic_write_json,
    load_json, project_candidates, validate_candidate_projection,
)
from tools.strategy_lab.generate import generate_dataset

RUNSPEC = REPO_ROOT / "tools/strategy_lab/runspecs/s03_adaptive_ma_symmetric.json"


def representative_indices(plan, projection):
    """Geometry only: every MA at endpoints and middle; stable fill to >=32."""
    _, index_by_id = candidate_identity_mappings(plan, projection)
    required = {index_by_id[i] for i in geometry_candidate_ids(projection)}
    for ma in plan.parameter_domains["maType3"].values:
        for length, count, band in ((25, 1, 0.2), (250, 7, 2.0), (125, 4, 1.0)):
            required.add(next(i for i in range(plan.deduped_candidate_count)
                              if all(plan.candidate_table.params_for_index(i)[key] == value
                                     for key, value in (("maType3", ma), ("maLength3", length),
                                                        ("closeCountLong", count), ("tBandLongPct", band)))))
    indices = deterministic_candidate_subset_indices(plan.deduped_candidate_count, 40, tuple(required))
    if len(indices) < 32 or not required.issubset(indices):
        raise ValueError("Representative subset lost required geometry or minimum size.")
    return indices


def assert_symmetric_projection(plan, projection):
    validate_candidate_projection(projection, plan)
    candidate_identity_mappings(plan, projection)
    assert plan.deduped_candidate_count == len(projection["candidates"]) == 2800
    assert projection["global_axis_names"] == ["maType3", "maLength3", "closeCountLong", "tBandLongPct"]
    counts = {}
    for row in projection["candidates"]:
        params = row["params"]
        assert params["closeCountLong"] == params["closeCountShort"]
        assert params["tBandLongPct"] == params["tBandShortPct"]
        counts[params["maType3"]] = counts.get(params["maType3"], 0) + 1
    assert list(counts) == list(plan.parameter_domains["maType3"].values)
    assert set(counts.values()) == {700}


def certify(data_root: Path, work_dir: Path):
    root = resolve_data_root(data_root)
    work = assert_certification_work_dir_allowed(work_dir)
    if work.exists():
        raise ValueError("Certification requires a fresh work directory.")
    if work.is_relative_to(root) or root.is_relative_to(work):
        raise ValueError("Certification work and market data roots must be disjoint.")
    spec = load_run_spec(RUNSPEC)
    plan = spec.plan
    projection = project_candidates(plan, spec.generation["strategy"])
    assert_symmetric_projection(plan, projection)
    indices = representative_indices(plan, projection)
    work.mkdir(parents=True)
    # Publish this recipe before any execution or inspection of PnL.
    atomic_write_json(work / "selection.json", {
        "recipe": "Every MA at (25,1,0.2), (250,7,2.0), (125,4,1.0); shared geometry edges and deterministic evenly spaced fill (limit 40).",
        "indices": indices,
        "candidate_ids": [plan.candidate_table.candidate_id_for_index(i) for i in indices],
    })
    sources, _ = validate_selected_sources(root, spec.inventory.entries, timeframe_minutes=30)
    source = next(s for s in sources if s.entry["canonical_symbol"] == "CRVUSDT")
    assert source.entry["cell"] == "dev"
    windows = build_authoritative_windows(spec, sources)[:3]
    parity = {}
    try:
        # Independent generator processes retain their own logs and host provenance.
        for name in ("clean-a", "clean-b"):
            command = [sys.executable, "-B", "-m", "tools.strategy_lab.generate",
                       str(RUNSPEC), "--data-root", str(root),
                       "--output-dir", str(work / name), "--ticker", "CRVUSDT",
                       "--window", "1", "--window", "2", "--window", "3"]
            print(f"Generating {name}", flush=True)
            with (work / f"{name}.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        hooks = GridV2StrategyHooks.from_strategy(strategy)
        reference_plan = replace(plan, settings=replace(plan.settings, prefer_compiled=False))
        for window in windows:
            for segment in SEGMENT_AXIS:
                label = f"{window.window_id}/{segment}"
                print(f"Parity {label}: {len(indices)} candidates", flush=True)
                prepared = prepare_segment(source, window, segment, warmup_bars=1000,
                                           timeframe_minutes=30, oos_period_months=1)
                kwargs = dict(candidate_indices=indices, compute_sharpe_daily=True,
                              compute_sqn=True, compute_max_drawdown_mtm=True)
                compiled = execute_grid_v2_candidates(plan, prepared.dataframe, prepared.trade_start_idx, hooks, **kwargs)
                reference = execute_grid_v2_candidates(reference_plan, prepared.dataframe, prepared.trade_start_idx, hooks, **kwargs)
                parity[label] = _compare_grid_runs(compiled, reference, plan,
                    expected_candidate_count=len(indices), segment_name=label)
        def interrupt_after_first(count, path):
            raise InterruptedError("Intentional smoke interruption after published group.")

        args = dict(data_root=root, output_dir=work / "resumed", ticker_selectors=["CRVUSDT"],
                    window_selectors=[1, 2, 3])
        try:
            generate_dataset(RUNSPEC, **args, _after_group=interrupt_after_first)
        except InterruptedError:
            pass
        else:
            raise AssertionError("Expected deliberate interruption.")
        partial = load_json(work / "resumed/manifest.partial.json", "partial")
        assert len(partial["groups"]) == 1
        resumed = generate_dataset(RUNSPEC, **args, resume=True)
        before = _snapshot_tree(work / "resumed")
        noop = generate_dataset(RUNSPEC, **args, resume=True)
        assert noop.no_op and before == _snapshot_tree(work / "resumed")
        manifests = [load_json(work / name / "manifest.json", name) for name in ("clean-a", "clean-b", "resumed")]
        deterministic = [_deterministic_manifest(m) for m in manifests]
        assert deterministic[0] == deterministic[1] == deterministic[2]
        groups = []
        for name, manifest in zip(("clean-a", "clean-b", "resumed"), manifests):
            assert manifest["schema_version"] == DATASET_SCHEMA_VERSION
            assert len(manifest["groups"]) == 3 and len(METRIC_AXIS) == 21
            assert_symmetric_projection(plan, load_json(work / name / "candidates.json", "projection"))
            for group in manifest["groups"]:
                matrix = np.load(work / name / group["path"], allow_pickle=False)
                facts = finite_mtm_group_facts(matrix, group_label=group["path"], candidate_count=2800)
                assert facts["finite_mtm_count"] == facts["mtm_value_count"] == 5600
                if name == "clean-a":
                    groups.append({**group, **facts})
        command = [sys.executable, "-B", "-m", "tools.strategy_lab.analysis.cli", "analyze",
                   "--dataset", str(work / "clean-a"), "--scope", "development",
                   "--allow-partial-scope", "--output", str(work / "analysis")]
        with (work / "analysis.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        evidence = {"scope": "partial development only: CRVUSDT windows 1,2,3",
                    "generation_sha256": spec.generation_sha256,
                    "pre_registration_sha256": spec.pre_registration_sha256,
                    "plan_fingerprint": plan.plan_fingerprint, "parity": parity,
                    "groups": groups, "resume_reused_groups": resumed.reused_groups,
                    "rerun_status": "verified_noop",
                    "deterministic_semantic_outputs_equal": True,
                    "source_snapshots": {s.entry["canonical_symbol"]: asdict(s.snapshot) for s in sources}}
    finally:
        preservation = verify_source_preservation(sources)
    evidence["source_preservation"] = preservation
    atomic_write_json(work / "evidence.json", evidence)
    print(f"Certified: {work / 'evidence.json'}", flush=True)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    certify(args.data_root, args.work_dir)


if __name__ == "__main__":
    main()
