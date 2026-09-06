"""Explicit real-pack HTTP WFA tie-back; intentionally not normally collected."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from core import storage
from core.storage import load_study_from_db, load_wfa_window_trials
from tools.strategy_lab.certify import (
    SELECTED_TRIAL_NET_PROFIT_BASIS,
    WINDOW_NET_PROFIT_BASIS,
    _snapshot_tree,
    allowed_roots_with_data_root,
    assert_path_contained,
    changed_snapshot_paths,
    select_primary_candidate,
    selected_trial_net_profit_from_lab,
    semantic_float_equal,
    utc_timestamps_equal,
    wfa_window_net_profit_from_lab,
)
from tools.strategy_lab.config import REPO_ROOT, canonical_json_bytes, load_run_spec
from tools.strategy_lab.dataset import METRIC_AXIS
from tools.strategy_lab.inventory import resolve_data_root
from ui import server_services
from ui.server import app
from tests.strategy_lab.certification_helpers import _normalized_params


PLAN_FINGERPRINT = "c0e40ede6521a1cc02063ef2c9245f58c0093ca97aeb4bd858b75b5d09c7f434"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setitem(app.config, "TESTING", True)
    with app.test_client() as test_client:
        yield test_client


def _required_path(environment_name: str) -> Path:
    value = os.environ.get(environment_name)
    if not value:
        pytest.fail(f"{environment_name} is required for explicit real WFA certification.")
    path = Path(value).resolve()
    if not path.exists():
        pytest.fail(f"{environment_name} does not exist: {path}")
    return path


@pytest.fixture
def protected_data_root():
    data_root = resolve_data_root(_required_path("MERLIN_STRATEGY_LAB_DATA_ROOT"))
    protected_roots = {
        "market_data": data_root,
        "production_storage_queue": REPO_ROOT / "src" / "storage",
        "production_presets": REPO_ROOT / "src" / "presets",
    }
    before = {
        label: _snapshot_tree(root)
        for label, root in protected_roots.items()
    }
    try:
        yield data_root
    finally:
        changed = []
        for label, root in protected_roots.items():
            changed.extend(
                changed_snapshot_paths(
                    before[label],
                    _snapshot_tree(root),
                    label=label,
                )
            )
        if changed:
            pytest.fail(
                "protected artifacts changed: "
                + ", ".join(changed[:20])
                + (f" (+{len(changed) - 20} more)" if len(changed) > 20 else "")
            )


def _payload(candidate_count: int) -> dict:
    axes = {
        "stopX": [1.0, 3.0, 0.5],
        "stopRR": [1.5, 3.0, 0.5],
        "stopLP": [2, 4, 2],
        "stopMaxPct": [2.0, 8.0, 2.0],
        "stopMaxDays": [2, 6, 2],
    }
    fixed = {
        "commissionPct": 0.05,
        "contractSize": 0.0001,
        "enableLong": True,
        "enableShort": True,
        "entryMode": "Reversal @ Triangle",
        "fastLength": 21,
        "fastSmooth": 7,
        "initialCapital": 1000.0,
        "riskPerTrade": 2.0,
        "slowLength": 112,
        "slowSmooth": 3,
        "thresholdOB": 20,
        "thresholdOS": 20,
        "useTrailMA": False,
        "dateFilter": True,
        "start": "2025-10-01T00:00:00Z",
        "end": "2026-08-01T00:00:00Z",
    }
    return {
        "strategy_id": "s06_r_trend_v02_b2",
        "optimization_mode": "grid",
        "enabled_params": {name: True for name in axes},
        "param_ranges": axes,
        "param_types": {
            "stopX": "float",
            "stopRR": "float",
            "stopLP": "int",
            "stopMaxPct": "float",
            "stopMaxDays": "int",
        },
        "fixed_params": fixed,
        "risk_per_trade_pct": 2.0,
        "contract_size": 0.0001,
        "commission_rate": 0.05,
        "worker_processes": 1,
        "filter_min_profit": False,
        "min_profit_threshold": 0.0,
        "objectives": ["net_profit_pct"],
        "primary_objective": "net_profit_pct",
        "constraints": [],
        "score_config": {},
        "grid_fast_objectives": ["net_profit_pct"],
        "grid_fast_primary_objective": "net_profit_pct",
        "grid_slow_refinement_enabled": False,
        "grid_slow_objectives": [],
        "grid_slow_primary_objective": None,
        "grid_v2_planning_policy": "full",
        "grid_budget": candidate_count,
        "grid_seed": 42,
        "grid_top_candidates": 1,
        "grid_enabled_modes": ["bracket"],
        "grid_diversity_enabled": False,
        "grid_diversity_max_per_group": 2,
        "grid_v2_prefer_compiled": True,
        "grid_v2_max_cache_mb": 512.0,
        "postProcess": {"enabled": False, "dsrEnabled": False, "stressTest": {"enabled": False}},
        "oosTest": {"enabled": False},
    }


def _recursive_values(value, key):
    found = []
    if isinstance(value, dict):
        for name, child in value.items():
            if name == key:
                found.append(child)
            found.extend(_recursive_values(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(_recursive_values(child, key))
    return found


def test_real_crvusdt_http_wfa_matches_existing_lab_smoke(
    client,
    tmp_path_factory,
    monkeypatch,
    protected_data_root,
):
    data_root = protected_data_root
    work_dir = _required_path("MERLIN_STRATEGY_LAB_CERT_WORK_DIR")
    smoke = work_dir / "smoke_one"
    if not (smoke / "manifest.json").is_file():
        pytest.fail("Phase 1-B smoke_one must exist before the WFA tie-back.")
    spec = load_run_spec(
        REPO_ROOT / "tools" / "strategy_lab" / "runspecs" / "s06_bracket_mvp.json"
    )
    assert spec.plan is not None
    expected_candidate_count = int(spec.plan.deduped_candidate_count)
    assert spec.plan.plan_fingerprint == PLAN_FINGERPRINT
    expected_window_count = int(
        spec.generation["windows"]["expected_window_count"]
    )
    candidates = json.loads(
        (smoke / "candidates.json").read_text(encoding="utf-8")
    )["candidates"]
    manifest = json.loads((smoke / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["identity"]["candidate_count"] == expected_candidate_count
    assert len(manifest["groups"]) == expected_window_count
    assert len(candidates) == expected_candidate_count
    crv_path = data_root / next(
        item["relative_filename"]
        for item in json.loads(
            (
                REPO_ROOT
                / "tools"
                / "strategy_lab"
                / "runspecs"
                / "tickers_current.json"
            ).read_text(encoding="utf-8")
        )["entries"]
        if item["canonical_symbol"] == "CRVUSDT"
    )

    isolated_root = tmp_path_factory.getbasetemp().resolve()
    assert_path_contained(
        storage._active_db_path,
        isolated_root,
        field="active WFA test storage",
    )
    monkeypatch.setattr(
        server_services,
        "CSV_ALLOWED_ROOTS",
        allowed_roots_with_data_root(server_services.CSV_ALLOWED_ROOTS, data_root),
    )
    response = client.post(
        "/api/walkforward",
        data={
            "strategy": "s06_r_trend_v02_b2",
            "csvPath": str(crv_path),
            "warmupBars": "1000",
            "config": json.dumps(_payload(expected_candidate_count)),
            "wf_period_unit": "months",
            "wf_is_period_months": "2",
            "wf_oos_period_months": "1",
            "wf_adaptive_mode": "false",
            "wf_store_top_n_trials": "50",
        },
    )
    assert response.status_code == 200, response.get_data(as_text=True)
    response_payload = response.get_json()
    assert response_payload["status"] == "success"
    study_id = response_payload["study_id"]
    loaded = load_study_from_db(study_id)
    assert loaded is not None
    windows = loaded["windows"]
    assert len(windows) == expected_window_count
    stored_fingerprints = _recursive_values(loaded, "grid_v2_plan_fingerprint")
    stored_candidate_counts = _recursive_values(loaded, "candidate_count")
    window_grid_diagnostics = [
        (window.get("module_status") or {}).get("grid_v2") or {}
        for window in windows
    ]
    stored_workers = sorted(
        {
            diagnostic.get("compiled_workers")
            for diagnostic in window_grid_diagnostics
            if diagnostic.get("compiled_workers") is not None
        }
    )
    missing_selected_identity = []
    for stored in windows:
        selected_trials = [
            trial
            for trial in load_wfa_window_trials(stored["window_id"]).get("optuna_is", [])
            if trial["is_selected"]
        ]
        if len(selected_trials) != 1:
            missing_selected_identity.append(int(stored["window_number"]))
            continue
        module_metrics = selected_trials[0].get("module_metrics") or {}
        if not module_metrics.get("semantic_key") or module_metrics.get("candidate_id") is None:
            missing_selected_identity.append(int(stored["window_number"]))
    blockers = []
    if PLAN_FINGERPRINT not in stored_fingerprints:
        blockers.append("stored WFA study has no Grid V2 plan fingerprint")
    if expected_candidate_count not in stored_candidate_counts:
        blockers.append("stored WFA study has no authoritative candidate count")
    if stored_workers != [1]:
        blockers.append(f"stored per-window compiled_workers is {stored_workers}, expected [1]")
    if missing_selected_identity:
        blockers.append(
            "selected stored trials omit semantic_key/candidate_id in windows "
            + ",".join(str(value) for value in missing_selected_identity)
        )
    if blockers:
        pytest.fail("Phase 1-B production WFA contract blocker: " + "; ".join(blockers))

    metric_index = {name: index for index, name in enumerate(METRIC_AXIS)}
    evidence_windows = []
    assert len(windows) == len(manifest["groups"])
    for stored, group_record in zip(windows, manifest["groups"]):
        matrix = np.load(smoke / group_record["path"], allow_pickle=False)
        profits = matrix[:, 0, metric_index["net_profit_pct"]]
        selected_index, top_ties = select_primary_candidate(profits, candidates)
        selected = candidates[selected_index]

        trial_groups = load_wfa_window_trials(stored["window_id"])
        selected_trials = [
            trial
            for trial in trial_groups.get("optuna_is", [])
            if trial["is_selected"]
        ]
        assert len(selected_trials) == 1
        selected_trial = selected_trials[0]
        assert selected_trial["module_metrics"]["semantic_key"] == selected["semantic_key"]
        assert (
            int(selected_trial["module_metrics"]["candidate_id"])
            == selected["candidate_id"]
        )
        assert _normalized_params(stored["best_params"]) == _normalized_params(
            selected["params"]
        )

        window_identity = manifest["identity"]["windows"][int(stored["window_number"]) - 1]
        for stored_name, lab_name in (
            ("is_start_ts", "is_start"), ("is_end_ts", "is_end"),
            ("oos_start_ts", "oos_start"), ("oos_end_ts", "oos_end"),
        ):
            assert utc_timestamps_equal(
                stored[stored_name],
                window_identity[lab_name],
                field=stored_name,
            ), f"{stored_name}: stored and Lab UTC timestamps differ."

        for segment_index, prefix in ((0, "is"), (1, "oos")):
            lab_profit = matrix[
                selected_index,
                segment_index,
                metric_index["net_profit_pct"],
            ]
            assert semantic_float_equal(
                stored[f"{prefix}_net_profit_pct"],
                wfa_window_net_profit_from_lab(lab_profit),
            ), f"{prefix} Net Profit differs on {WINDOW_NET_PROFIT_BASIS} basis."
            assert semantic_float_equal(
                stored[f"{prefix}_max_drawdown_pct"],
                matrix[
                    selected_index,
                    segment_index,
                    metric_index["max_drawdown_pct"],
                ],
            )
            assert stored[f"{prefix}_total_trades"] == int(
                matrix[
                    selected_index,
                    segment_index,
                    metric_index["total_trades"],
                ]
            )
            assert semantic_float_equal(
                stored[f"{prefix}_sharpe_daily"],
                matrix[
                    selected_index,
                    segment_index,
                    metric_index["sharpe_daily"],
                ],
            )
            for suffix in ("sharpe_daily_observations", "sharpe_daily_active_days"):
                expected = matrix[selected_index, segment_index, metric_index[suffix]]
                expected = None if np.isnan(expected) else int(expected)
                assert stored[f"{prefix}_{suffix}"] == expected

        assert semantic_float_equal(
            selected_trial["net_profit_pct"],
            selected_trial_net_profit_from_lab(
                matrix[selected_index, 0, metric_index["net_profit_pct"]]
            ),
        ), (
            "selected trial Net Profit differs on "
            f"{SELECTED_TRIAL_NET_PROFIT_BASIS} basis."
        )
        evidence_windows.append({
            "window_number": int(stored["window_number"]),
            "candidate_id": int(selected["candidate_id"]),
            "semantic_key": selected["semantic_key"],
            "top_primary_tie_count": top_ties,
            "structural_status": "passed",
        })

    evidence = {
        "status": "passed",
        "route": "POST /api/walkforward",
        "study_id": study_id,
        "isolated_storage_root": str(storage._active_db_path.parent),
        "plan_fingerprint": PLAN_FINGERPRINT,
        "candidate_count": expected_candidate_count,
        "window_count": expected_window_count,
        "window_net_profit_basis": WINDOW_NET_PROFIT_BASIS,
        "selected_trial_net_profit_basis": SELECTED_TRIAL_NET_PROFIT_BASIS,
        "windows": evidence_windows,
    }
    (work_dir / "wfa_certification.json").write_bytes(canonical_json_bytes(evidence) + b"\n")
