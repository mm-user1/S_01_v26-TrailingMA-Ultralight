from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.strategy_lab.config import load_run_spec


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNSPEC_PATH = REPO_ROOT / "tools" / "strategy_lab" / "runspecs" / "s06_bracket_mvp.json"


def test_current_plan_rebuilds_all_frozen_identity_pins_without_market_data():
    spec = load_run_spec(RUNSPEC_PATH, validate_inventory=False)
    plan = spec.plan
    expected = spec.generation["planning"]
    digest = hashlib.sha256()
    for index in range(plan.deduped_candidate_count):
        digest.update(plan.candidate_table.semantic_key_for_index(index).encode("utf-8"))
        digest.update(b"\n")

    assert plan.deduped_candidate_count == expected["expected_candidate_count"] == 480
    assert plan.metadata["planning"]["effective_policy"] == "full"
    assert plan.plan_fingerprint == expected["expected_plan_fingerprint"]
    assert digest.hexdigest() == expected["expected_semantic_key_digest"]


def test_plan_versions_axes_order_and_semantic_rows_are_frozen():
    spec = load_run_spec(RUNSPEC_PATH, validate_inventory=False)
    plan = spec.plan
    strategy = spec.generation["strategy"]
    planning = spec.generation["planning"]

    assert plan.metadata["engine_version"] == strategy["grid_v2_engine_version"]
    assert plan.metadata["planning"]["plan_identity_schema_version"] == strategy["plan_identity_schema_version"]
    assert plan.metadata["planning"]["semantic_identity_schema_version"] == strategy["semantic_identity_schema_version"]
    assert plan.metadata["planning"]["runtime_contract_version"] == strategy["runtime_contract_version"]
    assert plan.metadata["planning"]["planning_policy_version"] == strategy["planning_policy_version"]
    assert plan.metadata["planning"]["allocator_version"] == strategy["allocator_version"]
    assert list(plan.candidate_table.axis_names) == planning["enabled_axes"]
    assert [plan.candidate_table.candidate_for_index(index).candidate_id for index in (0, 1, 478, 479)] == [1, 2, 479, 480]
    assert all(
        json.loads(plan.candidate_table.semantic_key_for_index(index))["engine"]
        == strategy["grid_v2_engine_version"]
        for index in (0, 479)
    )
