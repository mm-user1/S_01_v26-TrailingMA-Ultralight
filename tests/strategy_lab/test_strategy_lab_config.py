from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from core.grid_v2 import GridV2Settings, build_grid_v2_plan
from strategies import get_strategy_config
from tools.strategy_lab.config import (
    ANALYSIS_SCOPES,
    EVIDENCE_CRITERIA,
    OBSERVATION_CONTRACT,
    StrategyLabConfigError,
    canonical_sha256,
    load_run_spec,
    semantic_key_digest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNSPEC_PATH = REPO_ROOT / "tools" / "strategy_lab" / "runspecs" / "s06_bracket_mvp.json"
INVENTORY_PATH = REPO_ROOT / "tools" / "strategy_lab" / "runspecs" / "tickers_current.json"


def _raw_runspec():
    return json.loads(RUNSPEC_PATH.read_text(encoding="utf-8"))


def _write_spec(tmp_path, value, *, inventory=None):
    if inventory is None:
        inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    (tmp_path / "inventory.json").write_text(
        json.dumps(inventory, allow_nan=False), encoding="utf-8"
    )
    value["generation"]["inventory"]["inventory_path"] = "inventory.json"
    path = tmp_path / "runspec.json"
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return path


def test_current_runspec_is_typed_bound_and_identity_checked():
    spec = load_run_spec(RUNSPEC_PATH)

    assert spec.strategy_id == "s06_r_trend_v02_b2"
    assert spec.plan.deduped_candidate_count == 480
    assert spec.plan.metadata["planning"]["effective_policy"] == "full"
    assert spec.inventory.sha256 == "87abcb709ad854b0b2f9d1049d34e38bdac26ad635a7b3650538fdb32426c588"
    assert spec.inventory.ticker_list_digest == "1cb80d62d30ddf5103b0b3fedb77cf193402991ecc6bc535923c3cfa73165fb5"
    assert spec.generation_sha256 == canonical_sha256(spec.generation)
    assert spec.pre_registration_sha256 == canonical_sha256(spec.raw)


def test_another_registered_v2_strategy_uses_the_same_generic_loader(tmp_path):
    raw = _raw_runspec()
    config = get_strategy_config("s06_r_trend_v02_regime_trendlines_b2")
    base_params = {
        name: declaration["default"]
        for name, declaration in config["parameters"].items()
        if declaration.get("role") != "runtime"
    }
    base_params["useTrailMA"] = False
    settings = GridV2Settings(
        enabled_variants=("bracket",),
        enabled_axes=(),
        planning_policy="full",
        prefer_compiled=True,
    )
    plan = build_grid_v2_plan(config, settings, base_params)
    strategy = raw["generation"]["strategy"]
    strategy["strategy_id"] = config["id"]
    strategy["strategy_version"] = config["version"]
    raw["generation"]["economics"]["base_params"] = base_params
    planning = raw["generation"]["planning"]
    planning["enabled_axes"] = []
    planning["axis_values"] = {}
    planning["expected_candidate_count"] = 1
    planning["expected_plan_fingerprint"] = plan.plan_fingerprint
    planning["expected_semantic_key_digest"] = semantic_key_digest(plan)
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    raw["generation"]["inventory"]["inventory_sha256"] = canonical_sha256(inventory)

    loaded = load_run_spec(_write_spec(tmp_path, raw, inventory=inventory), repo_root=tmp_path)

    assert loaded.strategy_id == "s06_r_trend_v02_regime_trendlines_b2"
    assert loaded.plan.deduped_candidate_count == 1


def test_v1_strategy_is_rejected_with_actionable_message(tmp_path):
    raw = _raw_runspec()
    raw["generation"]["strategy"]["strategy_id"] = "s06_r_trend_v02"

    with pytest.raises(StrategyLabConfigError, match="Backtester V1.*requires.*V2"):
        load_run_spec(_write_spec(tmp_path, raw), repo_root=tmp_path)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("{", "malformed JSON"),
        ("[]", "expected an object"),
    ],
)
def test_malformed_and_non_object_json_are_concise(tmp_path, contents, message):
    path = tmp_path / "bad.json"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(StrategyLabConfigError, match=message):
        load_run_spec(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: raw.pop("run_name"), "missing required field.*run_name"),
        (lambda raw: raw.__setitem__("schema_version", "future"), "schema_version"),
        (
            lambda raw: raw["generation"]["strategy"].__setitem__("runtime_contract_version", "future"),
            "runtime_contract_version",
        ),
        (
            lambda raw: raw["generation"]["inventory"].__setitem__("expected_ticker_count", True),
            "booleans are invalid",
        ),
        (
            lambda raw: raw["generation"]["windows"].__setitem__("warmup_bars", True),
            "booleans are invalid",
        ),
        (
            lambda raw: raw["generation"]["windows"].__setitem__("requested_end", "2025-10-01"),
            "dates must increase",
        ),
        (
            lambda raw: raw["preregistration"]["split"].__setitem__("holdout_ticker_count", 93),
            "must equal expected_ticker_count",
        ),
        (
            lambda raw: raw["preregistration"]["analysis_scopes"][0].__setitem__("requires_unlock", True),
            "four-scope contract",
        ),
        (
            lambda raw: raw["generation"]["planning"].__setitem__("enabled_variants", ["unknown"]),
            "plan construction failed|unsupported variant",
        ),
        (
            lambda raw: (
                raw["generation"]["planning"].__setitem__("enabled_axes", ["unknownAxis"]),
                raw["generation"]["planning"].__setitem__("axis_values", {"unknownAxis": [1]}),
            ),
            "plan construction failed",
        ),
        (
            lambda raw: raw["generation"]["planning"].__setitem__("expected_plan_fingerprint", "0" * 64),
            "plan fingerprint mismatch",
        ),
        (
            lambda raw: raw["generation"]["execution"].__setitem__("boundary", "none"),
            "does not match resolved V2 variant",
        ),
    ],
)
def test_invalid_contract_facts_are_rejected(tmp_path, mutate, message):
    raw = _raw_runspec()
    mutate(raw)

    with pytest.raises(StrategyLabConfigError, match=message):
        load_run_spec(_write_spec(tmp_path, raw), repo_root=tmp_path)


def test_inventory_digest_and_count_binding_are_strict(tmp_path):
    raw = _raw_runspec()
    raw["generation"]["inventory"]["inventory_sha256"] = "0" * 64
    with pytest.raises(StrategyLabConfigError, match="inventory canonical digest mismatch"):
        load_run_spec(_write_spec(tmp_path, raw), repo_root=tmp_path)

    raw = _raw_runspec()
    raw["generation"]["inventory"]["ticker_list_digest"] = "0" * 64
    with pytest.raises(StrategyLabConfigError, match="ordered ticker digest mismatch"):
        load_run_spec(_write_spec(tmp_path, raw), repo_root=tmp_path)


def test_evidence_and_scope_facts_round_trip_exactly():
    spec = load_run_spec(RUNSPEC_PATH)

    assert spec.preregistration["analysis_scopes"] == list(ANALYSIS_SCOPES)
    assert spec.preregistration["observation_contract"] == OBSERVATION_CONTRACT
    assert spec.preregistration["evidence_criteria"] == EVIDENCE_CRITERIA
    assert spec.preregistration["maximum_nominated_rules"] == 3


def test_canonical_digest_is_stable_in_two_independent_processes():
    code = (
        "import json; from pathlib import Path; "
        "from tools.strategy_lab.config import canonical_sha256; "
        f"p=Path({str(RUNSPEC_PATH)!r}); "
        "print(canonical_sha256(json.loads(p.read_text(encoding='utf-8'))))"
    )
    outputs = [
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().splitlines()[-1]
        for _ in range(2)
    ]

    assert outputs[0] == outputs[1] == canonical_sha256(_raw_runspec())
