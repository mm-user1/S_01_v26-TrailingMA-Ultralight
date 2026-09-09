import copy
import json
from dataclasses import replace

import pytest

from core.grid_v2 import build_grid_v2_plan
from strategies.s03_reversal_v16_4_a_adaptive_ma_b2 import strategy
from tools.strategy_lab.config import StrategyLabConfigError, canonical_json_bytes, load_run_spec
from tools.strategy_lab.certify_s03_smoke import assert_symmetric_projection, representative_indices
from tools.strategy_lab.dataset import DatasetError, project_candidates, validate_candidate_projection
from tools.strategy_lab.generate import _assert_plan_contract
from tests.strategy_lab.s03_helpers import RUNSPEC, pin_small_plan, small_raw, write_spec


def test_portable_s03_population_identity_projection_and_geometry(tmp_path):
    spec = load_run_spec(RUNSPEC)
    assert spec.plan.plan_fingerprint == "abd173f523cf956862041c00ebd60ba1d36616cbe7b7b7c83dfeecb5ad6eec7f"
    assert spec.generation_sha256 == "75877f88565dfd328dab64f3d17596b3c243738f572a6d111e3d6c2d8bd6b7c6"
    assert spec.pre_registration_sha256 == "3a4988828c8445b4af71545e361fb89f0ef1639907fbec7d310f31ad49aa8d27"
    assert spec.generation["planning"]["expected_semantic_key_digest"] == "1a8aab968ff6165ae23c55a735642a587700a959b073374046a4591e0ceab17b"
    assert spec.plan.settings.enabled_tie_groups == ("symmetricLongShort",)
    assert spec.plan.settings.enabled_variants is None
    assert spec.plan.candidate_table.variant_names == ("plain",)
    assert spec.generation["windows"]["expected_window_count"] == 8
    projection = project_candidates(spec.plan, spec.generation["strategy"])
    assert_symmetric_projection(spec.plan, projection)
    path = write_spec(tmp_path / "projection.json", projection)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert_symmetric_projection(spec.plan, saved)
    selected = representative_indices(spec.plan, saved)
    assert selected == representative_indices(spec.plan, projection) and len(selected) >= 32
    params = [spec.plan.candidate_table.params_for_index(i) for i in selected]
    for ma in spec.plan.parameter_domains["maType3"].values:
        for length, count, band in ((25, 1, 0.2), (250, 7, 2.0), (125, 4, 1.0)):
            assert any(p["maType3"] == ma and p["maLength3"] == length and
                       p["closeCountLong"] == count and p["tBandLongPct"] == band for p in params)
    saved["candidates"][0]["params"]["closeCountShort"] = 2
    with pytest.raises(DatasetError):
        assert_symmetric_projection(spec.plan, saved)


@pytest.mark.parametrize("value", [None, "symmetricLongShort", 3, {}, [None], [1], [""],
                                    ["unknown"], ["symmetricLongShort", "symmetricLongShort"]])
def test_tie_input_validation_is_strict(tmp_path, value):
    raw = small_raw()
    raw["generation"]["planning"]["enabled_tie_groups"] = value
    with pytest.raises(StrategyLabConfigError, match="enabled_tie_groups"):
        load_run_spec(write_spec(tmp_path / "invalid.json", raw))


def test_absent_and_explicit_empty_ties_preserve_presence_and_no_tie_plan(tmp_path):
    raw = small_raw()
    raw["generation"]["planning"].pop("enabled_tie_groups")
    pin_small_plan(raw)
    before = canonical_json_bytes(raw)
    absent = load_run_spec(write_spec(tmp_path / "absent.json", raw))
    assert canonical_json_bytes(absent.raw) == before
    raw["generation"]["planning"]["enabled_tie_groups"] = []
    empty = load_run_spec(write_spec(tmp_path / "empty.json", raw))
    assert empty.raw["generation"]["planning"]["enabled_tie_groups"] == []
    assert absent.plan.plan_fingerprint == empty.plan.plan_fingerprint
    assert absent.plan.settings.enabled_tie_groups == empty.plan.settings.enabled_tie_groups == ()
    assert absent.generation_sha256 != empty.generation_sha256
    assert absent.pre_registration_sha256 != empty.pre_registration_sha256


@pytest.mark.parametrize("ties", [None, [], ["symmetricLongShort"]])
@pytest.mark.parametrize("emergency", [False, True])
def test_internal_selector_is_resolved_independently_of_ties(tmp_path, ties, emergency):
    raw = small_raw()
    g = raw["generation"]
    if ties is None:
        g["planning"].pop("enabled_tie_groups")
    else:
        g["planning"]["enabled_tie_groups"] = ties
    g["economics"]["base_params"]["useEmergencySL"] = emergency
    variant = "emergency" if emergency else "plain"
    g["planning"]["enabled_variants"] = [variant]
    g["execution"] = dict(strategy.load_profile().variants[variant].modes)
    pin_small_plan(raw)
    spec = load_run_spec(write_spec(tmp_path / "run.json", raw))
    assert spec.plan.candidate_table.variant_names == (variant,)
    assert spec.plan.settings.enabled_variants is None
    assert spec.plan.candidate_table.params_for_index(0)["useEmergencySL"] is emergency


@pytest.mark.parametrize("variants", [["emergency"], ["unknown"], ["plain", "emergency"], [], None, ["plain", "plain"]])
def test_internal_variant_mismatch_precedes_fingerprint_error(tmp_path, variants):
    raw = small_raw()
    raw["generation"]["planning"].update(enabled_variants=variants, expected_plan_fingerprint="0" * 64)
    with pytest.raises(StrategyLabConfigError, match="enabled_variants"):
        load_run_spec(write_spec(tmp_path / "bad.json", raw))


def test_internal_selector_changes_identity_and_execution_is_still_checked(tmp_path):
    plain_raw = small_raw()
    plain = load_run_spec(write_spec(tmp_path / "plain.json", plain_raw))
    emergency_raw = copy.deepcopy(plain_raw)
    emergency_raw["generation"]["economics"]["base_params"]["useEmergencySL"] = True
    emergency_raw["generation"]["planning"]["enabled_variants"] = ["emergency"]
    pin_small_plan(emergency_raw)
    with pytest.raises(StrategyLabConfigError, match="generation.execution"):
        load_run_spec(write_spec(tmp_path / "bad.json", emergency_raw))
    emergency_raw["generation"]["execution"] = dict(strategy.load_profile().variants["emergency"].modes)
    emergency = load_run_spec(write_spec(tmp_path / "emergency.json", emergency_raw))
    assert plain.plan.plan_fingerprint != emergency.plan.plan_fingerprint


@pytest.mark.parametrize("change", ["unknown_planning", "missing_required", "optional_elsewhere", "target_axis_values"])
def test_optional_key_does_not_weaken_other_schema_contracts(tmp_path, change):
    raw = small_raw()
    g = raw["generation"]
    if change == "unknown_planning":
        g["planning"]["unrelated"] = []
    elif change == "missing_required":
        del g["planning"]["enabled_variants"]
    elif change == "optional_elsewhere":
        g["resources"]["enabled_tie_groups"] = []
    else:
        g["planning"]["axis_values"]["closeCountShort"] = [7]
    with pytest.raises(StrategyLabConfigError):
        load_run_spec(write_spec(tmp_path / "bad.json", raw))


@pytest.mark.parametrize("name", ["start", "end"])
def test_generation_accepts_null_runtime_defaults_but_rejects_window_dates(tmp_path, name):
    spec = load_run_spec(write_spec(tmp_path / "run.json", small_raw()))
    _assert_plan_contract(spec)
    assert spec.plan.candidate_table.params_for_index(0)[name] is None
    params = dict(spec.generation["economics"]["base_params"], **{name: "2025-10-01"})
    dated = build_grid_v2_plan(strategy.load_config(), spec.plan.settings, params)
    with pytest.raises(DatasetError, match="per-window"):
        _assert_plan_contract(replace(spec, plan=dated))
