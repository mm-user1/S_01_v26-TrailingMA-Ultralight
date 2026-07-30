import json
from dataclasses import replace
from pathlib import Path

import pytest

from core.engine_v2.diagnostics import V2Diagnostic, V2ValidationError
from core.engine_v2.profile import (
    ProfileValidationError,
    active_mode_values,
    active_parameter_names,
    canonical_selector_key,
    inactive_parameter_names,
    is_v2_config,
    mode_binding_for,
    parse_execution_profile,
    resolve_variant,
    validate_parameter_roles,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_structured_diagnostic_shape_and_error_payload_are_stable():
    diagnostic = V2Diagnostic(
        severity="error",
        code="V2_TEST_DIAGNOSTIC",
        strategy_id="fixture",
        path="execution.stop",
        variant="bracket",
        message="fixture diagnostic",
    )

    assert list(diagnostic.to_dict()) == [
        "severity",
        "code",
        "strategy_id",
        "path",
        "variant",
        "message",
    ]
    assert V2ValidationError(diagnostic).to_dict() == {
        "diagnostics": [diagnostic.to_dict()]
    }


def _param(role, default, *, optimize=True, depends_on=None):
    payload = {
        "type": "float",
        "default": default,
        "role": role,
        "optimize": {"enabled": optimize},
    }
    if depends_on is not None:
        payload["depends_on"] = depends_on
    return payload


def _variant_config():
    return {
        "id": "generic_variant_fixture",
        "engine": "v2",
        "execution": {
            "entryOrder": "market_next_open",
            "stop": "atr_swing",
            "sizing": "risk_per_trade",
            "maxDays": True,
            "variantSelector": {
                "param": "selector",
                "mapping": {False: "mode_a", True: "mode_b"},
            },
            "variants": {
                "mode_a": {"target": "rr", "trail": "none", "trailActivation": "none"},
                "mode_b": {"target": "none", "trail": "ma", "trailActivation": "rr"},
            },
        },
        "parameters": {
            "signalLen": _param("signal", 14),
            "selector": {"type": "bool", "default": False, "role": "execution", "optimize": {"enabled": True}},
            "stopX": _param("execution", 2.0),
            "stopLP": _param("execution", 2),
            "stopMaxPct": _param("execution", 6.0),
            "stopRR": _param("execution", 2.0),
            "trailRR": _param("execution", 1.0),
            "trailMAType": {"type": "select", "default": "SMA", "role": "execution", "optimize": {"enabled": True}},
            "trailMALength": _param("execution", 150),
            "trailMAOffsetEx": _param("execution", 0.0),
            "riskPerTrade": _param("execution", 2.0),
            "contractSize": _param("execution", 0.01),
            "stopMaxDays": _param("execution", 6),
            "unboundExec": _param("execution", 1.0, optimize=False),
            "runtimeOnly": _param("runtime", "2025-01-01", optimize=False),
        },
    }


def test_no_variants_creates_implicit_default_variant():
    config = {
        "id": "implicit_variant_fixture",
        "engine": "v2",
        "execution": {
            "entryOrder": "market_next_open",
            "stop": "atr_swing",
            "target": "rr",
            "trail": "none",
            "trailActivation": "none",
            "sizing": "risk_per_trade",
        },
        "parameters": {
            "signalLen": _param("signal", 5),
            "stopX": _param("execution", 2.0),
            "stopLP": _param("execution", 2),
            "stopMaxPct": _param("execution", 6.0),
            "stopRR": _param("execution", 2.0),
            "riskPerTrade": _param("execution", 2.0),
            "contractSize": _param("execution", 0.01),
        },
    }

    profile = parse_execution_profile(config)

    assert list(profile.variants) == ["default"]
    assert active_mode_values(profile, {})["target"] == "rr"
    assert active_parameter_names(profile, {}) == {
        "signalLen", "stopX", "stopLP", "stopMaxPct", "stopRR",
        "riskPerTrade", "contractSize",
    }


def test_variant_selector_resolves_bool_mapping_keys_and_arbitrary_names():
    profile = parse_execution_profile(_variant_config())

    assert profile.variant_selector.mapping == {"false": "mode_a", "true": "mode_b"}
    assert resolve_variant(profile, {"selector": False}).name == "mode_a"
    assert resolve_variant(profile, {"selector": True}).name == "mode_b"


def test_canonical_selector_key_collapses_integral_numbers():
    assert canonical_selector_key(True) == "true"
    assert canonical_selector_key(False) == "false"
    assert canonical_selector_key(1.0) == "1"
    assert canonical_selector_key(1.25) == "1.25"


def test_numeric_mapping_keys_are_canonicalized():
    config = _variant_config()
    config["execution"]["variantSelector"] = {
        "param": "selector",
        "mapping": {1.0: "mode_a", 2: "mode_b"},
    }
    config["parameters"]["selector"] = {
        "type": "select", "default": 1.0, "options": [1.0, 2],
        "role": "execution", "optimize": {"enabled": True},
    }
    profile = parse_execution_profile(config)

    assert profile.variant_selector.mapping == {"1": "mode_a", "2": "mode_b"}
    assert resolve_variant(profile, {}).name == "mode_a"


def test_json_round_trip_numeric_mapping_keys_are_canonicalized():
    config = _variant_config()
    config["execution"]["variantSelector"] = {
        "param": "selector",
        "mapping": {"1.0": "mode_a", "0.10": "mode_b"},
    }
    config["parameters"]["selector"] = {
        "type": "select", "default": 1.0, "options": [1.0, 0.1],
        "role": "execution", "optimize": {"enabled": True},
    }
    loaded = json.loads(json.dumps(config))

    profile = parse_execution_profile(loaded)

    assert profile.variant_selector.mapping == {"1": "mode_a", "0.1": "mode_b"}
    assert resolve_variant(profile, {}).name == "mode_a"
    assert resolve_variant(profile, {"selector": 0.1}).name == "mode_b"


def test_string_selector_mapping_still_resolves():
    config = _variant_config()
    config["execution"]["variantSelector"] = {
        "param": "selector",
        "mapping": {"reversal": "mode_a", "trend": "mode_b"},
    }
    config["parameters"]["selector"] = {
        "type": "select",
        "default": "reversal",
        "role": "execution",
        "optimize": {"enabled": True},
    }

    profile = parse_execution_profile(json.loads(json.dumps(config)))

    assert resolve_variant(profile, {}).name == "mode_a"
    assert resolve_variant(profile, {"selector": "trend"}).name == "mode_b"


def test_active_and_inactive_params_come_from_mode_bindings():
    profile = parse_execution_profile(_variant_config())

    mode_a_active = active_parameter_names(profile, {"selector": False})
    mode_a_inactive = inactive_parameter_names(profile, {"selector": False})
    mode_b_active = active_parameter_names(profile, {"selector": True})
    mode_b_inactive = inactive_parameter_names(profile, {"selector": True})

    assert "stopRR" in mode_a_active
    assert {"trailRR", "trailMAType", "trailMALength", "trailMAOffsetEx"} <= mode_a_inactive
    assert {"trailRR", "trailMAType", "trailMALength", "trailMAOffsetEx"} <= mode_b_active
    assert "stopRR" in mode_b_inactive
    assert "signalLen" in mode_a_active
    assert "runtimeOnly" not in mode_a_active
    assert "runtimeOnly" not in mode_a_inactive


def test_binding_table_exposes_expected_phase_1_modes():
    assert mode_binding_for("target", "rr").consumes_params == ("stopRR",)
    assert mode_binding_for("target", "none").consumes_params == ()
    assert mode_binding_for("trail", "ma").consumes_params == (
        "trailRR",
        "trailMAType",
        "trailMALength",
        "trailMAOffsetEx",
    )
    assert mode_binding_for("trail", "none").consumes_params == ()


def test_fixed_unbound_execution_param_warns_and_is_not_active():
    profile = parse_execution_profile(_variant_config())

    assert "unboundExec" not in active_parameter_names(profile, {"selector": False})
    assert any("unboundExec" in warning for warning in profile.validation_warnings)
    diagnostic = next(item for item in profile.diagnostics if "unboundExec" in item.message)
    assert diagnostic.code == "V2_UNBOUND_FIXED_EXECUTION_PARAM"
    assert diagnostic.severity == "warning"


def test_certified_but_unselected_fixed_param_is_info_not_warning():
    config = _variant_config()
    config["parameters"]["tickSize"] = _param("execution", 0.01, optimize=False)
    profile = parse_execution_profile(config)

    diagnostic = next(item for item in profile.diagnostics if item.path == "parameters.tickSize")
    assert diagnostic.code == "V2_UNSELECTED_MODE_EXECUTION_PARAM"
    assert diagnostic.severity == "info"
    assert "priceRounding=tick_outward" in diagnostic.message
    assert diagnostic.message not in profile.validation_warnings
    assert "tickSize" not in active_parameter_names(profile, {"selector": False})


def test_certified_but_unselected_optimized_param_fails_actionably():
    config = _variant_config()
    config["parameters"]["tickSize"] = _param("execution", 0.01, optimize=True)

    with pytest.raises(ProfileValidationError) as exc_info:
        parse_execution_profile(config)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "V2_UNSELECTED_MODE_OPTIMIZED_EXECUTION_PARAM"
    assert diagnostic.path == "parameters.tickSize"
    assert "priceRounding=tick_outward" in diagnostic.message


def test_tick_outward_selects_and_consumes_tick_size():
    config = _variant_config()
    config["execution"]["priceRounding"] = "tick_outward"
    config["parameters"]["tickSize"] = _param("execution", 0.01, optimize=True)
    profile = parse_execution_profile(config)

    assert "tickSize" in active_parameter_names(profile, {"selector": False})
    assert not any(item.path == "parameters.tickSize" for item in profile.diagnostics)


def test_later_only_consumer_is_unbound_and_names_uncertified_mode():
    config = _variant_config()
    config["parameters"]["trailAtrMult"] = _param(
        "execution", 2.0, optimize=False
    )
    profile = parse_execution_profile(config)

    diagnostic = next(
        item for item in profile.diagnostics if item.path == "parameters.trailAtrMult"
    )
    assert diagnostic.code == "V2_UNBOUND_FIXED_EXECUTION_PARAM"
    assert diagnostic.severity == "warning"
    assert "uncertified" in diagnostic.message
    assert "trail=atr" in diagnostic.message


def test_selector_missing_from_params_uses_config_default():
    profile = parse_execution_profile(_variant_config())

    assert resolve_variant(profile, {}).name == "mode_a"


def test_selector_missing_without_default_raises():
    config = _variant_config()
    del config["parameters"]["selector"]["default"]
    profile = parse_execution_profile(config)

    with pytest.raises(ProfileValidationError) as exc_info:
        resolve_variant(profile, {})
    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.strategy_id == "generic_variant_fixture"
    assert diagnostic.code == "V2_INVALID_SELECTOR"
    assert diagnostic.path == "parameters.selector"


def test_resolve_variant_defensive_errors_are_structured():
    profile = parse_execution_profile(_variant_config())

    with pytest.raises(ProfileValidationError) as missing_selector:
        resolve_variant(replace(profile, variant_selector=None), {})
    diagnostic = missing_selector.value.diagnostics[0]
    assert (diagnostic.strategy_id, diagnostic.code, diagnostic.path) == (
        "generic_variant_fixture",
        "V2_INVALID_SELECTOR",
        "execution.variantSelector",
    )

    with pytest.raises(ProfileValidationError) as unmapped:
        resolve_variant(profile, {"selector": "not-mapped"})
    diagnostic = unmapped.value.diagnostics[0]
    assert (diagnostic.strategy_id, diagnostic.code, diagnostic.path) == (
        "generic_variant_fixture",
        "V2_INVALID_SELECTOR",
        "execution.variantSelector.mapping",
    )


def test_selector_param_typo_fails_at_parse_time():
    config = _variant_config()
    config["execution"]["variantSelector"]["param"] = "missingSelector"

    with pytest.raises(
        ProfileValidationError,
        match="generic_variant_fixture: variantSelector parameter 'missingSelector' is not declared",
    ):
        parse_execution_profile(config)


def test_undeclared_consumed_binding_parameter_fails_at_parse_time():
    config = {
        "id": "no_role_fixture",
        "engine": "v2",
        "execution": {
            "variantSelector": {
                "param": "selector",
                "mapping": {False: "mode_a", True: "mode_b"},
            },
            "variants": {
                "mode_a": {"target": "rr", "trail": "none"},
                "mode_b": {"target": "none", "trail": "ma"},
            },
        },
        "parameters": {
            "selector": {"type": "bool", "default": False, "optimize": {"enabled": False}},
        },
    }

    with pytest.raises(ProfileValidationError, match="must be a non-runtime bool/select execution parameter"):
        parse_execution_profile(config)


def test_bound_roleless_declared_param_is_active_when_consumed():
    config = {
        "id": "roleless_bound_fixture",
        "engine": "v2",
        "execution": {
            "entryOrder": "market_next_open",
            "stop": "atr_swing",
            "target": "rr",
            "trail": "none",
            "trailActivation": "none",
            "sizing": "risk_per_trade",
        },
        "parameters": {
            "stopX": {"type": "float", "default": 2.0, "optimize": {"enabled": False}},
            "stopLP": {"type": "int", "default": 2, "optimize": {"enabled": False}},
            "stopMaxPct": {"type": "float", "default": 6.0, "optimize": {"enabled": False}},
            "stopRR": {"type": "float", "default": 2.0, "optimize": {"enabled": False}},
            "riskPerTrade": {"type": "float", "default": 2.0, "optimize": {"enabled": False}},
            "contractSize": {"type": "float", "default": 0.01, "optimize": {"enabled": False}},
        },
    }

    profile = parse_execution_profile(config)

    assert active_parameter_names(profile, {}) == {
        "stopX", "stopLP", "stopMaxPct", "stopRR", "riskPerTrade", "contractSize"
    }
    assert inactive_parameter_names(profile, {}) == set()


def test_optimized_parameter_without_role_fails_for_v2():
    config = _variant_config()
    del config["parameters"]["signalLen"]["role"]

    with pytest.raises(ProfileValidationError, match="signalLen"):
        parse_execution_profile(config)


def test_real_v1_config_does_not_trigger_v2_validation():
    with (REPO_ROOT / "src" / "strategies" / "s06_r_trend_v02" / "config.json").open(
        encoding="utf-8"
    ) as handle:
        config = json.load(handle)

    assert is_v2_config(config) is False
    validate_parameter_roles(config)


def test_cross_role_depends_on_fails_validation():
    config = _variant_config()
    config["parameters"]["stopX"]["depends_on"] = "signalLen"

    with pytest.raises(ProfileValidationError, match="cross-role depends_on"):
        parse_execution_profile(config)


def test_within_role_depends_on_is_accepted():
    config = _variant_config()
    config["parameters"]["trailRR"]["depends_on"] = "selector"

    parse_execution_profile(config)


def test_core_profile_modules_do_not_contain_strategy_specific_branches():
    forbidden = ("s06", "s06_r_trend_v02", "useTrailMA")
    for path in (REPO_ROOT / "src" / "core" / "engine_v2").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text


def test_all_production_profiles_validate_and_topology_wide_consumers_do_not_warn():
    strategy_ids = (
        "s06_r_trend_v02_b2",
        "s06_r_trend_v02_regime_trendlines_b2",
        "s03_reversal_v11_regime_er_b2",
    )
    for strategy_id in strategy_ids:
        with (REPO_ROOT / "src" / "strategies" / strategy_id / "config.json").open(
            encoding="utf-8"
        ) as handle:
            profile = parse_execution_profile(json.load(handle))
        warning_text = " ".join(profile.validation_warnings)
        for name in ("initialCapital", "commissionPct", "enableLong", "enableShort"):
            assert name not in warning_text
        assert profile.validation_warnings == ()
        if strategy_id.startswith("s06_"):
            tick = next(
                item for item in profile.diagnostics if item.path == "parameters.tickSize"
            )
            assert tick.code == "V2_UNSELECTED_MODE_EXECUTION_PARAM"
            assert tick.severity == "info"


def test_truly_unbound_optimized_execution_parameter_is_fatal():
    config = _variant_config()
    config["parameters"]["unboundExec"]["optimize"]["enabled"] = True

    with pytest.raises(ProfileValidationError) as exc_info:
        parse_execution_profile(config)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "V2_UNBOUND_OPTIMIZED_EXECUTION_PARAM"
    assert diagnostic.path == "parameters.unboundExec"


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda config: config["execution"].update(unknownMode="x"), "unknown"),
        (lambda config: config["execution"].update(margin="simulate"), "margin"),
        (
            lambda config: config["execution"]["variants"]["mode_a"].update(
                trail="ma", trailActivation="rr"
            ),
            "target=rr",
        ),
        (
            lambda config: config["execution"]["variants"]["mode_b"].update(
                trailActivation="none"
            ),
            "trailActivation",
        ),
        (
            lambda config: config["execution"]["variants"]["mode_a"].update(
                trailActivation="rr"
            ),
            "target=rr",
        ),
        (lambda config: config["execution"].pop("stop"), "stop"),
    ],
)
def test_position_mode_contract_fails_at_profile_parse(mutate, match):
    config = _variant_config()
    mutate(config)
    with pytest.raises(ProfileValidationError, match=match):
        parse_execution_profile(config)


def test_signal_reversal_positive_and_incompatible_combination():
    path = REPO_ROOT / "src" / "strategies" / "s03_reversal_v11_regime_er_b2" / "config.json"
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    assert set(parse_execution_profile(config).variants) == {"plain", "emergency"}
    config["execution"]["target"] = "rr"
    with pytest.raises(ProfileValidationError, match="signal_reversal.*target"):
        parse_execution_profile(config)


def test_declared_mode_consumer_is_required():
    config = _variant_config()
    del config["parameters"]["stopLP"]
    with pytest.raises(ProfileValidationError) as exc_info:
        parse_execution_profile(config)
    assert exc_info.value.diagnostics[0].code == "V2_UNDECLARED_CONSUMED_PARAMETER"
    assert "stopLP" in str(exc_info.value)


def test_selector_rejects_unreachable_duplicate_normalized_and_reserved_forms():
    unreachable = _variant_config()
    unreachable["execution"]["variantSelector"]["mapping"] = {False: "mode_a"}
    with pytest.raises(ProfileValidationError, match="cannot reach"):
        parse_execution_profile(unreachable)

    duplicate = _variant_config()
    duplicate["parameters"]["selector"].update(type="select", options=[1, 2], default=1)
    duplicate["execution"]["variantSelector"]["mapping"] = {1: "mode_a", "1.0": "mode_b"}
    with pytest.raises(ProfileValidationError, match="duplicate normalized key"):
        parse_execution_profile(duplicate)

    reserved = _variant_config()
    reserved["parameters"].update({"dateFilter": {
        "type": "bool", "default": True, "role": "runtime", "optimize": {"enabled": False}
    }})
    reserved["execution"]["variantSelector"]["param"] = "dateFilter"
    with pytest.raises(ProfileValidationError, match="incompatible|runtime"):
        parse_execution_profile(reserved)


@pytest.mark.parametrize("kind", ["unknown", "non_boolean", "self", "duplicate", "runtime"])
def test_dependency_contract_rejects_unsafe_shapes(kind):
    config = _variant_config()
    if kind == "unknown":
        config["parameters"]["trailRR"]["depends_on"] = "missing"
    elif kind == "non_boolean":
        config["parameters"]["trailRR"]["depends_on"] = "stopX"
    elif kind == "self":
        config["parameters"]["trailRR"]["depends_on"] = "trailRR"
    elif kind == "duplicate":
        config["parameters"]["trailRR"]["depends_on"] = ["selector", "selector"]
    else:
        config["parameters"]["runtimeOnly"]["depends_on"] = "selector"
    with pytest.raises(ProfileValidationError) as exc_info:
        parse_execution_profile(config)
    assert exc_info.value.diagnostics[0].code in {
        "V2_INVALID_DEPENDENCY", "V2_INCOMPATIBLE_RUNTIME_DECLARATION"
    }


def test_dependency_contract_rejects_cycles_and_roleless_parents():
    cycle = _variant_config()
    cycle["parameters"].update(
        {
            "a": {"type": "bool", "default": True, "role": "execution", "depends_on": "b"},
            "b": {"type": "bool", "default": True, "role": "execution", "depends_on": "a"},
        }
    )
    with pytest.raises(ProfileValidationError, match="cycle"):
        parse_execution_profile(cycle)

    roleless = _variant_config()
    roleless["parameters"].update(
        {
            "parent": {"type": "bool", "default": True},
            "child": {"type": "bool", "default": True, "role": "signal", "depends_on": "parent"},
        }
    )
    with pytest.raises(ProfileValidationError, match="must declare a role"):
        parse_execution_profile(roleless)
