from __future__ import annotations

import copy

import pytest

from core.engine_v2.runtime_contract import (
    V2_REBASABLE_DATE_PARAM_NAMES,
    V2_RESERVED_RUNTIME_PARAM_NAMES,
    V2_RUNTIME_CONTRACT_VERSION,
    V2RuntimeValidationError,
    normalize_v2_runtime_values,
    runtime_contract_payload,
    validate_v2_runtime_declarations,
)
from strategies import get_strategy_config

from runtime_test_helpers import canonical_v2_runtime_declarations


def _config() -> dict:
    return {
        "id": "runtime_fixture",
        "engine": "v2",
        "execution": {},
        "parameters": canonical_v2_runtime_declarations(),
    }


def test_runtime_contract_version_names_and_rebasable_dates_are_stable():
    assert V2_RUNTIME_CONTRACT_VERSION == "v2_runtime_contract_v1"
    assert V2_RESERVED_RUNTIME_PARAM_NAMES == (
        "dateFilter", "start", "end", "warmupBars"
    )
    assert V2_REBASABLE_DATE_PARAM_NAMES == {"dateFilter", "start", "end"}
    payload = runtime_contract_payload()
    assert payload["version"] == V2_RUNTIME_CONTRACT_VERSION
    assert [field["name"] for field in payload["fields"]] == list(
        V2_RESERVED_RUNTIME_PARAM_NAMES
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (False, False),
        (True, True),
        (0, False),
        (1, True),
        ("false", False),
        (" TRUE ", True),
        ("no", False),
        ("on", True),
    ],
)
def test_runtime_boolean_parsing_is_strict_and_deterministic(raw, expected):
    assert normalize_v2_runtime_values({"dateFilter": raw})["dateFilter"] is expected


@pytest.mark.parametrize("raw", ["sometimes", 2, -1, object()])
def test_runtime_boolean_rejects_unknown_values(raw):
    with pytest.raises(V2RuntimeValidationError, match="recognized boolean"):
        normalize_v2_runtime_values({"dateFilter": raw})


def test_runtime_missing_defaults_and_explicit_materialized_date_filter():
    assert normalize_v2_runtime_values({}) == {
        "dateFilter": False,
        "start": None,
        "end": None,
        "warmupBars": 1000,
    }
    assert normalize_v2_runtime_values({}, missing_date_filter=True)["dateFilter"] is True
    assert normalize_v2_runtime_values({"dateFilter": False}, missing_date_filter=True)[
        "dateFilter"
    ] is False


def test_runtime_bounds_are_canonical_utc_and_ordered_when_active():
    values = normalize_v2_runtime_values(
        {
            "dateFilter": True,
            "start": "2025-01-01 08:00:00+08:00",
            "end": "2025-01-02T00:00:00Z",
        }
    )
    assert values["start"] == "2025-01-01T00:00:00Z"
    assert values["end"] == "2025-01-02T00:00:00Z"
    with pytest.raises(V2RuntimeValidationError, match="end must be later"):
        normalize_v2_runtime_values(
            {"dateFilter": True, "start": "2025-01-02", "end": "2025-01-01"}
        )


@pytest.mark.parametrize("value", [100, 1000, 5000])
def test_user_boundary_warmup_accepts_inclusive_contract(value):
    assert normalize_v2_runtime_values({"warmupBars": value})["warmupBars"] == value


@pytest.mark.parametrize("value", [99, 5001, "abc", 100.5, True])
def test_user_boundary_warmup_rejects_out_of_range_or_malformed_values(value):
    with pytest.raises(V2RuntimeValidationError):
        normalize_v2_runtime_values({"warmupBars": value})


@pytest.mark.parametrize("value", [5, 20])
def test_internal_warmup_accepts_non_negative_small_values(value):
    assert normalize_v2_runtime_values(
        {"warmupBars": value}, user_boundary=False
    )["warmupBars"] == value


def test_reserved_declarations_are_optional_but_canonical_declarations_pass():
    validate_v2_runtime_declarations({"id": "minimal", "engine": "v2", "parameters": {}})
    validate_v2_runtime_declarations(_config())


@pytest.mark.parametrize(
    ("name", "mutation"),
    [
        ("dateFilter", lambda spec: spec.update(role="execution")),
        ("dateFilter", lambda spec: spec.update(type="string")),
        ("dateFilter", lambda spec: spec.update(default=False)),
        ("start", lambda spec: spec.update(options=[None])),
        ("end", lambda spec: spec["optimize"].update(enabled=True)),
        ("warmupBars", lambda spec: spec.update(min=0)),
        ("warmupBars", lambda spec: spec["optimize"].update(min=100, max=5000)),
        ("warmupBars", lambda spec: spec.update(gridValues=[1000])),
    ],
)
def test_incompatible_reserved_declarations_fail(name, mutation):
    config = _config()
    mutation(config["parameters"][name])
    with pytest.raises(V2RuntimeValidationError) as exc_info:
        validate_v2_runtime_declarations(config)
    assert exc_info.value.diagnostics[0].code == "V2_INCOMPATIBLE_RUNTIME_DECLARATION"
    assert exc_info.value.diagnostics[0].path == f"parameters.{name}"


def test_reserved_fields_cannot_be_selectors_dependencies_or_option_parameters():
    selector = _config()
    selector["execution"] = {
        "variantSelector": {"param": "dateFilter", "mapping": {"true": "x"}}
    }
    with pytest.raises(V2RuntimeValidationError):
        validate_v2_runtime_declarations(selector)

    dependency = _config()
    dependency["parameters"]["child"] = {
        "type": "bool", "default": True, "role": "signal", "depends_on": "dateFilter"
    }
    with pytest.raises(V2RuntimeValidationError):
        validate_v2_runtime_declarations(dependency)

    options = _config()
    options["parameters"]["start_options"] = {"default": [None]}
    with pytest.raises(V2RuntimeValidationError):
        validate_v2_runtime_declarations(options)


def test_v1_declarations_bypass_v2_runtime_validation():
    config = _config()
    config["engine"] = "v1"
    config["parameters"]["warmupBars"]["role"] = "execution"
    validate_v2_runtime_declarations(config)


def test_all_production_v2_configs_validate_runtime_declarations():
    for strategy_id in (
        "s06_r_trend_v02_b2",
        "s06_r_trend_v02_regime_trendlines_b2",
        "s03_reversal_v11_regime_er_b2",
    ):
        validate_v2_runtime_declarations(copy.deepcopy(get_strategy_config(strategy_id)))

