from __future__ import annotations

import copy

import pandas as pd
import pytest

from core.backtest_engine import align_date_bounds, prepare_dataset_with_warmup
from core.engine_v2.runtime_contract import (
    V2_REBASABLE_DATE_PARAM_NAMES,
    V2_RESERVED_RUNTIME_PARAM_NAMES,
    V2_RUNTIME_CONTRACT_VERSION,
    V2RuntimeValidationError,
    normalize_v2_runtime_field_value,
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


def test_scalar_runtime_normalizer_is_exported_through_v2_facade():
    from core.engine_v2 import normalize_v2_runtime_field_value as facade_normalizer

    assert facade_normalizer is normalize_v2_runtime_field_value


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


def test_runtime_date_only_bounds_use_start_and_inclusive_end_of_utc_day():
    values = normalize_v2_runtime_values(
        {
            "dateFilter": True,
            "start": "2025-06-30",
            "end": "2025-06-30",
        }
    )
    assert values["start"] == "2025-06-30T00:00:00Z"
    assert values["end"] == "2025-06-30T23:59:59.999999Z"

    assert normalize_v2_runtime_field_value(
        "end", "2025-06-30T23:59:59.999999Z"
    ) == values["end"]


def test_runtime_date_only_reversed_range_and_non_date_only_equal_instants_fail():
    with pytest.raises(V2RuntimeValidationError, match="end must be later"):
        normalize_v2_runtime_values(
            {
                "dateFilter": True,
                "start": "2025-07-01",
                "end": "2025-06-30",
            }
        )

    with pytest.raises(V2RuntimeValidationError, match="end must be later"):
        normalize_v2_runtime_values(
            {
                "dateFilter": True,
                "start": "2025-06-30T12:00:00Z",
                "end": "2025-06-30T12:00:00+00:00",
            }
        )


def test_date_only_canonical_bounds_preserve_legacy_15_minute_alignment():
    index = pd.date_range(
        "2025-06-01T00:00:00Z",
        "2025-06-30T23:45:00Z",
        freq="15min",
    )
    frame = pd.DataFrame({"value": range(len(index))}, index=index)
    legacy_start, legacy_end = align_date_bounds(
        index, "2025-06-01", "2025-06-30"
    )
    runtime = normalize_v2_runtime_values(
        {
            "dateFilter": True,
            "start": "2025-06-01",
            "end": "2025-06-30",
        }
    )
    canonical_start, canonical_end = align_date_bounds(
        index, runtime["start"], runtime["end"]
    )
    legacy_frame, _ = prepare_dataset_with_warmup(
        frame, legacy_start, legacy_end, 0
    )
    canonical_frame, _ = prepare_dataset_with_warmup(
        frame, canonical_start, canonical_end, 0
    )

    assert legacy_end == index[-1]
    assert canonical_end == pd.Timestamp("2025-06-30T23:59:59.999999Z")
    assert len(legacy_frame) == len(canonical_frame) == 2_880
    assert legacy_frame.index[-1] == canonical_frame.index[-1] == index[-1]

    same_day = normalize_v2_runtime_values(
        {
            "dateFilter": True,
            "start": "2025-06-30",
            "end": "2025-06-30",
        }
    )
    same_start, same_end = align_date_bounds(
        index, same_day["start"], same_day["end"]
    )
    same_frame, _ = prepare_dataset_with_warmup(frame, same_start, same_end, 0)
    assert len(same_frame) == 96
    assert same_frame.index[-1] == index[-1]


def test_grid_and_optuna_preparation_share_canonical_date_only_window(monkeypatch):
    from core import grid_engine, optuna_engine
    from core.optuna_engine import OptimizationConfig, OptunaConfig, OptunaOptimizer
    import strategies

    index = pd.date_range(
        "2025-06-01T00:00:00Z",
        "2025-06-30T23:45:00Z",
        freq="15min",
    )
    frame = pd.DataFrame(
        {
            "Open": 1.0,
            "High": 1.0,
            "Low": 1.0,
            "Close": 1.0,
            "Volume": 1.0,
        },
        index=index,
    )
    runtime = normalize_v2_runtime_values(
        {
            "dateFilter": True,
            "start": "2025-06-01",
            "end": "2025-06-30",
        }
    )
    config = OptimizationConfig(
        csv_file="synthetic.csv",
        strategy_id="s06_r_trend_v02_b2",
        enabled_params={},
        param_ranges={},
        param_types={},
        fixed_params={
            "dateFilter": runtime["dateFilter"],
            "start": runtime["start"],
            "end": runtime["end"],
        },
        warmup_bars=0,
    )

    monkeypatch.setattr(grid_engine, "load_data", lambda _source: frame)
    grid_frame, grid_trade_start, grid_start, grid_end = (
        grid_engine._prepare_grid_dataframe(config)
    )

    class DummyStrategy:
        pass

    monkeypatch.setattr(optuna_engine, "load_data", lambda _source: frame)
    monkeypatch.setattr(strategies, "get_strategy", lambda _strategy_id: DummyStrategy)
    optimizer = OptunaOptimizer(config, OptunaConfig())
    optimizer._prepare_data_and_strategy()

    assert grid_start == optimizer.df.index[0] == index[0]
    assert grid_end == pd.Timestamp("2025-06-30T23:59:59.999999Z")
    assert grid_frame.index[-1] == optimizer.df.index[-1] == index[-1]
    assert grid_trade_start == optimizer.trade_start_idx == 0
    assert len(grid_frame) == len(optimizer.df) == 2_880


@pytest.mark.parametrize("value", [100, 1000, 5000])
def test_user_boundary_warmup_accepts_inclusive_contract(value):
    assert normalize_v2_runtime_values({"warmupBars": value})["warmupBars"] == value


@pytest.mark.parametrize("value", [99, 5001, "abc", 100.5, True])
def test_user_boundary_warmup_rejects_out_of_range_or_malformed_values(value):
    with pytest.raises(V2RuntimeValidationError):
        normalize_v2_runtime_values({"warmupBars": value})


def test_user_boundary_warmup_error_uses_canonical_contract_bounds():
    warmup = next(
        field for field in runtime_contract_payload()["fields"]
        if field["name"] == "warmupBars"
    )
    with pytest.raises(V2RuntimeValidationError) as exc_info:
        normalize_v2_runtime_values({"warmupBars": warmup["minimum"] - 1})

    assert (
        f"between {warmup['minimum']} and {warmup['maximum']}"
        in exc_info.value.diagnostics[0].message
    )


@pytest.mark.parametrize("value", [5, 20])
def test_internal_warmup_accepts_non_negative_small_values(value):
    assert normalize_v2_runtime_values(
        {"warmupBars": value}, user_boundary=False
    )["warmupBars"] == value


def test_reserved_declarations_are_optional_but_canonical_declarations_pass():
    validate_v2_runtime_declarations({"id": "minimal", "engine": "v2", "parameters": {}})
    validate_v2_runtime_declarations(_config())


def test_reserved_declaration_must_be_a_mapping():
    config = _config()
    config["parameters"]["start"] = []

    with pytest.raises(V2RuntimeValidationError) as exc_info:
        validate_v2_runtime_declarations(config)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "V2_INCOMPATIBLE_RUNTIME_DECLARATION"
    assert diagnostic.path == "parameters.start"
    assert "must be a mapping" in diagnostic.message


@pytest.mark.parametrize(
    ("name", "mutation", "reason"),
    [
        ("dateFilter", lambda spec: spec.update(role="execution"), "role must be 'runtime'"),
        ("dateFilter", lambda spec: spec.update(type="string"), "type must be 'bool'"),
        ("dateFilter", lambda spec: spec.update(default=False), "default must be true"),
        ("start", lambda spec: spec.update(options=[None]), "options are forbidden"),
        ("end", lambda spec: spec["optimize"].update(enabled=True), "optimize.enabled must be false"),
        (
            "end",
            lambda spec: spec["optimize"].update(default_enabled=True),
            "optimize.default_enabled must be false",
        ),
        ("warmupBars", lambda spec: spec.update(min=0), "min must be 100"),
        ("warmupBars", lambda spec: spec.update(max=6000), "max must be 5000"),
        (
            "warmupBars",
            lambda spec: spec["optimize"].update(min=100, max=5000),
            "optimization axis metadata is forbidden",
        ),
        (
            "warmupBars",
            lambda spec: spec.update(gridValues=[1000]),
            "optimization axis metadata is forbidden",
        ),
    ],
)
def test_incompatible_reserved_declarations_report_actionable_reason(
    name, mutation, reason
):
    config = _config()
    mutation(config["parameters"][name])
    with pytest.raises(V2RuntimeValidationError) as exc_info:
        validate_v2_runtime_declarations(config)
    assert exc_info.value.diagnostics[0].code == "V2_INCOMPATIBLE_RUNTIME_DECLARATION"
    assert exc_info.value.diagnostics[0].path == f"parameters.{name}"
    assert reason in exc_info.value.diagnostics[0].message


@pytest.mark.parametrize("optimize", ["yes", [], 1, None])
def test_present_non_mapping_runtime_optimize_is_rejected(optimize):
    config = _config()
    config["parameters"]["warmupBars"]["optimize"] = optimize

    with pytest.raises(V2RuntimeValidationError) as exc_info:
        validate_v2_runtime_declarations(config)

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "V2_INCOMPATIBLE_RUNTIME_DECLARATION"
    assert diagnostic.path == "parameters.warmupBars"
    assert "optimize must be a mapping when present" in diagnostic.message


@pytest.mark.parametrize("optimize", [pytest.param(None, id="absent"), {}, {"enabled": False, "default_enabled": False}])
def test_absent_or_compatible_mapping_runtime_optimize_is_disabled(optimize):
    config = _config()
    if optimize is None:
        del config["parameters"]["warmupBars"]["optimize"]
    else:
        config["parameters"]["warmupBars"]["optimize"] = optimize

    validate_v2_runtime_declarations(config)


def test_runtime_declaration_reasons_have_stable_property_order():
    config = _config()
    spec = config["parameters"]["warmupBars"]
    spec.update(role="execution", type="float", default=10, min=0, max=10)
    spec["optimize"] = "yes"
    spec["depends_on"] = "dateFilter"

    with pytest.raises(V2RuntimeValidationError) as exc_info:
        validate_v2_runtime_declarations(config)

    message = exc_info.value.diagnostics[0].message
    fragments = (
        "role must be 'runtime'",
        "type must be 'int'",
        "default must be 1000",
        "min must be 100",
        "max must be 5000",
        "optimize must be a mapping",
        "depends_on is forbidden",
    )
    positions = [message.index(fragment) for fragment in fragments]
    assert positions == sorted(positions)


def test_reserved_fields_cannot_be_selectors_dependencies_or_option_parameters():
    selector = _config()
    selector["execution"] = {
        "variantSelector": {"param": "dateFilter", "mapping": {"true": "x"}}
    }
    with pytest.raises(V2RuntimeValidationError, match="variant selectors"):
        validate_v2_runtime_declarations(selector)

    dependency = _config()
    dependency["parameters"]["child"] = {
        "type": "bool", "default": True, "role": "signal", "depends_on": "dateFilter"
    }
    with pytest.raises(V2RuntimeValidationError, match="participate in depends_on"):
        validate_v2_runtime_declarations(dependency)

    options = _config()
    options["parameters"]["start_options"] = {"default": [None]}
    with pytest.raises(V2RuntimeValidationError, match="option parameters"):
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
