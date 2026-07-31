from __future__ import annotations

import json
import pickle
from copy import deepcopy

import pytest

from core.engine_v2 import (
    V2_RUNTIME_CONTRACT_VERSION,
    V2_RUNTIME_METADATA_SCHEMA_VERSION,
    V2Diagnostic,
    V2ValidationError,
    build_v2_runtime_metadata,
    parse_v2_runtime_metadata,
    resolve_stored_v2_runtime,
)
from core.optuna_engine import OptimizationConfig
from core.storage import _prepare_study_config_payload
from ui.server_services import (
    ResolvedStoredExecutionContext,
    ResolvedStrategyContext,
    _resolve_stored_execution_context,
)


STRATEGY_ID = "s06_r_trend_v02_b2"


def _values(**overrides):
    values = {
        "dateFilter": False,
        "start": None,
        "end": None,
        "warmupBars": 1000,
    }
    values.update(overrides)
    return values


def _metadata(**overrides):
    return build_v2_runtime_metadata(_values(**overrides), strategy_id=STRATEGY_ID)


def test_runtime_metadata_builder_exact_shape_json_copy_and_pickle() -> None:
    diagnostic = V2Diagnostic(
        severity="info",
        code="V2_INFO",
        strategy_id=STRATEGY_ID,
        path="parameters.tickSize",
        variant=None,
        message="Informational.",
    )
    values = _values(
        dateFilter=True,
        start="2025-01-01T00:00:00Z",
        end="2025-01-02T00:00:00Z",
        warmupBars=1200,
    )
    payload = build_v2_runtime_metadata(values, (diagnostic,), strategy_id=STRATEGY_ID)

    assert list(payload) == [
        "schema_version",
        "contract_version",
        "values",
        "diagnostics",
        "validation_warnings",
    ]
    assert list(payload["values"]) == ["dateFilter", "start", "end", "warmupBars"]
    assert payload == {
        "schema_version": V2_RUNTIME_METADATA_SCHEMA_VERSION,
        "contract_version": V2_RUNTIME_CONTRACT_VERSION,
        "values": values,
        "diagnostics": [diagnostic.to_dict()],
        "validation_warnings": [],
    }
    values["warmupBars"] = 5000
    assert payload["values"]["warmupBars"] == 1200
    assert json.loads(json.dumps(payload)) == payload
    assert deepcopy(payload) == payload
    assert pickle.loads(pickle.dumps(payload)) == payload


def test_runtime_metadata_builder_rejects_errors_and_noncanonical_values() -> None:
    fatal = V2Diagnostic(
        severity="error",
        code="V2_ERROR",
        strategy_id=STRATEGY_ID,
        path="runtime",
        variant=None,
        message="Fatal.",
    )
    with pytest.raises(ValueError, match="fatal diagnostics"):
        build_v2_runtime_metadata(_values(), (fatal,), strategy_id=STRATEGY_ID)
    with pytest.raises(ValueError, match="complete and canonical"):
        build_v2_runtime_metadata(
            _values(warmupBars="1000"), strategy_id=STRATEGY_ID
        )


def test_runtime_metadata_parser_current_legacy_default_and_unavailable() -> None:
    current = _metadata(warmupBars=0)
    resolved = resolve_stored_v2_runtime(
        {
            "strategy_id": STRATEGY_ID,
            "warmup_bars": 999,
            "config_json": {
                "v2_runtime": current,
                "fixed_params": {"dateFilter": True},
            },
        }
    )
    assert (resolved.source, resolved.usable, resolved.values) == (
        "current",
        True,
        current["values"],
    )

    legacy = resolve_stored_v2_runtime(
        {
            "strategy_id": STRATEGY_ID,
            "warmup_bars": 0,
            "config_json": {
                "fixed_params": {"dateFilter": False, "start": "", "end": None},
                "warmup_bars": 777,
            },
        }
    )
    assert legacy.source == "legacy"
    assert legacy.values == _values(warmupBars=0)

    defaulted = resolve_stored_v2_runtime(
        {
            "strategy_id": STRATEGY_ID,
            "warmup_bars": None,
            "config_json": {},
        }
    )
    assert defaulted.source == "defaulted"
    assert defaulted.usable is True
    assert defaulted.values == _values()

    malformed = deepcopy(current)
    malformed["schema_version"] = "future"
    unavailable = resolve_stored_v2_runtime(
        {
            "strategy_id": STRATEGY_ID,
            "warmup_bars": None,
            "config_json": {"v2_runtime": malformed},
        }
    )
    assert unavailable.source == "unavailable"
    assert unavailable.usable is False
    assert unavailable.diagnostics[0].code == "V2_STORED_RUNTIME_METADATA_INCOMPATIBLE"

    fallback = resolve_stored_v2_runtime(
        {
            "strategy_id": STRATEGY_ID,
            "warmup_bars": 20,
            "config_json": {"v2_runtime": malformed},
        }
    )
    assert fallback.source == "legacy"
    assert fallback.values["warmupBars"] == 20
    assert fallback.validation_warnings


def test_current_parser_rejects_unknown_version_and_warning_mismatch() -> None:
    raw = _metadata()
    raw["validation_warnings"] = ["invented"]
    resolution = parse_v2_runtime_metadata(raw, strategy_id=STRATEGY_ID)
    assert resolution.source == "unavailable"
    assert resolution.usable is False


def test_writer_payload_omits_v1_none_and_validates_present_metadata() -> None:
    config = OptimizationConfig(
        csv_file="x.csv",
        strategy_id="s03_reversal_v11",
        enabled_params={},
        param_ranges={},
        param_types={},
        fixed_params={},
    )
    payload = _prepare_study_config_payload(config)
    assert "v2_runtime" not in payload

    config.strategy_id = STRATEGY_ID
    config.v2_runtime = _metadata()
    assert _prepare_study_config_payload(config)["v2_runtime"] == config.v2_runtime

    config.v2_runtime = {"schema_version": "future"}
    with pytest.raises(V2ValidationError, match="invalid V2 runtime metadata") as caught:
        _prepare_study_config_payload(config)
    assert caught.value.diagnostics[0].code == "V2_RUNTIME_METADATA_INVALID"
    assert caught.value.diagnostics[0].path == "config_json.v2_runtime"


def test_stored_execution_precedence_strips_candidate_runtime_and_applies_dates_last() -> None:
    study = {
        "strategy_id": STRATEGY_ID,
        "warmup_bars": 999,
        "config_json": {
            "fixed_params": {
                "maLength": 21,
                "dateFilter": True,
                "warmupBars": 333,
            },
            "v2_runtime": _metadata(
                dateFilter=False,
                start="2025-01-01T00:00:00Z",
                end="2025-01-05T00:00:00Z",
                warmupBars=20,
            ),
        },
    }
    context = _resolve_stored_execution_context(
        study,
        explicit_runtime={"warmupBars": 5},
    )
    assert context.warmup_bars == 5
    params = context.params_for(
        {
            "maLength": 50,
            "dateFilter": False,
            "start": "candidate",
            "end": "candidate",
            "warmupBars": 9000,
        },
        operation_start="2025-02-01",
        operation_end="2025-02-02",
    )
    assert params["maLength"] == 50
    assert params["dateFilter"] is True
    assert params["start"] == "2025-02-01T00:00:00Z"
    assert params["end"] == "2025-02-02T23:59:59.999999Z"
    assert "warmupBars" not in params


def test_runtime_metadata_future_versions_hard_block_legacy_fallback() -> None:
    future_schema = _metadata()
    future_schema["schema_version"] = "v2_runtime_metadata_v2"
    future_contract = _metadata()
    future_contract["contract_version"] = "v2_runtime_contract_v2"

    for metadata in (future_schema, future_contract):
        resolution = resolve_stored_v2_runtime(
            {
                "strategy_id": STRATEGY_ID,
                "warmup_bars": 20,
                "config_json": {
                    "v2_runtime": metadata,
                    "fixed_params": {"dateFilter": False},
                },
            }
        )
        assert resolution.source == "unavailable"
        assert resolution.usable is False


@pytest.mark.parametrize(
    "contract_version",
    ["v2_runtime_contract_v2", "v2_runtime_contract_v1_typo", "", None, 2],
)
def test_any_present_unsupported_contract_hard_blocks_fallback(
    contract_version,
) -> None:
    metadata = _metadata()
    metadata["contract_version"] = contract_version
    resolution = resolve_stored_v2_runtime(
        {
            "strategy_id": STRATEGY_ID,
            "warmup_bars": 20,
            "config_json": {
                "v2_runtime": metadata,
                "fixed_params": {"dateFilter": False},
            },
        }
    )

    assert resolution.source == "unavailable"
    assert resolution.usable is False


def test_missing_contract_can_use_independent_legacy_fallback() -> None:
    metadata = _metadata()
    del metadata["contract_version"]

    resolution = resolve_stored_v2_runtime(
        {
            "strategy_id": STRATEGY_ID,
            "warmup_bars": 20,
            "config_json": {
                "v2_runtime": metadata,
                "fixed_params": {"dateFilter": False},
            },
        }
    )

    assert resolution.source == "legacy"
    assert resolution.usable is True
    assert resolution.values == _values(warmupBars=20)


@pytest.mark.parametrize("raw_config", [None, "", "   \t\r\n"])
def test_sql_null_and_blank_config_are_absent_not_corrupt(raw_config) -> None:
    study = {
        "strategy_id": STRATEGY_ID,
        "warmup_bars": None,
        "config_json": raw_config,
    }

    resolution = resolve_stored_v2_runtime(study)
    context = _resolve_stored_execution_context(study)

    assert resolution.source == "defaulted"
    assert resolution.usable is True
    assert resolution.values == _values()
    assert context.runtime_source == "defaulted"
    assert context.runtime_values == _values()


@pytest.mark.parametrize(
    "raw_config",
    ["not-json", "[]", "1", "true", [], 1, True, object()],
)
def test_invalid_raw_config_is_nonraising_for_view_and_strict_for_execution(raw_config) -> None:
    study = {"strategy_id": STRATEGY_ID, "config_json": raw_config}
    resolution = resolve_stored_v2_runtime(study)
    assert resolution.source == "unavailable"
    assert resolution.usable is False
    assert resolution.diagnostics[0].path == "config_json"

    with pytest.raises(V2ValidationError) as caught:
        _resolve_stored_execution_context(study)
    diagnostic = caught.value.diagnostics[0]
    assert diagnostic.code == "V2_STORED_CONFIG_INCOMPATIBLE"
    assert diagnostic.path == "config_json"


def test_raw_json_object_config_is_parsed_once_and_live_projection_is_detached() -> None:
    metadata = _metadata(warmupBars=20)
    study = {
        "strategy_id": STRATEGY_ID,
        "config_json": json.dumps(
            {
                "fixed_params": {
                    "fastLength": 21,
                    "slowLength": 70,
                    "trailMAType_options": ["SMA", "EMA"],
                    "undeclaredControl": "blocked",
                    "dateFilter": True,
                },
                "v2_runtime": metadata,
            }
        ),
    }
    context = _resolve_stored_execution_context(study)
    candidate = {
        "fastLength": 50,
        "stopX": 2.5,
        "candidateControl": "blocked",
        "dateFilter": True,
        "start": "hostile",
        "end": "hostile",
        "warmupBars": 9999,
    }

    assert context.warmup_bars == 20
    assert context.live_params_for(candidate) == {
        "fastLength": 50,
        "slowLength": 70,
        "stopX": 2.5,
        "dateFilter": False,
        "start": None,
        "end": None,
    }
    assert candidate["start"] == "hostile"


def test_v1_live_projection_is_candidate_only_and_registry_filtered() -> None:
    study = {
        "strategy_id": "s03_reversal_v10",
        "warmup_bars": 20,
        "config_json": {
            "fixed_params": {
                "maType3": "HMA",
                "maType3_options": ["SMA", "HMA"],
                "undeclaredControl": "blocked",
            }
        },
    }
    candidate = {
        "maLength3": 75,
        "useTBands": True,
        "candidateControl": "blocked",
        "dateFilter": True,
        "warmupBars": 9999,
    }
    original = deepcopy(candidate)

    context = _resolve_stored_execution_context(study)

    assert context.live_params_for(candidate) == {
        "maLength3": 75,
        "useTBands": True,
        "dateFilter": False,
        "start": None,
        "end": None,
    }
    assert candidate == original


def test_live_projection_excludes_options_even_if_registry_declares_the_name() -> None:
    strategy = ResolvedStrategyContext(
        strategy_id="synthetic_v2",
        engine="v2",
        config={"parameters": {"legitimate": {}, "legitimate_options": {}}},
        profile=None,
        diagnostics=(),
        validation_error=None,
    )
    context = ResolvedStoredExecutionContext(
        strategy=strategy,
        base_params={"legitimate": 1, "legitimate_options": [1, 2]},
        runtime_values=_values(),
        warmup_bars=1000,
        runtime_source="current",
        diagnostics=(),
    )

    assert context.live_params_for(
        {"legitimate": 2, "legitimate_options": [2, 3]}
    ) == {
        "legitimate": 2,
        "dateFilter": False,
        "start": None,
        "end": None,
    }


def test_stored_unknown_strategy_uses_engine_neutral_diagnostic() -> None:
    with pytest.raises(V2ValidationError) as caught:
        _resolve_stored_execution_context(
            {"strategy_id": "removed_strategy", "config_json": {}}
        )
    diagnostic = caught.value.diagnostics[0]
    assert diagnostic.code == "STORED_STRATEGY_UNAVAILABLE"
    assert diagnostic.strategy_id == "removed_strategy"
    assert diagnostic.path == "study.strategy_id"
