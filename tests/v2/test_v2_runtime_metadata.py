from __future__ import annotations

import json
import pickle
from copy import deepcopy

import pytest

from core.engine_v2 import (
    V2_RUNTIME_CONTRACT_VERSION,
    V2_RUNTIME_METADATA_SCHEMA_VERSION,
    V2Diagnostic,
    build_v2_runtime_metadata,
    parse_v2_runtime_metadata,
    resolve_stored_v2_runtime,
)
from core.optuna_engine import OptimizationConfig
from core.storage import _prepare_study_config_payload
from ui.server_services import _resolve_stored_execution_context


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
    with pytest.raises(ValueError, match="Invalid V2 runtime metadata"):
        _prepare_study_config_payload(config)


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

