"""Server runtime contracts."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import pandas as pd

from strategies import get_strategy_config

from ._helpers import (
    _grid_sidebar_config,
    _s03_regime_er_grid_preview_payload,
    _v2_runtime_diagnostic,
)


def test_v2_strategy_context_alias_agreement_conflict_missing_and_unknown():
    from core.engine_v2.diagnostics import V2ValidationError
    from ui.server_services import _resolve_strategy_context

    context = _resolve_strategy_context(
        [
            ("config.strategy_id", True, "s06_r_trend_v02_b2"),
            ("payload.strategyId", True, "s06_r_trend_v02_b2"),
            ("form.strategyId", True, ""),
        ]
    )
    assert context.strategy_id == "s06_r_trend_v02_b2"
    assert context.engine == "v2"
    assert context.profile is not None

    cases = [
        (
            [("form.strategy", False, None), ("json.strategy", True, "")],
            "V2_MISSING_STRATEGY_ID",
            "<unknown strategy>",
        ),
        (
            [
                ("config.strategy_id", True, "s06_r_trend_v02_b2"),
                ("payload.strategyId", True, "s03_reversal_v10"),
            ],
            "V2_CONFLICTING_STRATEGY_ID",
            "s06_r_trend_v02_b2",
        ),
        (
            [("form.strategy", True, "not_registered")],
            "V2_UNKNOWN_STRATEGY_ID",
            "not_registered",
        ),
    ]
    for aliases, code, strategy_id in cases:
        with pytest.raises(V2ValidationError) as raised:
            _resolve_strategy_context(aliases)
        diagnostic = raised.value.diagnostics[0]
        assert (diagnostic.code, diagnostic.strategy_id, diagnostic.path) == (
            code,
            strategy_id,
            "strategy_id",
        )


@pytest.mark.parametrize(
    ("endpoint", "data", "json_payload"),
    [
        ("/api/grid/preview", None, {"config": _s03_regime_er_grid_preview_payload(strategy_id="")}),
        ("/api/backtest", {"payload": "{}"}, None),
        ("/api/backtest/trades", {"payload": "{}"}, None),
        ("/api/optimize", {"config": json.dumps(_s03_regime_er_grid_preview_payload())}, None),
        ("/api/walkforward", {"config": json.dumps(_s03_regime_er_grid_preview_payload())}, None),
    ],
)
def test_run_surfaces_require_strategy_before_work(client, endpoint, data, json_payload):
    response = (
        client.post(endpoint, json=json_payload)
        if json_payload is not None
        else client.post(endpoint, data=data)
    )
    assert response.status_code == 400
    diagnostic = _v2_runtime_diagnostic(response)
    assert diagnostic["code"] == "V2_MISSING_STRATEGY_ID"
    assert diagnostic["path"] == "strategy_id"


def test_run_routes_accept_only_normative_strategy_aliases(client):
    json_alias = client.post(
        "/api/backtest",
        json={"strategy": "s06_r_trend_v02_b2"},
    )
    assert json_alias.status_code == 400
    assert "CSV path is required" in json_alias.get_data(as_text=True)
    assert "diagnostics" not in (json_alias.get_json(silent=True) or {})

    rejected_form_alias = client.post(
        "/api/backtest",
        data={"strategyId": "s06_r_trend_v02_b2", "payload": "{}"},
    )
    assert _v2_runtime_diagnostic(rejected_form_alias)["code"] == "V2_MISSING_STRATEGY_ID"

    preview_payload = _s03_regime_er_grid_preview_payload()
    preview_payload.pop("strategy_id")
    rejected_preview_alias = client.post(
        "/api/grid/preview",
        json={"config": preview_payload, "strategy_id": "s03_reversal_v11_regime_er_b2"},
    )
    assert _v2_runtime_diagnostic(rejected_preview_alias)["code"] == "V2_MISSING_STRATEGY_ID"


def test_unknown_strategy_config_is_structured_json_404(client):
    response = client.get("/api/strategy/not_registered/config")
    assert response.status_code == 404
    assert response.is_json
    diagnostic = _v2_runtime_diagnostic(response)
    assert diagnostic["code"] == "V2_UNKNOWN_STRATEGY_ID"
    assert diagnostic["strategy_id"] == "not_registered"


def test_unknown_run_strategy_is_structured_400_before_work(client):
    response = client.post(
        "/api/backtest",
        data={"strategy": "not_registered", "payload": "{}"},
    )
    assert response.status_code == 400
    diagnostic = _v2_runtime_diagnostic(response)
    assert diagnostic["code"] == "V2_UNKNOWN_STRATEGY_ID"
    assert diagnostic["strategy_id"] == "not_registered"
    assert diagnostic["path"] == "strategy_id"


@pytest.mark.parametrize("endpoint", ["/api/optimize", "/api/walkforward"])
def test_v2_metadata_build_failure_is_structured_before_csv_work(
    client, monkeypatch, endpoint
):
    from core.engine_v2 import V2Diagnostic, V2ValidationError
    from ui import server_routes_run

    monkeypatch.setattr(
        server_routes_run,
        "build_v2_runtime_metadata",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            V2ValidationError(
                V2Diagnostic(
                    severity="error",
                    code="V2_RUNTIME_METADATA_INVALID",
                    strategy_id="s03_reversal_v11_regime_er_b2",
                    path="v2_runtime.values",
                    variant=None,
                    message="metadata build failed",
                )
            )
        ),
    )
    response = client.post(
        endpoint,
        data={
            "strategy": "s03_reversal_v11_regime_er_b2",
            "config": json.dumps(_s03_regime_er_grid_preview_payload()),
        },
    )
    assert response.status_code == 400
    diagnostic = _v2_runtime_diagnostic(response)
    assert diagnostic["code"] == "V2_RUNTIME_METADATA_INVALID"
    assert diagnostic["path"] == "v2_runtime.values"


def test_v2_config_api_adds_runtime_readiness_without_mutating_registry(client):
    before = deepcopy(get_strategy_config("s06_r_trend_v02_b2"))

    first = client.get("/api/strategy/s06_r_trend_v02_b2/config")
    second = client.get("/api/strategy/s06_r_trend_v02_b2/config")

    assert first.status_code == second.status_code == 200
    payload = first.get_json()
    assert payload["runtime_contract"] == {
        "version": "v2_runtime_contract_v1",
        "fields": [
            {"name": "dateFilter", "type": "bool", "ui_default": True, "legacy_default": False, "minimum": None, "maximum": None},
            {"name": "start", "type": "datetime", "ui_default": None, "legacy_default": None, "minimum": None, "maximum": None},
            {"name": "end", "type": "datetime", "ui_default": None, "legacy_default": None, "minimum": None, "maximum": None},
            {"name": "warmupBars", "type": "int", "ui_default": 1000, "legacy_default": 1000, "minimum": 100, "maximum": 5000},
        ],
    }
    assert payload["runtime_values"] == {
        "dateFilter": True,
        "start": None,
        "end": None,
        "warmupBars": 1000,
    }
    assert [item["severity"] for item in payload["diagnostics"]] == ["info"]
    assert payload["diagnostics"][0]["code"] == "V2_UNSELECTED_MODE_EXECUTION_PARAM"
    assert payload["validation_warnings"] == []
    assert second.get_json() == payload
    assert get_strategy_config("s06_r_trend_v02_b2") == before
    assert "runtime_contract" not in get_strategy_config("s06_r_trend_v02_b2")


def test_v1_config_api_shape_has_no_v2_readiness_keys(client):
    response = client.get("/api/strategy/s03_reversal_v10/config")
    assert response.status_code == 200
    payload = response.get_json()
    for key in ("runtime_contract", "runtime_values", "diagnostics", "validation_warnings"):
        assert key not in payload


def test_invalid_v2_profile_is_422_on_config_and_400_on_preview(monkeypatch, client):
    import strategies

    original = strategies.get_strategy_config
    invalid = deepcopy(original("s06_r_trend_v02_b2"))
    invalid["execution"]["entryOrder"] = "unsupported"
    monkeypatch.setattr(
        strategies,
        "get_strategy_config",
        lambda strategy_id: invalid if strategy_id == invalid["id"] else original(strategy_id),
    )

    config_response = client.get(f"/api/strategy/{invalid['id']}/config")
    assert config_response.status_code == 422
    config_diagnostics = config_response.get_json()["diagnostics"]

    preview_payload = _s03_regime_er_grid_preview_payload(
        strategy_id=invalid["id"],
        enabled_params={},
        fixed_params={"dateFilter": False},
    )
    preview_response = client.post("/api/grid/preview", json=preview_payload)
    assert preview_response.status_code == 400
    assert preview_response.get_json()["diagnostics"] == config_diagnostics

    run_responses = [
        client.post(
            endpoint,
            data={"strategy": invalid["id"], "payload": "{}"},
        )
        for endpoint in ("/api/backtest", "/api/backtest/trades")
    ]
    for mode in ("optuna", "grid"):
        run_payload = deepcopy(preview_payload)
        run_payload["optimization_mode"] = mode
        run_responses.append(
            client.post(
                "/api/optimize",
                data={"strategy": invalid["id"], "config": json.dumps(run_payload)},
            )
        )
    for run_response in run_responses:
        assert run_response.status_code == 400
        assert run_response.get_json()["diagnostics"] == config_diagnostics

    monkeypatch.setattr(strategies, "get_strategy_config", original)
    assert client.get(f"/api/strategy/{invalid['id']}/config").status_code == 200


def test_v2_runtime_adapter_preserves_presence_and_calls_core_once(monkeypatch):
    from ui import server_services

    context = server_services._resolve_strategy_context(
        [("strategy_id", True, "s06_r_trend_v02_b2")]
    )
    original = server_services.normalize_v2_runtime_values
    calls = []

    def counted(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(server_services, "normalize_v2_runtime_values", counted)
    runtime = server_services._normalize_v2_request_runtime(
        context,
        [
            ("dateFilter", "fixed_params.dateFilter", "0"),
            ("start", "fixed_params.start", ""),
            ("end", "fixed_params.end", "2025-05-01T08:00:00+08:00"),
            ("warmupBars", "warmupBars", "100"),
        ],
        missing_date_filter=False,
    )
    assert list(runtime.values) == ["dateFilter", "start", "end", "warmupBars"]
    assert runtime.values == {
        "dateFilter": False,
        "start": None,
        "end": "2025-05-01T00:00:00Z",
        "warmupBars": 100,
    }
    assert runtime.execution_projection == {
        "dateFilter": False,
        "start": None,
        "end": "2025-05-01T00:00:00Z",
    }
    assert len(calls) == 1

    omitted = server_services._normalize_v2_request_runtime(
        context,
        [("warmupBars", "warmupBars", 5000)],
        missing_date_filter=False,
    )
    assert omitted.execution_projection == {}
    assert omitted.values["dateFilter"] is False


@pytest.mark.parametrize(
    ("name", "first", "second", "expected"),
    [
        ("warmupBars", 1000, "1000", 1000),
        ("dateFilter", False, "0", False),
        (
            "start",
            "2025-05-01T00:00",
            "2025-05-01T00:00:00Z",
            "2025-05-01T00:00:00Z",
        ),
        (
            "end",
            "2025-06-30",
            "2025-06-30T23:59:59.999999Z",
            "2025-06-30T23:59:59.999999Z",
        ),
    ],
)
def test_v2_runtime_adapter_duplicate_sources_compare_canonical_meaning(
    monkeypatch,
    name,
    first,
    second,
    expected,
):
    from ui import server_services

    context = server_services._resolve_strategy_context(
        [("strategy_id", True, "s06_r_trend_v02_b2")]
    )
    original = server_services.normalize_v2_runtime_values
    calls = []

    def counted(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(server_services, "normalize_v2_runtime_values", counted)
    runtime = server_services._normalize_v2_request_runtime(
        context,
        [
            (name, f"first.{name}", first),
            (name, f"second.{name}", second),
        ],
        missing_date_filter=False,
    )

    assert runtime.values[name] == expected
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("name", "first", "second"),
    [
        ("warmupBars", 1000, 1001),
        ("dateFilter", False, True),
        ("start", "2025-05-01T00:00:00Z", "2025-05-01T00:15:00Z"),
    ],
)
def test_v2_runtime_adapter_duplicate_conflicts_name_both_paths(
    name, first, second
):
    from core.engine_v2.diagnostics import V2ValidationError
    from ui import server_services

    context = server_services._resolve_strategy_context(
        [("strategy_id", True, "s06_r_trend_v02_b2")]
    )
    with pytest.raises(V2ValidationError) as raised:
        server_services._normalize_v2_request_runtime(
            context,
            [
                (name, f"first.{name}", first),
                (name, f"second.{name}", second),
            ],
            missing_date_filter=False,
        )

    diagnostic = raised.value.diagnostics[0]
    assert diagnostic.path == f"second.{name}"
    assert f"first.{name}" in diagnostic.message
    assert f"second.{name}" in diagnostic.message


def test_v2_runtime_adapter_invalid_duplicate_uses_invalid_source_path():
    from core.engine_v2.diagnostics import V2ValidationError
    from ui import server_services

    context = server_services._resolve_strategy_context(
        [("strategy_id", True, "s06_r_trend_v02_b2")]
    )
    with pytest.raises(V2ValidationError) as raised:
        server_services._normalize_v2_request_runtime(
            context,
            [
                ("warmupBars", "payload.warmupBars", 1000),
                ("warmupBars", "config.warmup_bars", "invalid"),
            ],
            missing_date_filter=False,
        )

    diagnostic = raised.value.diagnostics[0]
    assert diagnostic.path == "config.warmup_bars"
    assert diagnostic.message.startswith(
        "s06_r_trend_v02_b2: config.warmup_bars"
    )


def test_v2_preview_accepts_equivalent_duplicate_warmup_sources(monkeypatch, client):
    from ui import server_routes_run

    payload = _s03_regime_er_grid_preview_payload()
    payload["warmup_bars"] = "1000"
    captured = []
    monkeypatch.setattr(
        server_routes_run,
        "preview_grid_parameter_space",
        lambda config: captured.append(config) or {"ok": True},
    )

    response = client.post(
        "/api/grid/preview",
        json={"config": payload, "strategyId": payload["strategy_id"], "warmupBars": 1000},
    )

    assert response.status_code == 200
    assert captured[0].warmup_bars == 1000


@pytest.mark.parametrize("warmup", [100, 5000])
def test_v2_preview_accepts_exact_warmup_boundaries(monkeypatch, client, warmup):
    from ui import server_routes_run

    payload = _s03_regime_er_grid_preview_payload()
    captured = []
    monkeypatch.setattr(
        server_routes_run,
        "preview_grid_parameter_space",
        lambda config: captured.append(config) or {"ok": True},
    )
    response = client.post(
        "/api/grid/preview",
        json={"config": payload, "strategyId": payload["strategy_id"], "warmupBars": warmup},
    )
    assert response.status_code == 200
    assert captured[0].warmup_bars == warmup


@pytest.mark.parametrize(
    ("name", "value", "expected_path"),
    [
        ("dateFilter", "maybe", "fixed_params.dateFilter"),
        ("start", "not-a-date", "fixed_params.start"),
        ("warmupBars", 99, "warmupBars"),
        ("warmupBars", 5001, "warmupBars"),
        ("warmupBars", True, "warmupBars"),
        ("warmupBars", 100.5, "warmupBars"),
        ("warmupBars", "", "warmupBars"),
        ("warmupBars", "bad", "warmupBars"),
    ],
)
def test_v2_runtime_adapter_rejects_malformed_values(name, value, expected_path):
    from core.engine_v2.diagnostics import V2ValidationError
    from ui import server_services

    context = server_services._resolve_strategy_context(
        [("strategy_id", True, "s06_r_trend_v02_b2")]
    )
    prefix = "fixed_params" if name != "warmupBars" else ""
    path = f"{prefix}.{name}" if prefix else name
    with pytest.raises(V2ValidationError) as raised:
        server_services._normalize_v2_request_runtime(
            context,
            [(name, path, value)],
            missing_date_filter=False,
        )
    diagnostic = raised.value.diagnostics[0]
    assert diagnostic.code == "V2_INVALID_RUNTIME_VALUE"
    assert diagnostic.strategy_id == "s06_r_trend_v02_b2"
    assert diagnostic.path == expected_path
    assert diagnostic.message


def test_v2_runtime_adapter_rejects_active_equal_or_reversed_range():
    from core.engine_v2.diagnostics import V2ValidationError
    from ui import server_services

    context = server_services._resolve_strategy_context(
        [("strategy_id", True, "s06_r_trend_v02_b2")]
    )
    for end in ("2025-05-01T00:00", "2025-04-30T23:59"):
        with pytest.raises(V2ValidationError) as raised:
            server_services._normalize_v2_request_runtime(
                context,
                [
                    ("dateFilter", "fixed_params.dateFilter", True),
                    ("start", "fixed_params.start", "2025-05-01T00:00"),
                    ("end", "fixed_params.end", end),
                ],
                missing_date_filter=False,
            )
        diagnostic = raised.value.diagnostics[0]
        assert diagnostic.path == "fixed_params.end"
        assert diagnostic.message.startswith(
            "s06_r_trend_v02_b2: fixed_params.end"
        )


def test_v2_runtime_adapter_accepts_same_day_date_only_and_timezone_input():
    from ui import server_services

    context = server_services._resolve_strategy_context(
        [("strategy_id", True, "s06_r_trend_v02_b2")]
    )
    runtime = server_services._normalize_v2_request_runtime(
        context,
        [
            ("dateFilter", "fixed_params.dateFilter", True),
            ("start", "fixed_params.start", "2025-06-30"),
            ("end", "fixed_params.end", "2025-06-30"),
        ],
        missing_date_filter=False,
    )
    assert runtime.execution_projection == {
        "dateFilter": True,
        "start": "2025-06-30T00:00:00Z",
        "end": "2025-06-30T23:59:59.999999Z",
    }

    timezone_runtime = server_services._normalize_v2_request_runtime(
        context,
        [
            ("start", "fixed_params.start", "2025-05-01T02:00:00+02:00"),
        ],
        missing_date_filter=False,
    )
    assert timezone_runtime.execution_projection["start"] == "2025-05-01T00:00:00Z"


@pytest.mark.parametrize(
    ("fixed_updates", "warmup", "expected_path"),
    [
        ({"dateFilter": "maybe"}, 1000, "fixed_params.dateFilter"),
        ({"start": "bad-date"}, 1000, "fixed_params.start"),
        ({"dateFilter": True, "start": "2025-05-02", "end": "2025-05-01"}, 1000, "fixed_params.end"),
        ({}, True, "payload.warmupBars"),
        ({}, 99, "payload.warmupBars"),
    ],
)
def test_v2_grid_preview_runtime_failures_precede_plan_work(
    monkeypatch,
    client,
    fixed_updates,
    warmup,
    expected_path,
):
    from ui import server_routes_run

    payload = _s03_regime_er_grid_preview_payload()
    payload["fixed_params"].update(fixed_updates)
    monkeypatch.setattr(
        server_routes_run,
        "preview_grid_parameter_space",
        lambda _config: pytest.fail("preview planning must not start"),
    )
    response = client.post(
        "/api/grid/preview",
        json={
            "config": payload,
            "strategyId": "s03_reversal_v11_regime_er_b2",
            "warmupBars": warmup,
        },
    )
    assert response.status_code == 400
    diagnostic = _v2_runtime_diagnostic(response)
    assert diagnostic["code"] == "V2_INVALID_RUNTIME_VALUE"
    assert diagnostic["path"] == expected_path


def test_v2_backtest_malformed_runtime_precedes_csv_access(monkeypatch, client):
    from ui import server_services

    monkeypatch.setattr(
        server_services,
        "_resolve_csv_path",
        lambda _raw: pytest.fail("CSV access must not start"),
    )
    response = client.post(
        "/api/backtest",
        data={
            "strategy": "s06_r_trend_v02_b2",
            "warmupBars": "99",
            "payload": json.dumps({"dateFilter": False}),
        },
    )
    assert response.status_code == 400
    diagnostic = _v2_runtime_diagnostic(response)
    assert diagnostic["code"] == "V2_INVALID_RUNTIME_VALUE"
    assert diagnostic["path"] == "warmupBars"


@pytest.mark.parametrize("optimization_mode", ["grid"])
@pytest.mark.parametrize(
    ("container", "name", "path"),
    [
        ("enabled_params", "dateFilter", "enabled_params.dateFilter"),
        ("enabled_params", "start", "enabled_params.start"),
        ("enabled_params", "end", "enabled_params.end"),
        ("enabled_params", "warmupBars", "enabled_params.warmupBars"),
        ("param_ranges", "dateFilter", "param_ranges.dateFilter"),
        ("param_ranges", "start", "param_ranges.start"),
        ("param_ranges", "end", "param_ranges.end"),
        ("param_ranges", "warmupBars", "param_ranges.warmupBars"),
        ("fixed_params", "dateFilter_options", "fixed_params.dateFilter_options"),
        ("fixed_params", "start_options", "fixed_params.start_options"),
        ("fixed_params", "end_options", "fixed_params.end_options"),
        ("fixed_params", "warmupBars_options", "fixed_params.warmupBars_options"),
        ("fixed_params", "warmupBars", "fixed_params.warmupBars"),
    ],
)
def test_v2_optimize_rejects_every_reserved_runtime_request_path(
    client,
    optimization_mode,
    container,
    name,
    path,
):
    payload = _s03_regime_er_grid_preview_payload(
        optimization_mode=optimization_mode,
        objectives=["net_profit_pct"],
        grid_fast_objectives=["net_profit_pct"],
    )
    payload[container][name] = False
    response = client.post(
        "/api/optimize",
        data={
            "strategy": "s03_reversal_v11_regime_er_b2",
            "config": json.dumps(payload),
        },
    )
    assert response.status_code == 400
    diagnostic = _v2_runtime_diagnostic(response)
    assert diagnostic["code"] == "V2_RESERVED_RUNTIME_AXIS"
    assert diagnostic["path"] == path
    assert diagnostic["strategy_id"] == "s03_reversal_v11_regime_er_b2"
    assert "core-owned" in diagnostic["message"]


@pytest.mark.parametrize("endpoint", ["/api/backtest", "/api/backtest/trades"])
@pytest.mark.parametrize("value", [False, 1000])
def test_v2_backtest_surfaces_reject_parameter_warmup_before_dataset_work(
    monkeypatch,
    client,
    endpoint,
    value,
):
    from ui import server_services

    monkeypatch.setattr(
        server_services,
        "load_data",
        lambda _source: pytest.fail("dataset load must not start"),
    )
    response = client.post(
        endpoint,
        data={
            "strategy": "s06_r_trend_v02_b2",
            "payload": json.dumps({"warmupBars": value}),
        },
    )
    assert response.status_code == 400
    diagnostic = _v2_runtime_diagnostic(response)
    assert diagnostic["code"] == "V2_RESERVED_RUNTIME_AXIS"
    assert diagnostic["path"] == "parameters.warmupBars"


def test_v1_preview_bypasses_v2_runtime_adapter_and_keeps_warmup_clamp(
    monkeypatch,
    client,
):
    from ui import server_routes_run, server_services

    original_builder = server_routes_run._build_optimization_config
    captured = []

    def capture_builder(*args, **kwargs):
        config = original_builder(*args, **kwargs)
        captured.append(config)
        return config

    monkeypatch.setattr(
        server_services,
        "normalize_v2_runtime_values",
        lambda *_args, **_kwargs: pytest.fail("V1 must bypass V2 normalization"),
    )
    monkeypatch.setattr(server_routes_run, "_build_optimization_config", capture_builder)
    monkeypatch.setattr(server_routes_run, "preview_grid_parameter_space", lambda _config: {"ok": True})
    payload = _grid_sidebar_config()
    payload["strategy_id"] = "s03_reversal_v10"
    response = client.post(
        "/api/grid/preview",
        json={"config": payload, "warmupBars": 5},
    )
    assert response.status_code == 200
    assert captured[0].strategy_id == "s03_reversal_v10"
    assert captured[0].warmup_bars == 100

    backtest = client.post(
        "/api/backtest",
        data={
            "strategy": "s03_reversal_v10",
            "payload": json.dumps({"warmupBars": 5}),
        },
    )
    assert backtest.status_code == 400
    assert "CSV path is required" in backtest.get_data(as_text=True)


def test_derive_grid_preview_internal_runtime_behavior_remains_deferred(monkeypatch):
    from ui import server_services

    captured = {}

    def fake_builder(csv_file, payload, worker_processes, strategy_id, warmup_bars):
        captured.update(
            csv_file=csv_file,
            payload=deepcopy(payload),
            worker_processes=worker_processes,
            strategy_id=strategy_id,
            warmup_bars=warmup_bars,
        )
        return SimpleNamespace()

    monkeypatch.setattr(server_services, "_build_optimization_config", fake_builder)
    monkeypatch.setattr(server_services, "preview_grid_parameter_space", lambda _config: {"ok": True})
    result = server_services._derive_grid_preview(
        {"warmup_bars": 20, "enabled_params": {}, "fixed_params": {}},
        {"strategy_id": "s06_r_trend_v02_b2"},
    )
    assert result == {"ok": True}
    assert captured == {
        "csv_file": "grid-sidebar.csv",
        "payload": {
            "warmup_bars": 20,
            "enabled_params": {},
            "fixed_params": {},
            "optimization_mode": "grid",
        },
        "worker_processes": 1,
        "strategy_id": "s06_r_trend_v02_b2",
        "warmup_bars": 20,
    }


@pytest.mark.parametrize(
    "fixed_params",
    [
        {"dateFilter": False, "start": "", "end": "", "useRegime": False, "useEmergencySL": False},
        {"useRegime": False, "useEmergencySL": False},
    ],
)
@pytest.mark.parametrize("planning_policy", ["full", "sampled"])
def test_v2_preview_and_direct_run_build_equal_runtime_and_grid_facts(
    monkeypatch,
    client,
    tmp_path,
    fixed_params,
    planning_policy,
):
    from ui import server_routes_run

    csv_path = tmp_path / "parity.csv"
    csv_path.write_text("timestamp,open,high,low,close,volume\n", encoding="utf-8")
    payload = _s03_regime_er_grid_preview_payload(
        strategy_id="s03_reversal_v11_regime_er_b2",
        fixed_params=deepcopy(fixed_params),
        grid_v2_planning_policy=planning_policy,
        grid_budget=11,
        grid_seed=17,
        grid_enabled_modes=[],
        grid_allocation_method="manual",
        grid_manual_percents={"cc_only": 20, "tbands_only": 30, "both": 50},
        grid_fast_objectives=["net_profit_pct"],
        grid_fast_primary_objective=None,
        param_ranges={"maType3": {"type": "select", "values": ["EMA", "SMA"]}},
    )
    original_builder = server_routes_run._build_optimization_config
    built = []

    def capture_builder(*args, **kwargs):
        config = original_builder(*args, **kwargs)
        built.append(config)
        return config

    monkeypatch.setattr(server_routes_run, "_build_optimization_config", capture_builder)
    monkeypatch.setattr(server_routes_run, "preview_grid_parameter_space", lambda _config: {"ok": True})
    monkeypatch.setattr(server_routes_run, "_resolve_csv_path", lambda _raw: csv_path)
    monkeypatch.setattr(server_routes_run, "run_optimization", lambda _config: ([], None))

    preview_response = client.post(
        "/api/grid/preview",
        json={
            "config": deepcopy(payload),
            "strategyId": "s03_reversal_v11_regime_er_b2",
            "warmupBars": 250,
        },
    )
    run_response = client.post(
        "/api/optimize",
        data={
            "strategy": "s03_reversal_v11_regime_er_b2",
            "warmupBars": "250",
            "csvPath": str(csv_path),
            "config": json.dumps(payload),
        },
    )
    assert preview_response.status_code == 200
    assert run_response.status_code == 200
    assert len(built) == 2
    preview_config, run_config = built
    facts = (
        "strategy_id",
        "fixed_params",
        "warmup_bars",
        "enabled_params",
        "param_ranges",
        "grid_v2_planning_policy",
        "grid_budget",
        "grid_seed",
        "grid_enabled_modes",
        "grid_allocation_method",
        "grid_manual_percents",
        "grid_fast_objectives",
        "grid_fast_primary_objective",
    )
    for name in facts:
        assert getattr(preview_config, name) == getattr(run_config, name), name
    assert "warmupBars" not in preview_config.fixed_params
    assert run_config.v2_runtime == {
        "schema_version": "v2_runtime_metadata_v1",
        "contract_version": "v2_runtime_contract_v1",
        "values": {
            "dateFilter": False,
            "start": None,
            "end": None,
            "warmupBars": 250,
        },
        "diagnostics": [],
        "validation_warnings": [],
    }
    if "start" in fixed_params:
        assert preview_config.fixed_params["start"] is None
        assert preview_config.fixed_params["end"] is None
    else:
        assert "start" not in preview_config.fixed_params
        assert "end" not in preview_config.fixed_params


@pytest.mark.parametrize("optimization_mode", ["grid"])
def test_v2_ft_oos_derivation_uses_core_canonical_dates_once(
    monkeypatch,
    client,
    tmp_path,
    optimization_mode,
):
    from ui import server_routes_run, server_services

    csv_path = tmp_path / f"derived_{optimization_mode}.csv"
    csv_path.write_text("placeholder", encoding="utf-8")
    payload = _s03_regime_er_grid_preview_payload(
        optimization_mode=optimization_mode,
        objectives=["net_profit_pct"],
        grid_fast_objectives=["net_profit_pct"],
        fixed_params={
            "dateFilter": False,
            "start": "2025-01-01T00:00:00Z",
            "end": "2025-03-31T23:59:59Z",
            "useRegime": False,
            "useEmergencySL": False,
        },
    )
    payload["postProcess"] = {"enabled": True, "ftPeriodDays": 10}
    payload["oosTest"] = {"enabled": True, "periodDays": 7}

    built = []
    periods = []
    complete_calls = []
    original_builder = server_routes_run._build_optimization_config
    original_periods = server_routes_run.calculate_period_dates
    original_normalizer = server_services.normalize_v2_runtime_values

    def capture_builder(*args, **kwargs):
        config = original_builder(*args, **kwargs)
        built.append(config)
        return config

    def capture_periods(*args, **kwargs):
        result = original_periods(*args, **kwargs)
        periods.append(result)
        return result

    def count_complete(*args, **kwargs):
        complete_calls.append((args, kwargs))
        return original_normalizer(*args, **kwargs)

    monkeypatch.setattr(server_routes_run, "_resolve_csv_path", lambda _raw: csv_path)
    monkeypatch.setattr(server_routes_run, "_build_optimization_config", capture_builder)
    monkeypatch.setattr(server_routes_run, "calculate_period_dates", capture_periods)
    monkeypatch.setattr(server_services, "normalize_v2_runtime_values", count_complete)
    monkeypatch.setattr(server_routes_run, "run_optimization", lambda _config: ([], None))

    response = client.post(
        "/api/optimize",
        data={
            "strategy": "s03_reversal_v11_regime_er_b2",
            "warmupBars": "1000",
            "csvPath": str(csv_path),
            "config": json.dumps(payload),
        },
    )

    assert response.status_code == 200
    assert len(complete_calls) == 1
    assert len(built) == len(periods) == 1
    config = built[0]
    period = periods[0]
    assert config.fixed_params["dateFilter"] is True
    assert config.fixed_params["start"] == "2025-01-01T00:00:00Z"
    assert config.fixed_params["end"] == "2025-03-14T23:59:59Z"
    assert "+00:00" not in config.fixed_params["start"]
    assert "+00:00" not in config.fixed_params["end"]
    assert config.is_period_days == period["is_days"] == 72
    assert config.ft_period_days == period["ft_days"] == 10
    assert period["oos_days"] == 7
    assert period["ft_start"] == pd.Timestamp("2025-03-14T23:59:59Z")
    assert period["ft_end"] == period["oos_start"] == pd.Timestamp(
        "2025-03-24T23:59:59Z"
    )
    assert period["oos_end"] == pd.Timestamp("2025-03-31T23:59:59Z")


def test_v2_ft_derived_runtime_failure_is_structured_and_stops_config_build(
    monkeypatch,
    client,
    tmp_path,
):
    from core.engine_v2.diagnostics import V2Diagnostic
    from core.engine_v2.runtime_contract import V2RuntimeValidationError
    from ui import server_routes_run

    csv_path = tmp_path / "derived_runtime_failure.csv"
    csv_path.write_text("placeholder", encoding="utf-8")
    payload = _s03_regime_er_grid_preview_payload(
        optimization_mode="grid",
        objectives=["net_profit_pct"],
        fixed_params={
            "dateFilter": False,
            "start": "2025-01-01T00:00:00Z",
            "end": "2025-03-31T23:59:59Z",
            "useRegime": False,
            "useEmergencySL": False,
        },
    )
    payload["postProcess"] = {"enabled": True, "ftPeriodDays": 10}

    def fail_derived_normalization(
        _name,
        _value,
        *,
        strategy_id,
        path,
        user_boundary,
    ):
        assert user_boundary is False
        raise V2RuntimeValidationError(
            V2Diagnostic(
                severity="error",
                code="V2_INVALID_RUNTIME_VALUE",
                strategy_id=strategy_id,
                path=path,
                variant=None,
                message=f"{strategy_id}: {path} could not be normalized.",
            )
        )

    monkeypatch.setattr(server_routes_run, "_resolve_csv_path", lambda _raw: csv_path)
    monkeypatch.setattr(
        server_routes_run,
        "normalize_v2_runtime_field_value",
        fail_derived_normalization,
    )
    monkeypatch.setattr(
        server_routes_run,
        "_build_optimization_config",
        lambda *_args, **_kwargs: pytest.fail(
            "optimization config must not be built after runtime validation fails"
        ),
    )
    monkeypatch.setattr(
        server_routes_run,
        "run_optimization",
        lambda *_args, **_kwargs: pytest.fail(
            "optimization must not run after runtime validation fails"
        ),
    )

    response = client.post(
        "/api/optimize",
        data={
            "strategy": "s03_reversal_v11_regime_er_b2",
            "warmupBars": "1000",
            "csvPath": str(csv_path),
            "config": json.dumps(payload),
        },
    )

    assert response.status_code == 400
    body = response.get_json()
    assert body["error"] == (
        "s03_reversal_v11_regime_er_b2: fixed_params.end could not be normalized."
    )
    assert body["diagnostics"] == [
        {
            "severity": "error",
            "code": "V2_INVALID_RUNTIME_VALUE",
            "strategy_id": "s03_reversal_v11_regime_er_b2",
            "path": "fixed_params.end",
            "variant": None,
            "message": (
                "s03_reversal_v11_regime_er_b2: fixed_params.end could not be "
                "normalized."
            ),
        }
    ]


def test_v1_ft_derivation_keeps_legacy_isoformat_representation(
    monkeypatch,
    client,
    tmp_path,
):
    from ui import server_routes_run, server_services

    csv_path = tmp_path / "derived_v1.csv"
    csv_path.write_text("placeholder", encoding="utf-8")
    payload = _grid_sidebar_config()
    payload.update(
        optimization_mode="optuna",
        objectives=["net_profit_pct"],
        primary_objective="net_profit_pct",
        fixed_params={
            "dateFilter": False,
            "start": "2025-01-01T00:00:00Z",
            "end": "2025-03-31T23:59:59Z",
        },
        postProcess={"enabled": True, "ftPeriodDays": 10},
    )
    built = []
    original_builder = server_routes_run._build_optimization_config

    def capture_builder(*args, **kwargs):
        config = original_builder(*args, **kwargs)
        built.append(config)
        return config

    monkeypatch.setattr(server_routes_run, "_resolve_csv_path", lambda _raw: csv_path)
    monkeypatch.setattr(server_routes_run, "_build_optimization_config", capture_builder)
    monkeypatch.setattr(server_routes_run, "run_optimization", lambda _config: ([], None))
    monkeypatch.setattr(
        server_services,
        "normalize_v2_runtime_values",
        lambda *_args, **_kwargs: pytest.fail("V1 must bypass V2 normalization"),
    )

    response = client.post(
        "/api/optimize",
        data={
            "strategy": "s03_reversal_v10",
            "warmupBars": "1000",
            "csvPath": str(csv_path),
            "config": json.dumps(payload),
        },
    )

    assert response.status_code == 200
    assert built[0].fixed_params["dateFilter"] is True
    assert built[0].fixed_params["start"] == "2025-01-01T00:00:00Z"
    assert built[0].fixed_params["end"].endswith("+00:00")


@pytest.mark.parametrize("endpoint", ["/api/backtest", "/api/backtest/trades"])
def test_v2_backtest_and_trade_download_share_canonical_runtime_projection(
    monkeypatch,
    client,
    tmp_path,
    endpoint,
):
    import strategies
    from ui import server_services

    csv_path = tmp_path / "backtest_runtime.csv"
    csv_path.write_text("placeholder", encoding="utf-8")
    df = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]},
        index=pd.to_datetime(["2025-05-01T00:00:00Z"]),
    )
    captured = []

    class DummyResult:
        trades = []

        def to_dict(self):
            return {"ok": True}

    class DummyStrategy:
        @staticmethod
        def run(_df, params, trade_start_idx):
            captured.append((deepcopy(params), trade_start_idx))
            return DummyResult()

    original_get_strategy = strategies.get_strategy
    monkeypatch.setattr(
        strategies,
        "get_strategy",
        lambda strategy_id: DummyStrategy
        if strategy_id == "s06_r_trend_v02_b2"
        else original_get_strategy(strategy_id),
    )
    monkeypatch.setattr(server_services, "_resolve_csv_path", lambda _raw: csv_path)
    monkeypatch.setattr(server_services, "load_data", lambda _source: df)
    monkeypatch.setattr(
        server_services,
        "prepare_dataset_with_warmup",
        lambda data, _start, _end, warmup: (data, warmup),
    )

    response = client.post(
        endpoint,
        data={
            "strategy": "s06_r_trend_v02_b2",
            "warmupBars": "250",
            "csvPath": str(csv_path),
            "payload": json.dumps(
                {
                    "dateFilter": "true",
                    "start": "2025-05-01T08:00:00+08:00",
                    "end": "2025-05-02T08:00:00+08:00",
                }
            ),
        },
    )
    assert response.status_code == 200
    assert captured == [
        (
            {
                "dateFilter": True,
                "start": "2025-05-01T00:00:00Z",
                "end": "2025-05-02T00:00:00Z",
            },
            250,
        )
    ]


def test_v2_backtest_date_only_range_reaches_final_day_strategy_execution(
    monkeypatch,
    client,
    tmp_path,
):
    import strategies
    from ui import server_services

    csv_path = tmp_path / "date_only_backtest.csv"
    csv_path.write_text("placeholder", encoding="utf-8")
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
    captured = {}

    class DummyResult:
        trades = []

        def to_dict(self):
            return {"ok": True}

    class DummyStrategy:
        @staticmethod
        def run(df, params, trade_start_idx):
            captured.update(
                rows=len(df),
                first=df.index[trade_start_idx],
                last=df.index[-1],
                params=deepcopy(params),
            )
            return DummyResult()

    original_get_strategy = strategies.get_strategy
    monkeypatch.setattr(
        strategies,
        "get_strategy",
        lambda strategy_id: DummyStrategy
        if strategy_id == "s06_r_trend_v02_b2"
        else original_get_strategy(strategy_id),
    )
    monkeypatch.setattr(server_services, "_resolve_csv_path", lambda _raw: csv_path)
    monkeypatch.setattr(server_services, "load_data", lambda _source: frame)

    response = client.post(
        "/api/backtest",
        data={
            "strategy": "s06_r_trend_v02_b2",
            "warmupBars": "100",
            "csvPath": str(csv_path),
            "payload": json.dumps(
                {
                    "dateFilter": True,
                    "start": "2025-06-01",
                    "end": "2025-06-30",
                }
            ),
        },
    )

    assert response.status_code == 200
    assert captured["rows"] == 2_880
    assert captured["first"] == index[0]
    assert captured["last"] == index[-1]
    assert captured["params"]["end"] == "2025-06-30T23:59:59.999999Z"


def test_v1_backtest_branch_uses_explicit_runtime_locals(monkeypatch, client, tmp_path):
    import strategies
    from ui import server_services

    csv_path = tmp_path / "v1_runtime_locals.csv"
    csv_path.write_text("placeholder", encoding="utf-8")
    index = pd.date_range("2025-06-01T00:00:00Z", periods=2, freq="15min")
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

    class DummyResult:
        trades = []

        def to_dict(self):
            return {"ok": True}

    class DummyStrategy:
        @staticmethod
        def run(_df, _params, _trade_start_idx):
            return DummyResult()

    original_get_strategy = strategies.get_strategy
    monkeypatch.setattr(
        strategies,
        "get_strategy",
        lambda strategy_id: DummyStrategy
        if strategy_id == "s03_reversal_v10"
        else original_get_strategy(strategy_id),
    )
    monkeypatch.setattr(server_services, "_resolve_csv_path", lambda _raw: csv_path)
    monkeypatch.setattr(server_services, "load_data", lambda _source: frame)

    response = client.post(
        "/api/backtest",
        data={
            "strategy": "s03_reversal_v10",
            "csvPath": str(csv_path),
            "payload": "{}",
        },
    )
    assert response.status_code == 200
