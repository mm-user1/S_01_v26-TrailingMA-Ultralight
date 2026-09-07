"""Server wfa contracts."""

import json
import logging
import re
from copy import deepcopy
from types import SimpleNamespace

import pytest
import pandas as pd

from ui.server import app

from ._helpers import (
    _build_minimal_optuna_payload,
    _s03_regime_er_grid_preview_payload,
    _v2_runtime_diagnostic,
)


def test_invalid_v2_profile_fails_walkforward_before_work(monkeypatch, client):
    import strategies
    from ui import server_routes_run

    original = strategies.get_strategy_config
    invalid = deepcopy(original("s06_r_trend_v02_b2"))
    invalid["execution"]["entryOrder"] = "unsupported"
    monkeypatch.setattr(
        strategies,
        "get_strategy_config",
        lambda strategy_id: invalid if strategy_id == invalid["id"] else original(strategy_id),
    )
    monkeypatch.setattr(
        server_routes_run,
        "_clear_cancelled_run",
        lambda _run_id: pytest.fail("run state must not be mutated"),
    )
    monkeypatch.setattr(
        server_routes_run,
        "_resolve_csv_path",
        lambda _path: pytest.fail("CSV resolution must not start"),
    )
    monkeypatch.setattr(
        server_routes_run,
        "_build_optimization_config",
        lambda *_args, **_kwargs: pytest.fail("window config must not be built"),
    )
    monkeypatch.setattr(
        server_routes_run,
        "load_data",
        lambda _source: pytest.fail("dataset load must not start"),
    )

    response = client.post(
        "/api/walkforward",
        data={"strategy": invalid["id"], "config": "{}"},
    )

    assert response.status_code == 400
    diagnostic = _v2_runtime_diagnostic(response)
    assert diagnostic["code"] == "V2_UNSUPPORTED_EXECUTION_MODE"
    assert diagnostic["strategy_id"] == invalid["id"]


@pytest.mark.parametrize(
    ("strategy_id", "config"),
    [
        ("s06_r_trend_v02_b2", {"optimization_mode": "grid", "fixed_params": {"dateFilter": False}}),
        ("s03_reversal_v10", {}),
    ],
)
def test_valid_v2_and_v1_walkforward_retain_existing_post_profile_path(
    client, strategy_id, config
):
    response = client.post(
        "/api/walkforward",
        data={"strategy": strategy_id, "config": json.dumps(config)},
    )
    assert response.status_code == 400
    assert response.get_json() == {"error": "CSV path is required."}


@pytest.mark.parametrize(
    ("period_fields", "fixed_params", "message"),
    [
        (
            {"wf_period_unit": "months", "wf_is_period_months": "2", "wf_oos_period_months": "1", "wf_adaptive_mode": "true"},
            {"dateFilter": True, "start": "2025-10-01", "end": "2026-08-01"},
            "Adaptive",
        ),
        (
            {"wf_period_unit": "months", "wf_is_period_months": "2", "wf_oos_period_months": "1"},
            {"dateFilter": False, "start": "2025-10-01", "end": "2026-08-01"},
            "Date Filter",
        ),
        (
            {"wf_period_unit": "months", "wf_is_period_months": "2", "wf_oos_period_months": "1"},
            {"dateFilter": True, "start": "2025-08-29", "end": "2026-08-01"},
            "calendar day 1 through 28",
        ),
        (
            {"wf_period_unit": "months", "wf_is_period_months": "2", "wf_oos_period_months": "1"},
            {"dateFilter": True, "start": "2025-08-30", "end": "2026-08-01"},
            "calendar day 1 through 28",
        ),
        (
            {"wf_period_unit": "months", "wf_is_period_months": "2", "wf_oos_period_months": "1"},
            {"dateFilter": True, "start": "2025-08-31", "end": "2026-08-01"},
            "calendar day 1 through 28",
        ),
        (
            {"wf_period_unit": "months", "wf_is_period_months": "1.5", "wf_oos_period_months": "1"},
            {"dateFilter": True, "start": "2025-10-01", "end": "2026-08-01"},
            "whole number",
        ),
        (
            {"wf_period_unit": "weeks"},
            {"dateFilter": True, "start": "2025-10-01", "end": "2026-08-01"},
            "days.*months",
        ),
    ],
)
def test_wfa_month_request_validation_precedes_csv_and_config_work(
    client, monkeypatch, period_fields, fixed_params, message
):
    from ui import server_routes_run

    monkeypatch.setattr(
        server_routes_run,
        "_resolve_csv_path",
        lambda _raw: pytest.fail("CSV resolution must not start"),
    )
    monkeypatch.setattr(
        server_routes_run,
        "_build_optimization_config",
        lambda *_args, **_kwargs: pytest.fail("optimization config must not be built"),
    )
    payload = _build_minimal_optuna_payload()
    payload["fixed_params"] = fixed_params

    response = client.post(
        "/api/walkforward",
        data={
            "strategy": "s01_trailing_ma",
            "csvPath": "unused.csv",
            "config": json.dumps(payload),
            **period_fields,
        },
    )

    assert response.status_code == 400
    assert re.search(message, response.get_json()["error"])


@pytest.mark.parametrize(
    ("strategy_id", "is_v2"),
    [("s01_trailing_ma", False), ("s03_reversal_v11_regime_er_b2", True)],
)
@pytest.mark.parametrize(
    ("requested_start", "expected_windows"),
    [("2025-10-01", 8), ("2025-10-15", 7), ("2025-10-28", 7)],
)
def test_wfa_month_request_uses_authoritative_month_fields(
    monkeypatch,
    client,
    tmp_path,
    strategy_id,
    is_v2,
    requested_start,
    expected_windows,
):
    from core import walkforward_engine
    from core.engine_v2.runtime_contract import normalize_v2_runtime_field_value
    from ui import server_routes_run

    csv_path = tmp_path / "calendar_wfa.csv"
    csv_path.write_text("unused", encoding="utf-8")
    df = pd.DataFrame(
        {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0},
        index=pd.date_range("2025-08-01", "2026-08-01", freq="h", tz="UTC"),
    )
    captured = {}

    class DummyWalkForwardEngine:
        def __init__(self, config, base_template, _optuna_settings, csv_file_path=None):
            captured["config"] = config
            captured["base_template"] = base_template

        def run_wf_optimization(self, routed_df):
            config = captured["config"]
            fixed = captured["base_template"]["fixed_params"]
            requested_end = normalize_v2_runtime_field_value("end", fixed["end"])
            windows = walkforward_engine.build_calendar_month_windows(
                routed_df.index,
                fixed["start"],
                pd.Timestamp(requested_end),
                config.is_period_months,
                config.oos_period_months,
            )
            captured["windows"] = windows
            stitched = SimpleNamespace(
                final_net_profit_pct=0.0,
                max_drawdown_pct=0.0,
                total_trades=0,
                wfe=0.0,
                oos_win_rate=0.0,
            )
            return SimpleNamespace(total_windows=len(windows), stitched_oos=stitched), "month-study"

    monkeypatch.setattr(server_routes_run, "_resolve_csv_path", lambda _raw: csv_path)
    monkeypatch.setattr(server_routes_run, "load_data", lambda _path: df)
    monkeypatch.setattr(walkforward_engine, "WalkForwardEngine", DummyWalkForwardEngine)
    if is_v2:
        payload = _s03_regime_er_grid_preview_payload(
            optimization_mode="grid",
            objectives=["net_profit_pct"],
            primary_objective="net_profit_pct",
        )
        payload["fixed_params"].update(
            {"dateFilter": True, "start": requested_start, "end": "2026-08-01"}
        )
    else:
        payload = _build_minimal_optuna_payload()
        payload["fixed_params"] = {
            "dateFilter": True,
            "start": requested_start,
            "end": "2026-08-01",
        }

    response = client.post(
        "/api/walkforward",
        data={
            "strategy": strategy_id,
            "csvPath": str(csv_path),
            "warmupBars": "1000",
            "config": json.dumps(payload),
            "wf_period_unit": "months",
            "wf_is_period_months": "2",
            "wf_oos_period_months": "1",
        },
    )

    assert response.status_code == 200
    assert response.get_json()["summary"]["total_windows"] == expected_windows
    anchor = pd.Timestamp(requested_start, tz="UTC")
    first_window = captured["windows"][0]
    assert first_window.is_start == anchor
    assert first_window.is_end == anchor + pd.DateOffset(months=2) - pd.Timedelta(hours=1)
    assert first_window.oos_start == anchor + pd.DateOffset(months=2)
    assert first_window.oos_end == anchor + pd.DateOffset(months=3) - pd.Timedelta(hours=1)
    assert captured["config"].is_period_days is None
    assert captured["config"].oos_period_days is None
    assert captured["base_template"]["wfa"]["period_unit"] == "months"
    assert captured["base_template"]["wfa"]["is_period_months"] == 2
    assert captured["base_template"]["wfa"]["oos_period_months"] == 1
    assert "is_period_days" not in captured["base_template"]["wfa"]
    assert "oos_period_days" not in captured["base_template"]["wfa"]
    state_wfa = client.get("/api/optimization/status").get_json()["wfa"]
    assert state_wfa["period_unit"] == "months"
    assert state_wfa["is_period_months"] == 2
    assert state_wfa["oos_period_months"] == 1
    assert "is_period_days" not in state_wfa
    assert "oos_period_days" not in state_wfa


def test_v2_walkforward_runtime_failure_precedes_cancellation_and_csv(
    monkeypatch,
    client,
):
    from ui import server_routes_run

    monkeypatch.setattr(
        server_routes_run,
        "_clear_cancelled_run",
        lambda _run_id: pytest.fail("cancellation state must not be touched"),
    )
    monkeypatch.setattr(
        server_routes_run,
        "_resolve_csv_path",
        lambda _raw: pytest.fail("CSV must not be resolved"),
    )
    response = client.post(
        "/api/walkforward",
        data={
            "strategy": "s06_r_trend_v02_b2",
            "runId": "early-v2",
            "csvPath": "missing.csv",
            "warmupBars": "99",
            "config": json.dumps(
                {
                    "strategy_id": "s06_r_trend_v02_b2",
                    "optimization_mode": "grid",
                    "fixed_params": {"dateFilter": False},
                }
            ),
        },
    )

    assert response.status_code == 400
    diagnostic = _v2_runtime_diagnostic(response)
    assert diagnostic["code"] == "V2_INVALID_RUNTIME_VALUE"
    assert diagnostic["path"] == "warmupBars"


def test_walkforward_cancelled_run_cleans_up_saved_study(client, monkeypatch, tmp_path):
    from ui import server_routes_run
    import core.walkforward_engine as walkforward_engine

    csv_path = tmp_path / "wfa_cancel.csv"
    csv_path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2026-01-01 00:00:00,1,1,1,1,1\n",
        encoding="utf-8",
    )

    df = pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.2],
            "high": [1.0, 1.1, 1.2],
            "low": [1.0, 1.1, 1.2],
            "close": [1.0, 1.1, 1.2],
            "volume": [1.0, 1.0, 1.0],
        },
        index=pd.to_datetime(
            ["2026-01-01 00:00:00", "2026-01-01 01:00:00", "2026-01-01 02:00:00"],
            utc=True,
        ),
    )

    class DummyWalkForwardEngine:
        def __init__(self, *_args, **_kwargs):
            pass

        def run_wf_optimization(self, _dataframe):
            return None, "study_cancel_wfa"

    deleted_studies = []
    monkeypatch.setattr(server_routes_run, "_resolve_csv_path", lambda _raw: csv_path)
    monkeypatch.setattr(server_routes_run, "load_data", lambda _path: df)
    monkeypatch.setattr(server_routes_run, "_is_run_cancelled", lambda run_id: run_id == "run_cancel_wfa")
    monkeypatch.setattr(
        server_routes_run,
        "delete_study",
        lambda study_id: deleted_studies.append(study_id) or True,
    )
    monkeypatch.setattr(walkforward_engine, "WalkForwardEngine", DummyWalkForwardEngine)

    payload = _build_minimal_optuna_payload()
    payload["primary_objective"] = "net_profit_pct"

    response = client.post(
        "/api/walkforward",
        data={
            "strategy": "s01_trailing_ma",
            "csvPath": str(csv_path),
            "runId": "run_cancel_wfa",
            "config": json.dumps(payload),
        },
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "cancelled"
    assert data["run_id"] == "run_cancel_wfa"
    assert data["study_id"] is None
    assert deleted_studies == ["study_cancel_wfa"]


@pytest.mark.parametrize(
    ("worker_fields", "expected_workers"),
    [
        ({}, 6),
        ({"worker_processes": 1}, 1),
        ({"workerProcesses": "2"}, 2),
        ({"worker_processes": "3", "workerProcesses": "4"}, 3),
        ({"worker_processes": None, "workerProcesses": "4"}, 4),
        ({"worker_processes": 0}, 1),
        ({"worker_processes": 99}, 32),
    ],
)
def test_walkforward_route_threads_requested_workers_through_shared_builder(
    client,
    monkeypatch,
    tmp_path,
    worker_fields,
    expected_workers,
):
    from core import walkforward_engine
    from ui import server_routes_run

    csv_path = tmp_path / "wfa_workers.csv"
    csv_path.write_text("placeholder", encoding="utf-8")
    df = pd.DataFrame(
        {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1.0]},
        index=pd.to_datetime(["2026-01-01T00:00:00Z"]),
    )
    captured = {}

    class DummyWalkForwardEngine:
        def __init__(self, _config, base_template, _optuna_settings, csv_file_path=None):
            captured["base_template"] = base_template

        def run_wf_optimization(self, _dataframe):
            stitched = SimpleNamespace(
                final_net_profit_pct=0.0,
                max_drawdown_pct=0.0,
                total_trades=0,
                wfe=0.0,
                oos_win_rate=0.0,
            )
            return SimpleNamespace(total_windows=0, stitched_oos=stitched), "worker-study"

    monkeypatch.setattr(server_routes_run, "_resolve_csv_path", lambda _raw: csv_path)
    monkeypatch.setattr(server_routes_run, "load_data", lambda _path: df)
    monkeypatch.setattr(walkforward_engine, "WalkForwardEngine", DummyWalkForwardEngine)
    payload = _build_minimal_optuna_payload()
    payload.update(worker_fields)

    response = client.post(
        "/api/walkforward",
        data={
            "strategy": "s01_trailing_ma",
            "csvPath": str(csv_path),
            "config": json.dumps(payload),
        },
    )

    assert response.status_code == 200
    assert captured["base_template"]["worker_processes"] == expected_workers


def test_walkforward_route_rejects_malformed_requested_workers_before_data_load(
    client, monkeypatch, tmp_path
):
    from ui import server_routes_run

    csv_path = tmp_path / "wfa_bad_workers.csv"
    csv_path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(server_routes_run, "_resolve_csv_path", lambda _raw: csv_path)
    monkeypatch.setattr(
        server_routes_run,
        "load_data",
        lambda _path: pytest.fail("malformed worker count must fail before data load"),
    )
    payload = _build_minimal_optuna_payload()
    payload["worker_processes"] = "not-an-integer"

    response = client.post(
        "/api/walkforward",
        data={
            "strategy": "s01_trailing_ma",
            "csvPath": str(csv_path),
            "config": json.dumps(payload),
        },
    )

    assert response.status_code == 400
    assert "invalid literal for int" in response.get_json()["error"]


def test_walkforward_route_logs_value_error_details(client, monkeypatch, caplog, tmp_path):
    from ui import server_routes_run
    import core.walkforward_engine as walkforward_engine

    csv_path = tmp_path / "wfa_value_error.csv"
    csv_path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2026-01-01 00:00:00,1,1,1,1,1\n",
        encoding="utf-8",
    )
    df = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]},
        index=pd.to_datetime(["2026-01-01 00:00:00"], utc=True),
    )
    error_text = (
        "Adaptive WFA window 12 IS optimization failed "
        "(2025-05-26 to 2025-07-25, optimizer=grid): "
        "Grid fast-vs-slow validation failed: {\"candidate_id\": 99}"
    )

    class DummyWalkForwardEngine:
        def __init__(self, *_args, **_kwargs):
            pass

        def run_wf_optimization(self, _dataframe):
            raise ValueError(error_text)

    monkeypatch.setattr(server_routes_run, "_resolve_csv_path", lambda _raw: csv_path)
    monkeypatch.setattr(server_routes_run, "load_data", lambda _path: df)
    monkeypatch.setattr(walkforward_engine, "WalkForwardEngine", DummyWalkForwardEngine)

    payload = _build_minimal_optuna_payload()
    payload["optimization_mode"] = "grid"
    payload["strategy_id"] = "s03_reversal_v10"
    payload["primary_objective"] = "net_profit_pct"

    with caplog.at_level(logging.WARNING, logger=app.logger.name):
        response = client.post(
            "/api/walkforward",
            data={
                "strategy": "s03_reversal_v10",
                "csvPath": str(csv_path),
                "runId": "run_wfa_error",
                "config": json.dumps(payload),
                "wf_adaptive_mode": "true",
            },
        )

    assert response.status_code == 400
    data = response.get_json()
    assert data["error"] == error_text
    assert any(error_text in record.getMessage() for record in caplog.records)


def test_walkforward_construction_validation_is_structured_before_state(
    client, monkeypatch, tmp_path
):
    import core.walkforward_engine as walkforward_engine
    from core.engine_v2 import V2Diagnostic, V2ValidationError
    from ui import server_routes_run

    csv_path = tmp_path / "wfa_construction_validation.csv"
    csv_path.write_text("placeholder", encoding="utf-8")
    df = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]},
        index=pd.to_datetime(["2026-01-01T00:00:00Z"]),
    )
    diagnostic = V2Diagnostic(
        severity="error",
        code="V2_INVALID_EXECUTION_PROFILE",
        strategy_id="s06_r_trend_v02_b2",
        path="execution",
        variant=None,
        message="Construction profile is invalid.",
    )

    class InvalidWalkForwardEngine:
        def __init__(self, *_args, **_kwargs):
            raise V2ValidationError(diagnostic)

    monkeypatch.setattr(server_routes_run, "_resolve_csv_path", lambda _raw: csv_path)
    monkeypatch.setattr(server_routes_run, "load_data", lambda _path: df)
    monkeypatch.setattr(walkforward_engine, "WalkForwardEngine", InvalidWalkForwardEngine)
    monkeypatch.setattr(
        server_routes_run,
        "_set_optimization_state",
        lambda *_args, **_kwargs: pytest.fail("run state must not begin"),
    )

    response = client.post(
        "/api/walkforward",
        data={
            "strategy": "s06_r_trend_v02_b2",
            "csvPath": str(csv_path),
            "config": json.dumps(
                {
                    "strategy_id": "s06_r_trend_v02_b2",
                    "optimization_mode": "grid",
                    "enabled_params": {},
                    "param_ranges": {},
                    "param_types": {},
                    "fixed_params": {"dateFilter": False},
                    "objectives": ["net_profit_pct"],
                }
            ),
        },
    )

    assert response.status_code == 400
    assert response.get_json() == {
        "error": "Construction profile is invalid.",
        "diagnostics": [diagnostic.to_dict()],
    }


def test_walkforward_route_parses_adaptive_cooldown_fields(client, monkeypatch, tmp_path):
    from ui import server_routes_run
    import core.walkforward_engine as walkforward_engine

    csv_path = tmp_path / "wfa_cooldown_route.csv"
    csv_path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2026-01-01 00:00:00,1,1,1,1,1\n"
        "2026-01-02 00:00:00,1,1,1,1,1\n"
        "2026-01-03 00:00:00,1,1,1,1,1\n",
        encoding="utf-8",
    )

    df = pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.2],
            "high": [1.0, 1.1, 1.2],
            "low": [1.0, 1.1, 1.2],
            "close": [1.0, 1.1, 1.2],
            "volume": [1.0, 1.0, 1.0],
        },
        index=pd.to_datetime(
            ["2026-01-01 00:00:00", "2026-01-02 00:00:00", "2026-01-03 00:00:00"],
            utc=True,
        ),
    )

    captured = {}

    class DummyWalkForwardEngine:
        def __init__(self, config, base_template, optuna_settings, csv_file_path=None):
            captured["config"] = config
            captured["base_template"] = base_template
            captured["optuna_settings"] = optuna_settings
            captured["csv_file_path"] = csv_file_path

        def run_wf_optimization(self, _dataframe):
            return (
                SimpleNamespace(
                    total_windows=1,
                    stitched_oos=SimpleNamespace(
                        final_net_profit_pct=0.0,
                        max_drawdown_pct=0.0,
                        total_trades=0,
                        wfe=0.0,
                        oos_win_rate=0.0,
                    ),
                ),
                "study_route_cooldown",
            )

    monkeypatch.setattr(server_routes_run, "_resolve_csv_path", lambda _raw: csv_path)
    monkeypatch.setattr(server_routes_run, "load_data", lambda _path: df)
    monkeypatch.setattr(walkforward_engine, "WalkForwardEngine", DummyWalkForwardEngine)

    payload = _build_minimal_optuna_payload()
    payload["primary_objective"] = "net_profit_pct"
    payload["grid_v2_prefer_compiled"] = False
    payload["grid_v2_max_cache_mb"] = 123.5

    response = client.post(
        "/api/walkforward",
        data={
            "strategy": "s01_trailing_ma",
            "csvPath": str(csv_path),
            "config": json.dumps(payload),
            "wf_adaptive_mode": "true",
            "wf_cooldown_enabled": "true",
            "wf_cooldown_days": "21",
        },
    )

    assert response.status_code == 200
    assert captured["config"].adaptive_mode is True
    assert captured["config"].cooldown_enabled is True
    assert captured["config"].cooldown_days == 21
    assert captured["base_template"]["cooldown_enabled"] is True
    assert captured["base_template"]["cooldown_days"] == 21
    assert captured["base_template"]["wfa"]["cooldown_enabled"] is True
    assert captured["base_template"]["wfa"]["cooldown_days"] == 21
    assert captured["base_template"]["grid_v2_prefer_compiled"] is False
    assert captured["base_template"]["grid_v2_max_cache_mb"] == 123.5


def test_walkforward_route_parses_ft_reject_policy_fields(client, monkeypatch, tmp_path):
    from ui import server_routes_run
    import core.walkforward_engine as walkforward_engine

    csv_path = tmp_path / "wfa_ft_reject_route.csv"
    csv_path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2026-01-01 00:00:00,1,1,1,1,1\n"
        "2026-01-02 00:00:00,1,1,1,1,1\n"
        "2026-01-03 00:00:00,1,1,1,1,1\n",
        encoding="utf-8",
    )

    df = pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.2],
            "high": [1.0, 1.1, 1.2],
            "low": [1.0, 1.1, 1.2],
            "close": [1.0, 1.1, 1.2],
            "volume": [1.0, 1.0, 1.0],
        },
        index=pd.to_datetime(
            ["2026-01-01 00:00:00", "2026-01-02 00:00:00", "2026-01-03 00:00:00"],
            utc=True,
        ),
    )

    captured = {}

    class DummyWalkForwardEngine:
        def __init__(self, config, base_template, optuna_settings, csv_file_path=None):
            captured["config"] = config
            captured["base_template"] = base_template
            captured["optuna_settings"] = optuna_settings
            captured["csv_file_path"] = csv_file_path

        def run_wf_optimization(self, _dataframe):
            return (
                SimpleNamespace(
                    total_windows=1,
                    stitched_oos=SimpleNamespace(
                        final_net_profit_pct=0.0,
                        max_drawdown_pct=0.0,
                        total_trades=0,
                        wfe=0.0,
                        oos_win_rate=0.0,
                    ),
                ),
                "study_route_ft_policy",
            )

    monkeypatch.setattr(server_routes_run, "_resolve_csv_path", lambda _raw: csv_path)
    monkeypatch.setattr(server_routes_run, "load_data", lambda _path: df)
    monkeypatch.setattr(walkforward_engine, "WalkForwardEngine", DummyWalkForwardEngine)

    payload = _build_minimal_optuna_payload()
    payload["primary_objective"] = "net_profit_pct"
    payload["postProcess"] = {
        "enabled": True,
        "ftPeriodDays": 14,
        "topK": 10,
        "sortMetric": "profit_degradation",
        "ftThresholdPct": 5.0,
        "ftRejectAction": "cooldown_reoptimize",
        "ftRejectCooldownDays": 7,
        "ftRejectMaxAttempts": 3,
        "ftRejectMinRemainingOosDays": 11,
    }

    response = client.post(
        "/api/walkforward",
        data={
            "strategy": "s01_trailing_ma",
            "csvPath": str(csv_path),
            "config": json.dumps(payload),
        },
    )

    assert response.status_code == 200
    assert captured["config"].post_process is not None
    assert captured["config"].post_process.ft_threshold_pct == 5.0
    assert captured["config"].post_process.ft_reject_action == "cooldown_reoptimize"
    assert captured["config"].post_process.ft_reject_cooldown_days == 7
    assert captured["config"].post_process.ft_reject_max_attempts == 3
    assert captured["config"].post_process.ft_reject_min_remaining_oos_days == 11
