"""Server run contracts."""

import json

import pytest

from strategies import get_strategy_config, list_strategies

from ._helpers import (
    _build_minimal_optuna_payload,
    _s03_regime_er_grid_preview_payload,
    _v2_runtime_diagnostic,
)


@pytest.mark.parametrize("endpoint", ["/api/optimize", "/api/walkforward"])
@pytest.mark.parametrize("strategy_id", ["s03_reversal_v10", "s03_reversal_v11_regime_er_b2"])
def test_grid_api_enforces_six_fast_objective_boundary(
    client,
    monkeypatch,
    tmp_path,
    endpoint,
    strategy_id,
):
    from ui import server_routes_run

    csv_path = tmp_path / "objective_boundary.csv"
    csv_path.write_text("placeholder", encoding="utf-8")
    payload = _s03_regime_er_grid_preview_payload(
        strategy_id=strategy_id,
        optimization_mode="grid",
    )
    seven = [
        "net_profit_pct",
        "max_drawdown_pct",
        "romad",
        "profit_factor",
        "win_rate",
        "sharpe_ratio",
        "sharpe_daily",
    ]
    payload.update(
        objectives=seven,
        primary_objective="sharpe_ratio",
        grid_fast_objectives=seven,
        grid_fast_primary_objective="sharpe_ratio",
    )
    monkeypatch.setattr(server_routes_run, "_resolve_csv_path", lambda _raw: csv_path)
    monkeypatch.setattr(
        server_routes_run,
        "_build_optimization_config",
        lambda *_args, **_kwargs: pytest.fail("seven objectives must fail before config construction"),
    )

    rejected = client.post(
        endpoint,
        data={"strategy": strategy_id, "csvPath": str(csv_path), "config": json.dumps(payload)},
    )

    assert rejected.status_code == 400
    assert "Maximum 6 objectives allowed" in rejected.get_data(as_text=True)

    accepted_objectives = seven[:4] + ["sharpe_ratio", "sqn"]
    payload.update(
        objectives=accepted_objectives,
        primary_objective="sqn",
        grid_fast_objectives=accepted_objectives,
        grid_fast_primary_objective="sqn",
    )
    monkeypatch.setattr(
        server_routes_run,
        "_build_optimization_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("six-objective boundary accepted")),
    )

    accepted = client.post(
        endpoint,
        data={"strategy": strategy_id, "csvPath": str(csv_path), "config": json.dumps(payload)},
    )

    assert accepted.status_code == 400
    assert "six-objective boundary accepted" in accepted.get_data(as_text=True)


@pytest.mark.parametrize("endpoint", ["/api/optimize", "/api/walkforward"])
@pytest.mark.parametrize(
    ("mode_present", "mode_value"),
    [
        (False, None),
        (True, None),
        (True, ""),
        (True, "   "),
        (True, "optuna"),
        (True, " OpTuNa "),
        (True, "other"),
    ],
)
def test_v2_optimizer_requests_require_explicit_grid_before_work(
    client, monkeypatch, endpoint, mode_present, mode_value
):
    from ui import server_routes_run

    payload = _s03_regime_er_grid_preview_payload(
        optimization_mode="grid",
        objectives=["net_profit_pct"],
        grid_fast_objectives=["net_profit_pct"],
    )
    if mode_present:
        payload["optimization_mode"] = mode_value
    else:
        payload.pop("optimization_mode", None)

    for name in (
        "_clear_cancelled_run",
        "_resolve_csv_path",
        "_build_optimization_config",
        "load_data",
        "run_optimization",
        "_set_optimization_state",
    ):
        monkeypatch.setattr(
            server_routes_run,
            name,
            lambda *_args, _name=name, **_kwargs: pytest.fail(
                f"{_name} must not run before V2 optimizer validation"
            ),
        )

    response = client.post(
        endpoint,
        data={
            "strategy": "s03_reversal_v11_regime_er_b2",
            "csvPath": "must-not-resolve.csv",
            "config": json.dumps(payload),
        },
    )

    assert response.status_code == 400
    diagnostic = _v2_runtime_diagnostic(response)
    assert diagnostic == {
        "severity": "error",
        "code": "V2_GRID_ONLY_OPTIMIZER",
        "strategy_id": "s03_reversal_v11_regime_er_b2",
        "path": "optimization_mode",
        "variant": None,
        "message": (
            "s03_reversal_v11_regime_er_b2: Backtester V2 supports Grid "
            "optimization only; set optimization_mode='grid'."
        ),
    }


def test_v2_grid_mode_normalization_and_optimizer_error_precedence():
    from ui import server_services

    context = server_services._resolve_strategy_context(
        [("strategy_id", True, "s06_r_trend_v02_b2")]
    )
    payload = {
        "optimization_mode": " GrId ",
        "fixed_params": {"dateFilter": False},
    }
    normalized, _runtime = server_services._normalize_v2_optimizer_payload(
        context,
        payload,
        warmup_members=[],
    )
    assert normalized["optimization_mode"] == "grid"
    assert payload == {
        "optimization_mode": " GrId ",
        "fixed_params": {"dateFilter": False},
    }

    with pytest.raises(Exception) as excinfo:
        server_services._normalize_v2_optimizer_payload(
            context,
            {
                "optimization_mode": "optuna",
                "enabled_params": {"warmupBars": True},
                "fixed_params": {"dateFilter": False},
            },
            warmup_members=[],
        )
    diagnostic = excinfo.value.diagnostics[0].to_dict()
    assert diagnostic["code"] == "V2_GRID_ONLY_OPTIMIZER"
    assert diagnostic["path"] == "optimization_mode"


def test_v2_optimize_normalizes_explicit_grid_before_config_construction(
    client, monkeypatch
):
    from ui import server_routes_run

    payload = _s03_regime_er_grid_preview_payload(
        optimization_mode=" GrId ",
        objectives=["net_profit_pct"],
        grid_fast_objectives=["net_profit_pct"],
    )
    captured = {}

    monkeypatch.setattr(
        server_routes_run,
        "_resolve_csv_path",
        lambda *_args, **_kwargs: "isolated.csv",
    )

    def capture_build(_data_source, config_payload, *_args, **_kwargs):
        captured.update(config_payload)
        raise ValueError("configuration construction reached")

    monkeypatch.setattr(server_routes_run, "_build_optimization_config", capture_build)

    response = client.post(
        "/api/optimize",
        data={
            "strategy": "s03_reversal_v11_regime_er_b2",
            "csvPath": "isolated.csv",
            "config": json.dumps(payload),
        },
    )

    assert response.status_code == 400
    assert response.get_data(as_text=True) == "configuration construction reached"
    assert captured["optimization_mode"] == "grid"


@pytest.mark.parametrize(
    "strategy_id",
    [
        item["id"]
        for item in list_strategies()
        if str(get_strategy_config(item["id"]).get("engine", "v1")).strip().lower()
        == "v2"
    ],
)
def test_every_registered_v2_strategy_rejects_new_optuna_optimize_request(
    client, strategy_id
):
    response = client.post(
        "/api/optimize",
        data={
            "strategy": strategy_id,
            "config": json.dumps(
                {
                    "optimization_mode": "optuna",
                    "fixed_params": {"dateFilter": False},
                }
            ),
        },
    )
    assert response.status_code == 400
    diagnostic = _v2_runtime_diagnostic(response)
    assert diagnostic["code"] == "V2_GRID_ONLY_OPTIMIZER"
    assert diagnostic["strategy_id"] == strategy_id


@pytest.mark.parametrize("mode_payload", [{}, {"optimization_mode": "   "}])
def test_v1_missing_and_blank_optimizer_mode_still_default_to_optuna(mode_payload):
    from ui import server_services

    config = server_services._build_optimization_config(
        "isolated.csv",
        {
            **mode_payload,
            "enabled_params": {},
            "param_ranges": {},
            "param_types": {},
            "fixed_params": {},
        },
        1,
        "s03_reversal_v10",
        1000,
    )
    assert config.optimization_mode == "optuna"


def test_v1_explicit_null_optimizer_mode_retains_existing_rejection():
    from ui import server_services

    with pytest.raises(ValueError, match="Unsupported optimization mode: none"):
        server_services._build_optimization_config(
            "isolated.csv",
            {
                "optimization_mode": None,
                "enabled_params": {},
                "param_ranges": {},
                "param_types": {},
                "fixed_params": {},
            },
            1,
            "s03_reversal_v10",
            1000,
        )


def test_build_optimization_config_requires_explicit_strategy_without_fallback(monkeypatch):
    import strategies
    from ui import server as server_module

    monkeypatch.setattr(
        strategies,
        "list_strategies",
        lambda: pytest.fail("strategy discovery fallback must not run"),
    )
    payload = _s03_regime_er_grid_preview_payload(strategy="s03_reversal_v10")
    with pytest.raises(ValueError, match="Strategy ID is required"):
        server_module._build_optimization_config(
            "dummy.csv",
            payload,
            strategy_id=None,
        )


def test_optuna_sanitize_defaults():
    from ui import server as server_module

    payload = _build_minimal_optuna_payload()
    config = server_module._build_optimization_config(
        "dummy.csv",
        payload,
        worker_processes=1,
        strategy_id="s01_trailing_ma",
        warmup_bars=1000,
    )
    assert config.sanitize_enabled is True
    assert config.sanitize_trades_threshold == 0


def test_optuna_coverage_mode_parsed():
    from ui import server as server_module

    payload = _build_minimal_optuna_payload()
    payload["coverage_mode"] = True
    config = server_module._build_optimization_config(
        "dummy.csv",
        payload,
        worker_processes=1,
        strategy_id="s01_trailing_ma",
        warmup_bars=1000,
    )
    assert config.coverage_mode is True


def test_optuna_trials_log_parsed():
    from ui import server as server_module

    payload = _build_minimal_optuna_payload()
    payload["trials_log"] = True
    config = server_module._build_optimization_config(
        "dummy.csv",
        payload,
        worker_processes=1,
        strategy_id="s01_trailing_ma",
        warmup_bars=1000,
    )
    assert config.trials_log is True


def test_optuna_dispatcher_controls_parsed():
    from ui import server as server_module

    payload = _build_minimal_optuna_payload()
    payload["dispatcher_batch_result_processing"] = False
    payload["dispatcher_soft_duplicate_cycle_limit_enabled"] = False
    payload["dispatcher_duplicate_cycle_limit"] = 42

    config = server_module._build_optimization_config(
        "dummy.csv",
        payload,
        worker_processes=1,
        strategy_id="s01_trailing_ma",
        warmup_bars=1000,
    )

    assert config.dispatcher_batch_result_processing is False
    assert config.dispatcher_soft_duplicate_cycle_limit_enabled is False
    assert config.dispatcher_duplicate_cycle_limit == 42


def test_optuna_save_study_payload_is_ignored(caplog):
    from ui import server as server_module

    payload = _build_minimal_optuna_payload()
    payload["optuna_save_study"] = True

    with caplog.at_level("WARNING"):
        config = server_module._build_optimization_config(
            "dummy.csv",
            payload,
            worker_processes=1,
            strategy_id="s01_trailing_ma",
            warmup_bars=1000,
        )

    assert not hasattr(config, "optuna_save_study")
    assert any("optuna_save_study" in record.message for record in caplog.records)


def test_optuna_score_config_migrates_legacy_consistency_bounds():
    from ui import server as server_module

    payload = _build_minimal_optuna_payload()
    payload["score_config"] = {
        "enabled_metrics": {"consistency": True},
        "weights": {"consistency": 1.0},
        "metric_bounds": {"consistency": {"min": 0.0, "max": 100.0}},
    }

    config = server_module._build_optimization_config(
        "dummy.csv",
        payload,
        worker_processes=1,
        strategy_id="s01_trailing_ma",
        warmup_bars=1000,
    )

    assert config.score_config["metric_bounds"]["consistency"] == {"min": -1.0, "max": 1.0}


def test_optimize_cancelled_run_cleans_up_saved_study(client, monkeypatch, tmp_path):
    from ui import server_routes_run

    csv_path = tmp_path / "opt_cancel.csv"
    csv_path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2026-01-01 00:00:00,1,1,1,1,1\n",
        encoding="utf-8",
    )

    deleted_studies = []

    monkeypatch.setattr(server_routes_run, "_resolve_csv_path", lambda _raw: csv_path)
    monkeypatch.setattr(server_routes_run, "run_optimization", lambda _cfg: ([], "study_cancel_opt"))
    monkeypatch.setattr(server_routes_run, "_is_run_cancelled", lambda run_id: run_id == "run_cancel_opt")
    monkeypatch.setattr(
        server_routes_run,
        "delete_study",
        lambda study_id: deleted_studies.append(study_id) or True,
    )

    payload = _build_minimal_optuna_payload()
    payload["primary_objective"] = "net_profit_pct"

    response = client.post(
        "/api/optimize",
        data={
            "strategy": "s01_trailing_ma",
            "csvPath": str(csv_path),
            "runId": "run_cancel_opt",
            "config": json.dumps(payload),
        },
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "cancelled"
    assert data["run_id"] == "run_cancel_opt"
    assert data["study_id"] is None
    assert deleted_studies == ["study_cancel_opt"]


@pytest.mark.parametrize("threshold", [-1, "bad"])
def test_optuna_sanitize_threshold_validation(threshold):
    from ui import server as server_module

    payload = _build_minimal_optuna_payload()
    payload["sanitize_trades_threshold"] = threshold

    with pytest.raises(ValueError):
        server_module._build_optimization_config(
            "dummy.csv",
            payload,
            worker_processes=1,
            strategy_id="s01_trailing_ma",
            warmup_bars=1000,
        )
