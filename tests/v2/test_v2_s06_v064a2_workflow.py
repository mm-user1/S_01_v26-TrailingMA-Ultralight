from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import storage
from core.walkforward_engine import WFConfig, WalkForwardEngine
from strategies import get_strategy, get_strategy_config
from strategies.s06_r_trend_v06_4_a2_b2 import strategy
from ui import server_services
from ui.server import app


STRATEGY_ID = "s06_r_trend_v06_4_a2_b2"


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def _workflow_frame(periods: int = 480) -> pd.DataFrame:
    x = np.arange(periods, dtype=float)
    close = 100.0 + np.sin(x / 4.0) * 7.0 + np.sin(x / 17.0) * 2.0
    return pd.DataFrame(
        {
            "Open": close + np.cos(x / 6.0) * 0.15,
            "High": close + 0.9,
            "Low": close - 0.9,
            "Close": close,
            "Volume": np.full(periods, 1000.0),
        },
        index=pd.date_range("2025-01-01", periods=periods, freq="30min", tz="UTC"),
    )


def _fixed_params() -> dict:
    params = strategy.normalized_params(
        {
            "dateFilter": False,
            "entryMode": "Trend @ Square",
            "trailMode": "Fixed-AF SAR",
            "trailRR": 1.5,
            "sarSpeed": 0.015,
            "initialCapital": 100.0,
            "commissionPct": 0.0,
        }
    )
    for name in ("start", "end", "warmupBars"):
        params.pop(name, None)
    return params


def _grid_request() -> dict:
    return {
        "strategy_id": STRATEGY_ID,
        "optimization_mode": "grid",
        "enabled_params": {"stopLP": True},
        "param_ranges": {},
        "param_types": {},
        "fixed_params": _fixed_params(),
        "grid_enabled_modes": ["fixed_af_sar"],
        "objectives": ["net_profit_pct"],
        "grid_fast_objectives": ["net_profit_pct"],
        "primary_objective": "net_profit_pct",
        "grid_top_candidates": 1,
        "grid_diversity_enabled": False,
        "grid_slow_refinement_enabled": False,
        "grid_v2_prefer_compiled": False,
        "worker_processes": 1,
        "warmupBars": 120,
    }


def test_new_strategy_discovery_dynamic_config_and_grid_preview(client):
    assert get_strategy(STRATEGY_ID).STRATEGY_ID == STRATEGY_ID
    assert get_strategy_config(STRATEGY_ID)["version"] == "v06-4-a2-b2"

    config_response = client.get(f"/api/strategy/{STRATEGY_ID}/config")
    assert config_response.status_code == 200
    config = config_response.get_json()
    assert config["id"] == STRATEGY_ID
    assert config["parameters"]["trailMode"]["options"] == [
        "Off (Bracket)", "R Trail", "Chandelier Exit", "Fixed-AF SAR"
    ]

    preview_response = client.post("/api/grid/preview", json=_grid_request())
    assert preview_response.status_code == 200
    preview = preview_response.get_json()["preview"]
    assert preview["full_candidate_count"] == 2
    assert preview["planned_candidate_count"] == 2
    assert [row["mode"] for row in preview["modes"]] == ["fixed_af_sar"]


def test_bounded_fixed_grid_wfa_queue_storage_results_and_analytics_round_trip(
    client, monkeypatch, tmp_path
):
    queue_file = tmp_path / "s06_v064a2_queue.json"
    monkeypatch.setattr(server_services, "_queue_storage_file_path", lambda: queue_file)
    request = _grid_request()
    queue_payload = {
        "items": [
            {
                "id": "s06_v064a2_workflow",
                "index": 1,
                "label": "#1 S06 v06-4-A2 fixed WFA",
                "strategyId": STRATEGY_ID,
                "mode": "wfa",
                "warmupBars": 120,
                "sources": [{"type": "path", "path": r"C:\isolated_s06_v064a2.csv"}],
                "config": request,
            }
        ],
        "nextIndex": 2,
        "runtime": {"active": False, "updatedAt": 0},
    }
    assert client.put("/api/queue", json=queue_payload).status_code == 200
    queued = client.get("/api/queue").get_json()["items"][0]
    queued_fixed = queued["config"]["fixed_params"]
    assert queued["strategyId"] == STRATEGY_ID
    assert queued_fixed["entryMode"] == "Trend @ Square"
    assert queued_fixed["trailMode"] == "Fixed-AF SAR"
    assert queued_fixed["trailRR"] == 1.5
    assert queued_fixed["sarSpeed"] == 0.015

    base_template = {
        **request,
        "risk_per_trade_pct": 2.0,
        "contract_size": 0.01,
        "commission_rate": 0.0,
        "filter_min_profit": False,
        "min_profit_threshold": 0.0,
        "constraints": [],
        "score_config": {},
    }
    engine = WalkForwardEngine(
        WFConfig(
            strategy_id=STRATEGY_ID,
            is_period_days=4,
            oos_period_days=2,
            warmup_bars=120,
        ),
        base_template,
        {},
        csv_file_path="isolated_s06_v064a2_30m.csv",
    )
    result, study_id = engine.run_wf_optimization(_workflow_frame())

    assert study_id is not None and result.windows
    assert all(
        window.module_status["grid_v2"]["candidate_count"] == 2
        for window in result.windows
    )
    loaded = storage.load_study_from_db(study_id)
    assert loaded is not None
    stored_fixed = loaded["study"]["config_json"]["fixed_params"]
    assert (stored_fixed["entryMode"], stored_fixed["trailMode"]) == (
        "Trend @ Square", "Fixed-AF SAR"
    )
    assert (stored_fixed["trailRR"], stored_fixed["sarSpeed"]) == (1.5, 0.015)

    results_response = client.get(f"/api/studies/{study_id}")
    assert results_response.status_code == 200
    assert results_response.get_json()["study"]["strategy_id"] == STRATEGY_ID

    analytics_response = client.get("/api/analytics/summary")
    assert analytics_response.status_code == 200
    analytics_rows = analytics_response.get_json()["studies"]
    assert any(row["study_id"] == study_id for row in analytics_rows)
