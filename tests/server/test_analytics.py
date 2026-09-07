"""Server analytics contracts."""

import json
import uuid

import pytest

from core.metrics import _calculate_r2_consistency
from core.storage import delete_study, get_active_db_name, get_db_connection

from ._helpers import _grid_sidebar_config, _temporary_active_db


def _insert_analytics_study(
    *,
    study_id: str,
    study_name: str,
    strategy_id: str = "s01_trailing_ma",
    strategy_version: str | None = "2.0",
    optimization_mode: str = "wfa",
    csv_file_name: str | None = "OKX_LINKUSDT.P, 15 2025.05.01-2025.11.20.csv",
    adaptive_mode: int | None = 1,
    is_period_days: int | None = 60,
    config_json: dict | None = None,
    dataset_start_date: str | None = "2025-01-01",
    dataset_end_date: str | None = "2025-01-31",
    stitched_oos_net_profit_pct: float | None = 5.0,
    stitched_oos_max_drawdown_pct: float | None = 2.0,
    stitched_oos_total_trades: int | None = 100,
    stitched_oos_winning_trades: int | None = 55,
    best_value: float | None = 1.2,
    profitable_windows: int | None = 3,
    total_windows: int | None = 5,
    stitched_oos_win_rate: float | None = 60.0,
    median_window_profit: float | None = 1.0,
    median_window_wr: float | None = 58.0,
    stitched_oos_equity_curve: list | None = None,
    stitched_oos_timestamps_json: list | None = None,
    created_at: str | None = None,
    completed_at: str | None = None,
):
    config_payload = config_json
    if config_payload is None:
        config_payload = {"wfa": {"oos_period_days": 30}}

    columns = [
        "study_id",
        "study_name",
        "strategy_id",
        "strategy_version",
        "optimization_mode",
        "csv_file_name",
        "adaptive_mode",
        "is_period_days",
        "config_json",
        "dataset_start_date",
        "dataset_end_date",
        "stitched_oos_net_profit_pct",
        "stitched_oos_max_drawdown_pct",
        "stitched_oos_total_trades",
        "stitched_oos_winning_trades",
        "best_value",
        "profitable_windows",
        "total_windows",
        "stitched_oos_win_rate",
        "median_window_profit",
        "median_window_wr",
        "stitched_oos_equity_curve",
        "stitched_oos_timestamps_json",
    ]
    values = [
        study_id,
        study_name,
        strategy_id,
        strategy_version,
        optimization_mode,
        csv_file_name,
        adaptive_mode,
        is_period_days,
        json.dumps(config_payload) if isinstance(config_payload, dict) else config_payload,
        dataset_start_date,
        dataset_end_date,
        stitched_oos_net_profit_pct,
        stitched_oos_max_drawdown_pct,
        stitched_oos_total_trades,
        stitched_oos_winning_trades,
        best_value,
        profitable_windows,
        total_windows,
        stitched_oos_win_rate,
        median_window_profit,
        median_window_wr,
        json.dumps(stitched_oos_equity_curve)
        if isinstance(stitched_oos_equity_curve, list)
        else stitched_oos_equity_curve,
        json.dumps(stitched_oos_timestamps_json)
        if isinstance(stitched_oos_timestamps_json, list)
        else stitched_oos_timestamps_json,
    ]

    if created_at is not None:
        columns.append("created_at")
        values.append(created_at)
    if completed_at is not None:
        columns.append("completed_at")
        values.append(completed_at)

    with get_db_connection() as conn:
        conn.execute(
            f"INSERT INTO studies ({', '.join(columns)}) VALUES ({', '.join(['?'] * len(values))})",
            tuple(values),
        )
        conn.commit()


def _insert_analytics_wfa_window(
    *,
    study_id: str,
    window_number: int,
    window_id: str | None = None,
    oos_start_ts: str | None = None,
    oos_start_date: str | None = None,
    is_end_ts: str | None = None,
    is_end_date: str | None = None,
):
    resolved_window_id = window_id or f"{study_id}_w{window_number}"
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO wfa_windows (
                window_id,
                study_id,
                window_number,
                best_params_json,
                oos_start_ts,
                oos_start_date,
                is_end_ts,
                is_end_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                resolved_window_id,
                study_id,
                int(window_number),
                json.dumps({}),
                oos_start_ts,
                oos_start_date,
                is_end_ts,
                is_end_date,
            ),
        )
        conn.commit()


def test_analytics_page_renders(client):
    response = client.get("/analytics")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Analytics" in body
    assert "Group Dates" in body


def test_analytics_window_boundaries_endpoint_success(client):
    with _temporary_active_db(f"analytics_boundaries_{uuid.uuid4().hex[:8]}"):
        study_id = "wfa_boundaries_1"
        _insert_analytics_study(
            study_id=study_id,
            study_name="WFA_BOUNDARIES_1",
            optimization_mode="wfa",
        )
        _insert_analytics_wfa_window(
            study_id=study_id,
            window_number=1,
            oos_start_date="2025-01-15",
        )
        _insert_analytics_wfa_window(
            study_id=study_id,
            window_number=2,
            oos_start_ts="2025-02-01T00:00:00+00:00",
        )
        _insert_analytics_wfa_window(
            study_id=study_id,
            window_number=3,
            is_end_date="2025-03-20",
        )
        _insert_analytics_wfa_window(
            study_id=study_id,
            window_number=4,
        )

        response = client.get(f"/api/analytics/studies/{study_id}/window-boundaries")
        assert response.status_code == 200
        payload = response.get_json()

        assert payload["study_id"] == study_id
        assert payload["boundaries"] == [
            {
                "window_id": f"{study_id}_w1",
                "window_number": 1,
                "time": "2025-01-15",
                "label": "W1",
            },
            {
                "window_id": f"{study_id}_w2",
                "window_number": 2,
                "time": "2025-02-01T00:00:00+00:00",
                "label": "W2",
            },
            {
                "window_id": f"{study_id}_w3",
                "window_number": 3,
                "time": "2025-03-20",
                "label": "W3",
            },
        ]


def test_analytics_window_boundaries_endpoint_rejects_non_wfa(client):
    with _temporary_active_db(f"analytics_boundaries_non_wfa_{uuid.uuid4().hex[:8]}"):
        study_id = "optuna_boundaries_1"
        _insert_analytics_study(
            study_id=study_id,
            study_name="OPTUNA_BOUNDARIES_1",
            optimization_mode="optuna",
        )

        response = client.get(f"/api/analytics/studies/{study_id}/window-boundaries")
        assert response.status_code == 400
        assert "only for WFA studies" in response.get_json()["error"]


def test_analytics_window_boundaries_endpoint_returns_404_for_missing_study(client):
    response = client.get("/api/analytics/studies/missing_study/window-boundaries")
    assert response.status_code == 404
    assert response.get_json()["error"] == "Study not found."


def test_analytics_equity_endpoint_success(client):
    with _temporary_active_db(f"analytics_equity_{uuid.uuid4().hex[:8]}"):
        _insert_analytics_study(
            study_id="wfa_eq_1",
            study_name="WFA_EQ_1",
            optimization_mode="wfa",
            stitched_oos_equity_curve=[100.0, 120.0],
            stitched_oos_timestamps_json=[
                "2025-01-01T00:00:00+00:00",
                "2025-02-15T00:00:00+00:00",
            ],
        )
        _insert_analytics_study(
            study_id="wfa_eq_2",
            study_name="WFA_EQ_2",
            optimization_mode="wfa",
            stitched_oos_equity_curve=[100.0, 80.0],
            stitched_oos_timestamps_json=[
                "2025-01-01T00:00:00+00:00",
                "2025-02-15T00:00:00+00:00",
            ],
        )

        response = client.post(
            "/api/analytics/equity",
            json={"study_ids": ["wfa_eq_1", "wfa_eq_2"]},
        )
        assert response.status_code == 200
        payload = response.get_json()

        assert isinstance(payload["curve"], list)
        assert len(payload["curve"]) == len(payload["timestamps"])
        assert payload["studies_used"] == 2
        assert payload["selected_count"] == 2
        assert payload["missing_study_ids"] == []
        assert payload["profit_pct"] == pytest.approx(0.0, abs=1e-6)
        assert payload["return_profile"] == {
            "stems": [20.0, -20.0],
            "source_count": 2,
            "display_count": 2,
            "is_binned": False,
        }


def test_analytics_equity_endpoint_rejects_missing_payload(client):
    response = client.post("/api/analytics/equity")
    assert response.status_code == 400
    assert "Expected JSON payload." in response.get_json()["error"]


def test_analytics_equity_endpoint_rejects_non_array_study_ids(client):
    response = client.post(
        "/api/analytics/equity",
        json={"study_ids": "not-array"},
    )
    assert response.status_code == 400
    assert "study_ids must be an array." in response.get_json()["error"]


def test_analytics_equity_endpoint_rejects_empty_study_ids(client):
    response = client.post(
        "/api/analytics/equity",
        json={"study_ids": []},
    )
    assert response.status_code == 400
    assert "non-empty array" in response.get_json()["error"]


def test_analytics_equity_endpoint_rejects_study_ids_over_cap(client):
    response = client.post(
        "/api/analytics/equity",
        json={"study_ids": [f"id_{i}" for i in range(5001)]},
    )
    assert response.status_code == 400
    assert "Maximum allowed is 5000" in response.get_json()["error"]


def test_analytics_equity_endpoint_no_overlap_returns_warning(client):
    with _temporary_active_db(f"analytics_equity_no_overlap_{uuid.uuid4().hex[:8]}"):
        _insert_analytics_study(
            study_id="wfa_eq_no_1",
            study_name="WFA_EQ_NO_1",
            optimization_mode="wfa",
            stitched_oos_equity_curve=[100.0, 110.0],
            stitched_oos_timestamps_json=[
                "2025-01-01T00:00:00+00:00",
                "2025-01-02T00:00:00+00:00",
            ],
        )
        _insert_analytics_study(
            study_id="wfa_eq_no_2",
            study_name="WFA_EQ_NO_2",
            optimization_mode="wfa",
            stitched_oos_equity_curve=[100.0, 95.0],
            stitched_oos_timestamps_json=[
                "2025-01-10T00:00:00+00:00",
                "2025-01-11T00:00:00+00:00",
            ],
        )

        response = client.post(
            "/api/analytics/equity",
            json={"study_ids": ["wfa_eq_no_1", "wfa_eq_no_2"]},
        )
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["curve"] is None
        assert payload["warning"] == "Selected studies have no overlapping time period."
        assert payload["return_profile"] == {
            "stems": [],
            "source_count": 0,
            "display_count": 0,
            "is_binned": False,
        }


def test_analytics_equity_batch_endpoint_success(client):
    with _temporary_active_db(f"analytics_equity_batch_{uuid.uuid4().hex[:8]}"):
        _insert_analytics_study(
            study_id="wfa_eq_b1",
            study_name="WFA_EQ_B1",
            optimization_mode="wfa",
            stitched_oos_equity_curve=[100.0, 105.0],
            stitched_oos_timestamps_json=[
                "2025-01-01T00:00:00+00:00",
                "2025-02-10T00:00:00+00:00",
            ],
        )
        _insert_analytics_study(
            study_id="wfa_eq_b2",
            study_name="WFA_EQ_B2",
            optimization_mode="wfa",
            stitched_oos_equity_curve=[100.0, 95.0],
            stitched_oos_timestamps_json=[
                "2025-01-01T00:00:00+00:00",
                "2025-02-10T00:00:00+00:00",
            ],
        )

        response = client.post(
            "/api/analytics/equity/batch",
            json={
                "groups": [
                    {"group_id": "all", "study_ids": ["wfa_eq_b1", "wfa_eq_b2"]},
                    {"group_id": "subset", "study_ids": ["wfa_eq_b1"]},
                    {"group_id": "empty", "study_ids": []},
                ]
            },
        )
        assert response.status_code == 200
        payload = response.get_json()
        assert isinstance(payload["results"], list)
        by_id = {item["group_id"]: item for item in payload["results"]}
        assert set(by_id.keys()) == {"all", "subset", "empty"}
        assert by_id["all"]["studies_used"] == 2
        assert by_id["subset"]["studies_used"] == 1
        assert by_id["empty"]["curve"] is None
        assert "return_profile" not in by_id["all"]
        assert "return_profile" not in by_id["subset"]
        assert "return_profile" not in by_id["empty"]


def test_analytics_summary_empty_db_returns_expected_message(client):
    with _temporary_active_db(f"analytics_empty_{uuid.uuid4().hex[:8]}"):
        response = client.get("/api/analytics/summary")
        assert response.status_code == 200
        payload = response.get_json()

        assert payload["studies"] == []
        assert payload["db_name"] == get_active_db_name()
        assert payload["research_info"]["total_studies"] == 0
        assert payload["research_info"]["wfa_studies"] == 0
        assert payload["research_info"]["message"] == "No WFA studies found in this database."


def test_analytics_summary_optuna_only_returns_expected_message(client):
    with _temporary_active_db(f"analytics_optuna_only_{uuid.uuid4().hex[:8]}"):
        _insert_analytics_study(
            study_id="optuna_only_1",
            study_name="OPTUNA_ONLY_1",
            optimization_mode="optuna",
            config_json={},
        )

        response = client.get("/api/analytics/summary")
        assert response.status_code == 200
        payload = response.get_json()

        assert payload["studies"] == []
        assert payload["research_info"]["total_studies"] == 1
        assert payload["research_info"]["wfa_studies"] == 0
        assert payload["research_info"]["message"] == (
            "Analytics requires WFA studies. This database contains only Optuna studies."
        )


@pytest.mark.parametrize("csv_prefix", ["", "C:\\data_dir\\", "/data_dir/", "C:\\data_dir/mixed\\"])
def test_analytics_summary_wfa_phase1_contract(client, csv_prefix):
    with _temporary_active_db(f"analytics_wfa_{uuid.uuid4().hex[:8]}"):
        _insert_analytics_study(
            study_id="wfa_a1",
            study_name="WFA_A1",
            strategy_id="s01_trailing_ma",
            strategy_version="2.1",
            optimization_mode="wfa",
            csv_file_name=csv_prefix + "OKX_LINKUSDT.P, 15 2025.05.01-2025.11.20.csv",
            adaptive_mode=1,
            is_period_days=60,
            config_json={"wfa": {"oos_period_days": 30}},
            dataset_start_date="2025-01-01",
            dataset_end_date="2025-01-31",
            stitched_oos_net_profit_pct=10.0,
            stitched_oos_max_drawdown_pct=4.0,
            stitched_oos_total_trades=120,
            stitched_oos_winning_trades=67,
            best_value=45.0,
            profitable_windows=4,
            total_windows=5,
            stitched_oos_win_rate=80.0,
            median_window_profit=2.0,
            median_window_wr=55.0,
            stitched_oos_equity_curve=[100.0, 101.5, 110.0],
            stitched_oos_timestamps_json=[
                "2025-01-01T00:00:00+00:00",
                "2025-01-10T00:00:00+00:00",
                "2025-01-31T00:00:00+00:00",
            ],
        )
        _insert_analytics_study(
            study_id="wfa_a2",
            study_name="WFA_A2",
            strategy_id="s02_breakout",
            strategy_version="v3.0",
            optimization_mode="wfa",
            csv_file_name="OKX_LINKUSDT.P, 15 2025.05.01-2025.11.20.csv",
            adaptive_mode=0,
            is_period_days=None,
            config_json={"wfa": {"oos_period_days": 30}},
            dataset_start_date="2025-01-01",
            dataset_end_date="2025-01-31",
            stitched_oos_net_profit_pct=-3.0,
            stitched_oos_max_drawdown_pct=2.5,
            stitched_oos_total_trades=80,
            stitched_oos_winning_trades=35,
            best_value=20.0,
            profitable_windows=1,
            total_windows=5,
            stitched_oos_win_rate=20.0,
            median_window_profit=-0.5,
            median_window_wr=48.0,
            stitched_oos_equity_curve=[100.0, 99.2],
            stitched_oos_timestamps_json=["2025-01-01T00:00:00+00:00"],
        )
        _insert_analytics_study(
            study_id="wfa_b1",
            study_name="WFA_B1",
            strategy_id="custom_strategy",
            strategy_version=None,
            optimization_mode="WFA",
            csv_file_name="OKX_BTCUSDT.P, 1h 2025.05.01-2025.11.20.csv",
            adaptive_mode=None,
            is_period_days=None,
            config_json={},
            dataset_start_date="2025-02-01",
            dataset_end_date="2025-02-28",
            stitched_oos_net_profit_pct=4.5,
            stitched_oos_max_drawdown_pct=1.1,
            stitched_oos_total_trades=40,
            stitched_oos_winning_trades=20,
            best_value=None,
            profitable_windows=0,
            total_windows=0,
            stitched_oos_win_rate=None,
            median_window_profit=0.0,
            median_window_wr=None,
            stitched_oos_equity_curve=None,
            stitched_oos_timestamps_json=None,
        )
        _insert_analytics_study(
            study_id="optuna_aux",
            study_name="OPTUNA_AUX",
            optimization_mode="optuna",
            config_json={},
        )

        response = client.get("/api/analytics/summary")
        assert response.status_code == 200
        payload = response.get_json()

        assert payload["db_name"] == get_active_db_name()
        assert payload["research_info"]["total_studies"] == 4
        assert payload["research_info"]["wfa_studies"] == 3
        assert "message" not in payload["research_info"]

        studies = payload["studies"]
        assert [row["study_id"] for row in studies] == ["wfa_a1", "wfa_a2", "wfa_b1"]

        first = studies[0]
        expected_first_full = _calculate_r2_consistency([100.0, 101.5, 110.0])
        assert first["strategy"] == "S01 v2.1"
        assert first["symbol"] == "LINKUSDT.P"
        assert first["tf"] == "15m"
        assert first["wfa_mode"] == "Adaptive"
        assert first["is_oos"] == "60/30"
        assert first["has_equity_curve"] is True
        assert first["equity_point_count"] == 3
        assert first["equity_start_ts"] == "2025-01-01T00:00:00+00:00"
        assert first["equity_end_ts"] == "2025-01-31T00:00:00+00:00"
        assert first["oos_span_days_exact"] == pytest.approx(30.0)
        assert first["consistency_full"] == pytest.approx(expected_first_full, abs=1e-6)
        assert first["consistency_recent"] is None
        assert "equity_curve" not in first
        assert "equity_timestamps" not in first

        second = studies[1]
        assert second["strategy"] == "S02 v3.0"
        assert second["wfa_mode"] == "Fixed"
        assert second["is_oos"] == "?/30"
        assert second["has_equity_curve"] is False
        assert second["equity_point_count"] == 0
        assert second["equity_start_ts"] is None
        assert second["equity_end_ts"] is None
        assert second["oos_span_days_exact"] is None

        third = studies[2]
        assert third["strategy"] == "custom_strategy"
        assert third["symbol"] == "BTCUSDT.P"
        assert third["tf"] == "1h"
        assert third["wfa_mode"] == "Unknown"
        assert third["is_oos"] == "N/A"

        info = payload["research_info"]
        assert info["strategies"] == ["S01 v2.1", "S02 v3.0", "custom_strategy"]
        assert info["symbols"] == ["BTCUSDT.P", "LINKUSDT.P"]
        assert info["timeframes"] == ["15m", "1h"]
        assert info["wfa_modes"] == ["Fixed", "Adaptive", "Unknown"]
        assert info["is_oos_periods"] == ["60/30", "?/30", "N/A"]
        assert info["data_periods"] == [
            {"start": "2025-01-01", "end": "2025-01-31", "days": 30, "count": 2},
            {"start": "2025-02-01", "end": "2025-02-28", "days": 27, "count": 1},
        ]


def test_analytics_summary_includes_study_name_and_timestamps(client):
    with _temporary_active_db(f"analytics_timestamps_{uuid.uuid4().hex[:8]}"):
        _insert_analytics_study(
            study_id="wfa_ts_1",
            study_name="S01_OKX_SOLUSDT.P, 1h 2025.01.01-2025.01.31_WFA (3)",
            dataset_start_date="2025-01-01",
            dataset_end_date="2025-01-31",
            created_at="2026-02-20 12:00:00",
            completed_at="2026-02-20 12:05:00",
        )
        _insert_analytics_study(
            study_id="wfa_ts_2",
            study_name="S03_OKX_ETHUSDT.P, 240 2025.01.01-2025.01.31_WFA",
            dataset_start_date="2025-01-01",
            dataset_end_date="2025-01-31",
            created_at=None,
            completed_at="2026-02-21 09:30:00",
        )
        with get_db_connection() as conn:
            conn.execute("UPDATE studies SET created_at = NULL WHERE study_id = ?", ("wfa_ts_2",))
            conn.commit()

        response = client.get("/api/analytics/summary")
        assert response.status_code == 200
        payload = response.get_json()
        studies = payload["studies"]
        by_id = {row["study_id"]: row for row in studies}

        first = by_id["wfa_ts_1"]
        assert first["study_name"] == "S01_OKX_SOLUSDT.P, 1h 2025.01.01-2025.01.31_WFA (3)"
        assert first["created_at"] == "2026-02-20 12:00:00"
        assert first["completed_at"] == "2026-02-20 12:05:00"
        assert isinstance(first["created_at_epoch"], int)
        assert isinstance(first["completed_at_epoch"], int)
        assert first["created_at_epoch"] > 0
        assert first["completed_at_epoch"] > 0
        assert first["wfa_settings"]["run_time_seconds"] == 300
        assert "equity_curve" not in first
        assert "equity_timestamps" not in first

        second = by_id["wfa_ts_2"]
        assert second["study_name"] == "S03_OKX_ETHUSDT.P, 240 2025.01.01-2025.01.31_WFA"
        assert second["created_at"] is None
        assert second["created_at_epoch"] is None
        assert second["completed_at"] == "2026-02-21 09:30:00"
        assert isinstance(second["completed_at_epoch"], int)
        assert second["completed_at_epoch"] > 0
        assert second["wfa_settings"]["run_time_seconds"] is None


def test_analytics_summary_distinguishes_calendar_month_periods(client):
    with _temporary_active_db(f"analytics_month_periods_{uuid.uuid4().hex[:8]}"):
        _insert_analytics_study(
            study_id="wfa_months",
            study_name="Calendar Months",
            adaptive_mode=0,
            is_period_days=None,
            config_json={
                "wfa": {
                    "period_unit": "months",
                    "is_period_months": 2,
                    "oos_period_months": 1,
                    "adaptive_mode": False,
                }
            },
        )
        _insert_analytics_study(
            study_id="wfa_days",
            study_name="Calendar Days",
            adaptive_mode=0,
            is_period_days=2,
            config_json={"wfa": {"is_period_days": 2, "oos_period_days": 1}},
        )

        response = client.get("/api/analytics/summary")
        assert response.status_code == 200
        payload = response.get_json()
        by_id = {study["study_id"]: study for study in payload["studies"]}

        assert by_id["wfa_months"]["is_oos"] == "2m/1m"
        month_settings = by_id["wfa_months"]["wfa_settings"]
        assert month_settings["period_unit"] == "months"
        assert month_settings["is_period_days"] is None
        assert month_settings["oos_period_days"] is None
        assert month_settings["is_period_months"] == 2
        assert month_settings["oos_period_months"] == 1
        assert by_id["wfa_days"]["is_oos"] == "2/1"
        assert by_id["wfa_days"]["wfa_settings"]["period_unit"] == "days"
        assert {"2m/1m", "2/1"}.issubset(payload["research_info"]["is_oos_periods"])


def test_analytics_study_equity_endpoint_returns_lazy_curve_payload(client):
    with _temporary_active_db(f"analytics_study_equity_{uuid.uuid4().hex[:8]}"):
        _insert_analytics_study(
            study_id="wfa_study_curve",
            study_name="WFA_STUDY_CURVE",
            optimization_mode="wfa",
            stitched_oos_equity_curve=[100.0, 110.0, 108.0],
            stitched_oos_timestamps_json=[
                "2025-01-01T00:00:00+00:00",
                "2025-01-05T00:00:00+00:00",
                "2025-01-08T00:00:00+00:00",
            ],
        )
        _insert_analytics_study(
            study_id="optuna_curve",
            study_name="OPTUNA_CURVE",
            optimization_mode="optuna",
        )

        response = client.get("/api/analytics/studies/wfa_study_curve/equity")
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["study_id"] == "wfa_study_curve"
        assert payload["has_equity_curve"] is True
        assert payload["point_count"] == 3
        assert payload["curve"] == [100.0, 110.0, 108.0]
        assert payload["timestamps"] == [
            "2025-01-01T00:00:00+00:00",
            "2025-01-05T00:00:00+00:00",
            "2025-01-08T00:00:00+00:00",
        ]

        missing = client.get("/api/analytics/studies/missing/equity")
        assert missing.status_code == 404

        non_wfa = client.get("/api/analytics/studies/optuna_curve/equity")
        assert non_wfa.status_code == 400
        assert "WFA" in non_wfa.get_json()["error"]


def test_analytics_set_and_all_studies_equity_endpoints_return_cached_payloads(client):
    with _temporary_active_db(f"analytics_cached_equity_{uuid.uuid4().hex[:8]}"):
        _insert_analytics_study(
            study_id="wfa_cache_a",
            study_name="WFA_CACHE_A",
            optimization_mode="wfa",
            stitched_oos_net_profit_pct=10.0,
            stitched_oos_max_drawdown_pct=4.0,
            stitched_oos_equity_curve=[100.0, 110.0],
            stitched_oos_timestamps_json=[
                "2025-01-01T00:00:00+00:00",
                "2025-01-10T00:00:00+00:00",
            ],
        )
        _insert_analytics_study(
            study_id="wfa_cache_b",
            study_name="WFA_CACHE_B",
            optimization_mode="wfa",
            stitched_oos_net_profit_pct=5.0,
            stitched_oos_max_drawdown_pct=2.0,
            stitched_oos_equity_curve=[100.0, 105.0],
            stitched_oos_timestamps_json=[
                "2025-01-01T00:00:00+00:00",
                "2025-01-10T00:00:00+00:00",
            ],
        )

        created = client.post(
            "/api/analytics/sets",
            json={"name": "Cached Set", "study_ids": ["wfa_cache_a", "wfa_cache_b"]},
        ).get_json()

        sets_payload = client.get("/api/analytics/sets").get_json()
        assert sets_payload["all_metrics"]["selected_count"] == 2
        assert sets_payload["all_metrics"]["has_curve"] is True
        assert sets_payload["sets"][0]["metrics"]["selected_count"] == 2
        assert sets_payload["sets"][0]["metrics"]["has_curve"] is True

        set_equity = client.get(f"/api/analytics/sets/{created['id']}/equity")
        assert set_equity.status_code == 200
        set_payload = set_equity.get_json()
        assert set_payload["selected_count"] == 2
        assert set_payload["has_curve"] is True
        assert len(set_payload["curve"]) == len(set_payload["timestamps"]) == 2

        all_equity = client.get("/api/analytics/all-studies/equity")
        assert all_equity.status_code == 200
        all_payload = all_equity.get_json()
        assert all_payload["selected_count"] == 2
        assert all_payload["has_curve"] is True
        assert len(all_payload["curve"]) == len(all_payload["timestamps"]) == 2

        update_response = client.put(
            f"/api/analytics/sets/{created['id']}",
            json={"study_ids": ["wfa_cache_a"]},
        )
        assert update_response.status_code == 200

        refreshed = client.get(f"/api/analytics/sets/{created['id']}/equity")
        assert refreshed.status_code == 200
        refreshed_payload = refreshed.get_json()
        assert refreshed_payload["selected_count"] == 1
        assert refreshed_payload["profit_pct"] == pytest.approx(10.0)


def test_analytics_sets_payload_includes_consistency_pair(client):
    with _temporary_active_db(f"analytics_consistency_{uuid.uuid4().hex[:8]}"):
        curve = [100.0, 104.0, 108.0, 112.0, 116.0, 120.0, 119.0, 117.0, 115.0]
        timestamps = [f"2025-03-{day:02d}T00:00:00+00:00" for day in range(1, 10)]
        expected_full = _calculate_r2_consistency(curve)
        expected_recent = _calculate_r2_consistency(curve[-3:])

        _insert_analytics_study(
            study_id="wfa_consistency_set",
            study_name="WFA_CONSISTENCY_SET",
            optimization_mode="wfa",
            stitched_oos_net_profit_pct=15.0,
            stitched_oos_max_drawdown_pct=4.1667,
            stitched_oos_equity_curve=curve,
            stitched_oos_timestamps_json=timestamps,
        )

        created = client.post(
            "/api/analytics/sets",
            json={"name": "Consistency Set", "study_ids": ["wfa_consistency_set"]},
        ).get_json()

        sets_payload = client.get("/api/analytics/sets")
        assert sets_payload.status_code == 200
        payload = sets_payload.get_json()
        assert payload["all_metrics"]["consistency_full"] == pytest.approx(expected_full, abs=1e-6)
        assert payload["all_metrics"]["consistency_recent"] == pytest.approx(expected_recent, abs=1e-6)
        assert payload["sets"][0]["metrics"]["consistency_full"] == pytest.approx(expected_full, abs=1e-6)
        assert payload["sets"][0]["metrics"]["consistency_recent"] == pytest.approx(expected_recent, abs=1e-6)

        equity_response = client.get(f"/api/analytics/sets/{created['id']}/equity")
        assert equity_response.status_code == 200
        equity_payload = equity_response.get_json()
        assert equity_payload["consistency_full"] == pytest.approx(expected_full, abs=1e-6)
        assert equity_payload["consistency_recent"] == pytest.approx(expected_recent, abs=1e-6)


def test_analytics_summary_includes_per_study_consistency_pair(client):
    with _temporary_active_db(f"analytics_summary_consistency_{uuid.uuid4().hex[:8]}"):
        curve = [100.0, 104.0, 108.0, 112.0, 116.0, 120.0, 119.0, 117.0, 115.0]
        timestamps = [f"2025-03-{day:02d}T00:00:00+00:00" for day in range(1, 10)]
        expected_full = _calculate_r2_consistency(curve)
        expected_recent = _calculate_r2_consistency(curve[-3:])

        _insert_analytics_study(
            study_id="wfa_summary_consistency",
            study_name="WFA_SUMMARY_CONSISTENCY",
            optimization_mode="wfa",
            stitched_oos_equity_curve=curve,
            stitched_oos_timestamps_json=timestamps,
        )

        response = client.get("/api/analytics/summary")
        assert response.status_code == 200
        payload = response.get_json()

        studies = payload["studies"]
        assert len(studies) == 1
        assert studies[0]["consistency_full"] == pytest.approx(expected_full, abs=1e-6)
        assert studies[0]["consistency_recent"] == pytest.approx(expected_recent, abs=1e-6)


def test_analytics_summary_includes_focus_settings_payload(client):
    with _temporary_active_db(f"analytics_focus_settings_{uuid.uuid4().hex[:8]}"):
        _insert_analytics_study(
            study_id="wfa_focus_1",
            study_name="WFA_FOCUS_1",
            optimization_mode="wfa",
            adaptive_mode=1,
            is_period_days=60,
            config_json={
                "objectives": ["net_profit_pct", "max_drawdown_pct"],
                "primary_objective": "net_profit_pct",
                "constraints": [{"enabled": True, "metric": "total_trades", "threshold": 30}],
                "worker_processes": 4,
                "filter_min_profit": True,
                "min_profit_threshold": 12.5,
                "score_config": {
                    "filter_enabled": True,
                    "min_score_threshold": 77.5,
                },
                "optuna_config": {
                    "budget_mode": "trials",
                    "n_trials": 500,
                    "time_limit": 3600,
                    "convergence_patience": 50,
                    "sampler": "tpe",
                    "enable_pruning": False,
                    "pruner": "median",
                    "warmup_trials": 132,
                    "coverage_mode": True,
                    "dispatcher_batch_result_processing": False,
                    "dispatcher_soft_duplicate_cycle_limit_enabled": True,
                    "dispatcher_duplicate_cycle_limit": 18,
                    "sanitize_enabled": True,
                    "sanitize_trades_threshold": 3,
                },
                "wfa": {
                    "is_period_days": 90,
                    "oos_period_days": 30,
                    "adaptive_mode": True,
                    "max_oos_period_days": 120,
                    "min_oos_trades": 7,
                    "check_interval_trades": 4,
                    "cusum_threshold": 5.5,
                    "dd_threshold_multiplier": 1.7,
                    "inactivity_multiplier": 6.2,
                },
                "postProcess": {
                    "enabled": True,
                    "ftPeriodDays": 14,
                    "topK": 10,
                    "sortMetric": "profit_degradation",
                    "ftThresholdPct": 4.0,
                    "ftRejectAction": "cooldown_reoptimize",
                    "ftRejectCooldownDays": 5,
                    "ftRejectMaxAttempts": 2,
                    "ftRejectMinRemainingOosDays": 10,
                    "dsrEnabled": True,
                    "dsrTopK": 18,
                    "stressTest": {
                        "enabled": True,
                        "topK": 7,
                        "failureThreshold": 0.65,
                        "sortMetric": "profit_retention",
                    },
                },
            },
        )
        _insert_analytics_study(
            study_id="wfa_focus_2",
            study_name="WFA_FOCUS_2",
            optimization_mode="wfa",
            adaptive_mode=None,
            is_period_days=None,
            config_json={},
        )
        with get_db_connection() as conn:
            conn.execute(
                """
                UPDATE studies
                SET
                    cooldown_enabled = 1,
                    cooldown_days = 15,
                    max_oos_period_days = 110,
                    min_oos_trades = 9,
                    check_interval_trades = 8,
                    cusum_threshold = 4.4,
                    dd_threshold_multiplier = 1.8,
                    inactivity_multiplier = 7.2,
                    optimization_time_seconds = 3661
                WHERE study_id = 'wfa_focus_1'
                """
            )
            conn.execute(
                """
                UPDATE studies
                SET
                    budget_mode = 'time',
                    time_limit = 1800,
                    sampler_type = 'random',
                    sanitize_enabled = 0,
                    sanitize_trades_threshold = 11,
                    filter_min_profit = 1,
                    min_profit_threshold = 9.0,
                    optimization_time_seconds = 95,
                    score_config_json = '{"filter_enabled": 1, "min_score_threshold": 73.5}'
                WHERE study_id = 'wfa_focus_2'
                """
            )
            conn.commit()

        response = client.get("/api/analytics/summary")
        assert response.status_code == 200
        payload = response.get_json()
        studies = payload["studies"]
        by_id = {row["study_id"]: row for row in studies}

        first = by_id["wfa_focus_1"]
        assert first["optimization_mode"] == "wfa"
        assert first["optuna_settings"]["objectives"] == ["net_profit_pct", "max_drawdown_pct"]
        assert first["optuna_settings"]["primary_objective"] == "net_profit_pct"
        assert first["optuna_settings"]["budget_mode"] == "trials"
        assert first["optuna_settings"]["n_trials"] == 500
        assert first["optuna_settings"]["sampler_type"] == "tpe"
        assert first["optuna_settings"]["enable_pruning"] is False
        assert first["optuna_settings"]["pruner"] == "median"
        assert first["optuna_settings"]["warmup_trials"] == 132
        assert first["optuna_settings"]["coverage_mode"] is True
        assert first["optuna_settings"]["dispatcher_batch_result_processing"] is False
        assert first["optuna_settings"]["dispatcher_soft_duplicate_cycle_limit_enabled"] is True
        assert first["optuna_settings"]["dispatcher_duplicate_cycle_limit"] == 18
        assert first["optuna_settings"]["workers"] == 4
        assert first["optuna_settings"]["sanitize_enabled"] is True
        assert first["optuna_settings"]["sanitize_trades_threshold"] == 3
        assert first["optuna_settings"]["filter_min_profit"] is True
        assert first["optuna_settings"]["min_profit_threshold"] == 12.5
        assert first["optuna_settings"]["score_filter_enabled"] is True
        assert first["optuna_settings"]["score_min_threshold"] == 77.5

        assert first["wfa_settings"]["is_period_days"] == 60
        assert first["wfa_settings"]["oos_period_days"] == 30
        assert first["wfa_settings"]["adaptive_mode"] is True
        assert first["wfa_settings"]["cooldown_enabled"] is True
        assert first["wfa_settings"]["cooldown_days"] == 15
        assert first["wfa_settings"]["max_oos_period_days"] == 110
        assert first["wfa_settings"]["min_oos_trades"] == 9
        assert first["wfa_settings"]["check_interval_trades"] == 8
        assert first["wfa_settings"]["cusum_threshold"] == 4.4
        assert first["wfa_settings"]["dd_threshold_multiplier"] == 1.8
        assert first["wfa_settings"]["inactivity_multiplier"] == 7.2
        assert first["wfa_settings"]["run_time_seconds"] == 3661
        assert first["post_process_settings"]["ft_enabled"] is True
        assert first["post_process_settings"]["ft_period_days"] == 14
        assert first["post_process_settings"]["ft_top_k"] == 10
        assert first["post_process_settings"]["ft_sort_metric"] == "profit_degradation"
        assert first["post_process_settings"]["ft_threshold_pct"] == 4.0
        assert first["post_process_settings"]["ft_reject_action"] == "cooldown_reoptimize"
        assert first["post_process_settings"]["ft_reject_cooldown_days"] == 5
        assert first["post_process_settings"]["ft_reject_max_attempts"] == 2
        assert first["post_process_settings"]["ft_reject_min_remaining_oos_days"] == 10
        assert first["post_process_settings"]["dsr_enabled"] is True
        assert first["post_process_settings"]["dsr_top_k"] == 18
        assert first["post_process_settings"]["st_enabled"] is True
        assert first["post_process_settings"]["st_top_k"] == 7
        assert first["post_process_settings"]["st_failure_threshold"] == 0.65
        assert first["post_process_settings"]["st_sort_metric"] == "profit_retention"

        second = by_id["wfa_focus_2"]
        assert second["optimization_mode"] == "wfa"
        assert second["optuna_settings"]["budget_mode"] == "time"
        assert second["optuna_settings"]["time_limit"] == 1800
        assert second["optuna_settings"]["sampler_type"] == "random"
        assert second["optuna_settings"]["warmup_trials"] is None
        assert second["optuna_settings"]["coverage_mode"] is None
        assert second["optuna_settings"]["dispatcher_batch_result_processing"] is None
        assert second["optuna_settings"]["dispatcher_soft_duplicate_cycle_limit_enabled"] is None
        assert second["optuna_settings"]["dispatcher_duplicate_cycle_limit"] is None
        assert second["optuna_settings"]["sanitize_enabled"] is False
        assert second["optuna_settings"]["sanitize_trades_threshold"] == 11
        assert second["optuna_settings"]["filter_min_profit"] is True
        assert second["optuna_settings"]["min_profit_threshold"] == 9.0
        assert second["optuna_settings"]["score_filter_enabled"] is True
        assert second["optuna_settings"]["score_min_threshold"] == 73.5
        assert second["wfa_settings"]["adaptive_mode"] is None
        assert second["wfa_settings"]["run_time_seconds"] == 95
        assert second["post_process_settings"]["ft_enabled"] is False
        assert second["post_process_settings"]["dsr_enabled"] is False
        assert second["post_process_settings"]["st_enabled"] is False


def test_analytics_summary_includes_wfa_grid_settings_from_config(client):
    with _temporary_active_db(f"analytics_grid_settings_{uuid.uuid4().hex[:8]}"):
        grid_config = _grid_sidebar_config()
        grid_config["wfa"] = {"is_period_days": 60, "oos_period_days": 30}
        _insert_analytics_study(
            study_id="wfa_grid_settings",
            study_name="WFA_GRID_SETTINGS",
            strategy_id="s03_reversal_v10",
            optimization_mode="wfa",
            adaptive_mode=0,
            is_period_days=60,
            config_json=grid_config,
        )
        _insert_analytics_study(
            study_id="wfa_optuna_settings",
            study_name="WFA_OPTUNA_SETTINGS",
            optimization_mode="wfa",
            adaptive_mode=0,
            is_period_days=60,
            config_json={"wfa": {"oos_period_days": 30}},
        )

        response = client.get("/api/analytics/summary")
        assert response.status_code == 200
        studies = {row["study_id"]: row for row in response.get_json()["studies"]}

        grid = studies["wfa_grid_settings"]["grid_settings"]
        rows = {row["key"]: row["val"] for row in grid["rows"]}
        allocation_rows = {row["key"]: row["val"] for row in grid["allocation_rows"]}

        assert grid["enabled"] is True
        assert grid["is_wfa_grid"] is True
        assert rows["Budget"] == "10 candidates"
        assert rows["Seed"] == "42"
        assert rows["Top Candidates"] == "5"
        assert rows["Workers"] == "6 Numba threads"
        assert rows["Fast Objectives"] == "Net Profit %"
        assert rows["Slow Refinement"] == "Off"
        assert "Runtime" not in rows
        assert allocation_rows["Allocation"] == "Auto sqrt-space"
        assert any("|" in value for key, value in allocation_rows.items() if key != "Allocation")
        assert studies["wfa_optuna_settings"]["grid_settings"] is None


def test_analytics_sets_crud_and_reorder(client):
    with _temporary_active_db(f"analytics_sets_{uuid.uuid4().hex[:8]}"):
        _insert_analytics_study(
            study_id="wfa_set_1",
            study_name="WFA_SET_1",
            optimization_mode="wfa",
        )
        _insert_analytics_study(
            study_id="wfa_set_2",
            study_name="WFA_SET_2",
            optimization_mode="wfa",
        )

        create_response = client.post(
            "/api/analytics/sets",
            json={"name": "First Set", "study_ids": ["wfa_set_1", "wfa_set_2"]},
        )
        assert create_response.status_code == 201
        created_first = create_response.get_json()
        assert created_first["name"] == "First Set"
        assert created_first["color_token"] is None
        assert created_first["study_ids"] == ["wfa_set_1", "wfa_set_2"]

        second_response = client.post(
            "/api/analytics/sets",
            json={"name": "Second Set", "study_ids": ["wfa_set_2"]},
        )
        assert second_response.status_code == 201
        created_second = second_response.get_json()

        list_response = client.get("/api/analytics/sets")
        assert list_response.status_code == 200
        payload = list_response.get_json()
        assert "all_metrics" in payload
        assert payload["all_metrics"]["selected_count"] == 2
        assert [item["name"] for item in payload["sets"]] == ["First Set", "Second Set"]
        assert all("metrics" in item for item in payload["sets"])

        update_response = client.put(
            f"/api/analytics/sets/{created_first['id']}",
            json={"name": "First Set Updated", "study_ids": ["wfa_set_1"], "color_token": "blue"},
        )
        assert update_response.status_code == 200
        assert update_response.get_json()["ok"] is True

        reorder_response = client.put(
            "/api/analytics/sets/reorder",
            json={"order": [created_second["id"], created_first["id"]]},
        )
        assert reorder_response.status_code == 200
        assert reorder_response.get_json()["ok"] is True

        list_after_reorder = client.get("/api/analytics/sets").get_json()
        assert [item["id"] for item in list_after_reorder["sets"]] == [
            created_second["id"],
            created_first["id"],
        ]
        by_id = {item["id"]: item for item in list_after_reorder["sets"]}
        assert by_id[created_first["id"]]["name"] == "First Set Updated"
        assert by_id[created_first["id"]]["color_token"] == "blue"
        assert by_id[created_first["id"]]["study_ids"] == ["wfa_set_1"]
        assert by_id[created_first["id"]]["metrics"]["selected_count"] == 1

        delete_response = client.delete(f"/api/analytics/sets/{created_first['id']}")
        assert delete_response.status_code == 200
        assert delete_response.get_json()["ok"] is True

        final_payload = client.get("/api/analytics/sets").get_json()
        assert [item["id"] for item in final_payload["sets"]] == [created_second["id"]]


def test_analytics_sets_create_auto_suffixes_duplicate_names(client):
    with _temporary_active_db(f"analytics_sets_duplicates_{uuid.uuid4().hex[:8]}"):
        _insert_analytics_study(
            study_id="wfa_dup_1",
            study_name="WFA_DUP_1",
            optimization_mode="wfa",
        )

        first = client.post(
            "/api/analytics/sets",
            json={"name": "Duplicate Set", "study_ids": ["wfa_dup_1"]},
        )
        assert first.status_code == 201
        assert first.get_json()["name"] == "Duplicate Set"

        second = client.post(
            "/api/analytics/sets",
            json={"name": "Duplicate Set", "study_ids": ["wfa_dup_1"]},
        )
        assert second.status_code == 201
        assert second.get_json()["name"] == "Duplicate Set (1)"

        third = client.post(
            "/api/analytics/sets",
            json={"name": "Duplicate Set", "study_ids": ["wfa_dup_1"]},
        )
        assert third.status_code == 201
        assert third.get_json()["name"] == "Duplicate Set (2)"

        payload = client.get("/api/analytics/sets").get_json()
        assert [item["name"] for item in payload["sets"]] == [
            "Duplicate Set",
            "Duplicate Set (1)",
            "Duplicate Set (2)",
        ]


def test_analytics_sets_rename_auto_suffixes_duplicate_names(client):
    with _temporary_active_db(f"analytics_sets_rename_duplicates_{uuid.uuid4().hex[:8]}"):
        _insert_analytics_study(
            study_id="wfa_rename_dup_1",
            study_name="WFA_RENAME_DUP_1",
            optimization_mode="wfa",
        )

        first = client.post(
            "/api/analytics/sets",
            json={"name": "Rename Duplicate", "study_ids": ["wfa_rename_dup_1"]},
        )
        assert first.status_code == 201
        assert first.get_json()["name"] == "Rename Duplicate"

        second = client.post(
            "/api/analytics/sets",
            json={"name": "Rename Duplicate", "study_ids": ["wfa_rename_dup_1"]},
        )
        assert second.status_code == 201
        assert second.get_json()["name"] == "Rename Duplicate (1)"

        target = client.post(
            "/api/analytics/sets",
            json={"name": "Rename Target", "study_ids": ["wfa_rename_dup_1"]},
        )
        assert target.status_code == 201
        target_id = target.get_json()["id"]

        rename_response = client.put(
            f"/api/analytics/sets/{target_id}",
            json={"name": "Rename Duplicate"},
        )
        assert rename_response.status_code == 200
        assert rename_response.get_json()["ok"] is True

        payload = client.get("/api/analytics/sets").get_json()
        assert [item["name"] for item in payload["sets"]] == [
            "Rename Duplicate",
            "Rename Duplicate (1)",
            "Rename Duplicate (2)",
        ]


def test_analytics_sets_reject_non_wfa_study_ids(client):
    with _temporary_active_db(f"analytics_sets_validation_{uuid.uuid4().hex[:8]}"):
        _insert_analytics_study(
            study_id="optuna_not_allowed",
            study_name="OPTUNA_NOT_ALLOWED",
            optimization_mode="optuna",
        )
        response = client.post(
            "/api/analytics/sets",
            json={"name": "Invalid Set", "study_ids": ["optuna_not_allowed"]},
        )
        assert response.status_code == 400
        payload = response.get_json()
        assert "non-WFA" in payload["error"]


def test_analytics_sets_reject_invalid_color_token(client):
    with _temporary_active_db(f"analytics_sets_bad_color_{uuid.uuid4().hex[:8]}"):
        _insert_analytics_study(
            study_id="wfa_color_1",
            study_name="WFA_COLOR_1",
            optimization_mode="wfa",
        )

        created = client.post(
            "/api/analytics/sets",
            json={"name": "Color Set", "study_ids": ["wfa_color_1"]},
        ).get_json()

        response = client.put(
            f"/api/analytics/sets/{created['id']}",
            json={"color_token": "magenta"},
        )
        assert response.status_code == 400
        assert "Unsupported set color" in response.get_json()["error"]


def test_analytics_sets_color_token_can_be_cleared(client):
    with _temporary_active_db(f"analytics_sets_clear_color_{uuid.uuid4().hex[:8]}"):
        _insert_analytics_study(
            study_id="wfa_color_clear_1",
            study_name="WFA_COLOR_CLEAR_1",
            optimization_mode="wfa",
        )

        created = client.post(
            "/api/analytics/sets",
            json={"name": "Clear Color Set", "study_ids": ["wfa_color_clear_1"], "color_token": "teal"},
        ).get_json()
        assert created["color_token"] == "teal"

        response = client.put(
            f"/api/analytics/sets/{created['id']}",
            json={"color_token": None},
        )
        assert response.status_code == 200

        payload = client.get("/api/analytics/sets").get_json()
        assert payload["sets"][0]["color_token"] is None


def test_analytics_sets_bulk_color_and_delete(client):
    with _temporary_active_db(f"analytics_sets_bulk_{uuid.uuid4().hex[:8]}"):
        _insert_analytics_study(
            study_id="wfa_bulk_1",
            study_name="WFA_BULK_1",
            optimization_mode="wfa",
        )
        _insert_analytics_study(
            study_id="wfa_bulk_2",
            study_name="WFA_BULK_2",
            optimization_mode="wfa",
        )

        first = client.post(
            "/api/analytics/sets",
            json={"name": "Bulk First", "study_ids": ["wfa_bulk_1"], "color_token": "blue"},
        ).get_json()
        second = client.post(
            "/api/analytics/sets",
            json={"name": "Bulk Second", "study_ids": ["wfa_bulk_2"], "color_token": "teal"},
        ).get_json()

        color_response = client.put(
            "/api/analytics/sets/bulk-color",
            json={"set_ids": [first["id"], second["id"]], "color_token": "amber"},
        )
        assert color_response.status_code == 200
        assert color_response.get_json()["ok"] is True

        payload = client.get("/api/analytics/sets").get_json()
        assert [item["color_token"] for item in payload["sets"]] == ["amber", "amber"]

        delete_response = client.post(
            "/api/analytics/sets/bulk-delete",
            json={"set_ids": [first["id"], second["id"]]},
        )
        assert delete_response.status_code == 200
        assert delete_response.get_json()["ok"] is True
        assert delete_response.get_json()["deleted"] == 2
        assert client.get("/api/analytics/sets").get_json()["sets"] == []


def test_analytics_sets_members_cascade_on_study_delete(client):
    with _temporary_active_db(f"analytics_sets_cascade_{uuid.uuid4().hex[:8]}"):
        _insert_analytics_study(
            study_id="wfa_cascade_1",
            study_name="WFA_CASCADE_1",
            optimization_mode="wfa",
        )
        create_response = client.post(
            "/api/analytics/sets",
            json={"name": "Cascade Set", "study_ids": ["wfa_cascade_1"]},
        )
        assert create_response.status_code == 201

        assert delete_study("wfa_cascade_1") is True

        payload = client.get("/api/analytics/sets").get_json()
        assert payload["sets"][0]["name"] == "Cascade Set"
        assert payload["sets"][0]["study_ids"] == []
        assert payload["sets"][0]["metrics"]["selected_count"] == 0


def test_analytics_sets_reorder_requires_complete_order(client):
    with _temporary_active_db(f"analytics_sets_reorder_{uuid.uuid4().hex[:8]}"):
        _insert_analytics_study(
            study_id="wfa_r1",
            study_name="WFA_R1",
            optimization_mode="wfa",
        )
        _insert_analytics_study(
            study_id="wfa_r2",
            study_name="WFA_R2",
            optimization_mode="wfa",
        )
        first = client.post(
            "/api/analytics/sets",
            json={"name": "R1", "study_ids": ["wfa_r1"]},
        ).get_json()
        client.post(
            "/api/analytics/sets",
            json={"name": "R2", "study_ids": ["wfa_r2"]},
        )

        bad_reorder = client.put(
            "/api/analytics/sets/reorder",
            json={"order": [first["id"]]},
        )
        assert bad_reorder.status_code == 400
        assert "exactly once" in bad_reorder.get_json()["error"]
