import io
import sys
import csv
import hashlib
import json
import uuid
import logging
from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ui.server import app
from core.backtest_engine import TradeRecord
from core.metrics import _calculate_r2_consistency
from core.walkforward_engine import OOSStitchedResult, WFConfig, WFResult, WindowResult
from core.storage import (
    create_new_db,
    delete_study,
    get_active_db_name,
    get_db_connection,
    save_wfa_study_to_db,
    set_active_db,
)
from strategies import get_strategy_config


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_core_logger_console_handler_is_configured_once():
    from ui import server as server_module

    core_logger = logging.getLogger("core")
    marked_handlers_before = [
        handler for handler in core_logger.handlers if getattr(handler, "_merlin_core_console_handler", False)
    ]

    server_module._configure_core_console_logging()

    marked_handlers_after = [
        handler for handler in core_logger.handlers if getattr(handler, "_merlin_core_console_handler", False)
    ]

    assert marked_handlers_before
    assert len(marked_handlers_after) == len(marked_handlers_before)
    assert core_logger.level <= logging.INFO


def test_grid_start_page_label_and_marker_are_compact():
    repo_root = Path(__file__).parent.parent
    index_html = (repo_root / "src" / "ui" / "templates" / "index.html").read_text(encoding="utf-8")
    ui_handlers_js = (repo_root / "src" / "ui" / "static" / "js" / "ui-handlers.js").read_text(encoding="utf-8")
    strategy_config_js = (repo_root / "src" / "ui" / "static" / "js" / "strategy-config.js").read_text(encoding="utf-8")
    results_html = (repo_root / "src" / "ui" / "templates" / "results.html").read_text(encoding="utf-8")
    results_tables_js = (repo_root / "src" / "ui" / "static" / "js" / "results-tables.js").read_text(encoding="utf-8")
    analytics_js = (repo_root / "src" / "ui" / "static" / "js" / "analytics.js").read_text(encoding="utf-8")
    queue_js = (repo_root / "src" / "ui" / "static" / "js" / "queue.js").read_text(encoding="utf-8")

    assert "Grid v1 is supported only for S03 Reversal v10." not in ui_handlers_js
    assert "No fast Grid backend is available." in ui_handlers_js
    assert "в–ѕ" not in index_html
    assert "&#9660;" in index_html
    assert "GRID SETTINGS" in index_html
    assert 'id="optuna-settings-section"' in results_html
    assert 'id="gridFastObjectivesSection"' in index_html
    assert 'class="grid-fast-objective-checkbox"' in index_html
    assert 'id="gridSlowRefinementEnabled"' in index_html
    assert 'class="grid-slow-objective-checkbox"' in index_html
    assert 'id="gridProfileModesSection"' in index_html
    assert 'id="gridV2PlanningPolicy"' in index_html
    assert '<option value="full" selected>Full enumeration</option>' in index_html
    assert 'id="gridV2ManualAllocation"' in index_html
    assert "grid_enabled_modes" in ui_handlers_js
    assert "function getUserFacingGridModes(metadata)" in ui_handlers_js
    assert "function hasUserFacingGridModes(metadata)" in ui_handlers_js
    assert "function isFullEnumerationProfile(profile)" in ui_handlers_js
    assert "profile === 'full_enumeration_v2'" in ui_handlers_js
    assert "modeSection.style.display = fullEnumeration && hasModes ? 'block' : 'none'" in ui_handlers_js
    assert "isFullEnumerationProfile(gridMeta.profile) && hasGridModes && !getSelectedGridModes().length" in ui_handlers_js
    assert "grid_enabled_modes: fullEnumeration && hasGridModes ? getSelectedGridModes() : []" in ui_handlers_js
    assert "grid_enabled_modes: fullEnumeration ? getSelectedGridModes() : []" not in ui_handlers_js
    assert "const GLOBAL_RUNTIME_PARAM_NAMES = new Set(['dateFilter', 'start', 'end'])" in ui_handlers_js
    assert "const GLOBAL_BACKTEST_CONTROL_PARAM_NAMES = new Set([...GLOBAL_RUNTIME_PARAM_NAMES, 'warmupBars'])" in ui_handlers_js
    assert "if (isGlobalBacktestControlParam(name)) return;" in ui_handlers_js
    assert "const DYNAMIC_BACKTEST_GLOBAL_PARAM_NAMES = new Set(['dateFilter', 'start', 'end', 'warmupBars'])" in strategy_config_js
    assert "function shouldRenderDynamicBacktestParam(paramName, paramDef = {})" in strategy_config_js
    assert "if (!shouldRenderDynamicBacktestParam(paramName, paramDef)) continue;" in strategy_config_js
    assert "collectGridObjectiveSelection('fast')" in ui_handlers_js
    assert "grid_fast_objectives" in ui_handlers_js
    assert "applyQueueGridConfig" in queue_js
    assert "grid_slow_refinement_enabled" in queue_js
    assert "grid_v2_planning_policy" in ui_handlers_js
    assert "const sameOrderedBlocks = blockNames.length === existingBlockNames.length" in ui_handlers_js
    assert "if (sameOrderedBlocks)" in ui_handlers_js
    assert ui_handlers_js.index("if (sameOrderedBlocks)") < ui_handlers_js.index("container.replaceChildren()")
    assert "Object.prototype.hasOwnProperty.call(pending, blockNames[index])" in ui_handlers_js
    assert "Object.prototype.hasOwnProperty.call(pending, blockName)" in ui_handlers_js
    assert "Object.prototype.hasOwnProperty.call(manual, blockName)" in queue_js
    assert "data-grid-block-name" not in index_html
    assert "td.textContent = String(value)" in ui_handlers_js
    assert "planned_candidate_count" in queue_js
    assert "Object.prototype.hasOwnProperty.call(item, 'warmupBars')" in queue_js
    run_start = queue_js.index("async function runQueue()")
    run_source = queue_js[run_start:]
    assert "formData.append('warmupBars', String(item.warmupBars))" not in run_source
    assert "appendQueueWarmupField(formData, item)" in run_source
    ensure_start = queue_js.index("async function ensureQueueStateLoaded()")
    ensure_end = queue_js.index("function hasPersistedQueueItems()", ensure_start)
    ensure_source = queue_js[ensure_start:ensure_end]
    assert "applyQueueState(null)" not in ensure_source
    assert "throw error" in ensure_source
    assert results_html.index('id="optuna-settings-section"') < results_html.index("Optuna Settings")
    assert results_html.index('id="optuna-settings-section"') > results_html.index("Status &amp; Controls")
    assert "setElementVisible('optuna-settings-section', gridRows.length === 0)" in results_tables_js
    assert "optunaSection.style.display = gridRows.length ? 'none' : ''" in analytics_js


@contextmanager
def _temporary_active_db(label: str):
    previous_db = get_active_db_name()
    create_new_db(label)
    try:
        yield
    finally:
        set_active_db(previous_db)


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


def _insert_lancelot_export_study(
    *,
    study_id: str,
    study_name: str,
    optimization_mode: str,
    csv_file_path: str,
    csv_file_name: str,
    strategy_id: str = "s03_reversal_v10",
    strategy_version: str = "v10",
    warmup_bars: int = 1000,
    config_json: dict | None = None,
):
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO studies (
                study_id,
                study_name,
                strategy_id,
                strategy_version,
                optimization_mode,
                csv_file_path,
                csv_file_name,
                warmup_bars,
                config_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                study_id,
                study_name,
                strategy_id,
                strategy_version,
                optimization_mode,
                csv_file_path,
                csv_file_name,
                int(warmup_bars),
                json.dumps(config_json or {"fixed_params": {}}),
            ),
        )
        conn.commit()


def _insert_lancelot_export_trial(*, study_id: str, trial_number: int, params: dict):
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO trials (
                study_id,
                trial_number,
                params_json,
                objective_values_json,
                constraint_values_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                study_id,
                int(trial_number),
                json.dumps(params),
                json.dumps([]),
                json.dumps([]),
            ),
        )
        conn.commit()


def _grid_sidebar_config() -> dict:
    return {
        "strategy_id": "s03_reversal_v10",
        "optimization_mode": "grid",
        "enabled_params": {
            "maType3": True,
            "maLength3": True,
            "maOffset3": False,
            "useCloseCount": True,
            "useTBands": True,
            "closeCountLong": True,
            "closeCountShort": True,
            "tBandLongPct": True,
            "tBandShortPct": True,
        },
        "param_ranges": {
            "maLength3": [3, 4, 1],
            "closeCountLong": [1, 2, 1],
            "closeCountShort": [1, 1, 1],
            "tBandLongPct": [0.5, 1.0, 0.5],
            "tBandShortPct": [0.5, 1.0, 0.5],
        },
        "param_types": {
            "maType3": "select",
            "maLength3": "int",
            "maOffset3": "float",
            "useCloseCount": "bool",
            "useTBands": "bool",
            "closeCountLong": "int",
            "closeCountShort": "int",
            "tBandLongPct": "float",
            "tBandShortPct": "float",
        },
        "fixed_params": {
            "maType3_options": ["SMA"],
            "useCloseCount_options": [True],
            "useTBands_options": [True],
            "contractSize": 0.01,
            "commissionPct": 0.05,
            "dateFilter": False,
        },
        "worker_processes": 6,
        "warmup_bars": 20,
        "risk_per_trade_pct": 2.0,
        "contract_size": 0.01,
        "commission_rate": 0.0005,
        "filter_min_profit": False,
        "min_profit_threshold": 0.0,
        "score_config": {},
        "objectives": ["net_profit_pct"],
        "primary_objective": None,
        "constraints": [],
        "grid_budget": 10,
        "grid_seed": 42,
        "grid_top_candidates": 5,
        "grid_allocation_method": "auto_sqrt_space",
        "grid_min_quota": 0.10,
        "grid_diversity_enabled": True,
        "grid_diversity_max_per_group": 2,
        "grid_strict_validation": True,
        "grid_config": {
            "budget": 10,
            "seed": 42,
            "top_candidates": 5,
            "allocation_method": "auto_sqrt_space",
            "min_quota": 0.10,
        },
    }


def _insert_lancelot_export_wfa_window(
    *,
    study_id: str,
    window_number: int,
    best_params: dict,
    best_params_source: str = "optuna_is",
    is_best_trial_number: int | None = None,
):
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO wfa_windows (
                window_id,
                study_id,
                window_number,
                best_params_json,
                best_params_source,
                is_best_trial_number
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                f"{study_id}_w{window_number}",
                study_id,
                int(window_number),
                json.dumps(best_params),
                best_params_source,
                is_best_trial_number,
            ),
        )
        conn.commit()


def _insert_lancelot_export_wfa_trial(
    *,
    study_id: str,
    window_number: int,
    module_type: str,
    trial_number: int,
    params: dict,
    is_selected: bool,
):
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO wfa_window_trials (
                window_id,
                module_type,
                trial_number,
                params_json,
                objective_values_json,
                constraint_values_json,
                is_selected,
                module_metrics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"{study_id}_w{window_number}",
                module_type,
                int(trial_number),
                json.dumps(params),
                json.dumps([]),
                json.dumps([]),
                1 if is_selected else 0,
                json.dumps({}),
            ),
        )
        conn.commit()


def _insert_stored_execution_inventory(
    *, study_id: str, csv_path: Path, config_json: dict | str
) -> None:
    _insert_lancelot_export_study(
        study_id=study_id,
        study_name=study_id,
        strategy_id="s06_r_trend_v02_b2",
        strategy_version="v02-b2",
        optimization_mode="optuna",
        csv_file_path=str(csv_path),
        csv_file_name="OKX_SUIUSDT.P, 30 2025.01.01-2026.02.01.csv",
        warmup_bars=999,
        config_json=config_json if isinstance(config_json, dict) else {},
    )
    with get_db_connection() as conn:
        if isinstance(config_json, str):
            conn.execute(
                "UPDATE studies SET config_json = ? WHERE study_id = ?",
                (config_json, study_id),
            )
        conn.execute(
            """
            UPDATE studies
            SET ft_enabled = 1,
                ft_start_date = '2025-05-01',
                ft_end_date = '2025-05-10',
                oos_test_enabled = 1,
                oos_test_start_date = '2025-06-01',
                oos_test_end_date = '2025-06-10'
            WHERE study_id = ?
            """,
            (study_id,),
        )
        conn.commit()
    _insert_lancelot_export_trial(
        study_id=study_id,
        trial_number=7,
        params={
            "fastLength": 50,
            "dateFilter": True,
            "start": "candidate-start",
            "end": "candidate-end",
            "warmupBars": 9999,
        },
    )


def test_csv_import_s01_parameters(client):
    csv_content = "parameter,value\nmaType,ema\nmaLength,45\n"

    response = client.post(
        "/api/presets/import-csv",
        data={
            "file": (io.BytesIO(csv_content.encode("utf-8")), "params.csv"),
            "strategy": "s01_trailing_ma",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["values"]["maType"] == "EMA"
    assert payload["values"]["maLength"] == 45


def test_csv_import_s04_parameters(client):
    csv_content = "parameter,value\nrsiLen,16\nstochLen,20\n"

    response = client.post(
        "/api/presets/import-csv",
        data={
            "file": (io.BytesIO(csv_content.encode("utf-8")), "params.csv"),
            "strategy": "s04_stochrsi",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["values"]["rsiLen"] == 16
    assert payload["values"]["stochLen"] == 20


def test_csv_import_without_strategy_uses_first_available(client, monkeypatch):
    import strategies

    monkeypatch.setattr(
        strategies,
        "list_strategies",
        lambda: [{"id": "s04_stochrsi", "name": "S04 StochRSI"}],
    )

    csv_content = "parameter,value\nrsiLen,16\n"

    response = client.post(
        "/api/presets/import-csv",
        data={"file": (io.BytesIO(csv_content.encode("utf-8")), "params.csv")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["values"]["rsiLen"] == 16


def test_csv_import_fails_when_no_strategy_typing(monkeypatch, client):
    import strategies

    # Simulate discovery failure (no strategies available).
    monkeypatch.setattr(strategies, "list_strategies", lambda: [])

    csv_content = "parameter,value\nmaLength,45\n"

    response = client.post(
        "/api/presets/import-csv",
        data={"file": (io.BytesIO(csv_content.encode("utf-8")), "params.csv")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    message = response.get_data(as_text=True).lower()
    assert "parameter types are unavailable" in message
    assert "maLength".lower() in message


def test_csv_import_fails_when_strategy_config_unloadable(monkeypatch, client):
    import strategies

    monkeypatch.setattr(strategies, "get_strategy_config", lambda _sid: (_ for _ in ()).throw(ValueError("boom")))

    csv_content = "parameter,value\nmaLength,45\n"

    response = client.post(
        "/api/presets/import-csv",
        data={
            "file": (io.BytesIO(csv_content.encode("utf-8")), "params.csv"),
            "strategy": "s01_trailing_ma",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    message = response.get_data(as_text=True)
    assert "s01_trailing_ma" in message
    assert "parameter types are unavailable" in message


def test_grid_availability_reason_uses_generic_backend_label(client):
    response = client.get("/api/strategy/s01_trailing_ma/config")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["grid_optimizer"]["reason"] == (
        "No fast Grid backend is available for this strategy."
    )


_S03_REGIME_ER_MA_TYPES = ["EMA", "SMA", "HMA", "WMA", "ALMA", "KAMA", "TMA", "T3", "DEMA", "VWMA"]
_S03_REGIME_ER_COUNT_AXES = {
    "maType3": True,
    "maLength3": True,
    "useCloseCount": True,
    "closeCountLong": True,
    "closeCountShort": True,
    "useTBands": True,
    "tBandLongPct": True,
    "tBandShortPct": True,
}


def _s03_regime_er_grid_preview_payload(**overrides):
    payload = {
        "strategy_id": "s03_reversal_v11_regime_er_b2",
        "optimization_mode": "grid",
        "enabled_params": dict(_S03_REGIME_ER_COUNT_AXES),
        "param_ranges": {},
        "param_types": {},
        "fixed_params": {
            "dateFilter": False,
            "maType3_options": list(_S03_REGIME_ER_MA_TYPES),
            "useRegime": False,
            "useEmergencySL": False,
        },
        "objectives": ["net_profit_pct"],
        "grid_fast_objectives": ["net_profit_pct"],
        "grid_budget": "200k",
        "grid_top_candidates": 5,
    }
    payload.update(overrides)
    return payload


def _v2_runtime_diagnostic(response):
    body = response.get_json()
    assert isinstance(body, dict)
    assert body.get("error")
    diagnostics = body.get("diagnostics")
    assert isinstance(diagnostics, list) and diagnostics
    return diagnostics[0]


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


def test_grid_preview_strategy_aliases_agree_and_conflict(client):
    payload = {
        "config": _s03_regime_er_grid_preview_payload(
            strategy_id="s03_reversal_v11_regime_er_b2"
        ),
        "strategyId": "s03_reversal_v11_regime_er_b2",
    }
    agreed = client.post("/api/grid/preview", json=payload)
    assert agreed.status_code == 200

    payload["strategyId"] = "s03_reversal_v10"
    conflict = client.post("/api/grid/preview", json=payload)
    assert conflict.status_code == 400
    diagnostic = _v2_runtime_diagnostic(conflict)
    assert diagnostic["code"] == "V2_CONFLICTING_STRATEGY_ID"
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
    "strategy_id", ["s06_r_trend_v02_b2", "s03_reversal_v10"]
)
def test_valid_v2_and_v1_walkforward_retain_existing_post_profile_path(
    client, strategy_id
):
    response = client.post(
        "/api/walkforward",
        data={"strategy": strategy_id, "config": "{}"},
    )
    assert response.status_code == 400
    assert response.get_json() == {"error": "CSV path is required."}


def test_config_api_unexpected_failure_uses_app_logger(monkeypatch, client, caplog):
    import strategies

    monkeypatch.setattr(
        strategies,
        "get_strategy_config",
        lambda _strategy_id: (_ for _ in ()).throw(RuntimeError("unexpected registry failure")),
    )
    with caplog.at_level(logging.ERROR, logger=app.logger.name):
        response = client.get("/api/strategy/s06_r_trend_v02_b2/config")
    assert response.status_code == 500
    assert response.is_json
    assert any("Failed to load config" in record.getMessage() for record in caplog.records)


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


@pytest.mark.parametrize("optimization_mode", ["optuna", "grid"])
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
                    "fixed_params": {"dateFilter": False},
                }
            ),
        },
    )

    assert response.status_code == 400
    diagnostic = _v2_runtime_diagnostic(response)
    assert diagnostic["code"] == "V2_INVALID_RUNTIME_VALUE"
    assert diagnostic["path"] == "warmupBars"


@pytest.mark.parametrize("optimization_mode", ["optuna", "grid"])
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
        optimization_mode="optuna",
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


def test_tz64a_request_runtime_row_digests_and_identity_pins():
    from core.grid_v2 import build_grid_v2_plan
    from ui import server_services

    strategy_id = "s06_r_trend_v02_b2"
    config = get_strategy_config(strategy_id)
    context = server_services._resolve_strategy_context(
        [("strategy_id", True, strategy_id)]
    )

    def ordered_row_digest(plan):
        digest = hashlib.sha256()
        for index in range(len(plan.candidate_table)):
            candidate = plan.candidate_for_index(index)
            row = {
                "candidate_id": candidate.candidate_id,
                "variant_name": candidate.variant_name,
                "grid_mode_name": candidate.grid_mode_name,
                "params": dict(candidate.params),
            }
            digest.update(
                (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode(
                    "utf-8"
                )
            )
        return digest.hexdigest()

    def semantic_digest(plan):
        digest = hashlib.sha256()
        for key in plan.candidate_table.semantic_keys_by_row or ():
            digest.update((key + "\n").encode("utf-8"))
        return digest.hexdigest()

    blank_payload, blank_runtime = server_services._normalize_v2_optimizer_payload(
        context,
        {"fixed_params": {"dateFilter": False, "start": "", "end": ""}},
        warmup_members=[],
    )
    dated_payload, dated_runtime = server_services._normalize_v2_optimizer_payload(
        context,
        {
            "fixed_params": {
                "dateFilter": True,
                "start": "2025-05-01T00:00",
                "end": "2025-11-20T00:00",
            }
        },
        warmup_members=[],
    )
    alias_payload, _alias_runtime = server_services._normalize_v2_optimizer_payload(
        context,
        {"fixed_params": {"dateFilter": "0"}},
        warmup_members=[],
    )
    date_only_payload, _date_only_runtime = (
        server_services._normalize_v2_optimizer_payload(
            context,
            {
                "fixed_params": {
                    "dateFilter": True,
                    "start": "2025-06-01",
                    "end": "2025-06-30",
                }
            },
            warmup_members=[],
        )
    )
    assert blank_payload["fixed_params"] == {
        "dateFilter": False,
        "start": None,
        "end": None,
    }
    assert dated_payload["fixed_params"] == {
        "dateFilter": True,
        "start": "2025-05-01T00:00:00Z",
        "end": "2025-11-20T00:00:00Z",
    }
    assert alias_payload["fixed_params"] == {"dateFilter": False}
    assert date_only_payload["fixed_params"] == {
        "dateFilter": True,
        "start": "2025-06-01T00:00:00Z",
        "end": "2025-06-30T23:59:59.999999Z",
    }
    assert blank_runtime.values["warmupBars"] == dated_runtime.values["warmupBars"] == 1000

    cases = [
        (None, "f6a6258a07f21102ae60a91a239b960a23a5c59f15790423ab3a3874c00bfa6f"),
        ({"dateFilter": False}, "e0ba87a4dbf66a843d462d0f73d8b1c991841b247acbb6051d999bd5767b5f7c"),
        (
            {"dateFilter": False, "start": "", "end": ""},
            "8c2ca227ef140b31700f358d16a84246f34d7edb0f46b63f139bef003a8a25a5",
        ),
        (
            blank_payload["fixed_params"],
            "c5645dfc07084855c4618053b12c58b55859637425fedafe849dd9f751f01ff6",
        ),
        (
            {
                "dateFilter": True,
                "start": "2025-05-01T00:00",
                "end": "2025-11-20T00:00",
            },
            "73d4665bffa3a4df39d84e71cce5778d50cae0cb43ecb1f6f340010d20c68784",
        ),
        (
            dated_payload["fixed_params"],
            "54a7709b31efc592bf6d329d45b7a0dd2bab38a78d6a7bdf1673d89f1829e19d",
        ),
        (
            {
                "dateFilter": True,
                "start": "2025-06-01",
                "end": "2025-06-30",
            },
            "b1900647b0afefb3f8f11b5150426ef04c4dc8164221f51eb260e19f1c46339c",
        ),
        (
            date_only_payload["fixed_params"],
            "9faf81224522a95727fb39961e3a7ea9da010c3dc4655e99ce84be11edd24e68",
        ),
    ]
    for base_params, expected_row_digest in cases:
        plan = build_grid_v2_plan(config, base_params=base_params)
        assert len(plan.candidate_table) == 48_480
        assert plan.per_variant_counts == {"bracket": 480, "trail": 48_000}
        assert ordered_row_digest(plan) == expected_row_digest
        assert semantic_digest(plan) == (
            "fc55d174e835e7196ae5fcf21427d318dc364241f6b10560aa32545e6910a08f"
        )
        assert plan.plan_fingerprint == (
            "0f8d001c380df5ee95d34ca4e25910c674e20e9e8f34886a1bd2f1c261f019b2"
        )

    from core.param_identity import create_display_param_id

    trading_params = {"maType": "EMA", "maLength": 50}
    raw_display_id = create_display_param_id(
        trading_params,
        fixed_params={
            "dateFilter": True,
            "start": "2025-06-01",
            "end": "2025-06-30",
            "warmupBars": 1000,
        },
    )
    canonical_display_id = create_display_param_id(
        trading_params,
        fixed_params={
            **date_only_payload["fixed_params"],
            "warmupBars": 5000,
        },
    )
    assert raw_display_id == canonical_display_id


def test_s03_regime_er_grid_metadata_hides_internal_variants(client):
    response = client.get("/api/strategy/s03_reversal_v11_regime_er_b2/config")

    assert response.status_code == 200
    grid_optimizer = response.get_json()["grid_optimizer"]
    assert grid_optimizer["profile"] == "full_enumeration_v2"
    assert grid_optimizer["modes"] == []
    assert grid_optimizer["diversity_group_fields"] == ["grid_mode_name"]


def test_s03_regime_er_grid_preview_uses_logical_modes_not_internal_variants(client):
    response = client.post("/api/grid/preview", json=_s03_regime_er_grid_preview_payload())

    assert response.status_code == 200
    preview = response.get_json()["preview"]
    modes = {row["mode"]: row for row in preview["modes"]}

    assert list(modes) == ["cc_only", "tbands_only", "both"]
    assert [row["label"] for row in preview["modes"]] == ["Close Count only", "T Bands only", "Both"]
    assert "plain" not in modes
    assert "emergency" not in modes
    assert preview["mode_space_sizes"] == {"cc_only": 7_200, "tbands_only": 20_000, "both": 720_000}
    assert preview["full_candidate_count"] == 747_200
    assert preview["allocation_method"] == "full_enumeration_v2"


def test_s03_regime_er_budgeted_grid_preview_uses_generic_logical_block_allocation(client):
    response = client.post(
        "/api/grid/preview",
        json=_s03_regime_er_grid_preview_payload(
            grid_v2_planning_policy="sampled",
            grid_budget=1_000,
            grid_allocation_method="manual",
            grid_manual_percents={"cc_only": 20, "tbands_only": 30, "both": 50},
        ),
    )

    assert response.status_code == 200
    preview = response.get_json()["preview"]
    assert preview["requested_planning_policy"] == "sampled"
    assert preview["effective_planning_policy"] == "sampled"
    assert preview["planned_candidate_count"] == 1_000
    assert preview["allocation_method"] == "manual"
    assert preview["mode_budgets"] == {"cc_only": 200, "tbands_only": 300, "both": 500}
    assert [row["mode"] for row in preview["modes"]] == ["cc_only", "tbands_only", "both"]


def test_s03_regime_er_manual_zero_percentages_round_trip_through_preview_api(client):
    response = client.post(
        "/api/grid/preview",
        json=_s03_regime_er_grid_preview_payload(
            grid_v2_planning_policy="sampled",
            grid_budget=1,
            grid_allocation_method="manual",
            grid_manual_percents={"cc_only": 0, "tbands_only": 0, "both": 100},
        ),
    )

    assert response.status_code == 200
    preview = response.get_json()["preview"]
    assert preview["allocation_method"] == "manual"
    assert preview["planned_candidate_count"] == 1
    assert preview["mode_budgets"] == {"cc_only": 0, "tbands_only": 0, "both": 1}


def test_s03_regime_er_budgeted_grid_preview_rejects_unsafe_budget_and_unknown_manual_block(client):
    too_small = client.post(
        "/api/grid/preview",
        json=_s03_regime_er_grid_preview_payload(
            grid_v2_planning_policy="sampled",
            grid_budget=2,
        ),
    )
    unknown = client.post(
        "/api/grid/preview",
        json=_s03_regime_er_grid_preview_payload(
            grid_v2_planning_policy="sampled",
            grid_budget=100,
            grid_allocation_method="manual",
            grid_manual_percents={"cc_only": 20, "tbands_only": 30, "both": 40, "ghost": 10},
        ),
    )

    assert too_small.status_code == 400
    assert "non-empty planning blocks" in too_small.get_json()["error"]
    assert unknown.status_code == 400
    assert "ghost" in unknown.get_json()["error"]


def test_s03_regime_er_grid_preview_uses_posted_numeric_ranges(client):
    base_response = client.post(
        "/api/grid/preview",
        json={
            "config": _s03_regime_er_grid_preview_payload(),
            "strategyId": "s03_reversal_v11_regime_er_b2",
            "warmupBars": 1000,
        },
    )
    ma_length_response = client.post(
        "/api/grid/preview",
        json={
            "config": _s03_regime_er_grid_preview_payload(
                param_ranges={"maLength3": [25, 250, 25]},
            ),
            "strategyId": "s03_reversal_v11_regime_er_b2",
            "warmupBars": 1000,
        },
    )
    close_count_response = client.post(
        "/api/grid/preview",
        json={
            "config": _s03_regime_er_grid_preview_payload(
                param_ranges={"closeCountLong": [2, 7, 2]},
            ),
            "strategyId": "s03_reversal_v11_regime_er_b2",
            "warmupBars": 1000,
        },
    )

    assert base_response.status_code == 200
    assert ma_length_response.status_code == 200
    assert close_count_response.status_code == 200

    base_preview = base_response.get_json()["preview"]
    ma_preview = ma_length_response.get_json()["preview"]
    close_preview = close_count_response.get_json()["preview"]

    assert base_preview["full_candidate_count"] == 747_200
    assert ma_preview["mode_space_sizes"] == {"cc_only": 3_600, "tbands_only": 10_000, "both": 360_000}
    assert ma_preview["full_candidate_count"] == 373_600
    assert close_preview["mode_space_sizes"] == {"cc_only": 3_600, "tbands_only": 20_000, "both": 360_000}
    assert close_preview["full_candidate_count"] == 383_600


def test_s03_regime_er_grid_enabled_modes_must_be_empty_for_internal_variants(client):
    from ui import server as server_module

    payload = _s03_regime_er_grid_preview_payload()
    config = server_module._build_optimization_config(
        "grid-preview.csv",
        payload,
        worker_processes=1,
        strategy_id="s03_reversal_v11_regime_er_b2",
        warmup_bars=1000,
    )
    explicit_empty = client.post(
        "/api/grid/preview",
        json=_s03_regime_er_grid_preview_payload(grid_enabled_modes=[]),
    )
    stale_modes = client.post(
        "/api/grid/preview",
        json=_s03_regime_er_grid_preview_payload(grid_enabled_modes=["plain"]),
    )

    assert config.grid_enabled_modes == []
    assert explicit_empty.status_code == 200
    assert stale_modes.status_code == 400
    assert "internal variant selector" in stale_modes.get_json()["error"]


def test_s03_regime_er_optimizer_defaults_are_user_facing_grid_defaults():
    config = get_strategy_config("s03_reversal_v11_regime_er_b2")
    params = config["parameters"]
    default_enabled = {
        name
        for name, spec in params.items()
        if spec.get("optimize", {}).get("enabled")
        and spec.get("optimize", {}).get("default_enabled", True) is not False
    }

    assert {
        "maType3",
        "maLength3",
        "useCloseCount",
        "closeCountLong",
        "closeCountShort",
        "useTBands",
        "tBandLongPct",
        "tBandShortPct",
        "regimeErLength",
        "regimeErThresh",
    } <= default_enabled
    assert "maOffset3" not in default_enabled
    assert "emergencySlPct" not in default_enabled
    assert params["useEmergencySL"]["optimize"] == {"enabled": False}
    assert params["useRegime"]["optimize"] == {"enabled": False}


def test_s06_b2_grid_preview_keeps_user_facing_modes(client):
    response = client.post(
        "/api/grid/preview",
        json={
            "strategy_id": "s06_r_trend_v02_b2",
            "optimization_mode": "grid",
            "enabled_params": {},
            "param_ranges": {},
            "param_types": {},
            "fixed_params": {},
            "objectives": ["net_profit_pct"],
            "grid_fast_objectives": ["net_profit_pct"],
            "grid_budget": "200k",
            "grid_top_candidates": 5,
        },
    )

    assert response.status_code == 200
    preview = response.get_json()["preview"]
    assert [row["mode"] for row in preview["modes"]] == ["bracket", "trail"]
    assert preview["mode_space_sizes"] == {"bracket": 480, "trail": 48_000}


def test_csv_import_rejects_invalid_int(client):
    csv_content = "parameter,value\nmaLength,abc\n"

    response = client.post(
        "/api/presets/import-csv",
        data={
            "file": (io.BytesIO(csv_content.encode("utf-8")), "params.csv"),
            "strategy": "s01_trailing_ma",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"] == "Invalid numeric values in CSV."
    assert any("maLength" in detail for detail in payload["details"])


def test_csv_import_rejects_invalid_float(client):
    csv_content = "parameter,value\nstopLongX,abc\n"

    response = client.post(
        "/api/presets/import-csv",
        data={
            "file": (io.BytesIO(csv_content.encode("utf-8")), "params.csv"),
            "strategy": "s01_trailing_ma",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"] == "Invalid numeric values in CSV."
    assert any("stopLongX" in detail for detail in payload["details"])


def test_csv_import_stops_on_mixed_valid_and_invalid_numbers(client):
    csv_content = "parameter,value\nmaLength,abc\nstopLongX,2.5\n"

    response = client.post(
        "/api/presets/import-csv",
        data={
            "file": (io.BytesIO(csv_content.encode("utf-8")), "params.csv"),
            "strategy": "s01_trailing_ma",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"] == "Invalid numeric values in CSV."
    assert any("maLength" in detail for detail in payload["details"])
    # Ensure even valid numeric fields do not get applied when an error is present.
    assert "values" not in payload


def _build_minimal_optuna_payload():
    return {
        "strategy": "s01_trailing_ma",
        "enabled_params": {},
        "param_ranges": {},
        "fixed_params": {},
        "objectives": ["net_profit_pct"],
        "primary_objective": None,
        "optuna_budget_mode": "trials",
        "optuna_n_trials": 10,
        "optuna_time_limit": 60,
        "optuna_convergence": 10,
    }


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


def _patch_queue_storage_path(monkeypatch, tmp_path: Path, filename: str) -> Path:
    from ui import server_services

    queue_file = tmp_path / filename

    monkeypatch.setattr(
        server_services,
        "_queue_storage_file_path",
        lambda: queue_file,
    )
    return queue_file


def test_queue_api_roundtrip_persists_in_file_storage(client, monkeypatch, tmp_path):
    queue_file = _patch_queue_storage_path(monkeypatch, tmp_path, "queue_roundtrip.json")

    payload = {
        "items": [
            {
                "id": "q_test_1",
                "index": 1,
                "label": "#1 example",
                "sources": [{"type": "path", "path": r"C:\data\file_1.csv"}],
                "sourceCursor": 0,
                "successCount": 0,
                "failureCount": 0,
            }
        ],
        "nextIndex": 2,
        "runtime": {"active": False, "updatedAt": 0},
    }

    response_put = client.put("/api/queue", json=payload)
    assert response_put.status_code == 200
    put_data = response_put.get_json()
    assert put_data["nextIndex"] == 2
    assert len(put_data["items"]) == 1
    assert queue_file.exists()

    response_get = client.get("/api/queue")
    assert response_get.status_code == 200
    get_data = response_get.get_json()
    assert len(get_data["items"]) == 1
    assert get_data["items"][0]["id"] == "q_test_1"
    assert get_data["items"][0]["sources"][0]["path"] == r"C:\data\file_1.csv"

    response_delete = client.delete("/api/queue")
    assert response_delete.status_code == 200
    delete_data = response_delete.get_json()
    assert delete_data["items"] == []
    assert delete_data["nextIndex"] == 1
    assert delete_data["runtime"]["active"] is False
    assert not queue_file.exists()


def test_queue_api_empty_items_removes_queue_file(client, monkeypatch, tmp_path):
    queue_file = _patch_queue_storage_path(monkeypatch, tmp_path, "queue_empty_cleanup.json")

    seed_payload = {
        "items": [
            {
                "id": "q_test_2",
                "index": 1,
                "label": "#1 seed",
                "sources": [{"type": "path", "path": r"C:\data\seed.csv"}],
            }
        ],
        "nextIndex": 2,
        "runtime": {"active": True, "updatedAt": 123},
    }

    response_seed = client.put("/api/queue", json=seed_payload)
    assert response_seed.status_code == 200
    assert queue_file.exists()

    response_clear = client.put(
        "/api/queue",
        json={
            "items": [],
            "nextIndex": 999,
            "runtime": {"active": True, "updatedAt": 999},
        },
    )
    assert response_clear.status_code == 200
    clear_data = response_clear.get_json()
    assert clear_data["items"] == []
    assert clear_data["nextIndex"] == 1
    assert clear_data["runtime"]["active"] is False
    assert clear_data["runtime"]["updatedAt"] == 0
    assert not queue_file.exists()


def test_queue_api_roundtrip_preserves_extended_item_metadata(client, monkeypatch, tmp_path):
    queue_file = _patch_queue_storage_path(monkeypatch, tmp_path, "queue_extended_metadata.json")

    payload = {
        "items": [
            {
                "id": "q_test_meta",
                "index": 73,
                "label": "#73 example",
                "mode": "wfa",
                "finalState": "completed",
                "dbTarget": "analytics_01.sqlite",
                "sources": [
                    {"type": "path", "path": r"C:\data\alpha_30m.csv"},
                    {"type": "path", "path": r"C:\data\beta_30m.csv"},
                ],
                "sourceCursor": 2,
                "successCount": 2,
                "failureCount": 0,
                "studySet": {
                    "autoCreate": True,
                    "completedStudyIds": ["study_a", "study_b"],
                    "createdSetId": 11,
                    "createdSetName": "#73 · S03 · 30m · NSGA-2 (357) · 1.5k · WFA-F 60/30",
                    "status": "created",
                    "error": "",
                    "lastUpdatedAt": "2026-03-12T10:15:00Z",
                },
                "uiSnapshot": {
                    "selectedTab": "optimizer",
                    "dbTarget": {"value": "analytics_01.sqlite"},
                },
                "wfa": {
                    "isPeriodDays": 60,
                    "oosPeriodDays": 30,
                    "storeTopNTrials": 50,
                    "adaptiveMode": True,
                    "cooldownEnabled": True,
                    "cooldownDays": 15,
                    "maxOosPeriodDays": 120,
                    "minOosTrades": 7,
                    "checkIntervalTrades": 4,
                    "cusumThreshold": 5.5,
                    "ddThresholdMultiplier": 1.7,
                    "inactivityMultiplier": 6.2,
                },
            }
        ],
        "nextIndex": 74,
        "runtime": {"active": False, "updatedAt": 0},
    }

    response_put = client.put("/api/queue", json=payload)
    assert response_put.status_code == 200
    stored = response_put.get_json()
    assert stored["items"][0]["finalState"] == "completed"
    assert stored["items"][0]["studySet"]["createdSetId"] == 11
    assert stored["items"][0]["studySet"]["completedStudyIds"] == ["study_a", "study_b"]
    assert stored["items"][0]["uiSnapshot"]["dbTarget"]["value"] == "analytics_01.sqlite"
    assert stored["items"][0]["wfa"]["cooldownEnabled"] is True
    assert stored["items"][0]["wfa"]["cooldownDays"] == 15

    response_get = client.get("/api/queue")
    assert response_get.status_code == 200
    loaded = response_get.get_json()
    assert loaded["items"][0]["studySet"]["createdSetName"].startswith("#73")
    assert loaded["items"][0]["dbTarget"] == "analytics_01.sqlite"
    assert loaded["items"][0]["wfa"]["cooldownEnabled"] is True
    assert loaded["items"][0]["wfa"]["cooldownDays"] == 15

    on_disk = json.loads(queue_file.read_text(encoding="utf-8"))
    assert on_disk["items"][0]["studySet"]["status"] == "created"
    assert on_disk["items"][0]["uiSnapshot"]["selectedTab"] == "optimizer"
    assert on_disk["items"][0]["wfa"]["cooldownEnabled"] is True
    assert on_disk["items"][0]["wfa"]["cooldownDays"] == 15


def test_queue_api_roundtrip_preserves_b2_wfa_grid_transport(client, monkeypatch, tmp_path):
    queue_file = _patch_queue_storage_path(monkeypatch, tmp_path, "queue_b2_wfa_grid.json")
    item = {
        "id": "q_b2_wfa_grid",
        "index": 22,
        "label": "#22 B2 WFA Grid",
        "strategyId": "s06_r_trend_v02_b2",
        "mode": "wfa",
        "warmupBars": 1500,
        "sources": [{"type": "path", "path": r"C:\data\b2.csv"}],
        "sourceCursor": 1,
        "successCount": 1,
        "failureCount": 0,
        "config": {
            "optimization_mode": "grid",
            "fixed_params": {
                "dateFilter": True,
                "start": "2026-01-01T00:00:00Z",
                "end": "2026-06-30T23:59:59.999999Z",
            },
            "grid_v2_planning_policy": "sampled",
            "grid_budget": 1200,
            "grid_seed": 41,
            "grid_allocation_method": "manual",
            "grid_manual_percents": {"bracket": 35, "trail": 65},
            "planned_candidate_count": 1200,
            "planned_candidate_policy": "sampled",
            "planning_policy_version": "grid_v2_planning_policy_v1",
            "future": {"nested": ["opaque", 7]},
        },
        "planned_candidate_count": 1200,
        "planned_candidate_policy": "sampled",
        "planning_policy_version": "grid_v2_planning_policy_v1",
        "wfa": {
            "isPeriodDays": 90,
            "oosPeriodDays": 30,
            "adaptiveMode": True,
            "cooldownEnabled": True,
            "cooldownDays": 15,
            "storeTopNTrials": 25,
        },
        "studySet": {"completedStudyIds": ["study_1"]},
        "forwardCompatible": {"schemaHint": "future-v3"},
    }
    payload = {
        "items": [item],
        "nextIndex": 23,
        "runtime": {"active": True, "updatedAt": 123456},
    }

    response = client.put("/api/queue", json=payload)
    assert response.status_code == 200
    assert response.get_json() == payload
    assert json.loads(queue_file.read_text(encoding="utf-8")) == payload
    assert client.get("/api/queue").get_json() == payload


def _assert_queue_get_preserves_unreadable_file(client, queue_file: Path, raw: bytes):
    queue_file.write_bytes(raw)
    before = (queue_file.read_bytes(), queue_file.stat().st_mtime_ns)

    response = client.get("/api/queue")

    assert response.status_code == 409
    assert response.get_json() == {
        "error": "Stored Queue state is unreadable. The source file was preserved."
    }
    assert (queue_file.read_bytes(), queue_file.stat().st_mtime_ns) == before


def test_queue_get_malformed_json_is_non_mutating(client, monkeypatch, tmp_path):
    queue_file = _patch_queue_storage_path(monkeypatch, tmp_path, "queue_malformed.json")
    _assert_queue_get_preserves_unreadable_file(client, queue_file, b'{"items": [')


def test_queue_get_invalid_utf8_is_non_mutating(client, monkeypatch, tmp_path):
    queue_file = _patch_queue_storage_path(monkeypatch, tmp_path, "queue_invalid_utf8.json")
    _assert_queue_get_preserves_unreadable_file(client, queue_file, b"\xff\xfe\x80")


@pytest.mark.parametrize("raw", [b"[]", b"null", b'{"items": null}', b'{"items": {}}'])
def test_queue_get_invalid_top_level_shape_is_non_mutating(
    client,
    monkeypatch,
    tmp_path,
    raw,
):
    queue_file = _patch_queue_storage_path(monkeypatch, tmp_path, "queue_invalid_shape.json")
    _assert_queue_get_preserves_unreadable_file(client, queue_file, raw)


def test_queue_get_accepts_utf8_bom_without_rewrite(client, monkeypatch, tmp_path):
    queue_file = _patch_queue_storage_path(monkeypatch, tmp_path, "queue_bom.json")
    raw = b"\xef\xbb\xbf" + json.dumps({"items": []}, separators=(",", ":")).encode("utf-8")
    queue_file.write_bytes(raw)
    before = (queue_file.read_bytes(), queue_file.stat().st_mtime_ns)

    response = client.get("/api/queue")

    assert response.status_code == 200
    assert response.get_json()["items"] == []
    assert (queue_file.read_bytes(), queue_file.stat().st_mtime_ns) == before


@pytest.mark.parametrize("payload", [{}, {"items": []}])
def test_queue_get_valid_empty_state_does_not_delete_or_rewrite(
    client,
    monkeypatch,
    tmp_path,
    payload,
):
    queue_file = _patch_queue_storage_path(monkeypatch, tmp_path, "queue_valid_empty.json")
    raw = json.dumps(payload, indent=2).encode("utf-8")
    queue_file.write_bytes(raw)
    before = (queue_file.read_bytes(), queue_file.stat().st_mtime_ns)

    response = client.get("/api/queue")

    assert response.status_code == 200
    assert response.get_json() == {
        "items": [],
        "nextIndex": 1,
        "runtime": {"active": False, "updatedAt": 0},
    }
    assert (queue_file.read_bytes(), queue_file.stat().st_mtime_ns) == before


def test_queue_get_keeps_lenient_item_normalization_without_rewriting(
    client,
    monkeypatch,
    tmp_path,
):
    queue_file = _patch_queue_storage_path(monkeypatch, tmp_path, "queue_lenient_items.json")
    raw = json.dumps({"items": [7, {"sources": []}], "nextIndex": 9}, indent=2).encode("utf-8")
    queue_file.write_bytes(raw)
    before = (queue_file.read_bytes(), queue_file.stat().st_mtime_ns)

    response = client.get("/api/queue")

    assert response.status_code == 200
    assert response.get_json()["items"] == []
    assert (queue_file.read_bytes(), queue_file.stat().st_mtime_ns) == before


def test_queue_progress_put_and_v1_transport_remain_compatible(client, monkeypatch, tmp_path):
    _patch_queue_storage_path(monkeypatch, tmp_path, "queue_v1_progress.json")
    payload = {
        "items": [{
            "id": "q_v1",
            "index": 1,
            "label": "#1 V1",
            "strategyId": "s03_reversal_v10",
            "mode": "grid",
            "warmupBars": 1000,
            "sources": [{"type": "path", "path": r"C:\data\v1.csv"}],
            "sourceCursor": 1,
            "successCount": 1,
            "failureCount": 0,
            "config": {"optimization_mode": "grid", "grid_seed": 7},
        }],
        "nextIndex": 2,
        "runtime": {"active": True, "updatedAt": 999},
    }

    response = client.put("/api/queue", json=payload)

    assert response.status_code == 200
    assert response.get_json() == payload
    assert client.get("/api/queue").get_json() == payload


def test_queue_legacy_missing_warmup_reaches_v2_runtime_default_once(
    monkeypatch,
    client,
    tmp_path,
):
    from ui import server_routes_run, server_services

    csv_path = tmp_path / "queue_legacy_warmup.csv"
    csv_path.write_text("placeholder", encoding="utf-8")
    payload = _s03_regime_er_grid_preview_payload(optimization_mode="optuna")
    captured = []
    normalization_calls = []
    original_builder = server_routes_run._build_optimization_config
    original_normalizer = server_services.normalize_v2_runtime_values

    def capture_builder(*args, **kwargs):
        config = original_builder(*args, **kwargs)
        captured.append(config)
        return config

    def count_normalization(*args, **kwargs):
        normalization_calls.append((args, kwargs))
        return original_normalizer(*args, **kwargs)

    monkeypatch.setattr(server_routes_run, "_resolve_csv_path", lambda _raw: csv_path)
    monkeypatch.setattr(server_routes_run, "_build_optimization_config", capture_builder)
    monkeypatch.setattr(server_routes_run, "run_optimization", lambda _config: ([], None))
    monkeypatch.setattr(server_services, "normalize_v2_runtime_values", count_normalization)

    response = client.post(
        "/api/optimize",
        data={
            "strategy": "s03_reversal_v11_regime_er_b2",
            "csvPath": str(csv_path),
            "config": json.dumps(payload),
        },
    )

    assert response.status_code == 200
    assert len(normalization_calls) == 1
    assert len(captured) == 1
    assert captured[0].warmup_bars == 1000


def test_queue_api_rejects_non_object_payload(client):
    response = client.put("/api/queue", json=["not", "an", "object"])
    assert response.status_code == 400
    payload = response.get_json()
    assert "json object" in payload["error"].lower()


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
                    "optimization_mode": "optuna",
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


def _build_params_from_config(strategy_id: str):
    config = get_strategy_config(strategy_id)
    parameters = config.get("parameters", {}) if isinstance(config, dict) else {}
    params = {}
    for name, spec in parameters.items():
        if not isinstance(spec, dict):
            continue
        default_value = spec.get("default")
        params[name] = default_value if default_value is not None else 0
    return params


def _create_wfa_study() -> str:
    data_path = Path(__file__).parent.parent / "data" / "raw" / "OKX_LINKUSDT.P, 15 2025.05.01-2025.11.20.csv"
    if not data_path.exists():
        pytest.skip("Sample data file not available for WFA API tests.")

    strategy_id = "s01_trailing_ma"
    params = _build_params_from_config(strategy_id)
    wf_config = WFConfig(strategy_id=strategy_id, is_period_days=30, oos_period_days=15, warmup_bars=10)

    window = WindowResult(
        window_id=1,
        is_start=pd.Timestamp("2025-05-01", tz="UTC"),
        is_end=pd.Timestamp("2025-05-30", tz="UTC"),
        oos_start=pd.Timestamp("2025-05-31", tz="UTC"),
        oos_end=pd.Timestamp("2025-06-14", tz="UTC"),
        best_params=params,
        param_id="test_params",
        is_net_profit_pct=0.0,
        is_max_drawdown_pct=0.0,
        is_total_trades=0,
        oos_net_profit_pct=0.0,
        oos_max_drawdown_pct=0.0,
        oos_total_trades=0,
        oos_equity_curve=[100.0],
        oos_timestamps=[pd.Timestamp("2025-05-31", tz="UTC")],
        is_equity_curve=[100.0],
        is_timestamps=[pd.Timestamp("2025-05-01", tz="UTC")],
        best_params_source="optuna_is",
        available_modules=["optuna_is"],
        optuna_is_trials=[
            {
                "trial_number": 1,
                "params": params,
                "param_id": "test_params",
                "net_profit_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "total_trades": 0,
                "win_rate": 0.0,
                "is_selected": True,
            }
        ],
    )

    stitched = OOSStitchedResult(
        final_net_profit_pct=0.0,
        max_drawdown_pct=0.0,
        total_trades=0,
        wfe=0.0,
        oos_win_rate=0.0,
        equity_curve=[100.0],
        timestamps=[pd.Timestamp("2025-05-31", tz="UTC")],
        window_ids=[1],
    )

    wf_result = WFResult(
        config=wf_config,
        windows=[window],
        stitched_oos=stitched,
        strategy_id=strategy_id,
        total_windows=1,
        trading_start_date=window.is_start,
        trading_end_date=window.oos_end,
        warmup_bars=wf_config.warmup_bars,
    )

    study_id = save_wfa_study_to_db(
        wf_result=wf_result,
        config={"fixed_params": {}},
        csv_file_path=str(data_path),
        start_time=0.0,
        score_config=None,
    )
    return study_id


def test_get_wfa_window_details(client):
    study_id = _create_wfa_study()
    response = client.get(f"/api/studies/{study_id}/wfa/windows/1")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["window"]["window_number"] == 1
    assert payload["window"]["oos_actual_days"] is None
    assert payload["window"]["trigger_type"] is None
    assert "optuna_is" in payload["modules"]


def test_generate_wfa_window_equity(client):
    study_id = _create_wfa_study()
    response = client.post(
        f"/api/studies/{study_id}/wfa/windows/1/equity",
        json={"period": "is"},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert "equity_curve" in payload


def test_download_wfa_window_trades(client):
    study_id = _create_wfa_study()
    response = client.post(
        f"/api/studies/{study_id}/wfa/windows/1/trades",
        json={"period": "oos"},
    )
    assert response.status_code == 200
    assert response.headers.get("Content-Type", "").startswith("text/csv")


def test_resolve_wfa_period_oos_prefers_precise_timestamp():
    from ui.server_services import _resolve_wfa_period

    window = {
        "oos_start_date": "2025-01-01",
        "oos_end_date": "2025-01-02",
        "oos_start_ts": "2025-01-01T00:00:00+00:00",
        "oos_end_ts": "2025-01-02T12:00:00+00:00",
    }
    start, end, error = _resolve_wfa_period(window, "oos")
    assert error is None
    assert start == "2025-01-01T00:00:00+00:00"
    assert end == "2025-01-02T12:00:00+00:00"

    # Backward compatibility: when ts fields are missing, date fields are used.
    legacy_window = {
        "oos_start_date": "2025-01-01",
        "oos_end_date": "2025-01-02",
    }
    legacy_start, legacy_end, legacy_error = _resolve_wfa_period(legacy_window, "oos")
    assert legacy_error is None
    assert legacy_start == "2025-01-01"
    assert legacy_end == "2025-01-02"


def test_download_wfa_window_trades_respects_stored_oos_trade_count(client, monkeypatch, tmp_path):
    import ui.server_routes_data as routes_data

    csv_path = tmp_path / "_tmp_wfa_window_trades.csv"
    csv_path.write_text("timestamp,open,high,low,close,volume\n", encoding="utf-8")

    try:
        study_id = "wfa_window_count_limit"
        study_data = {
            "study": {
                "study_id": study_id,
                "study_name": "wfa_window_count_limit",
                "optimization_mode": "wfa",
                "adaptive_mode": 1,
                "strategy_id": "s01_trailing_ma",
                "csv_file_path": str(csv_path),
                "csv_file_name": "OKX_TESTUSDT.csv",
                "warmup_bars": 0,
                "config_json": {"fixed_params": {}},
            },
            "windows": [
                {
                    "window_number": 1,
                    "window_id": f"{study_id}_w1",
                    "best_params": {},
                    "oos_start_ts": "2025-01-01T00:00:00+00:00",
                    "oos_end_ts": "2025-01-02T23:00:00+00:00",
                    "oos_start_date": "2025-01-01",
                    "oos_end_date": "2025-01-02",
                    "oos_total_trades": 1,
                }
            ],
        }

        monkeypatch.setattr(
            routes_data,
            "load_study_from_db",
            lambda sid: study_data if sid == study_id else None,
        )

        fake_trades = [
            TradeRecord(
                direction="long",
                entry_time=pd.Timestamp("2025-01-02 10:00:00+00:00"),
                exit_time=pd.Timestamp("2025-01-02 10:30:00+00:00"),
                entry_price=100.0,
                exit_price=101.0,
                size=1.0,
            ),
            TradeRecord(
                direction="long",
                entry_time=pd.Timestamp("2025-01-02 20:00:00+00:00"),
                exit_time=pd.Timestamp("2025-01-02 20:30:00+00:00"),
                entry_price=100.0,
                exit_price=99.0,
                size=1.0,
            ),
        ]

        monkeypatch.setattr(
            routes_data,
            "_run_trade_export",
            lambda **_kwargs: (fake_trades, None),
        )

        response = client.post(
            f"/api/studies/{study_id}/wfa/windows/1/trades",
            json={"period": "oos"},
        )
        assert response.status_code == 200

        rows = list(csv.reader(io.StringIO(response.get_data(as_text=True))))
        # Header + 2 rows (entry/exit) for exactly one trade after cap.
        assert len(rows) == 3
    finally:
        if csv_path.exists():
            csv_path.unlink()


def test_download_wfa_trades_uses_precise_oos_bounds(client, monkeypatch, tmp_path):
    import strategies
    import ui.server_routes_data as routes_data

    csv_path = tmp_path / "_tmp_wfa_precise_bounds.csv"
    csv_path.write_text("timestamp,open,high,low,close,volume\n", encoding="utf-8")

    try:
        study_id = "wfa_precise_bounds"
        study_data = {
            "study": {
                "study_id": study_id,
                "study_name": "wfa_precise_bounds",
                "optimization_mode": "wfa",
                "adaptive_mode": 1,
                "strategy_id": "s01_trailing_ma",
                "csv_file_path": str(csv_path),
                "csv_file_name": "OKX_TESTUSDT.csv",
                "warmup_bars": 0,
                "config_json": {"fixed_params": {}},
            },
            "windows": [
                {
                    "window_number": 1,
                    "best_params": {},
                    "oos_start_date": "2025-01-01",
                    "oos_end_date": "2025-01-02",
                    "oos_start_ts": "2025-01-01T00:00:00+00:00",
                    "oos_end_ts": "2025-01-02T12:00:00+00:00",
                    "oos_total_trades": 1,
                }
            ],
        }

        monkeypatch.setattr(
            routes_data,
            "load_study_from_db",
            lambda sid: study_data if sid == study_id else None,
        )

        index = pd.date_range("2025-01-01 00:00:00+00:00", periods=72, freq="h")
        df = pd.DataFrame(
            {
                "Open": 1.0,
                "High": 1.0,
                "Low": 1.0,
                "Close": 1.0,
                "Volume": 1.0,
            },
            index=index,
        )
        monkeypatch.setattr(routes_data, "load_data", lambda _: df)
        monkeypatch.setattr(routes_data, "prepare_dataset_with_warmup", lambda data, *_: (data, 0))

        class FakeResult:
            def __init__(self, trades):
                self.trades = trades

        class FakeStrategy:
            @staticmethod
            def run(*_args, **_kwargs):
                return FakeResult(
                    [
                        TradeRecord(
                            direction="long",
                            entry_time=pd.Timestamp("2025-01-02 10:00:00+00:00"),
                            exit_time=pd.Timestamp("2025-01-02 10:30:00+00:00"),
                            entry_price=100.0,
                            exit_price=101.0,
                            size=1.0,
                        ),
                        TradeRecord(
                            direction="long",
                            entry_time=pd.Timestamp("2025-01-02 20:00:00+00:00"),
                            exit_time=pd.Timestamp("2025-01-02 20:30:00+00:00"),
                            entry_price=100.0,
                            exit_price=99.0,
                            size=1.0,
                        ),
                    ]
                )

        monkeypatch.setattr(strategies, "get_strategy", lambda _sid: FakeStrategy)

        response = client.post(f"/api/studies/{study_id}/wfa/trades")
        assert response.status_code == 200

        rows = list(csv.reader(io.StringIO(response.get_data(as_text=True))))
        # Header + 2 rows (entry/exit) for exactly one trade within precise OOS end.
        assert len(rows) == 3
    finally:
        if csv_path.exists():
            csv_path.unlink()


def test_stored_execution_endpoint_inventory_uses_shared_runtime_and_reads_tests(
    client, monkeypatch, tmp_path
):
    from core.storage import get_study_trial, load_study_from_db
    from ui import server_routes_data

    csv_path = tmp_path / "stored_execution_inventory.csv"
    csv_path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2025-01-01T00:00:00Z,1,1,1,1,1\n",
        encoding="utf-8",
    )
    values = {
        "dateFilter": False,
        "start": "2025-01-01T00:00:00Z",
        "end": "2025-02-01T00:00:00Z",
        "warmupBars": 20,
    }
    config = {
        "fixed_params": {"slowLength": 70},
        "v2_runtime": {
            "schema_version": "v2_runtime_metadata_v1",
            "contract_version": "v2_runtime_contract_v1",
            "values": values,
            "diagnostics": [],
            "validation_warnings": [],
        },
    }
    frame = pd.DataFrame(
        {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0},
        index=pd.date_range("2025-01-01", "2025-08-01", freq="D", tz="UTC"),
    )
    captured = []

    def fake_period_test(**kwargs):
        candidate = kwargs["trials"][0]["params"]
        captured.append(
            (
                "manual",
                kwargs["execution_params_resolver"](
                    candidate, kwargs["start_ts"], kwargs["end_ts"]
                ),
                kwargs["warmup_bars"],
            )
        )
        return []

    def fake_trade_export(**kwargs):
        captured.append(("trade", deepcopy(kwargs["params"]), kwargs["warmup_bars"]))
        return [], None

    monkeypatch.setattr(server_routes_data, "load_data", lambda _path: frame)
    monkeypatch.setattr(server_routes_data, "run_period_test_for_trials", fake_period_test)
    monkeypatch.setattr(server_routes_data, "_run_trade_export", fake_trade_export)

    with _temporary_active_db(f"stored_inventory_{uuid.uuid4().hex[:8]}"):
        study_id = "stored_execution_inventory"
        _insert_stored_execution_inventory(
            study_id=study_id,
            csv_path=csv_path,
            config_json=config,
        )
        original_study = deepcopy(load_study_from_db(study_id)["study"])
        original_trial = deepcopy(get_study_trial(study_id, 7))

        manual = client.post(
            f"/api/studies/{study_id}/test",
            json={
                "dataSource": "original_csv",
                "startDate": "2025-07-01",
                "endDate": "2025-07-03",
                "trialNumbers": [7],
                "sourceTab": "optuna",
                "testName": "inventory",
            },
        )
        assert manual.status_code == 200
        test_id = manual.get_json()["test_id"]
        listed = client.get(f"/api/studies/{study_id}/tests")
        detail = client.get(f"/api/studies/{study_id}/tests/{test_id}")
        assert listed.status_code == detail.status_code == 200
        assert listed.get_json()["tests"][0]["id"] == test_id
        assert detail.get_json()["results_json"]["config"]["start_date"] == "2025-07-01"

        routes = [
            f"/api/studies/{study_id}/trials/7/trades",
            f"/api/studies/{study_id}/trials/7/ft-trades",
            f"/api/studies/{study_id}/trials/7/oos-trades",
            f"/api/studies/{study_id}/tests/{test_id}/trials/7/mt-trades",
        ]
        for route in routes:
            assert client.post(route).status_code == 200

        assert captured[0] == (
            "manual",
            {
                "slowLength": 70,
                "fastLength": 50,
                "dateFilter": True,
                "start": "2025-07-01T00:00:00Z",
                "end": "2025-07-03T00:00:00Z",
            },
            20,
        )
        exported = [item[1] for item in captured[1:]]
        assert exported[0] == {
            "slowLength": 70,
            "fastLength": 50,
            "dateFilter": False,
            "start": "2025-01-01T00:00:00Z",
            "end": "2025-02-01T00:00:00Z",
        }
        assert exported[1]["start"] == "2025-05-01T00:00:00Z"
        assert exported[1]["end"] == "2025-05-10T23:59:59.999999Z"
        assert exported[2]["start"] == "2025-06-01T00:00:00Z"
        assert exported[2]["end"] == "2025-06-10T23:59:59.999999Z"
        assert exported[3]["start"] == "2025-07-01T00:00:00Z"
        assert exported[3]["end"] == "2025-07-03T23:59:59.999999Z"
        assert all(item[2] == 20 for item in captured)
        assert load_study_from_db(study_id)["study"] == original_study
        assert get_study_trial(study_id, 7) == original_trial


@pytest.mark.parametrize(
    ("route_suffix", "payload"),
    [
        (
            "/test",
            {
                "dataSource": "original_csv",
                "startDate": "2025-07-01",
                "endDate": "2025-07-03",
                "trialNumbers": [7],
                "sourceTab": "optuna",
            },
        ),
        ("/trials/7/trades", None),
        ("/trials/7/ft-trades", None),
        ("/trials/7/oos-trades", None),
        ("/tests/1/trials/7/mt-trades", None),
    ],
)
def test_stored_execution_inventory_corruption_stops_before_csv(
    client, tmp_path, route_suffix, payload
):
    csv_path = tmp_path / "must_not_be_read.csv"
    assert not csv_path.exists()
    with _temporary_active_db(f"stored_corrupt_{uuid.uuid4().hex[:8]}"):
        study_id = "stored_corrupt"
        _insert_stored_execution_inventory(
            study_id=study_id,
            csv_path=csv_path,
            config_json="{not-json",
        )
        response = client.post(
            f"/api/studies/{study_id}{route_suffix}",
            json=payload,
        )

    assert response.status_code == 400
    diagnostic = _v2_runtime_diagnostic(response)
    assert diagnostic["code"] == "V2_STORED_CONFIG_INCOMPATIBLE"
    assert diagnostic["path"] == "config_json"


def test_v1_stored_trade_export_keeps_legacy_merge_semantics(
    client, monkeypatch, tmp_path
):
    from ui import server_routes_data

    csv_path = tmp_path / "v1_stored_trade.csv"
    csv_path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2025-01-01T00:00:00Z,1,1,1,1,1\n",
        encoding="utf-8",
    )
    captured = {}
    monkeypatch.setattr(
        server_routes_data,
        "_run_trade_export",
        lambda **kwargs: (captured.update(deepcopy(kwargs)) or [], None),
    )
    with _temporary_active_db(f"v1_stored_trade_{uuid.uuid4().hex[:8]}"):
        study_id = "v1_stored_trade"
        _insert_lancelot_export_study(
            study_id=study_id,
            study_name=study_id,
            strategy_id="s03_reversal_v10",
            strategy_version="v10",
            optimization_mode="optuna",
            csv_file_path=str(csv_path),
            csv_file_name="OKX_LINKUSDT.P, 15 2025.01.01-2025.01.02.csv",
            warmup_bars=20,
            config_json={
                "fixed_params": {
                    "maType3": "HMA",
                    "maType3_options": ["SMA", "HMA"],
                    "dateFilter": False,
                }
            },
        )
        _insert_lancelot_export_trial(
            study_id=study_id,
            trial_number=7,
            params={"maLength3": 75, "dateFilter": True, "start": "candidate"},
        )

        response = client.post(f"/api/studies/{study_id}/trials/7/trades")

    assert response.status_code == 200
    assert captured["params"] == {
        "maType3": "HMA",
        "maType3_options": ["SMA", "HMA"],
        "dateFilter": True,
        "maLength3": 75,
        "start": "candidate",
    }
    assert captured["warmup_bars"] == 20


def test_export_lancelot_bundle_from_optuna_trial(client, tmp_path):
    csv_path = tmp_path / "_tmp_lancelot_export_optuna.csv"
    csv_path.write_text(
        "timestamp,open,high,low,close,volume\n2025-01-01T00:00:00Z,1,1,1,1,1\n",
        encoding="utf-8",
    )

    try:
        with _temporary_active_db(f"bundle_export_optuna_{uuid.uuid4().hex[:8]}"):
            study_id = "bundle_optuna_1"
            _insert_lancelot_export_study(
                study_id=study_id,
                study_name="bundle_optuna_1",
                optimization_mode="optuna",
                csv_file_path=str(csv_path),
                csv_file_name="OKX_LINKUSDT.P, 15 2025.05.01-2025.11.20.csv",
                config_json={
                    "fixed_params": {
                        "maType3": "HMA",
                        "maType3_options": ["SMA", "HMA"],
                        "undeclaredControl": "must-not-export",
                        "dateFilter": True,
                        "warmupBars": 9999,
                    }
                },
            )
            _insert_lancelot_export_trial(
                study_id=study_id,
                trial_number=42,
                params={"closeCountLong": 3, "useTBands": True},
            )

            response = client.post(
                f"/api/studies/{study_id}/export/lancelot",
                json={"trialNumber": 42},
            )

            assert response.status_code == 200
            payload = response.get_json()
            assert payload["bundleSchemaVersion"] == 2
            assert payload["strategyId"] == "s03_reversal_v10"
            assert payload["strategyVersion"] == "v10"
            assert payload["symbol"] == "LINK/USDT:USDT"
            assert payload["timeframe"] == "15m"
            assert payload["warmupBars"] == 1000
            assert payload["exportMode"] == "live"
            assert payload["params"] == {
                "closeCountLong": 3,
                "useTBands": True,
                "dateFilter": False,
                "start": None,
                "end": None,
            }
            assert payload["source"]["studyId"] == study_id
            assert payload["source"]["trialNumber"] == 42
            assert payload["source"]["studyName"] == "bundle_optuna_1"
            assert payload["source"]["exportedAt"].endswith("Z")
            assert payload["source"]["merlinVersion"]
            assert payload["source"]["merlinCommit"]
            assert payload["source"]["dataFingerprint"] == (
                f"sha256:{hashlib.sha256(csv_path.read_bytes()).hexdigest()}"
            )
    finally:
        if csv_path.exists():
            csv_path.unlink()


def test_export_lancelot_bundle_from_grid_candidate(client, tmp_path):
    csv_path = tmp_path / "_tmp_lancelot_export_grid.csv"
    csv_path.write_text(
        "timestamp,open,high,low,close,volume\n2025-01-01T00:00:00Z,1,1,1,1,1\n",
        encoding="utf-8",
    )

    try:
        with _temporary_active_db(f"bundle_export_grid_{uuid.uuid4().hex[:8]}"):
            study_id = "bundle_grid_1"
            _insert_lancelot_export_study(
                study_id=study_id,
                study_name="bundle_grid_1",
                optimization_mode="grid",
                csv_file_path=str(csv_path),
                csv_file_name="OKX_SOLUSDT.P, 5 2025.05.01-2025.11.20.csv",
                warmup_bars=500,
                config_json={
                    "fixed_params": {
                        "maType3": "HMA",
                        "useCloseCount_options": [True, False],
                        "undeclaredControl": "must-not-export",
                    }
                },
            )
            _insert_lancelot_export_trial(
                study_id=study_id,
                trial_number=9,
                params={"maType3": "EMA", "maLength3": 75, "useCloseCount": True},
            )

            response = client.post(
                f"/api/studies/{study_id}/export/lancelot",
                json={"trialNumber": 9},
            )

            assert response.status_code == 200
            payload = response.get_json()
            assert payload["symbol"] == "SOL/USDT:USDT"
            assert payload["timeframe"] == "5m"
            assert payload["warmupBars"] == 500
            assert payload["params"] == {
                "maType3": "EMA",
                "maLength3": 75,
                "useCloseCount": True,
                "dateFilter": False,
                "start": None,
                "end": None,
            }
            assert payload["source"]["trialNumber"] == 9
    finally:
        if csv_path.exists():
            csv_path.unlink()


@pytest.mark.parametrize(
    ("strategy_id", "strategy_version", "optimization_mode", "selection"),
    [
        ("s06_r_trend_v02", "v02", "optuna", {"trialNumber": 3}),
        ("s06_r_trend_v02_b2", "v02-b2", "wfa", {"windowNumber": 1}),
    ],
)
def test_lancelot_rejects_unsupported_v1_and_v2_before_export_work(
    client,
    monkeypatch,
    tmp_path,
    strategy_id,
    strategy_version,
    optimization_mode,
    selection,
):
    from core import storage
    from ui import server_routes_data

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("unsupported Lancelot export continued past strategy gate")

    for name in (
        "load_study_from_db",
        "_resolve_stored_execution_context",
        "_resolve_csv_path",
        "get_study_trial",
        "_find_wfa_window",
        "load_wfa_window_trials",
        "build_lancelot_partial_bundle",
    ):
        monkeypatch.setattr(server_routes_data, name, fail_if_called)
    monkeypatch.setattr(storage, "backfill_stitched_oos_metadata", fail_if_called)

    missing_csv = tmp_path / "must_not_be_read.csv"
    assert not missing_csv.exists()
    with _temporary_active_db(f"unsupported_lancelot_{uuid.uuid4().hex[:8]}"):
        study_id = f"unsupported_{strategy_id}_{optimization_mode}"
        _insert_lancelot_export_study(
            study_id=study_id,
            study_name=study_id,
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            optimization_mode=optimization_mode,
            csv_file_path=str(missing_csv),
            csv_file_name="must_not_be_read.csv",
            config_json={"fixed_params": {}},
        )
        response = client.post(
            f"/api/studies/{study_id}/export/lancelot", json=selection
        )

    assert response.status_code == 400
    payload = response.get_json()
    assert "currently supports only 's03_reversal_v10'" in payload["error"]
    diagnostic = _v2_runtime_diagnostic(response)
    assert diagnostic == {
        "severity": "error",
        "code": "LANCELOT_EXPORT_STRATEGY_UNSUPPORTED",
        "strategy_id": strategy_id,
        "path": "strategy_id",
        "variant": None,
        "message": payload["error"],
    }


def test_lancelot_supported_identity_disappearing_before_full_load_returns_404(
    client, monkeypatch
):
    from ui import server_routes_data

    monkeypatch.setattr(
        server_routes_data,
        "load_study_identity_from_db",
        lambda _study_id: {
            "study_id": "disappearing-study",
            "strategy_id": "s03_reversal_v10",
        },
    )
    monkeypatch.setattr(server_routes_data, "load_study_from_db", lambda _study_id: None)

    response = client.post(
        "/api/studies/disappearing-study/export/lancelot",
        json={"trialNumber": 1},
    )

    assert response.status_code == 404
    assert response.get_json() == {"error": "Study not found."}


def test_lancelot_strategy_change_between_identity_and_full_load_is_rejected(
    client, monkeypatch
):
    from ui import server_routes_data

    monkeypatch.setattr(
        server_routes_data,
        "load_study_identity_from_db",
        lambda _study_id: {
            "study_id": "changed-study",
            "strategy_id": "s03_reversal_v10",
        },
    )
    monkeypatch.setattr(
        server_routes_data,
        "load_study_from_db",
        lambda _study_id: {"study": {"strategy_id": "s06_r_trend_v02_b2"}},
    )

    response = client.post(
        "/api/studies/changed-study/export/lancelot",
        json={"trialNumber": 1},
    )

    assert response.status_code == 400
    diagnostic = _v2_runtime_diagnostic(response)
    assert diagnostic["code"] == "LANCELOT_EXPORT_STRATEGY_UNSUPPORTED"
    assert diagnostic["strategy_id"] == "s06_r_trend_v02_b2"
    assert diagnostic["path"] == "strategy_id"


def test_study_endpoint_includes_single_grid_settings(client):
    preview = {
        "total_space": 20,
        "total_space_label": "20",
        "actual_budget": 10,
        "actual_budget_label": "10",
        "coverage_pct": 50.0,
        "coverage_label": "50.0%",
        "allocation_method": "auto_sqrt_space",
        "modes": [
            {
                "mode": "cc_only",
                "space_size": 8,
                "space_label": "8",
                "budget": 4,
                "budget_label": "4",
                "coverage_pct": 50.0,
                "coverage_label": "50.0%",
                "generation": "LHS",
            }
        ],
    }
    config = _grid_sidebar_config()
    summary = {
        "requested_budget": 10,
        "actual_budget": 10,
        "grid": {"preview": preview},
        "optimization_time_seconds": 26,
    }

    with _temporary_active_db(f"grid_settings_single_{uuid.uuid4().hex[:8]}"):
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO studies (
                    study_id,
                    study_name,
                    strategy_id,
                    optimization_mode,
                    optimizer_mode,
                    config_json,
                    grid_requested_budget,
                    grid_actual_budget,
                    grid_coverage_pct,
                    grid_top_candidates,
                    grid_summary_json,
                    optimization_time_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "grid_settings_single",
                    "GRID_SETTINGS_SINGLE",
                    "s03_reversal_v10",
                    "grid",
                    "grid",
                    json.dumps(config),
                    10,
                    10,
                    50.0,
                    5,
                    json.dumps(summary),
                    26,
                ),
            )
            conn.commit()

        response = client.get("/api/studies/grid_settings_single")
        assert response.status_code == 200
        study = response.get_json()["study"]
        grid = study["grid_settings"]
        rows = {row["key"]: row["val"] for row in grid["rows"]}
        allocation = {row["key"]: row["val"] for row in grid["allocation_rows"]}

        assert grid["enabled"] is True
        assert grid["is_wfa_grid"] is False
        assert rows["Budget"] == "10 candidates"
        assert rows["Parameter Space"] == "20 combinations"
        assert rows["Coverage"] == "50.0%"
        assert rows["Workers"] == "6 Numba threads"
        assert rows["Fast Objectives"] == "Net Profit %"
        assert rows["Fast Primary"] == "Net Profit %"
        assert rows["Slow Refinement"] == "Off"
        assert rows["Runtime"] == "26s"
        assert allocation["Allocation"] == "Auto sqrt-space"
        assert allocation["CC only"] == "4 / 8 | 50.0% | LHS"


def test_study_endpoint_isolates_only_unexpected_grid_enrichment_failure(
    client, monkeypatch
):
    from ui import server_routes_data

    logged = []
    monkeypatch.setattr(
        server_routes_data,
        "build_grid_settings_view",
        lambda _study: (_ for _ in ()).throw(RuntimeError("preview exploded")),
    )
    monkeypatch.setattr(app.logger, "exception", lambda *args: logged.append(args))

    with _temporary_active_db(f"grid_isolation_{uuid.uuid4().hex[:8]}"):
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO studies (
                    study_id, study_name, strategy_id, optimization_mode, config_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "grid_isolation",
                    "GRID_ISOLATION",
                    "s03_reversal_v10",
                    "grid",
                    json.dumps(_grid_sidebar_config()),
                ),
            )
            conn.commit()

        response = client.get("/api/studies/grid_isolation")
        assert response.status_code == 200
        study = response.get_json()["study"]
        assert study["study_id"] == "grid_isolation"
        assert "grid_settings" not in study
        assert len(logged) == 1
        assert logged[0][1:] == ("grid_isolation", "s03_reversal_v10")


def test_export_lancelot_bundle_from_wfa_window_uses_window_trial_number(client, tmp_path):
    csv_path = tmp_path / "_tmp_lancelot_export_wfa.csv"
    csv_path.write_text(
        "timestamp,open,high,low,close,volume\n2025-01-01T00:00:00Z,1,1,1,1,1\n",
        encoding="utf-8",
    )

    try:
        with _temporary_active_db(f"bundle_export_wfa_{uuid.uuid4().hex[:8]}"):
            study_id = "bundle_wfa_1"
            _insert_lancelot_export_study(
                study_id=study_id,
                study_name="bundle_wfa_1",
                optimization_mode="wfa",
                csv_file_path=str(csv_path),
                csv_file_name="OKX_BTCUSDT.P, 1h 2025.05.01-2025.11.20.csv",
                warmup_bars=1500,
                config_json={
                    "fixed_params": {
                        "maType3": "HMA",
                        "useTBands_options": [True, False],
                        "undeclaredControl": "must-not-export",
                    }
                },
            )
            _insert_lancelot_export_wfa_window(
                study_id=study_id,
                window_number=1,
                best_params={"maLength3": 75},
                is_best_trial_number=7,
            )

            response = client.post(
                f"/api/studies/{study_id}/export/lancelot",
                json={"windowNumber": 1},
            )

            assert response.status_code == 200
            payload = response.get_json()
            assert payload["symbol"] == "BTC/USDT:USDT"
            assert payload["timeframe"] == "1h"
            assert payload["warmupBars"] == 1500
            assert payload["params"] == {
                "maLength3": 75,
                "dateFilter": False,
                "start": None,
                "end": None,
            }
            assert payload["source"]["trialNumber"] == 7
    finally:
        if csv_path.exists():
            csv_path.unlink()


def test_export_lancelot_bundle_from_wfa_window_falls_back_to_selected_module_trial(client, tmp_path):
    csv_path = tmp_path / "_tmp_lancelot_export_wfa_selected.csv"
    csv_path.write_text(
        "timestamp,open,high,low,close,volume\n2025-01-01T00:00:00Z,1,1,1,1,1\n",
        encoding="utf-8",
    )

    try:
        with _temporary_active_db(f"bundle_export_wfa_sel_{uuid.uuid4().hex[:8]}"):
            study_id = "bundle_wfa_selected_1"
            _insert_lancelot_export_study(
                study_id=study_id,
                study_name="bundle_wfa_selected_1",
                optimization_mode="wfa",
                csv_file_path=str(csv_path),
                csv_file_name="OKX_ETHUSDT.P, 60 2025.05.01-2025.11.20.csv",
            )
            _insert_lancelot_export_wfa_window(
                study_id=study_id,
                window_number=1,
                best_params={"maLength3": 125},
                best_params_source="forward_test",
                is_best_trial_number=None,
            )
            _insert_lancelot_export_wfa_trial(
                study_id=study_id,
                window_number=1,
                module_type="forward_test",
                trial_number=19,
                params={"maLength3": 125},
                is_selected=True,
            )

            response = client.post(
                f"/api/studies/{study_id}/export/lancelot",
                json={"windowNumber": 1},
            )

            assert response.status_code == 200
            payload = response.get_json()
            assert payload["symbol"] == "ETH/USDT:USDT"
            assert payload["timeframe"] == "1h"
            assert payload["source"]["trialNumber"] == 19
    finally:
        if csv_path.exists():
            csv_path.unlink()


def test_export_lancelot_bundle_requires_selection_payload(client, tmp_path):
    csv_path = tmp_path / "_tmp_lancelot_export_missing.csv"
    csv_path.write_text(
        "timestamp,open,high,low,close,volume\n2025-01-01T00:00:00Z,1,1,1,1,1\n",
        encoding="utf-8",
    )

    try:
        with _temporary_active_db(f"bundle_export_missing_{uuid.uuid4().hex[:8]}"):
            study_id = "bundle_missing_selection"
            _insert_lancelot_export_study(
                study_id=study_id,
                study_name="bundle_missing_selection",
                optimization_mode="optuna",
                csv_file_path=str(csv_path),
                csv_file_name="OKX_LINKUSDT.P, 15 2025.05.01-2025.11.20.csv",
            )

            response = client.post(
                f"/api/studies/{study_id}/export/lancelot",
                json={},
            )

            assert response.status_code == 400
            assert response.get_json()["error"] == "trialNumber is required for bundle export."
    finally:
        if csv_path.exists():
            csv_path.unlink()


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


def test_analytics_summary_wfa_phase1_contract(client):
    with _temporary_active_db(f"analytics_wfa_{uuid.uuid4().hex[:8]}"):
        _insert_analytics_study(
            study_id="wfa_a1",
            study_name="WFA_A1",
            strategy_id="s01_trailing_ma",
            strategy_version="2.1",
            optimization_mode="wfa",
            csv_file_name="OKX_LINKUSDT.P, 15 2025.05.01-2025.11.20.csv",
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


# ---------------------------------------------------------------------------
# Workstream D: Constraints row in the shared Grid Settings sidebar
# ---------------------------------------------------------------------------

from ui.server_services import build_grid_settings_view


_TWO_ENABLED_CONSTRAINTS = [
    {"metric": "total_trades", "threshold": 30, "enabled": True},
    {"metric": "max_drawdown_pct", "threshold": 30.0, "enabled": True},
    {"metric": "net_profit_pct", "threshold": 5.0, "enabled": False},
]
_EXPECTED_CONSTRAINTS_TEXT = "Total Trades >= 30, Min Drawdown % <= 30"


def _constraints_row(view):
    assert view is not None
    return next((row["val"] for row in view["rows"] if row["key"] == "Constraints"), None)


def _standalone_grid_study(**overrides):
    study = {
        "optimization_mode": "grid",
        "config_json": {
            "grid_budget": 10,
            "grid_seed": 42,
            "grid_top_candidates": 5,
            "grid_fast_objectives": ["net_profit_pct"],
        },
    }
    study.update(overrides)
    return study


def _saved_v2_full_enumeration_grid_summary():
    return {
        "grid": {
            "preview": {
                "full_candidate_count": 48_480,
            },
            "allocation": {
                "allocation_method": "full_enumeration_v2",
                "mode_space_sizes": {"bracket": 480, "trail": 48_000},
                "mode_budgets": {"bracket": 480, "trail": 48_000},
                "mode_coverage_pct": {"bracket": 100.0, "trail": 100.0},
            },
        },
    }


def _grid_settings_rows(view):
    assert view is not None
    return {row["key"]: row["val"] for row in view["rows"]}


def _grid_settings_allocation_rows(view):
    assert view is not None
    return {row["key"]: row["val"] for row in view["allocation_rows"]}


def test_grid_settings_derivation_memo_ignores_study_row_identity(monkeypatch):
    from ui import server_services

    calls = []

    def fake_derive(config, study):
        calls.append((deepcopy(config), deepcopy(study)))
        return {"candidate_count": 10, "modes": []}

    monkeypatch.setattr(server_services, "_derive_grid_preview", fake_derive)
    config = _grid_sidebar_config()
    first = {
        "study_id": "grid-one",
        "strategy_id": "s03_reversal_v10",
        "optimization_mode": "grid",
        "config_json": config,
    }
    second = deepcopy(first)
    second["study_id"] = "grid-two"
    memo = {}

    assert build_grid_settings_view(first, memo=memo)["available"] is True
    assert build_grid_settings_view(second, memo=memo)["available"] is True
    assert len(calls) == 1


def test_grid_settings_memo_caches_expected_failure_and_keeps_envelope_truthful(
    monkeypatch,
):
    from ui import server_services

    calls = []

    def fail_derive(_config, study):
        calls.append(study["study_id"])
        raise server_services._grid_settings_error(
            "stored_runtime_unavailable",
            study["strategy_id"],
            "config_json.v2_runtime",
            "Stored runtime is unavailable.",
        )

    monkeypatch.setattr(server_services, "_derive_grid_preview", fail_derive)
    base = {
        "study_id": "one",
        "strategy_id": "s06_r_trend_v02_b2",
        "optimization_mode": "grid",
        "config_json": _grid_sidebar_config(),
        "grid_summary": {"requested_budget": 10},
    }
    memo = {}
    first = build_grid_settings_view(base, memo=memo)
    second_study = deepcopy(base)
    second_study["study_id"] = "two"
    second = build_grid_settings_view(second_study, memo=memo)

    assert calls == ["one"]
    for view in (first, second):
        assert view["available"] is True
        assert view["unavailable_reason"] is None
        assert view["derivation_status"] == "unavailable"
        assert view["derivation_unavailable_reason"] == "stored_runtime_unavailable"
        assert view["diagnostics"][0]["code"] == "V2_STORED_RUNTIME_METADATA_INCOMPATIBLE"
    first["diagnostics"].clear()
    assert second["diagnostics"]


def test_grid_settings_without_memo_skips_memo_key_serialization(monkeypatch):
    from ui import server_services

    monkeypatch.setattr(
        server_services, "_derive_grid_preview", lambda _config, _study: {"modes": []}
    )
    monkeypatch.setattr(
        server_services,
        "_grid_settings_memo_key",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("memo key computed")
        ),
    )
    view = build_grid_settings_view(
        {
            "strategy_id": "s03_reversal_v10",
            "optimization_mode": "grid",
            "config_json": _grid_sidebar_config(),
        }
    )
    assert view["available"] is True
    assert view["derivation_status"] == "succeeded"


@pytest.mark.parametrize("noncanonical", [float("nan"), float("inf"), float("-inf")])
def test_grid_settings_noncanonical_memo_key_derives_without_caching(
    monkeypatch, noncanonical
):
    from ui import server_services

    calls = []
    monkeypatch.setattr(
        server_services,
        "_derive_grid_preview",
        lambda _config, _study: calls.append(True) or {"modes": []},
    )
    config = _grid_sidebar_config()
    config["noncanonicalMemoValue"] = noncanonical
    memo = {"sentinel": object()}

    study = {
        "strategy_id": "s03_reversal_v10",
        "optimization_mode": "grid",
        "config_json": config,
    }
    first_view = build_grid_settings_view(study, memo=memo)
    second_view = build_grid_settings_view(study, memo=memo)

    assert first_view["available"] is True
    assert first_view["derivation_status"] == "succeeded"
    assert second_view == first_view
    assert calls == [True, True]
    assert set(memo) == {"sentinel"}
    assert None not in memo


def test_grid_settings_unusable_derivation_sets_only_unavailable_reason(monkeypatch):
    from ui import server_services

    monkeypatch.setattr(
        server_services,
        "_derive_grid_preview",
        lambda _config, study: (_ for _ in ()).throw(
            server_services._grid_settings_error(
                "stored_runtime_unavailable",
                study["strategy_id"],
                "config_json.v2_runtime",
                "Stored runtime is unavailable.",
            )
        ),
    )
    view = build_grid_settings_view(
        {
            "strategy_id": "s06_r_trend_v02_b2",
            "optimization_mode": "grid",
            "config_json": _grid_sidebar_config(),
        }
    )
    assert view["available"] is False
    assert view["unavailable_reason"] == "stored_runtime_unavailable"
    assert view["derivation_unavailable_reason"] == "stored_runtime_unavailable"


def test_grid_settings_unknown_strategy_is_viewable_without_substitution():
    study = {
        "strategy_id": "removed_strategy",
        "optimization_mode": "grid",
        "config_json": _grid_sidebar_config(),
    }
    view = build_grid_settings_view(study)

    assert view["available"] is False
    assert view["unavailable_reason"] == "strategy_unavailable"
    assert view["diagnostics"][0]["strategy_id"] == "removed_strategy"


def _assert_saved_v2_full_enumeration_grid_settings(view):
    rows = _grid_settings_rows(view)
    allocation_rows = _grid_settings_allocation_rows(view)

    assert rows["Sampling"] == "Full enumeration"
    assert "Seed" not in rows
    assert rows["Parameter Space"] != "-"
    assert rows["Diversity"] == "On, max 2 / strategy group"
    assert allocation_rows["Allocation"] == "Full enumeration"
    assert "Bracket" in allocation_rows
    assert "Trail" in allocation_rows


def test_grid_settings_saved_v2_full_enumeration_labels():
    study = _standalone_grid_study(grid_summary=_saved_v2_full_enumeration_grid_summary())

    view = build_grid_settings_view(study)

    assert view["is_wfa_grid"] is False
    _assert_saved_v2_full_enumeration_grid_settings(view)


def test_grid_settings_saved_wfa_v2_full_enumeration_labels():
    study = {
        "optimization_mode": "wfa",
        "optimizer_mode": "grid",
        "config_json": {"optimization_mode": "grid", "grid_budget": 10},
        "grid_summary": _saved_v2_full_enumeration_grid_summary(),
    }

    view = build_grid_settings_view(study)

    assert view["is_wfa_grid"] is True
    _assert_saved_v2_full_enumeration_grid_settings(view)


def test_grid_settings_s03_v1_keeps_mode_ma_period_diversity_wording():
    study = _standalone_grid_study(
        config_json={
            "strategy_id": "s03_reversal_v10",
            "grid_v2_planning_policy": "full",
            "grid_allocation_method": "auto_sqrt_space",
            "grid_budget": 10,
            "grid_seed": 42,
            "grid_diversity_enabled": True,
            "grid_diversity_max_per_group": 2,
        },
    )

    rows = _grid_settings_rows(build_grid_settings_view(study))

    assert rows["Diversity"] == "On, max 2 / Mode+MA+Period"


def test_grid_settings_effective_full_overrides_configured_automatic_allocation():
    study = _standalone_grid_study(
        config_json={
            "grid_v2_planning_policy": "sampled",
            "grid_allocation_method": "auto_sqrt_space",
            "grid_budget": 100_000,
        },
        grid_summary={
            "engine": "v2",
            "grid_v2_plan_fingerprint": "current-v2-plan",
            "grid": {
                "backend": {"engine": "v2", "profile": "full_enumeration_v2"},
                "planning": {
                    "requested_policy": "sampled",
                    "effective_policy": "full",
                    "effective_allocation_method": "full_enumeration_v2",
                    "planned_candidate_count": 30,
                    "requested_budget": 100_000,
                    "plan_identity_schema_version": "grid_v2_plan_identity_v2",
                    "planning_policy_version": "grid_v2_planning_policy_v2",
                },
                "preview": {
                    "full_candidate_count": 30,
                    "planned_candidate_count": 30,
                    "coverage_pct": 100.0,
                },
            },
        },
    )

    view = build_grid_settings_view(study)
    rows = _grid_settings_rows(view)
    allocation_rows = _grid_settings_allocation_rows(view)
    assert rows["Planning"].startswith("sampled")
    assert rows["Planning"].endswith("full")
    assert rows["Diversity"] == "On, max 2 / strategy group"
    assert "Seed" not in rows
    assert allocation_rows["Allocation"] == "Full enumeration"


def test_grid_settings_sampled_v2_uses_generic_diversity_wording():
    study = _standalone_grid_study(
        config_json={
            "grid_v2_planning_policy": "sampled",
            "grid_allocation_method": "proportional_space",
            "grid_seed": 42,
            "grid_diversity_enabled": True,
            "grid_diversity_max_per_group": 2,
        },
        grid_summary={
            "grid": {
                "planning": {"requested_policy": "sampled", "effective_policy": "sampled"},
                "preview": {
                    "engine": "v2",
                    "profile": "full_enumeration_v2",
                    "full_candidate_count": 100,
                    "planned_candidate_count": 10,
                },
                "allocation": {"allocation_method": "proportional_space"},
            },
        },
    )

    rows = _grid_settings_rows(build_grid_settings_view(study))
    assert rows["Diversity"] == "On, max 2 / strategy group"
    assert rows["Seed"] == "42"


@pytest.mark.parametrize(
    ("marker_section", "marker_key", "marker_value"),
    [
        ("preview", "engine", "v2"),
        ("summary", "engine", "v2"),
        ("backend", "engine", "v2"),
        ("summary", "grid_v2_plan_fingerprint", "v2-plan-fingerprint"),
        ("planning", "plan_identity_schema_version", "grid_v2_plan_identity_v2"),
        ("planning", "planning_policy_version", "grid_v2_planning_policy_v2"),
        ("preview", "profile", "full_enumeration_v2"),
        ("backend", "profile", "full_enumeration_v2"),
        ("allocation", "allocation_method", "full_enumeration_v2"),
    ],
)
def test_grid_settings_recognizes_authoritative_v2_markers(
    marker_section,
    marker_key,
    marker_value,
):
    grid_summary = {
        "grid": {
            "backend": {},
            "planning": {"requested_policy": "sampled", "effective_policy": "sampled"},
            "preview": {"full_candidate_count": 100, "planned_candidate_count": 10},
            "allocation": {"allocation_method": "proportional_space"},
        },
    }
    marker_target = grid_summary if marker_section == "summary" else grid_summary["grid"][marker_section]
    marker_target[marker_key] = marker_value
    study = _standalone_grid_study(grid_summary=grid_summary)

    rows = _grid_settings_rows(build_grid_settings_view(study))

    assert rows["Diversity"] == "On, max 2 / strategy group"
    assert rows["Seed"] == "42"


def test_grid_settings_prefers_v2_planning_json_over_ambiguous_legacy_columns():
    study = _standalone_grid_study(
        grid_requested_budget=999,
        grid_actual_budget=888,
        grid_coverage_pct=88.8,
        grid_summary={
            "requested_budget": 100,
            "actual_budget": 50,
            "grid": {
                "planning": {
                    "requested_policy": "sampled",
                    "effective_policy": "sampled",
                    "requested_budget": 100,
                    "planned_candidate_count": 50,
                },
                "preview": {
                    "full_candidate_count": 1_000,
                    "planned_candidate_count": 50,
                    "coverage_pct": 5.0,
                    "requested_planning_policy": "sampled",
                    "effective_planning_policy": "sampled",
                },
            },
        },
    )

    rows = _grid_settings_rows(build_grid_settings_view(study))
    assert rows["Planning"] == "sampled"
    assert rows["Budget"] == "50 candidates"
    assert rows["Parameter Space"] == "1k combinations"
    assert rows["Coverage"] == "5.0%"


def test_grid_settings_constraints_standalone_two_enabled():
    study = _standalone_grid_study(constraints=list(_TWO_ENABLED_CONSTRAINTS))
    assert _constraints_row(build_grid_settings_view(study)) == _EXPECTED_CONSTRAINTS_TEXT


def test_grid_settings_constraints_wfa_grid_two_enabled():
    study = {
        "optimization_mode": "wfa",
        "optimizer_mode": "grid",
        "config_json": {"optimization_mode": "grid", "grid_budget": 10},
        "constraints_json": json.dumps(_TWO_ENABLED_CONSTRAINTS),
    }
    view = build_grid_settings_view(study)
    assert view["is_wfa_grid"] is True
    assert _constraints_row(view) == _EXPECTED_CONSTRAINTS_TEXT


def test_grid_settings_constraints_none_when_absent():
    assert _constraints_row(build_grid_settings_view(_standalone_grid_study())) == "None"


def test_grid_settings_constraints_excludes_disabled():
    disabled = [
        {"metric": "total_trades", "threshold": 30, "enabled": False},
        {"metric": "max_drawdown_pct", "threshold": 30.0, "enabled": False},
    ]
    study = _standalone_grid_study(constraints=disabled)
    assert _constraints_row(build_grid_settings_view(study)) == "None"


def test_grid_settings_constraints_from_constraints_json_string():
    # Analytics passes the raw constraints_json column (a JSON string).
    study = _standalone_grid_study(constraints_json=json.dumps(_TWO_ENABLED_CONSTRAINTS))
    assert _constraints_row(build_grid_settings_view(study)) == _EXPECTED_CONSTRAINTS_TEXT


def test_grid_settings_constraints_config_fallback_shapes():
    top_level = _standalone_grid_study()
    top_level["config_json"]["constraints"] = list(_TWO_ENABLED_CONSTRAINTS)
    assert _constraints_row(build_grid_settings_view(top_level)) == _EXPECTED_CONSTRAINTS_TEXT

    nested = _standalone_grid_study()
    nested["config_json"]["optuna_config"] = {"constraints": list(_TWO_ENABLED_CONSTRAINTS)}
    assert _constraints_row(build_grid_settings_view(nested)) == _EXPECTED_CONSTRAINTS_TEXT


def test_grid_settings_constraints_results_and_analytics_row_identical():
    # Results loads constraints into a parsed list; Analytics keeps the raw JSON
    # string column.  Both must yield an identical Constraints row.
    results_shape = _standalone_grid_study(
        constraints=list(_TWO_ENABLED_CONSTRAINTS),
        constraints_json=list(_TWO_ENABLED_CONSTRAINTS),
    )
    analytics_shape = _standalone_grid_study(
        constraints_json=json.dumps(_TWO_ENABLED_CONSTRAINTS),
    )
    assert _constraints_row(build_grid_settings_view(results_shape)) == _constraints_row(
        build_grid_settings_view(analytics_shape)
    )
