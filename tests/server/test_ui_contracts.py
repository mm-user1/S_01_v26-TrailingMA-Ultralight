"""Server ui contracts contracts."""

import logging
import re

from ui.server import app
from core.grid_engine import GRID_SUPPORTED_FAST_OBJECTIVES, GRID_SUPPORTED_SLOW_OBJECTIVES

from ._helpers import REPO_ROOT


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
    repo_root = REPO_ROOT
    index_html = (repo_root / "src" / "ui" / "templates" / "index.html").read_text(encoding="utf-8")
    ui_handlers_js = (repo_root / "src" / "ui" / "static" / "js" / "ui-handlers.js").read_text(encoding="utf-8")
    strategy_config_js = (repo_root / "src" / "ui" / "static" / "js" / "strategy-config.js").read_text(encoding="utf-8")
    results_html = (repo_root / "src" / "ui" / "templates" / "results.html").read_text(encoding="utf-8")
    results_tables_js = (repo_root / "src" / "ui" / "static" / "js" / "results-tables.js").read_text(encoding="utf-8")
    analytics_js = (repo_root / "src" / "ui" / "static" / "js" / "analytics.js").read_text(encoding="utf-8")
    wfa_results_js = (repo_root / "src" / "ui" / "static" / "js" / "wfa-results-ui.js").read_text(encoding="utf-8")
    queue_js = (repo_root / "src" / "ui" / "static" / "js" / "queue.js").read_text(encoding="utf-8")
    run_routes_py = (repo_root / "src" / "ui" / "server_routes_run.py").read_text(encoding="utf-8")

    assert 'id="optimizerModeOptunaLabel"' in index_html
    assert "getElementById('optimizerModeOptunaLabel')" in ui_handlers_js
    assert "Grid v1 is supported only for S03 Reversal v10." not in ui_handlers_js
    assert "No fast Grid backend is available." in ui_handlers_js
    assert "в–ѕ" not in index_html
    assert "&#9660;" in index_html
    assert "GRID SETTINGS" in index_html
    assert 'id="optuna-settings-section"' in results_html
    assert 'id="gridFastObjectivesSection"' in index_html
    assert 'class="grid-fast-objective-checkbox"' in index_html
    assert index_html.count('class="grid-fast-objective-checkbox"') == 8
    assert 'data-objective="sharpe_ratio"' in index_html
    assert index_html.count('class="objective-checkbox" data-objective="sharpe_daily"') == 1
    assert index_html.count('class="grid-fast-objective-checkbox" data-objective="sharpe_daily"') == 1
    assert index_html.count('class="grid-slow-objective-checkbox" data-objective="sharpe_daily"') == 0
    assert 'data-objective="sqn"' in index_html
    assert "Select 1-6 objectives." in index_html
    assert "const GRID_MAX_FAST_OBJECTIVES = 6" in ui_handlers_js
    assert "'sharpe_ratio'" in ui_handlers_js
    assert "'sqn'" in ui_handlers_js
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
    assert 'id="wfCalendarMonths"' in index_html
    assert 'id="wfIsPeriodLabel"' in index_html
    assert 'id="wfOosPeriodLabel"' in index_html
    assert "function syncWfaModeUi()" in ui_handlers_js
    assert "isPeriodMonths" in ui_handlers_js
    assert "wf_is_period_months" in ui_handlers_js
    assert 'id="wfa-is-period-key"' in results_html
    assert 'id="wfa-oos-period-key"' in results_html
    assert "isPeriodMonths" in results_tables_js
    assert "IS (${periodUnit})" in analytics_js
    assert "appendQueueWfaPeriodFields" in queue_js
    assert "WFA-F'} ${facts.compact}" in queue_js
    assert run_routes_py.count("**wfa_period_values") == 4
    wfa_queue_load = queue_js[queue_js.index("const isWfaMode = item.mode === 'wfa'"):]
    assert wfa_queue_load.index("setCheckboxValue('wfCalendarMonths'") < wfa_queue_load.index(
        "setInputValue(\n      'wfIsPeriodDays'"
    )
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
    assert "sharpe_daily: window.is_sharpe_daily" in wfa_results_js
    assert "sharpe_daily: window.oos_sharpe_daily" in wfa_results_js
    assert "objective_values: isObjectiveValues" in wfa_results_js
    assert "objective_values: oosObjectiveValues" in wfa_results_js


def test_common_fast_objectives_match_javascript_and_start_page_controls():
    repo_root = REPO_ROOT
    index_html = (repo_root / "src" / "ui" / "templates" / "index.html").read_text(encoding="utf-8")
    ui_handlers_js = (repo_root / "src" / "ui" / "static" / "js" / "ui-handlers.js").read_text(
        encoding="utf-8"
    )

    declaration = re.search(
        r"const\s+GRID_SUPPORTED_OBJECTIVES\s*=\s*new Set\(\[(.*?)\]\);",
        ui_handlers_js,
        flags=re.DOTALL,
    )
    assert declaration is not None
    javascript_objectives = set(re.findall(r"['\"]([a-z0-9_]+)['\"]", declaration.group(1)))

    control_objectives = []
    for tag in re.findall(r"<input\b[^>]*>", index_html, flags=re.IGNORECASE):
        class_attribute = re.search(r"class=['\"]([^'\"]*)['\"]", tag, flags=re.IGNORECASE)
        if class_attribute is None or "grid-fast-objective-checkbox" not in class_attribute.group(1).split():
            continue
        objective_attribute = re.search(r"data-objective=['\"]([^'\"]+)['\"]", tag, flags=re.IGNORECASE)
        assert objective_attribute is not None
        control_objectives.append(objective_attribute.group(1))

    python_objectives = set(GRID_SUPPORTED_FAST_OBJECTIVES)
    assert python_objectives <= javascript_objectives
    for objective in python_objectives:
        assert control_objectives.count(objective) == 1

    assert "sharpe_daily" not in GRID_SUPPORTED_SLOW_OBJECTIVES


def _javascript_function_source(source: str, signature: str, next_signature: str) -> str:
    start = source.index(signature)
    end = source.index(next_signature, start)
    return source[start:end]


def test_strategy_readiness_uses_captured_requested_identity_and_clears_stale_state():
    repo_root = REPO_ROOT
    source = (repo_root / "src" / "ui" / "static" / "js" / "strategy-config.js").read_text(
        encoding="utf-8"
    )
    load_source = _javascript_function_source(
        source,
        "async function loadStrategyConfig(strategyId)",
        "function updateStrategyInfo(config)",
    )

    assert "window.currentStrategyConfigId = null" in source
    assert "function isCurrentStrategyConfigReady()" in source
    assert "window.currentStrategyConfigId === window.currentStrategyId" in source
    readiness_source = _javascript_function_source(
        source,
        "function isCurrentStrategyConfigReady()",
        "function clearStrategyGeneratedState()",
    )
    assert "config.id" not in readiness_source
    assert "const requestedStrategyId = String(strategyId || '').trim()" in load_source
    assert "window.currentStrategyConfigId = requestedStrategyId" in load_source
    provisional_config = load_source.index("window.currentStrategyConfig = config")
    optimizer_render = load_source.index("generateOptimizerForm(config)")
    accepted_identity = load_source.index("window.currentStrategyConfigId = requestedStrategyId")
    assert provisional_config < optimizer_render < accepted_identity
    assert "window.currentStrategyConfig === provisionalConfig" in load_source
    assert "clearStrategyGeneratedState()" in load_source


def test_strategy_readiness_reset_targets_generated_fields_and_preview_only():
    repo_root = REPO_ROOT
    source = (repo_root / "src" / "ui" / "static" / "js" / "strategy-config.js").read_text(
        encoding="utf-8"
    )
    reset_source = _javascript_function_source(
        source,
        "function clearStrategyGeneratedState()",
        "function shouldRenderDynamicBacktestParam",
    )

    for element_id in (
        "backtestParamsContent",
        "optimizerParamsContainer",
        "strategyInfo",
        "strategyName",
        "strategyVersion",
        "strategyDescription",
        "strategyParamCount",
        "gridEnabledModes",
        "gridV2ManualAllocation",
    ):
        assert element_id in reset_source
    assert "delete gridModesContainer.dataset.profileKey" in reset_source
    assert "resetGridPreviewState()" in reset_source
    for preserved_id in (
        "dateFilter",
        "startDate",
        "startTime",
        "endDate",
        "endTime",
        "warmupBars",
        "csvDirectory",
        "dbTarget",
        "gridBudget",
        "gridSeed",
        "gridV2PlanningPolicy",
        "gridAllocAuto",
        "gridFastPrimaryObjective",
        "wfIsPeriodDays",
    ):
        assert f"getElementById('{preserved_id}')" not in reset_source


def test_optimizer_mode_sync_waits_for_strategy_config_before_grid_availability_check():
    repo_root = REPO_ROOT
    source = (repo_root / "src" / "ui" / "static" / "js" / "ui-handlers.js").read_text(
        encoding="utf-8"
    )
    sync_source = _javascript_function_source(
        source,
        "function syncOptimizerModeUI()",
        "async function submitOptimization",
    )

    assert sync_source.index("if (!strategyConfig)") < sync_source.index(
        "const gridMeta = getEnabledGridMetadata()"
    )


def test_form_action_readiness_guards_precede_request_building_and_wfa_dispatch():
    repo_root = REPO_ROOT
    ui_source = (repo_root / "src" / "ui" / "static" / "js" / "ui-handlers.js").read_text(
        encoding="utf-8"
    )
    queue_source = (repo_root / "src" / "ui" / "static" / "js" / "queue.js").read_text(
        encoding="utf-8"
    )

    backtest = _javascript_function_source(
        ui_source,
        "async function executeBacktestRun(",
        "async function runBacktest(event)",
    )
    preview = _javascript_function_source(
        ui_source,
        "async function updateGridPreview()",
        "function syncGridParameterOptions()",
    )
    submit = ui_source[ui_source.index("async function submitOptimization(event)") :]
    collect = _javascript_function_source(
        queue_source,
        "function collectQueueItem()",
        "function addToQueue(item)",
    )
    walkforward = _javascript_function_source(
        ui_source,
        "async function runWalkForward(",
        "async function triggerDownloadFromResponse(",
    )

    assert backtest.index("if (!isCurrentStrategyConfigReady())") < backtest.index(
        "runBacktestRequest(formData)"
    )
    assert preview.index("if (!isCurrentStrategyConfigReady()) return") < preview.index(
        "gatherFormState()"
    )
    assert collect.index("if (!isCurrentStrategyConfigReady())") < collect.index(
        "buildCurrentOptimizerConfig"
    )
    assert submit.index("await runQueue()") < submit.index("if (!isCurrentStrategyConfigReady())")
    assert submit.index("if (!isCurrentStrategyConfigReady())") < submit.index("if (wfEnabled)")
    assert "isCurrentStrategyConfigReady" not in walkforward
    assert "STRATEGY_CONFIG_NOT_READY_MESSAGE" in backtest
    assert "STRATEGY_CONFIG_NOT_READY_MESSAGE" in submit
    assert "STRATEGY_CONFIG_NOT_READY_MESSAGE" in collect


def test_queue_form_load_requires_readiness_but_persisted_queue_execution_does_not():
    repo_root = REPO_ROOT
    queue_source = (repo_root / "src" / "ui" / "static" / "js" / "queue.js").read_text(
        encoding="utf-8"
    )
    ensure = _javascript_function_source(
        queue_source,
        "async function ensureQueueItemStrategyLoaded(item)",
        "async function loadQueueItemIntoForm(",
    )
    run_queue = queue_source[queue_source.index("async function runQueue()") :]

    assert "const loaded = await loadStrategyConfig(strategyId)" in ensure
    assert "!loaded || !isCurrentStrategyConfigReady()" in ensure
    assert "isCurrentStrategyConfigReady" not in run_queue


def test_strategy_config_api_reuses_shared_error_reader_without_magic_use_count():
    repo_root = REPO_ROOT
    api_source = (repo_root / "src" / "ui" / "static" / "js" / "api.js").read_text(
        encoding="utf-8"
    )
    api_test_source = (repo_root / "tests" / "js" / "test_api_error_message.js").read_text(
        encoding="utf-8"
    )
    fetch_config = _javascript_function_source(
        api_source,
        "async function fetchStrategyConfig(strategyId)",
        "async function readApiErrorMessage(",
    )

    assert "await readApiErrorMessage(" in fetch_config
    assert "Server returned ${response.status}" not in fetch_config
    assert "helperUses.length" not in api_test_source
    for name in (
        "fetchStrategyConfig",
        "runBacktestRequest",
        "downloadBacktestTradesRequest",
        "runOptimizationRequest",
    ):
        assert name in api_test_source


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
