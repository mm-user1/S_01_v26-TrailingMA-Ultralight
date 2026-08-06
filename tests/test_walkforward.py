import hashlib
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core import storage
from core.walkforward_engine import (
    ISPipelineResult,
    MAX_WFA_PERIOD_MONTHS,
    OOSStitchedResult,
    WFConfig,
    WFResult,
    WalkForwardEngine,
    WindowResult,
    build_calendar_month_windows,
)
from core.optuna_engine import OptimizationResult
from core.post_process import (
    DSRConfig,
    DSRResult,
    PostProcessConfig,
    StressTestConfig,
    _ft_result_from_payload,
    _ft_worker_entry,
)
from core.engine_v2 import build_v2_runtime_metadata
from core.backtest_engine import StrategyResult, TradeRecord
from core.backtest_engine import load_data
from strategies import get_strategy_config
from strategies.s03_reversal_v11_regime_er_b2.strategy import normalized_params as normalized_s03_params


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


def _build_wf_result(strategy_id: str):
    wf_config = WFConfig(strategy_id=strategy_id, is_period_days=180, oos_period_days=60)
    params = _build_params_from_config(strategy_id)
    engine = WalkForwardEngine(wf_config, {}, {})
    param_id = engine._create_param_id(params)

    window_result = WindowResult(
        window_id=1,
        is_start=pd.Timestamp("2025-01-01", tz="UTC"),
        is_end=pd.Timestamp("2025-06-29", tz="UTC"),
        oos_start=pd.Timestamp("2025-06-30", tz="UTC"),
        oos_end=pd.Timestamp("2025-08-28", tz="UTC"),
        best_params=params,
        param_id=param_id,
        is_net_profit_pct=1.0,
        is_max_drawdown_pct=0.5,
        is_total_trades=5,
        oos_net_profit_pct=2.0,
        oos_max_drawdown_pct=1.0,
        oos_total_trades=4,
        oos_equity_curve=[100.0, 102.0],
        oos_timestamps=[
            pd.Timestamp("2025-06-30", tz="UTC"),
            pd.Timestamp("2025-08-28", tz="UTC"),
        ],
    )

    stitched = OOSStitchedResult(
        final_net_profit_pct=2.0,
        max_drawdown_pct=1.0,
        total_trades=4,
        wfe=100.0,
        oos_win_rate=100.0,
        equity_curve=[100.0, 102.0],
        timestamps=[pd.Timestamp("2025-06-30", tz="UTC"), pd.Timestamp("2025-08-28", tz="UTC")],
        window_ids=[1, 1],
    )

    result = WFResult(
        config=wf_config,
        windows=[window_result],
        stitched_oos=stitched,
        strategy_id=strategy_id,
        total_windows=1,
        trading_start_date=window_result.is_start,
        trading_end_date=window_result.oos_end,
        warmup_bars=wf_config.warmup_bars,
    )

    return result, params, param_id


def _capture_grid_wfa_window_error(monkeypatch, message):
    engine = WalkForwardEngine(
        WFConfig(strategy_id="s03_reversal_v10", is_period_days=30, oos_period_days=10),
        {
            "optimization_mode": "grid",
            "grid_fast_objectives": ["net_profit_pct"],
            "objectives": ["net_profit_pct"],
        },
        {},
    )
    source_error = ValueError(message)
    monkeypatch.setattr(
        engine,
        "_run_optuna_on_window",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(source_error),
    )
    frame = pd.DataFrame(index=pd.date_range("2025-01-01", periods=2, freq="D", tz="UTC"))

    with pytest.raises(ValueError) as captured:
        engine._run_window_is_pipeline(
            frame,
            pd.Timestamp("2025-01-01T00:00:00Z"),
            pd.Timestamp("2025-01-31T00:00:00Z"),
            7,
        )
    return source_error, captured.value


@pytest.mark.parametrize(
    ("ranking_error", "expected", "excluded"),
    [
        (
            "Grid fast screening produced no candidates with usable objective values for: sqn",
            (
                "WFA window 7",
                "Grid fast screening",
                "for: sqn. SQN is undefined below 30 completed trades.",
            ),
            ("Sharpe can be undefined",),
        ),
        (
            "Grid fast screening produced no candidates with usable objective values for: sharpe_ratio",
            (
                "WFA window 7",
                "Grid fast screening",
                "for: sharpe_ratio. Sharpe can be undefined",
            ),
            ("SQN", "30 completed trades"),
        ),
        (
            "Grid V2 fast screening produced no candidates with usable objective values for: sharpe_ratio, sqn",
            ("WFA window 7", "Grid V2 fast screening", "sharpe_ratio, sqn", "Sharpe", "30 completed trades"),
            (),
        ),
        (
            "Grid V2 slow refinement produced no candidates with usable objective values for: ulcer_index",
            ("WFA window 7", "Grid V2 slow refinement", "for: ulcer_index."),
            ("net_profit_pct", "Sharpe can be undefined", "SQN", "30 completed trades"),
        ),
        (
            "Grid fast screening produced no candidates with usable objective values for: consistency_score",
            ("WFA window 7", "Grid fast screening", "for: consistency_score."),
            ("Sharpe can be undefined", "SQN", "30 completed trades"),
        ),
        (
            "Grid fast screening produced no candidates with usable objective values for: sqn.",
            ("for: sqn. SQN is undefined below 30 completed trades.",),
            ("sqn..",),
        ),
    ],
)
def test_grid_wfa_no_usable_objective_error_preserves_ranking_context(
    monkeypatch,
    ranking_error,
    expected,
    excluded,
):
    source_error, wrapped_error = _capture_grid_wfa_window_error(monkeypatch, ranking_error)
    message = str(wrapped_error)

    assert wrapped_error.__cause__ is source_error
    assert message.endswith((".", "!", "?"))
    for text in expected:
        assert text in message
    for text in excluded:
        assert text not in message


def test_grid_wfa_unrelated_value_error_is_reraised_unchanged(monkeypatch):
    source_error, reraised_error = _capture_grid_wfa_window_error(
        monkeypatch,
        "Grid candidate construction failed",
    )

    assert reraised_error is source_error
    assert reraised_error.__cause__ is None


def test_param_id_generation_s01():
    engine = WalkForwardEngine(WFConfig(strategy_id="s01_trailing_ma"), {}, {})
    params = {"maType": "EMA", "maLength": 45, "closeCountLong": 7}

    expected_hash = hashlib.md5(json.dumps(params, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:8]
    assert engine._create_param_id(params) == f"EMA 45_{expected_hash}"


def test_param_id_generation_s04():
    engine = WalkForwardEngine(WFConfig(strategy_id="s04_stochrsi"), {}, {})
    params = {"rsiLen": 16, "stochLen": 20, "kLen": 3}

    expected_hash = hashlib.md5(json.dumps(params, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:8]
    assert engine._create_param_id(params) == f"16 20_{expected_hash}"


def test_param_id_ignores_runtime_fields_for_display_identity():
    engine = WalkForwardEngine(WFConfig(strategy_id="s03_reversal_v10"), {}, {})
    base = {"maType3": "HMA", "maLength3": 300, "useCloseCount": True, "closeCountLong": 2}
    with_runtime = {
        **base,
        "riskPerTrade": 0.0,
        "commissionRate": 0.05,
        "dateFilter": True,
        "start": "2025-01-01",
        "end": "2025-02-01",
    }
    different_strategy_params = {**base, "maLength3": 301}

    assert engine._create_param_id(base) == engine._create_param_id(with_runtime)
    assert engine._create_param_id(base) != engine._create_param_id(different_strategy_params)
    assert engine._create_param_id(base).startswith("HMA 300_")


def test_wfa_window_runtime_family_comes_from_registry_not_metadata_carrier():
    start = pd.Timestamp("2025-01-01 12:00", tz="UTC")
    end = pd.Timestamp("2025-01-02 12:00", tz="UTC")
    v2 = WalkForwardEngine(
        WFConfig(strategy_id="s06_r_trend_v02_b2"),
        {"fixed_params": {}},
        {},
    )
    v1 = WalkForwardEngine(
        WFConfig(strategy_id="s03_reversal_v10"),
        {"fixed_params": {}, "v2_runtime": {"lookalike": True}},
        {},
    )

    assert v2._window_execution_params({}, start, end) == {
        "dateFilter": True,
        "start": "2025-01-01T12:00:00Z",
        "end": "2025-01-02T12:00:00Z",
    }
    assert v1._window_execution_params({}, start, end) == {
        "dateFilter": True,
        "start": start,
        "end": end,
    }
    assert v1._window_execution_params(
        {}, start, end, legacy_isoformat=True
    )["start"] == start.isoformat()


def test_wfa_unknown_strategy_failure_order_remains_explicit():
    with pytest.raises(ValueError, match="Failed to load strategy 'missing_strategy'"):
        WalkForwardEngine(WFConfig(strategy_id="missing_strategy"), {}, {})


def test_real_ft_result_converts_and_persists_as_wfa_candidate_params():
    strategy_id = "s06_r_trend_v02_b2"
    candidate = _build_params_from_config(strategy_id)
    for name in ("dateFilter", "start", "end", "warmupBars"):
        candidate.pop(name, None)
    task = {
        "trial_number": 7,
        "source_rank": 1,
        "params": candidate,
        "is_metrics": {},
    }
    csv_path = (
        Path(__file__).parent.parent
        / "data"
        / "raw"
        / "OKX_SUIUSDT.P, 30 2025.01.01-2026.02.01.csv"
    )

    payload = _ft_worker_entry(
        str(csv_path),
        strategy_id,
        task,
        "2025-05-01",
        "2025-06-15",
        200,
        90,
        45,
    )
    assert payload is not None
    ft_result = _ft_result_from_payload(payload)
    ft_result.ft_passes_threshold = True
    engine = WalkForwardEngine(WFConfig(strategy_id=strategy_id), {}, {})
    ft_trials = engine._convert_ft_results_for_storage([ft_result], 1, {})

    json.dumps(ft_result.params)
    json.dumps(ft_trials)
    wf_result, _, _ = _build_wf_result(strategy_id)
    window = wf_result.windows[0]
    window.best_params = dict(ft_result.params)
    window.param_id = engine._create_param_id(window.best_params)
    window.best_params_source = "forward_test"
    window.forward_test_trials = ft_trials
    window.stress_test_trials = [
        {
            "trial_number": 7,
            "params": dict(ft_result.params),
            "param_id": window.param_id,
            "is_selected": True,
        }
    ]
    metadata = build_v2_runtime_metadata(
        {
            "dateFilter": False,
            "start": None,
            "end": None,
            "warmupBars": 200,
        },
        strategy_id=strategy_id,
    )

    study_id = storage.save_wfa_study_to_db(
        wf_result,
        {
            "strategy_id": strategy_id,
            "fixed_params": {},
            "v2_runtime": metadata,
        },
        str(csv_path),
        0.0,
    )
    loaded = storage.load_study_from_db(study_id)
    saved_window = loaded["windows"][0]
    assert saved_window["best_params"] == candidate
    modules = storage.load_wfa_window_trials(saved_window["window_id"])
    assert modules["forward_test"][0]["params"] == candidate
    assert modules["stress_test"][0]["params"] == candidate
    assert task["params"] == candidate


def test_param_id_falls_back_and_logs_warning(monkeypatch, caplog):
    """
    Ensure _create_param_id logs and falls back to hash when strategy config cannot be read.
    """

    engine = WalkForwardEngine(WFConfig(strategy_id="s01_trailing_ma"), {}, {})
    params = {"maType": "EMA", "maLength": 45, "closeCountLong": 7}
    expected_hash = hashlib.md5(json.dumps(params, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:8]

    def raise_value_error(strategy_id):  # noqa: ARG001
        raise ValueError("boom")

    monkeypatch.setattr("strategies.get_strategy_config", raise_value_error)

    with caplog.at_level(logging.WARNING, logger="core.walkforward_engine"):
        param_id = engine._create_param_id(params)

    assert param_id == expected_hash
    assert any("Falling back to hash-only param_id" in record.message for record in caplog.records)


def test_s06_grid_wfa_threads_full_enumeration_modes(monkeypatch):
    captured = {}
    result = OptimizationResult(
        params={"useTrailMA": False},
        net_profit_pct=1.0,
        max_drawdown_pct=0.5,
        total_trades=1,
        optuna_trial_number=1,
    )

    def fake_run_grid(config, save_study=False, grid_v2_plan_cache=None):
        captured["config"] = config
        captured["save_study"] = save_study
        captured["grid_v2_plan_cache"] = grid_v2_plan_cache
        config.grid_summary = {"grid": {"backend": {"profile": "full_enumeration"}}}
        config.optuna_all_results = []
        return [result], None

    monkeypatch.setattr("core.walkforward_engine.run_grid_optimization", fake_run_grid)
    base_template = {
        "enabled_params": {},
        "param_ranges": {},
        "param_types": {},
        "fixed_params": {"dateFilter": True},
        "worker_processes": 2,
        "risk_per_trade_pct": 2.0,
        "contract_size": 0.01,
        "commission_rate": 0.05,
        "filter_min_profit": False,
        "min_profit_threshold": 0.0,
        "optimization_mode": "grid",
        "objectives": ["net_profit_pct"],
        "grid_enabled_modes": ["bracket"],
        "grid_budget": 1,
        "grid_seed": 999,
        "grid_top_candidates": 1,
        "grid_v2_prefer_compiled": False,
        "grid_v2_max_cache_mb": 2048.0,
    }
    engine = WalkForwardEngine(
        WFConfig(strategy_id="s06_r_trend_v02", warmup_bars=20),
        base_template,
        {},
    )
    index = pd.date_range("2025-01-01", periods=80, freq="30min", tz="UTC")
    df = pd.DataFrame(
        {
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
            "Volume": 1.0,
        },
        index=index,
    )
    selected, all_results = engine._run_optuna_on_window(
        df,
        index[20],
        index[-1],
    )

    assert selected == [result]
    assert all_results == []
    assert captured["save_study"] is False
    assert captured["grid_v2_plan_cache"] is None
    assert engine._grid_v2_plan_cache is None
    assert captured["config"].strategy_id == "s06_r_trend_v02"
    assert captured["config"].grid_enabled_modes == ["bracket"]
    assert captured["config"].grid_v2_prefer_compiled is False
    assert captured["config"].grid_v2_max_cache_mb == 2048.0
    assert captured["config"].fixed_params["start"] == index[20].isoformat()
    assert captured["config"].fixed_params["end"] == index[-1].isoformat()


def test_s06_b2_grid_wfa_plan_reuse_matches_cache_disabled_windows():
    from core.grid_v2 import GridV2PlanReuseCache

    data_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "raw"
        / "OKX_SUIUSDT.P, 30 2025.01.01-2026.02.01.csv"
    )
    if not data_path.exists():
        pytest.skip("S06 B2 raw data is not present in this checkout.")

    df = load_data(str(data_path)).iloc[:460]
    if len(df) < 430:
        pytest.skip("S06 B2 raw data fixture is too short for the WFA reuse smoke test.")

    base_template = {
        "enabled_params": {"stopX": True},
        "param_ranges": {},
        "param_types": {},
        "fixed_params": {"dateFilter": False},
        "worker_processes": 1,
        "risk_per_trade_pct": 2.0,
        "contract_size": 0.01,
        "commission_rate": 0.0005,
        "filter_min_profit": False,
        "min_profit_threshold": 0.0,
        "optimization_mode": "grid",
        "objectives": ["net_profit_pct"],
        "primary_objective": "net_profit_pct",
        "constraints": [],
        "score_config": {},
        "grid_top_candidates": 2,
        "grid_v2_planning_policy": "sampled",
        "grid_budget": 3,
        "grid_seed": 77,
        "grid_enabled_modes": ["bracket"],
        "grid_diversity_enabled": False,
        "grid_slow_refinement_enabled": False,
        "grid_v2_prefer_compiled": False,
    }
    wf_config = WFConfig(strategy_id="s06_r_trend_v02_b2", warmup_bars=200)
    reuse_engine = WalkForwardEngine(wf_config, base_template, {})
    no_reuse_engine = WalkForwardEngine(wf_config, base_template, {})
    no_reuse_engine._grid_v2_plan_cache = GridV2PlanReuseCache(enabled=False)

    first_start, first_end = df.index[220], df.index[310]
    second_start, second_end = df.index[310], df.index[420]
    reuse_first, reuse_first_all = reuse_engine._run_optuna_on_window(df, first_start, first_end)
    first_summary = reuse_engine._last_grid_summary
    first_materialized_start = reuse_first_all[0].params["start"]
    reuse_second, reuse_second_all = reuse_engine._run_optuna_on_window(df, second_start, second_end)
    second_summary = reuse_engine._last_grid_summary
    no_reuse_first, no_reuse_first_all = no_reuse_engine._run_optuna_on_window(
        df, first_start, first_end
    )
    no_reuse_second, no_reuse_second_all = no_reuse_engine._run_optuna_on_window(
        df, second_start, second_end
    )

    assert first_summary["grid"]["plan_reuse_enabled"] is True
    assert first_summary["grid"]["planning"]["effective_policy"] == "sampled"
    assert first_summary["planned_candidate_count"] == 3
    assert first_summary["grid"]["plan_reuse_hit"] is False
    assert second_summary["grid"]["plan_reuse_enabled"] is True
    assert second_summary["grid"]["plan_reuse_hit"] is True
    assert second_summary["grid"]["plan_build_count"] == 1
    assert second_summary["grid"]["plan_reuse_hit_count"] == 1
    assert second_summary["grid"]["plan_reuse_miss_count"] == 1
    assert reuse_first_all[0].params["start"] == first_materialized_start
    assert reuse_first_all[0].params["start"] == first_start.isoformat().replace("+00:00", "Z")
    assert reuse_second_all[0].params["start"] == second_start.isoformat().replace("+00:00", "Z")
    assert reuse_second_all[0].params["end"] == second_end.isoformat().replace("+00:00", "Z")
    assert no_reuse_engine._last_grid_summary["grid"]["plan_reuse_enabled"] is False

    def assert_same_results(left, right):
        assert len(left) == len(right)
        for lhs, rhs in zip(left, right):
            assert lhs.candidate_id == rhs.candidate_id
            assert lhs.params.get("start") == rhs.params.get("start")
            assert lhs.params.get("end") == rhs.params.get("end")
            assert lhs.total_trades == rhs.total_trades
            assert lhs.net_profit_pct == pytest.approx(rhs.net_profit_pct)
            assert lhs.max_drawdown_pct == pytest.approx(rhs.max_drawdown_pct)

    assert_same_results(reuse_first, no_reuse_first)
    assert_same_results(reuse_second, no_reuse_second)
    assert_same_results(reuse_first_all, no_reuse_first_all)
    assert_same_results(reuse_second_all, no_reuse_second_all)


def test_split_data_rolls_forward():
    index = pd.date_range("2025-01-01", periods=40, freq="D", tz="UTC")
    base_row = {"Open": 1.0, "High": 1.1, "Low": 0.9, "Close": 1.0, "Volume": 100}
    df = pd.DataFrame([base_row for _ in range(len(index))], index=index)

    wf_config = WFConfig(strategy_id="s01_trailing_ma", is_period_days=10, oos_period_days=5)
    engine = WalkForwardEngine(wf_config, {}, {})

    windows = engine.split_data(df, df.index[0], df.index[-1])

    assert len(windows) >= 2
    assert windows[0].oos_start == windows[0].is_end + pd.Timedelta(days=1)
    assert windows[1].is_start == windows[0].is_start + pd.Timedelta(days=5)


def test_day_period_unit_default_and_explicit_preserve_exact_windows():
    index = pd.date_range("2025-01-01", periods=40, freq="D", tz="UTC")
    df = pd.DataFrame({"Close": np.ones(len(index))}, index=index)
    expected = [
        (
            pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=5 * offset),
            pd.Timestamp("2025-01-10", tz="UTC") + pd.Timedelta(days=5 * offset),
            pd.Timestamp("2025-01-11", tz="UTC") + pd.Timedelta(days=5 * offset),
            pd.Timestamp("2025-01-15", tz="UTC") + pd.Timedelta(days=5 * offset),
        )
        for offset in range(6)
    ]

    actual = []
    for unit in (None, "days"):
        kwargs = {} if unit is None else {"period_unit": unit}
        engine = WalkForwardEngine(
            WFConfig(
                strategy_id="s01_trailing_ma",
                is_period_days=10,
                oos_period_days=5,
                **kwargs,
            ),
            {},
            {},
        )
        actual.append(
            [
                (window.is_start, window.is_end, window.oos_start, window.oos_end)
                for window in engine.split_data(df, index[0], index[-1])
            ]
        )

    assert actual == [expected, expected]


def _calendar_engine(is_months=2, oos_months=1, start="2025-10-01"):
    return WalkForwardEngine(
        WFConfig(
            strategy_id="s01_trailing_ma",
            period_unit="months",
            is_period_days=None,
            oos_period_days=None,
            is_period_months=is_months,
            oos_period_months=oos_months,
        ),
        {"fixed_params": {"dateFilter": True, "start": start}},
        {},
    )


def test_calendar_month_windows_use_requested_anchor_and_complete_oos_periods():
    index = pd.date_range("2025-08-01", "2026-08-01", freq="D", tz="UTC")
    df = pd.DataFrame(index=index)
    engine = _calendar_engine()

    july_end = engine.split_data(
        df,
        pd.Timestamp("2025-10-01", tz="UTC"),
        pd.Timestamp("2026-07-31", tz="UTC"),
    )
    august_start = engine.split_data(
        df,
        pd.Timestamp("2025-10-01", tz="UTC"),
        pd.Timestamp("2026-08-01", tz="UTC"),
    )

    assert len(july_end) == len(august_start) == 8
    assert july_end == august_start
    assert (july_end[0].is_start, july_end[0].is_end) == (
        pd.Timestamp("2025-10-01", tz="UTC"),
        pd.Timestamp("2025-11-30", tz="UTC"),
    )
    assert (july_end[0].oos_start, july_end[0].oos_end) == (
        pd.Timestamp("2025-12-01", tz="UTC"),
        pd.Timestamp("2025-12-31", tz="UTC"),
    )
    assert (july_end[-1].oos_start, july_end[-1].oos_end) == (
        pd.Timestamp("2026-07-01", tz="UTC"),
        pd.Timestamp("2026-07-31", tz="UTC"),
    )
    assert all(
        left.oos_end < right.oos_start
        for left, right in zip(july_end, july_end[1:])
    )
    assert engine.split_data(df, index[0], pd.Timestamp("2026-07-15", tz="UTC")) == july_end[:-1]


def test_calendar_month_windows_clamp_late_end_to_last_available_calendar_date():
    index = pd.date_range("2025-08-01", "2026-07-15", freq="D", tz="UTC")
    requested_ends = [
        pd.Timestamp("2026-08-01", tz="UTC"),
        pd.Timestamp("2026-08-01T23:59:59", tz="UTC"),
        pd.Timestamp("2026-09-30T23:59:59.999999Z"),
    ]

    outputs = [
        build_calendar_month_windows(index, "2025-10-01", end, 2, 1)
        for end in requested_ends
    ]

    assert outputs[0] == outputs[1] == outputs[2]
    assert len(outputs[0]) == 7
    assert outputs[0][-1].oos_start == pd.Timestamp("2026-06-01", tz="UTC")
    assert outputs[0][-1].oos_end == pd.Timestamp("2026-06-30", tz="UTC")


def test_calendar_month_windows_keep_logical_boundaries_when_bars_are_missing():
    index = pd.date_range("2027-01-03", "2027-06-01", freq="D", tz="UTC")
    index = index[index != pd.Timestamp("2027-03-01", tz="UTC")]
    df = pd.DataFrame(index=index)
    engine = _calendar_engine(is_months=1, oos_months=1, start="2027-01-01")

    windows = engine.split_data(df, index[0], pd.Timestamp("2027-05-31", tz="UTC"))

    assert windows[0].is_start == pd.Timestamp("2027-01-03", tz="UTC")
    assert windows[0].oos_start == pd.Timestamp("2027-02-01", tz="UTC")
    assert windows[0].oos_end == pd.Timestamp("2027-02-28", tz="UTC")
    assert windows[1].oos_start == pd.Timestamp("2027-03-02", tz="UTC")
    assert windows[1].is_start == pd.Timestamp("2027-02-01", tz="UTC")


@pytest.mark.parametrize(
    ("year", "expected_end"),
    [(2027, "2027-02-28"), (2028, "2028-02-29")],
)
def test_calendar_month_windows_follow_real_february(year, expected_end):
    index = pd.date_range(f"{year}-01-01", f"{year}-06-01", freq="D", tz="UTC")
    windows = build_calendar_month_windows(index, f"{year}-01-01", index[-1], 1, 1)
    assert windows[0].oos_end == pd.Timestamp(expected_end, tz="UTC")
    assert windows[1].oos_start == pd.Timestamp(f"{year}-03-01", tz="UTC")


def test_calendar_month_windows_do_not_fabricate_missing_leap_day():
    index = pd.date_range("2028-01-01", "2028-06-01", freq="D", tz="UTC")
    index = index[index != pd.Timestamp("2028-02-29", tz="UTC")]
    windows = build_calendar_month_windows(index, "2028-01-01", index[-1], 1, 1)
    assert windows[0].oos_end == pd.Timestamp("2028-02-28", tz="UTC")
    assert windows[1].oos_start == pd.Timestamp("2028-03-01", tz="UTC")


def test_calendar_month_windows_cross_year_without_drift():
    index = pd.date_range("2025-11-01", "2026-05-01", freq="D", tz="UTC")
    windows = build_calendar_month_windows(index, "2025-11-01", index[-1], 1, 1)
    assert windows[0].oos_start == pd.Timestamp("2025-12-01", tz="UTC")
    assert windows[1].oos_start == pd.Timestamp("2026-01-01", tz="UTC")


@pytest.mark.parametrize(
    "fixed_params, message",
    [
        ({"dateFilter": False, "start": "2025-10-01"}, "Date Filter"),
        ({"dateFilter": True}, "requested Start"),
        ({"dateFilter": True, "start": "2025-09-28"}, "first calendar day"),
    ],
)
def test_calendar_month_engine_rejects_invalid_requested_anchor(fixed_params, message):
    index = pd.date_range("2025-10-01", "2026-04-01", freq="D", tz="UTC")
    engine = _calendar_engine()
    engine.base_config_template["fixed_params"] = fixed_params
    with pytest.raises(ValueError, match=message):
        engine.split_data(pd.DataFrame(index=index), index[0], index[-1])


@pytest.mark.parametrize("value", [True, 1.5, "", "0", "-1", "1.0", 121])
def test_calendar_month_config_rejects_invalid_counts(value):
    with pytest.raises(ValueError, match=f"1 to {MAX_WFA_PERIOD_MONTHS}"):
        WFConfig(
            period_unit="months",
            is_period_days=None,
            oos_period_days=None,
            is_period_months=value,
            oos_period_months=1,
        )


def test_calendar_month_config_rejects_unknown_unit_and_adaptive_mode():
    with pytest.raises(ValueError, match="days.*months"):
        WFConfig(period_unit="weeks")
    with pytest.raises(ValueError, match="Adaptive"):
        WFConfig(
            period_unit="months",
            is_period_days=None,
            oos_period_days=None,
            is_period_months=2,
            oos_period_months=1,
            adaptive_mode=True,
        )


def _assert_empty_calendar_slice(error, *, window, stage, start, end):
    message = str(error.value)
    assert f"Window {window}" in message
    assert stage in message
    assert start in message
    assert end in message
    assert "no market bars" in message


def test_calendar_month_windows_reject_start_before_available_data():
    index = pd.date_range("2025-04-01", "2025-12-31", freq="D", tz="UTC")

    with pytest.raises(ValueError) as error:
        build_calendar_month_windows(index, "2025-01-01", index[-1], 2, 1)

    _assert_empty_calendar_slice(
        error,
        window=1,
        stage="IS",
        start="2025-01-01T00:00:00+00:00",
        end="2025-03-01T00:00:00+00:00",
    )


def test_calendar_month_windows_reject_empty_interior_is_slice():
    index = pd.date_range("2025-01-01", "2025-09-30", freq="D", tz="UTC")
    index = index[index.month != 5]

    with pytest.raises(ValueError) as error:
        build_calendar_month_windows(index, "2025-01-01", index[-1], 1, 2)

    _assert_empty_calendar_slice(
        error,
        window=3,
        stage="IS",
        start="2025-05-01T00:00:00+00:00",
        end="2025-06-01T00:00:00+00:00",
    )


def test_calendar_month_windows_reject_empty_interior_oos_slice():
    index = pd.date_range("2025-01-01", "2025-07-31", freq="D", tz="UTC")
    index = index[index.month != 4]

    with pytest.raises(ValueError) as error:
        build_calendar_month_windows(index, "2025-01-01", index[-1], 1, 1)

    _assert_empty_calendar_slice(
        error,
        window=3,
        stage="OOS",
        start="2025-04-01T00:00:00+00:00",
        end="2025-05-01T00:00:00+00:00",
    )


def test_calendar_month_windows_ignore_incomplete_trailing_month():
    index = pd.date_range("2025-01-01", "2025-05-15", freq="D", tz="UTC")

    windows = build_calendar_month_windows(index, "2025-01-01", index[-1], 1, 1)

    assert len(windows) == 3
    assert windows[-1].oos_start == pd.Timestamp("2025-04-01", tz="UTC")
    assert windows[-1].oos_end == pd.Timestamp("2025-04-30", tz="UTC")


def test_calendar_month_windows_still_require_two_complete_nonempty_windows():
    index = pd.date_range("2025-01-01", "2025-02-28", freq="D", tz="UTC")
    with pytest.raises(ValueError, match=r"IS=1m, OOS=1m"):
        build_calendar_month_windows(index, "2025-01-01", index[-1], 1, 1)


def test_stitched_equity_skips_duplicate_points():
    wf_config = WFConfig(strategy_id="s01_trailing_ma", is_period_days=180, oos_period_days=60)
    engine = WalkForwardEngine(wf_config, {}, {})

    windows = [
        WindowResult(
            window_id=1,
            is_start=pd.Timestamp("2025-01-01", tz="UTC"),
            is_end=pd.Timestamp("2025-06-29", tz="UTC"),
            oos_start=pd.Timestamp("2025-06-30", tz="UTC"),
            oos_end=pd.Timestamp("2025-08-28", tz="UTC"),
            best_params={},
            param_id="p1",
            is_net_profit_pct=10.0,
            is_max_drawdown_pct=2.0,
            is_total_trades=5,
            oos_net_profit_pct=5.0,
            oos_max_drawdown_pct=1.0,
            oos_total_trades=3,
            oos_equity_curve=[100.0, 110.0, 120.0],
            oos_timestamps=[
                pd.Timestamp("2025-06-30", tz="UTC"),
                pd.Timestamp("2025-07-30", tz="UTC"),
                pd.Timestamp("2025-08-28", tz="UTC"),
            ],
        ),
        WindowResult(
            window_id=2,
            is_start=pd.Timestamp("2025-06-30", tz="UTC"),
            is_end=pd.Timestamp("2025-12-28", tz="UTC"),
            oos_start=pd.Timestamp("2025-12-29", tz="UTC"),
            oos_end=pd.Timestamp("2026-02-26", tz="UTC"),
            best_params={},
            param_id="p2",
            is_net_profit_pct=8.0,
            is_max_drawdown_pct=2.5,
            is_total_trades=4,
            oos_net_profit_pct=7.0,
            oos_max_drawdown_pct=1.2,
            oos_total_trades=2,
            oos_equity_curve=[100.0, 110.0, 130.0],
            oos_timestamps=[
                pd.Timestamp("2025-12-29", tz="UTC"),
                pd.Timestamp("2026-01-29", tz="UTC"),
                pd.Timestamp("2026-02-26", tz="UTC"),
            ],
        ),
    ]

    stitched = engine._build_stitched_oos_equity(windows)

    assert stitched.equity_curve == pytest.approx([100.0, 110.0, 120.0, 132.0, 156.0])
    assert pytest.approx(stitched.final_net_profit_pct, rel=1e-4) == 56.0


def test_wfe_is_annualized():
    wf_config = WFConfig(strategy_id="s01_trailing_ma", is_period_days=180, oos_period_days=60)
    engine = WalkForwardEngine(wf_config, {}, {})

    windows = [
        WindowResult(
            window_id=1,
            is_start=pd.Timestamp("2025-01-01", tz="UTC"),
            is_end=pd.Timestamp("2025-06-29", tz="UTC"),
            oos_start=pd.Timestamp("2025-06-30", tz="UTC"),
            oos_end=pd.Timestamp("2025-08-28", tz="UTC"),
            best_params={},
            param_id="p1",
            is_net_profit_pct=50.0,
            is_max_drawdown_pct=2.0,
            is_total_trades=5,
            oos_net_profit_pct=20.0,
            oos_max_drawdown_pct=1.0,
            oos_total_trades=3,
            oos_equity_curve=[100.0, 120.0],
            oos_timestamps=[pd.Timestamp("2025-06-30", tz="UTC"), pd.Timestamp("2025-08-28", tz="UTC")],
        ),
        WindowResult(
            window_id=2,
            is_start=pd.Timestamp("2025-06-30", tz="UTC"),
            is_end=pd.Timestamp("2025-12-28", tz="UTC"),
            oos_start=pd.Timestamp("2025-12-29", tz="UTC"),
            oos_end=pd.Timestamp("2026-02-26", tz="UTC"),
            best_params={},
            param_id="p2",
            is_net_profit_pct=50.0,
            is_max_drawdown_pct=2.0,
            is_total_trades=5,
            oos_net_profit_pct=20.0,
            oos_max_drawdown_pct=1.0,
            oos_total_trades=3,
            oos_equity_curve=[100.0, 120.0],
            oos_timestamps=[pd.Timestamp("2025-12-29", tz="UTC"), pd.Timestamp("2026-02-26", tz="UTC")],
        ),
    ]

    stitched = engine._build_stitched_oos_equity(windows)

    assert pytest.approx(stitched.wfe, rel=1e-3) == 120.0


def test_calendar_month_wfe_uses_nominal_month_annualization():
    wf_config = WFConfig(
        strategy_id="s01_trailing_ma",
        period_unit="months",
        is_period_days=None,
        oos_period_days=None,
        is_period_months=2,
        oos_period_months=1,
    )
    engine = WalkForwardEngine(wf_config, {}, {})
    windows = [
        WindowResult(
            window_id=index,
            is_start=pd.Timestamp(f"2028-0{index}-01", tz="UTC"),
            is_end=pd.Timestamp(f"2028-0{index + 1}-29", tz="UTC"),
            oos_start=pd.Timestamp(f"2028-0{index + 2}-01", tz="UTC"),
            oos_end=pd.Timestamp(f"2028-0{index + 2}-28", tz="UTC"),
            best_params={},
            param_id=f"p{index}",
            is_net_profit_pct=50.0,
            is_max_drawdown_pct=2.0,
            is_total_trades=5,
            oos_net_profit_pct=20.0,
            oos_max_drawdown_pct=1.0,
            oos_total_trades=3,
            oos_equity_curve=[100.0, 120.0],
            oos_timestamps=[
                pd.Timestamp(f"2028-0{index + 2}-01", tz="UTC"),
                pd.Timestamp(f"2028-0{index + 2}-28", tz="UTC"),
            ],
        )
        for index in (1, 2)
    ]

    stitched = engine._build_stitched_oos_equity(windows)

    assert stitched.wfe == pytest.approx(80.0)


def test_store_top_n_trials_limit():
    engine = WalkForwardEngine(WFConfig(strategy_id="s01_trailing_ma"), {}, {})
    results = []
    for i in range(5):
        results.append(
            OptimizationResult(
                params={"maType": "EMA", "maLength": 10 + i, "closeCountLong": 7},
                net_profit_pct=1.0 + i,
                max_drawdown_pct=0.5,
                total_trades=5,
                optuna_trial_number=i,
            )
        )
    stored = engine._convert_optuna_results_for_storage(results, limit=2)
    assert len(stored) == 2
    assert stored[0]["trial_number"] == 0


def test_run_optuna_on_window_forwards_coverage_mode(monkeypatch):
    index = pd.date_range("2025-01-01", periods=40, freq="D", tz="UTC")
    base_row = {"Open": 1.0, "High": 1.1, "Low": 0.9, "Close": 1.0, "Volume": 100}
    df = pd.DataFrame([base_row for _ in range(len(index))], index=index)

    wf_config = WFConfig(strategy_id="s01_trailing_ma", is_period_days=10, oos_period_days=5)
    base_template = {
        "enabled_params": {},
        "param_ranges": {},
        "param_types": {},
        "fixed_params": {"dateFilter": False},
        "risk_per_trade_pct": 2.0,
        "contract_size": 0.01,
        "commission_rate": 0.0005,
        "worker_processes": 1,
        "filter_min_profit": False,
        "min_profit_threshold": 0.0,
        "score_config": {},
        "csv_original_name": "OKX_LINKUSDT.P, 30 2025.05.01-2025.11.20.csv",
        "detailed_log": True,
        "trials_log": True,
        "dispatcher_batch_result_processing": False,
        "dispatcher_soft_duplicate_cycle_limit_enabled": False,
        "dispatcher_duplicate_cycle_limit": 42,
        "objectives": ["net_profit_pct"],
        "primary_objective": None,
        "constraints": [],
        "sampler_type": "tpe",
        "population_size": 50,
        "crossover_prob": 0.9,
        "mutation_prob": None,
        "swapping_prob": 0.5,
        "n_startup_trials": 33,
        "coverage_mode": True,
    }
    optuna_settings = {
        "objectives": ["net_profit_pct"],
        "primary_objective": None,
        "constraints": [],
        "budget_mode": "trials",
        "n_trials": 20,
        "time_limit": 3600,
        "convergence_patience": 50,
        "enable_pruning": True,
        "sampler": "tpe",
        "population_size": 50,
        "crossover_prob": 0.9,
        "mutation_prob": None,
        "swapping_prob": 0.5,
        "pruner": "median",
        "warmup_trials": 33,
        "coverage_mode": True,
        "save_study": False,
    }
    engine = WalkForwardEngine(wf_config, base_template, optuna_settings)

    captured = {}

    def fake_run_optuna_optimization(base_config, optuna_cfg):
        captured["base_config"] = base_config
        captured["optuna_cfg"] = optuna_cfg
        return [], None

    monkeypatch.setattr("core.walkforward_engine.run_optuna_optimization", fake_run_optuna_optimization)

    engine._run_optuna_on_window(df, index[0], index[10])

    assert captured["base_config"].coverage_mode is True
    assert captured["base_config"].n_startup_trials == 33
    assert captured["base_config"].csv_original_name == "OKX_LINKUSDT.P, 30 2025.05.01-2025.11.20.csv"
    assert captured["base_config"].detailed_log is True
    assert captured["base_config"].trials_log is True
    assert captured["base_config"].dispatcher_batch_result_processing is False
    assert captured["base_config"].dispatcher_soft_duplicate_cycle_limit_enabled is False
    assert captured["base_config"].dispatcher_duplicate_cycle_limit == 42
    assert captured["optuna_cfg"].coverage_mode is True
    assert captured["optuna_cfg"].warmup_trials == 33


@pytest.mark.slow
def test_run_optuna_on_window_multiprocess_uses_in_memory_worker_csv():
    data_path = Path(__file__).parent.parent / "data" / "raw" / "OKX_LINKUSDT.P, 15 2025.05.01-2025.11.20.csv"
    if not data_path.exists():
        pytest.skip("Sample data file not available for WFA multiprocess test.")

    df = load_data(str(data_path)).iloc[:1200].copy()
    wf_config = WFConfig(
        strategy_id="s01_trailing_ma",
        is_period_days=60,
        oos_period_days=30,
        warmup_bars=200,
    )
    base_template = {
        "enabled_params": {"maLength": True},
        "param_ranges": {"maLength": (10, 40, 10)},
        "param_types": {"maLength": "int"},
        "fixed_params": {
            "dateFilter": False,
            "maType": "EMA",
            "closeCountLong": 2,
            "closeCountShort": 2,
        },
        "risk_per_trade_pct": 2.0,
        "contract_size": 0.01,
        "commission_rate": 0.0005,
        "worker_processes": 2,
        "filter_min_profit": False,
        "min_profit_threshold": 0.0,
        "score_config": {},
        "objectives": ["net_profit_pct"],
        "primary_objective": None,
        "constraints": [],
        "sampler_type": "tpe",
        "population_size": 50,
        "crossover_prob": 0.9,
        "mutation_prob": None,
        "swapping_prob": 0.5,
        "n_startup_trials": 20,
        "coverage_mode": False,
    }
    optuna_settings = {
        "objectives": ["net_profit_pct"],
        "primary_objective": None,
        "constraints": [],
        "budget_mode": "trials",
        "n_trials": 2,
        "time_limit": 3600,
        "convergence_patience": 10,
        "enable_pruning": False,
        "sampler": "tpe",
        "population_size": 50,
        "crossover_prob": 0.9,
        "mutation_prob": None,
        "swapping_prob": 0.5,
        "pruner": "median",
        "warmup_trials": 20,
        "coverage_mode": False,
    }
    engine = WalkForwardEngine(wf_config, base_template, optuna_settings)

    before_entries = {path.name for path in storage.JOURNAL_DIR.iterdir()}
    results, all_results = engine._run_optuna_on_window(df, df.index[200], df.index[-1])
    after_entries = {path.name for path in storage.JOURNAL_DIR.iterdir()}

    assert after_entries == before_entries
    assert isinstance(results, list)
    assert isinstance(all_results, list)


def test_best_params_source_tracked(monkeypatch):
    index = pd.date_range("2025-01-01", periods=40, freq="D", tz="UTC")
    base_row = {"Open": 1.0, "High": 1.1, "Low": 0.9, "Close": 1.0, "Volume": 100}
    df = pd.DataFrame([base_row for _ in range(len(index))], index=index)

    optuna_result = OptimizationResult(
        params={"maType": "EMA", "maLength": 20, "closeCountLong": 7},
        net_profit_pct=5.0,
        max_drawdown_pct=1.0,
        total_trades=4,
        optuna_trial_number=12,
        constraints_satisfied=False,
        is_pareto_optimal=True,
    )

    def fake_optuna(self, df_slice, start_time, end_time):  # noqa: ARG001
        return [optuna_result], [optuna_result]

    def fake_dsr(*args, **kwargs):  # noqa: ARG001
        dsr = DSRResult(
            trial_number=12,
            optuna_rank=1,
            params=optuna_result.params,
            original_result=optuna_result,
            dsr_probability=0.5,
            dsr_rank=1,
        )
        return [dsr], {}

    monkeypatch.setattr(WalkForwardEngine, "_run_optuna_on_window", fake_optuna)
    monkeypatch.setattr("core.walkforward_engine.run_dsr_analysis", fake_dsr)

    wf_config = WFConfig(
        strategy_id="s01_trailing_ma",
        is_period_days=10,
        oos_period_days=5,
        warmup_bars=5,
        dsr_config=DSRConfig(enabled=True, top_k=1),
    )
    base_template = {
        "fixed_params": {"dateFilter": False},
        "risk_per_trade_pct": 2.0,
        "contract_size": 0.01,
        "commission_rate": 0.0005,
        "worker_processes": 1,
        "filter_min_profit": False,
        "min_profit_threshold": 0.0,
        "score_config": {},
    }
    engine = WalkForwardEngine(wf_config, base_template, {})

    class FakeStrategy:
        @staticmethod
        def run(df_slice, params, trade_start_idx):  # noqa: ARG001
            return StrategyResult(
                trades=[],
                equity_curve=[100.0],
                balance_curve=[100.0],
                timestamps=[df_slice.index[min(trade_start_idx, len(df_slice) - 1)]],
            )

    engine.strategy_class = FakeStrategy
    result, _study_id = engine.run_wf_optimization(df)

    assert result.windows[0].best_params_source == "dsr"
    assert result.windows[0].is_pareto_optimal is True
    assert result.windows[0].constraints_satisfied is False


def test_grid_wfa_dsr_candidate_replaces_objective_winner(monkeypatch):
    index = pd.date_range("2025-01-01", periods=40, freq="D", tz="UTC")
    df = pd.DataFrame(
        {"Open": 1.0, "High": 1.1, "Low": 0.9, "Close": 1.0, "Volume": 100},
        index=index,
    )

    objective_result = OptimizationResult(
        params={"source": "objective", "maType3": "SMA", "maLength3": 20},
        net_profit_pct=5.0,
        max_drawdown_pct=1.0,
        total_trades=4,
        optuna_trial_number=1,
        objective_values=[5.0],
        constraints_satisfied=True,
    )
    objective_result.candidate_id = 1
    objective_result.grid_rank = 1
    objective_result.selection_sources = ["objective"]
    objective_result.is_objective_selected = True

    dsr_result = OptimizationResult(
        params={"source": "dsr", "maType3": "EMA", "maLength3": 50},
        net_profit_pct=1.0,
        max_drawdown_pct=1.0,
        total_trades=4,
        optuna_trial_number=5,
        objective_values=[1.0],
        constraints_satisfied=True,
    )
    dsr_result.candidate_id = 5
    dsr_result.grid_rank = 5
    dsr_result.selection_sources = ["dsr"]
    dsr_result.is_dsr_selected = True
    dsr_result.dsr_rank = 1
    dsr_result.dsr_source_rank = 2
    dsr_result.dsr_probability = 0.99

    def fake_grid_window(self, df_slice, start_time, end_time):  # noqa: ARG001
        return [objective_result, dsr_result], [objective_result, dsr_result]

    def unexpected_dsr(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("Grid WFA must use precomputed Grid DSR metadata.")

    monkeypatch.setattr(WalkForwardEngine, "_run_optuna_on_window", fake_grid_window)
    monkeypatch.setattr("core.walkforward_engine.run_dsr_analysis", unexpected_dsr)

    wf_config = WFConfig(
        strategy_id="s03_reversal_v10",
        is_period_days=10,
        oos_period_days=5,
        warmup_bars=5,
        dsr_config=DSRConfig(enabled=True, top_k=1),
    )
    base_template = {
        "optimization_mode": "grid",
        "fixed_params": {"dateFilter": False},
        "risk_per_trade_pct": 2.0,
        "contract_size": 0.01,
        "commission_rate": 0.0005,
        "worker_processes": 1,
        "filter_min_profit": False,
        "min_profit_threshold": 0.0,
        "score_config": {},
    }
    engine = WalkForwardEngine(wf_config, base_template, {})
    backtest_params = []

    class FakeStrategy:
        @staticmethod
        def run(df_slice, params, trade_start_idx):
            backtest_params.append(dict(params))
            return StrategyResult(
                trades=[],
                equity_curve=[100.0],
                balance_curve=[100.0],
                timestamps=[df_slice.index[min(trade_start_idx, len(df_slice) - 1)]],
            )

    engine.strategy_class = FakeStrategy
    result, _study_id = engine.run_wf_optimization(df)

    window = result.windows[0]
    assert window.best_params_source == "dsr"
    assert window.best_params["source"] == "dsr"
    assert window.is_best_trial_number == 5
    assert window.selection_chain["dsr"] == 5
    assert window.dsr_trials[0]["source_rank"] == 2
    assert window.dsr_trials[0]["module_rank"] == 1
    assert backtest_params
    assert all(params.get("source") == "dsr" for params in backtest_params)


def test_grid_wfa_threads_window_dsr_summary_and_candidate_moments(monkeypatch):
    index = pd.date_range("2025-01-01", periods=40, freq="D", tz="UTC")
    df = pd.DataFrame(
        {"Open": 1.0, "High": 1.1, "Low": 0.9, "Close": 1.0, "Volume": 100},
        index=index,
    )

    def make_candidate(candidate_id: int, *, dsr_rank=None) -> OptimizationResult:
        result = _grid_post_process_result(candidate_id, f"candidate-{candidate_id}", grid_rank=candidate_id)
        result.dsr_skewness = 0.10 + candidate_id
        result.dsr_kurtosis = 3.0 + candidate_id
        result.dsr_track_length = 12 + candidate_id
        result.dsr_probability = 0.90 - (candidate_id * 0.01)
        result.dsr_luck_share_pct = 5.0 + candidate_id
        if dsr_rank is not None:
            result.is_dsr_selected = True
            result.dsr_rank = dsr_rank
            result.dsr_source_rank = candidate_id
            result.selection_sources = ["objective", "dsr"]
        return result

    def fake_grid_window(self, df_slice, start_time, end_time):  # noqa: ARG001
        candidates = [make_candidate(1, dsr_rank=1), make_candidate(2)]
        summary = {
            "candidate_count": 48_480,
            "valid_candidate_count": 2,
            "selected_candidate_count": 2,
            "grid": {
                "engine": "v2",
                "backend_kind": "compiled_numba",
                "compiled_batch_used": True,
                "compiled_workers": 6,
                "candidate_count": 48_480,
                "valid_candidate_count": 2,
                "selected_candidate_count": 2,
                "cache_estimate": {"estimated_total_mb": 11.5, "estimated_signal_mb": 9.25},
                "cache_stats": {"signal_misses": 1, "dataprep_misses": 3},
                "chunk_count": 3,
                "max_chunk_candidates": 16_160,
                "max_chunk_estimated_mb": 4.2,
                "chunk_estimated_mb": 4.2,
                "full_run_estimated_signal_mb": 9.25,
                "configured_limit_mb": 512.0,
                "signal_stack_rows_built": 12,
                "signal_stack_rows_peak": 4,
                "compiled_config_packing": "mapping",
                "full_population_result_object_note": "Full-population materialization remains deferred.",
                "candidates_per_second": 1234.5,
                "plan_reuse_enabled": True,
                "plan_reuse_hit": True,
                "plan_cache_key_hash": "abc123",
                "plan_build_count": 1,
                "plan_reuse_hit_count": 2,
                "plan_reuse_miss_count": 1,
                "timings": {
                    "candidate_generation_seconds": 0.11,
                    "plan_build_seconds": 0.0,
                    "plan_reuse_lookup_seconds": 0.01,
                    "runtime_rebase_seconds": 0.02,
                    "data_prepare_seconds": 0.22,
                    "cache_key_build_seconds": 0.07,
                    "signal_build_seconds": 0.08,
                    "stack_build_seconds": 0.09,
                    "compiled_batch_seconds": 0.10,
                    "fast_evaluation_seconds": 0.33,
                    "fast_result_materialization_seconds": 0.04,
                    "ranking_seconds": 0.05,
                    "slow_validation_seconds": 0.44,
                    "slow_refinement_seconds": 0.06,
                    "total_seconds": 1.10,
                },
                "dsr": {
                    "enabled": True,
                    "top_k": 2,
                    "dsr_n_trials": 10,
                    "dsr_mean_sharpe": 0.25,
                    "dsr_var_sharpe": 0.04,
                    "dsr_sr0": 0.70,
                }
            },
        }
        return candidates, list(candidates), summary

    monkeypatch.setattr(WalkForwardEngine, "_run_optuna_on_window", fake_grid_window)

    wf_config = WFConfig(
        strategy_id="s03_reversal_v10",
        is_period_days=10,
        oos_period_days=5,
        warmup_bars=5,
        dsr_config=DSRConfig(enabled=True, top_k=1),
    )
    base_template = {
        "optimization_mode": "grid",
        "fixed_params": {"dateFilter": False},
        "risk_per_trade_pct": 2.0,
        "contract_size": 0.01,
        "commission_rate": 0.0005,
        "worker_processes": 1,
        "filter_min_profit": False,
        "min_profit_threshold": 0.0,
        "score_config": {},
    }
    engine = WalkForwardEngine(wf_config, base_template, {}, csv_file_path="dummy.csv")

    class FakeStrategy:
        @staticmethod
        def run(df_slice, params, trade_start_idx):  # noqa: ARG001
            return StrategyResult(
                trades=[],
                equity_curve=[100.0],
                balance_curve=[100.0],
                timestamps=[df_slice.index[min(trade_start_idx, len(df_slice) - 1)]],
            )

    engine.strategy_class = FakeStrategy
    result, study_id = engine.run_wf_optimization(df)

    assert result.windows
    for window in result.windows:
        diagnostics = window.module_status["grid_v2"]
        assert diagnostics["engine"] == "v2"
        assert diagnostics["backend_kind"] == "compiled_numba"
        assert diagnostics["compiled_batch_used"] is True
        assert diagnostics["compiled_workers"] == 6
        assert diagnostics["candidate_count"] == 48_480
        assert diagnostics["valid_candidate_count"] == 2
        assert diagnostics["selected_candidate_count"] == 2
        assert diagnostics["cache_estimate"]["estimated_total_mb"] == pytest.approx(11.5)
        assert diagnostics["cache_estimate"]["estimated_signal_mb"] == pytest.approx(9.25)
        assert diagnostics["cache_stats"]["signal_misses"] == 1
        assert diagnostics["cache_stats"]["dataprep_misses"] == 3
        assert diagnostics["candidate_generation_seconds"] == pytest.approx(0.11)
        assert diagnostics["plan_build_seconds"] == pytest.approx(0.0)
        assert diagnostics["plan_reuse_lookup_seconds"] == pytest.approx(0.01)
        assert diagnostics["runtime_rebase_seconds"] == pytest.approx(0.02)
        assert diagnostics["data_prepare_seconds"] == pytest.approx(0.22)
        assert diagnostics["cache_key_build_seconds"] == pytest.approx(0.07)
        assert diagnostics["signal_build_seconds"] == pytest.approx(0.08)
        assert diagnostics["stack_build_seconds"] == pytest.approx(0.09)
        assert diagnostics["compiled_batch_seconds"] == pytest.approx(0.10)
        assert diagnostics["fast_evaluation_seconds"] == pytest.approx(0.33)
        assert diagnostics["fast_result_materialization_seconds"] == pytest.approx(0.04)
        assert diagnostics["ranking_seconds"] == pytest.approx(0.05)
        assert diagnostics["slow_validation_seconds"] == pytest.approx(0.44)
        assert diagnostics["slow_refinement_seconds"] == pytest.approx(0.06)
        assert diagnostics["total_seconds"] == pytest.approx(1.10)
        assert diagnostics["candidates_per_second"] == pytest.approx(1234.5)
        assert diagnostics["chunk_count"] == 3
        assert diagnostics["max_chunk_candidates"] == 16_160
        assert diagnostics["max_chunk_estimated_mb"] == pytest.approx(4.2)
        assert diagnostics["chunk_estimated_mb"] == pytest.approx(4.2)
        assert diagnostics["full_run_estimated_signal_mb"] == pytest.approx(9.25)
        assert diagnostics["configured_limit_mb"] == pytest.approx(512.0)
        assert diagnostics["signal_stack_rows_built"] == 12
        assert diagnostics["signal_stack_rows_peak"] == 4
        assert diagnostics["compiled_config_packing"] == "mapping"
        assert diagnostics["full_population_result_object_note"] == "Full-population materialization remains deferred."
        assert diagnostics["plan_reuse_enabled"] is True
        assert diagnostics["plan_reuse_hit"] is True
        assert diagnostics["plan_cache_key_hash"] == "abc123"
        assert diagnostics["plan_build_count"] == 1
        assert diagnostics["plan_reuse_hit_count"] == 2
        assert diagnostics["plan_reuse_miss_count"] == 1
        assert window.grid_dsr_enabled is True
        assert window.grid_dsr_top_k == 2
        assert window.grid_dsr_n_trials == 10
        assert window.grid_dsr_mean_sharpe == pytest.approx(0.25)
        assert window.grid_dsr_var_sharpe == pytest.approx(0.04)
        assert window.grid_dsr_sr0 == pytest.approx(0.70)
        assert window.grid_valid_candidate_count == 2
        assert window.grid_selected_candidate_count == 2
        for trial in window.optuna_is_trials:
            module_metrics = trial["module_metrics"]
            assert module_metrics["semantic_key"].startswith("candidate:")
            assert module_metrics["candidate_id"] in {1, 2}
            assert "dsr_skewness" in module_metrics
            assert "dsr_kurtosis" in module_metrics
            assert "dsr_track_length" in module_metrics

    loaded = storage.load_study_from_db(study_id)
    persisted = loaded["windows"][0]["module_status"]["grid_v2"]
    assert persisted["backend_kind"] == "compiled_numba"
    assert persisted["candidate_generation_seconds"] == pytest.approx(0.11)
    assert persisted["plan_reuse_hit"] is True


def test_grid_v2_diagnostics_uses_sibling_candidates_per_second():
    summary = {
        "engine": "v2",
        "candidate_count": 10,
        "valid_candidate_count": 10,
        "selected_candidate_count": 2,
        "grid": {
            "backend_kind": "compiled_numba",
            "compiled_batch_used": True,
            "compiled_workers": 6,
            "plan_reuse_enabled": True,
            "plan_reuse_hit": False,
            "plan_build_count": 1,
            "plan_reuse_hit_count": 0,
            "plan_reuse_miss_count": 1,
            "timings": {
                "candidate_generation_seconds": 0.1,
                "plan_build_seconds": 0.08,
                "plan_reuse_lookup_seconds": 0.01,
                "runtime_rebase_seconds": 0.0,
                "data_prepare_seconds": 0.2,
                "fast_evaluation_seconds": 0.3,
                "fast_result_materialization_seconds": 0.04,
                "ranking_seconds": 0.05,
                "slow_validation_seconds": 0.4,
                "slow_refinement_seconds": 0.06,
                "total_seconds": 1.0,
                "candidates_per_second": 999.0,
            },
            "candidates_per_second": 1234.5,
        },
    }

    diagnostics = WalkForwardEngine._grid_v2_diagnostics_from_summary(summary)

    assert diagnostics["fast_evaluation_seconds"] == pytest.approx(0.3)
    assert diagnostics["plan_build_seconds"] == pytest.approx(0.08)
    assert diagnostics["plan_reuse_lookup_seconds"] == pytest.approx(0.01)
    assert diagnostics["runtime_rebase_seconds"] == pytest.approx(0.0)
    assert diagnostics["fast_result_materialization_seconds"] == pytest.approx(0.04)
    assert diagnostics["ranking_seconds"] == pytest.approx(0.05)
    assert diagnostics["slow_refinement_seconds"] == pytest.approx(0.06)
    assert diagnostics["candidates_per_second"] == pytest.approx(1234.5)
    assert diagnostics["plan_reuse_enabled"] is True
    assert diagnostics["plan_reuse_hit"] is False
    assert diagnostics["plan_build_count"] == 1
    assert diagnostics["plan_reuse_hit_count"] == 0
    assert diagnostics["plan_reuse_miss_count"] == 1


def test_s03_grid_v2_wfa_smoke_persists_signal_chunk_diagnostics():
    periods = 240
    x = np.arange(periods, dtype=float)
    close = 100.0 + np.sin(x / 3.0) * 6.0 + np.sin(x / 11.0) * 2.0
    df = pd.DataFrame(
        {
            "Open": close + np.cos(x / 5.0) * 0.1,
            "High": close + 0.75,
            "Low": close - 0.75,
            "Close": close,
            "Volume": np.full(periods, 1000.0),
        },
        index=pd.date_range("2025-01-01", periods=periods, freq="30min", tz="UTC"),
    )
    fixed_params = normalized_s03_params(
        {
            "dateFilter": False,
            "maType3_options": ["EMA", "SMA"],
            "maLength3": 8,
            "maOffset3": 0.0,
            "useCloseCount": True,
            "closeCountLong": 1,
            "closeCountShort": 1,
            "useTBands": False,
            "useRegime": True,
            "regimeErLength": 6,
            "regimeErThresh": 0.25,
            "useEmergencySL": False,
            "positionPct": 100.0,
            "contractSize": 0.01,
            "initialCapital": 100.0,
            "commissionPct": 0.0,
        }
    )
    base_template = {
        "enabled_params": {"maType3": True, "maLength3": True, "regimeErLength": True},
        "param_ranges": {},
        "param_types": {},
        "fixed_params": fixed_params,
        "worker_processes": 1,
        "risk_per_trade_pct": 2.0,
        "contract_size": 0.01,
        "commission_rate": 0.0,
        "filter_min_profit": False,
        "min_profit_threshold": 0.0,
        "optimization_mode": "grid",
        "objectives": ["net_profit_pct"],
        "primary_objective": "net_profit_pct",
        "constraints": [],
        "score_config": {},
        "grid_top_candidates": 1,
        "grid_diversity_enabled": False,
        "grid_slow_refinement_enabled": False,
        "grid_v2_prefer_compiled": False,
        "grid_v2_max_cache_mb": 0.03,
    }
    engine = WalkForwardEngine(
        WFConfig(
            strategy_id="s03_reversal_v11_regime_er_b2",
            is_period_days=2,
            oos_period_days=1,
            warmup_bars=20,
        ),
        base_template,
        {},
        csv_file_path="synthetic_s03_grid_v2_wfa.csv",
    )

    result, study_id = engine.run_wf_optimization(df)

    assert study_id is not None
    assert result.windows
    assert any(window.module_status["grid_v2"]["chunk_count"] > 1 for window in result.windows)
    for window in result.windows:
        diagnostics = window.module_status["grid_v2"]
        assert diagnostics["engine"] == "v2"
        assert diagnostics["backend_kind"] == "reference"
        assert diagnostics["candidate_count"] == 120
        assert diagnostics["chunk_count"] > 1
        assert diagnostics["max_chunk_candidates"] < diagnostics["candidate_count"]
        assert diagnostics["full_run_estimated_signal_mb"] > diagnostics["configured_limit_mb"]
        assert diagnostics["full_population_result_object_note"]

    loaded = storage.load_study_from_db(study_id)
    persisted = loaded["windows"][0]["module_status"]["grid_v2"]
    assert persisted["chunk_count"] > 1
    assert persisted["max_chunk_candidates"] < persisted["candidate_count"]


def test_grid_wfa_dsr_disabled_leaves_replay_dsr_fields_empty(monkeypatch):
    index = pd.date_range("2025-01-01", periods=40, freq="D", tz="UTC")
    df = pd.DataFrame(
        {"Open": 1.0, "High": 1.1, "Low": 0.9, "Close": 1.0, "Volume": 100},
        index=index,
    )

    def fake_grid_window(self, df_slice, start_time, end_time):  # noqa: ARG001
        candidate = _grid_post_process_result(1, "candidate-1", grid_rank=1)
        summary = {
            "valid_candidate_count": 1,
            "selected_candidate_count": 1,
            "grid": {
                "dsr": {
                    "enabled": False,
                    "top_k": None,
                    "dsr_n_trials": None,
                    "dsr_mean_sharpe": None,
                    "dsr_var_sharpe": None,
                    "dsr_sr0": None,
                }
            },
        }
        return [candidate], [candidate], summary

    monkeypatch.setattr(WalkForwardEngine, "_run_optuna_on_window", fake_grid_window)

    wf_config = WFConfig(
        strategy_id="s03_reversal_v10",
        is_period_days=10,
        oos_period_days=5,
        warmup_bars=5,
    )
    base_template = {
        "optimization_mode": "grid",
        "fixed_params": {"dateFilter": False},
        "risk_per_trade_pct": 2.0,
        "contract_size": 0.01,
        "commission_rate": 0.0005,
        "worker_processes": 1,
        "filter_min_profit": False,
        "min_profit_threshold": 0.0,
        "score_config": {},
    }
    engine = WalkForwardEngine(wf_config, base_template, {})

    class FakeStrategy:
        @staticmethod
        def run(df_slice, params, trade_start_idx):  # noqa: ARG001
            return StrategyResult(
                trades=[],
                equity_curve=[100.0],
                balance_curve=[100.0],
                timestamps=[df_slice.index[min(trade_start_idx, len(df_slice) - 1)]],
            )

    engine.strategy_class = FakeStrategy
    result, _study_id = engine.run_wf_optimization(df)

    assert result.windows
    for window in result.windows:
        assert window.grid_dsr_enabled is None
        assert window.grid_dsr_top_k is None
        assert window.grid_dsr_n_trials is None
        assert window.grid_dsr_mean_sharpe is None
        assert window.grid_dsr_var_sharpe is None
        assert window.grid_dsr_sr0 is None
        assert window.grid_valid_candidate_count == 1
        assert window.grid_selected_candidate_count == 1
        module_metrics = window.optuna_is_trials[0]["module_metrics"]
        assert module_metrics["semantic_key"] == "candidate:1"
        assert module_metrics["candidate_id"] == 1
        assert "dsr_skewness" not in module_metrics
        assert "dsr_kurtosis" not in module_metrics
        assert "dsr_track_length" not in module_metrics


def _grid_post_process_result(candidate_id: int, source: str, *, grid_rank: int) -> OptimizationResult:
    result = OptimizationResult(
        params={"source": source, "maType3": "HMA", "maLength3": 100 + candidate_id},
        net_profit_pct=10.0 - candidate_id,
        max_drawdown_pct=1.0,
        total_trades=4,
        optuna_trial_number=candidate_id,
        objective_values=[10.0 - candidate_id],
        constraints_satisfied=True,
    )
    result.candidate_id = candidate_id
    result.semantic_key = f"candidate:{candidate_id}"
    result.grid_rank = grid_rank
    result.selection_sources = ["objective"]
    result.is_objective_selected = True
    return result


def _grid_post_process_engine(*, dsr: bool = False, ft: bool = False, st: bool = False) -> WalkForwardEngine:
    wf_config = WFConfig(
        strategy_id="s03_reversal_v10",
        is_period_days=20,
        oos_period_days=5,
        warmup_bars=5,
        dsr_config=DSRConfig(enabled=dsr, top_k=1) if dsr else None,
        post_process=PostProcessConfig(enabled=ft, ft_period_days=5, ft_threshold_pct=-99.0) if ft else None,
        stress_test_config=StressTestConfig(enabled=st, top_k=1) if st else None,
    )
    return WalkForwardEngine(
        wf_config,
        {
            "optimization_mode": "grid",
            "fixed_params": {"dateFilter": False},
            "risk_per_trade_pct": 2.0,
            "contract_size": 0.01,
            "commission_rate": 0.0005,
            "worker_processes": 1,
            "filter_min_profit": False,
            "min_profit_threshold": 0.0,
            "score_config": {},
        },
        {},
        csv_file_path="dummy.csv",
    )


def test_grid_wfa_optuna_is_trials_include_grid_audit_module_metrics():
    engine = _grid_post_process_engine()
    grid_result = _grid_post_process_result(3, "objective", grid_rank=123)
    grid_result.slow_refinement_rank = 4
    grid_result.grid_mode_name = "both"
    grid_result.grid_generation_mode = "lhs"
    grid_result.diversity_group = "both|HMA|275"
    grid_result.selection_sources = ["objective"]

    trials = engine._convert_optuna_results_for_storage([grid_result], 1)
    module_metrics = trials[0]["module_metrics"]

    assert module_metrics == {
        "grid_rank": 123,
        "slow_refinement_rank": 4,
        "grid_mode_name": "both",
        "grid_generation_mode": "lhs",
        "diversity_group": "both|HMA|275",
        "selection_sources": ["objective"],
        "semantic_key": "candidate:3",
        "candidate_id": 3,
    }

    optuna_result = OptimizationResult(
        params={"source": "optuna"},
        net_profit_pct=1.0,
        max_drawdown_pct=1.0,
        total_trades=1,
        optuna_trial_number=1,
    )
    optuna_trials = engine._convert_optuna_results_for_storage([optuna_result], 1)

    assert "module_metrics" not in optuna_trials[0]


def test_grid_wfa_forward_test_selects_ft_rank_one(monkeypatch):
    objective = _grid_post_process_result(1, "objective", grid_rank=1)
    ft_selected_trial = 1

    monkeypatch.setattr(WalkForwardEngine, "_run_optuna_on_window", lambda *args, **kwargs: ([objective], [objective]))

    def fake_forward_test(**kwargs):
        assert [item.params["source"] for item in kwargs["optuna_results"]] == ["objective"]
        return [
            SimpleNamespace(
                trial_number=ft_selected_trial,
                source_rank=1,
                params={"source": "ft", "maType3": "HMA", "maLength3": 101},
                ft_rank=1,
                ft_passes_threshold=True,
                ft_net_profit_pct=3.0,
                ft_max_drawdown_pct=1.0,
                ft_total_trades=2,
                ft_win_rate=50.0,
            )
        ]

    monkeypatch.setattr("core.walkforward_engine.run_forward_test", fake_forward_test)
    engine = _grid_post_process_engine(ft=True)
    df = pd.DataFrame(
        {"Open": 1.0, "High": 1.1, "Low": 0.9, "Close": 1.0, "Volume": 100},
        index=pd.date_range("2025-01-01", periods=40, freq="D", tz="UTC"),
    )

    pipeline = engine._run_window_is_pipeline(df, df.index[0], df.index[19], 1)

    assert pipeline.best_params_source == "forward_test"
    assert pipeline.best_params["source"] == "ft"
    assert pipeline.best_trial_number == ft_selected_trial
    assert pipeline.selection_chain["forward_test"] == ft_selected_trial
    assert pipeline.module_status["forward_test"]["reason"] is None


def test_grid_wfa_dsr_then_forward_test_uses_dsr_candidates(monkeypatch):
    objective = _grid_post_process_result(1, "objective", grid_rank=1)
    dsr_base = _grid_post_process_result(5, "dsr", grid_rank=5)
    dsr_base.selection_sources = ["objective", "dsr"]
    dsr_base.is_dsr_selected = True
    dsr_base.dsr_rank = 1
    dsr_base.dsr_probability = 0.99

    monkeypatch.setattr(WalkForwardEngine, "_run_optuna_on_window", lambda *args, **kwargs: ([objective, dsr_base], [objective, dsr_base]))

    def fake_forward_test(**kwargs):
        assert [item.params["source"] for item in kwargs["optuna_results"]] == ["dsr"]
        return [
            SimpleNamespace(
                trial_number=5,
                source_rank=1,
                params={"source": "ft_after_dsr", "maType3": "HMA", "maLength3": 105},
                ft_rank=1,
                ft_passes_threshold=True,
                ft_net_profit_pct=4.0,
                ft_max_drawdown_pct=1.0,
                ft_total_trades=2,
                ft_win_rate=50.0,
            )
        ]

    monkeypatch.setattr("core.walkforward_engine.run_forward_test", fake_forward_test)
    engine = _grid_post_process_engine(dsr=True, ft=True)
    df = pd.DataFrame(
        {"Open": 1.0, "High": 1.1, "Low": 0.9, "Close": 1.0, "Volume": 100},
        index=pd.date_range("2025-01-01", periods=40, freq="D", tz="UTC"),
    )

    pipeline = engine._run_window_is_pipeline(df, df.index[0], df.index[19], 1)

    assert pipeline.best_params_source == "forward_test"
    assert pipeline.best_params["source"] == "ft_after_dsr"
    assert pipeline.selection_chain["dsr"] == 5
    assert pipeline.selection_chain["forward_test"] == 5


def test_grid_wfa_stress_test_selects_st_rank_one(monkeypatch):
    objective = _grid_post_process_result(1, "objective", grid_rank=1)
    monkeypatch.setattr(WalkForwardEngine, "_run_optuna_on_window", lambda *args, **kwargs: ([objective], [objective]))

    def fake_stress_test(**kwargs):
        assert [item.params["source"] for item in kwargs["source_results"]] == ["objective"]
        return [
            SimpleNamespace(
                trial_number=1,
                source_rank=1,
                st_rank=1,
                status="ok",
                base_net_profit_pct=2.0,
                base_max_drawdown_pct=1.0,
                base_romad=2.0,
                base_sharpe_ratio=None,
            )
        ], {}

    monkeypatch.setattr("core.walkforward_engine.run_stress_test", fake_stress_test)
    engine = _grid_post_process_engine(st=True)
    df = pd.DataFrame(
        {"Open": 1.0, "High": 1.1, "Low": 0.9, "Close": 1.0, "Volume": 100},
        index=pd.date_range("2025-01-01", periods=40, freq="D", tz="UTC"),
    )

    pipeline = engine._run_window_is_pipeline(df, df.index[0], df.index[19], 1)

    assert pipeline.best_params_source == "stress_test"
    assert pipeline.best_params["source"] == "objective"
    assert pipeline.best_trial_number == 1
    assert pipeline.selection_chain["stress_test"] == 1


def test_grid_wfa_forward_test_then_stress_test_uses_ft_candidates(monkeypatch):
    objective = _grid_post_process_result(1, "objective", grid_rank=1)
    monkeypatch.setattr(WalkForwardEngine, "_run_optuna_on_window", lambda *args, **kwargs: ([objective], [objective]))

    def fake_forward_test(**kwargs):
        return [
            SimpleNamespace(
                trial_number=1,
                source_rank=1,
                params={"source": "ft", "maType3": "HMA", "maLength3": 101},
                ft_rank=1,
                ft_passes_threshold=True,
                ft_net_profit_pct=4.0,
                ft_max_drawdown_pct=1.0,
                ft_total_trades=2,
                ft_win_rate=50.0,
            )
        ]

    def fake_stress_test(**kwargs):
        assert [item.params["source"] for item in kwargs["source_results"]] == ["ft"]
        return [
            SimpleNamespace(
                trial_number=1,
                source_rank=1,
                st_rank=1,
                status="ok",
                base_net_profit_pct=2.0,
                base_max_drawdown_pct=1.0,
                base_romad=2.0,
                base_sharpe_ratio=None,
            )
        ], {}

    monkeypatch.setattr("core.walkforward_engine.run_forward_test", fake_forward_test)
    monkeypatch.setattr("core.walkforward_engine.run_stress_test", fake_stress_test)
    engine = _grid_post_process_engine(ft=True, st=True)
    df = pd.DataFrame(
        {"Open": 1.0, "High": 1.1, "Low": 0.9, "Close": 1.0, "Volume": 100},
        index=pd.date_range("2025-01-01", periods=40, freq="D", tz="UTC"),
    )

    pipeline = engine._run_window_is_pipeline(df, df.index[0], df.index[19], 1)

    assert pipeline.best_params_source == "stress_test"
    assert pipeline.best_params["source"] == "ft"
    assert pipeline.selection_chain["forward_test"] == 1
    assert pipeline.selection_chain["stress_test"] == 1


def test_fixed_wfa_is_failure_includes_window_context(monkeypatch):
    index = pd.date_range("2025-01-01", periods=40, freq="D", tz="UTC")
    df = pd.DataFrame(
        {"Open": 1.0, "High": 1.1, "Low": 0.9, "Close": 1.0, "Volume": 100},
        index=index,
    )

    wf_config = WFConfig(
        strategy_id="s03_reversal_v10",
        is_period_days=10,
        oos_period_days=5,
        warmup_bars=5,
    )
    engine = WalkForwardEngine(
        wf_config,
        {"optimization_mode": "grid", "fixed_params": {"dateFilter": False}},
        {},
    )

    def fail_pipeline(self, df, is_start, is_end, window_id):  # noqa: ARG001
        raise ValueError(
            "Grid fast-vs-slow validation failed: "
            '{"candidate_id": 7, "semantic_key": "abc", "diffs": {"net_profit_pct": 1.2}}'
        )

    monkeypatch.setattr(WalkForwardEngine, "_run_window_is_pipeline", fail_pipeline)

    with pytest.raises(ValueError) as excinfo:
        engine.run_wf_optimization(df)

    message = str(excinfo.value)
    assert "Fixed WFA window 1 IS optimization failed" in message
    assert "2025-01-01 to 2025-01-10" in message
    assert "optimizer=grid" in message
    assert "Grid fast-vs-slow validation failed" in message
    assert '"candidate_id": 7' in message


def test_fixed_wfa_ft_retry_delays_entry_and_trades_remaining_window(monkeypatch):
    index = pd.date_range("2025-01-01", periods=70, freq="D", tz="UTC")
    df = pd.DataFrame(
        {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0},
        index=index,
    )

    config = WFConfig(
        strategy_id="s01_trailing_ma",
        is_period_days=20,
        oos_period_days=10,
        post_process=PostProcessConfig(
            enabled=True,
            ft_period_days=5,
            ft_threshold_pct=-5.0,
            ft_reject_action="cooldown_reoptimize",
            ft_reject_cooldown_days=5,
            ft_reject_max_attempts=2,
            ft_reject_min_remaining_oos_days=3,
        ),
    )
    engine = WalkForwardEngine(config, {"fixed_params": {"dateFilter": False}}, {})

    call_counter = {"count": 0}

    def fake_pipeline(self, df, is_start, is_end, window_id):  # noqa: ARG001
        call_counter["count"] += 1
        return ISPipelineResult(
            best_result=OptimizationResult(
                params={"maType": "EMA", "maLength": 20, "closeCountLong": 7},
                net_profit_pct=5.0,
                max_drawdown_pct=1.0,
                total_trades=3,
                optuna_trial_number=window_id + call_counter["count"],
            ),
            best_params={"maType": "EMA", "maLength": 20, "closeCountLong": 7},
            param_id=f"retry_{call_counter['count']}",
            best_trial_number=window_id + call_counter["count"],
            best_params_source="forward_test",
            is_pareto_optimal=None,
            constraints_satisfied=True,
            available_modules=["optuna_is", "forward_test"],
            module_status={
                "optuna_is": {"enabled": True, "ran": True, "reason": None},
                "forward_test": {"enabled": True, "ran": True, "reason": None},
            },
            selection_chain={"optuna_is": 1},
            optimization_start=is_start,
            optimization_end=is_end - pd.Timedelta(days=5),
            ft_start=is_end - pd.Timedelta(days=5),
            ft_end=is_end,
            optuna_is_trials=[],
            dsr_trials=None,
            forward_test_trials=[],
            stress_test_trials=None,
            ft_gate_failed=call_counter["count"] == 1,
            ft_pass_count=0 if call_counter["count"] == 1 else 1,
        )

    def fake_backtest(self, df, start, end, params):  # noqa: ARG001
        trade = TradeRecord(
            entry_time=start,
            exit_time=end,
            net_pnl=5.0,
            profit_pct=5.0,
        )
        return StrategyResult(
            trades=[trade],
            equity_curve=[100.0, 105.0],
            balance_curve=[100.0, 105.0],
            timestamps=[start, end],
        )

    monkeypatch.setattr(WalkForwardEngine, "_run_window_is_pipeline", fake_pipeline)
    monkeypatch.setattr(WalkForwardEngine, "_run_period_backtest", fake_backtest)

    result, _study_id = engine.run_wf_optimization(df)

    first_window = result.windows[0]
    assert first_window.window_status == "traded"
    assert first_window.entry_delay_days == pytest.approx(5.0)
    assert first_window.ft_retry_attempts_used == 1
    assert first_window.trade_start == pd.Timestamp("2025-01-26", tz="UTC")
    assert first_window.is_end == pd.Timestamp("2025-01-25", tz="UTC")
    assert call_counter["count"] >= 2

def test_walkforward_integration_with_sample_data(monkeypatch):
    data_path = Path(__file__).parent.parent / "data" / "raw" / "OKX_LINKUSDT.P, 15 2025.05.01-2025.11.20.csv"
    if not data_path.exists():
        pytest.skip("Sample data file not available for integration test.")

    df = load_data(str(data_path))

    strategy_id = "s01_trailing_ma"
    default_params = _build_params_from_config(strategy_id)

    class FakeResult:
        def __init__(self, params):
            self.params = params

    def fake_optuna(self, df_slice, start_time, end_time):  # noqa: ARG001
        fake_results = [FakeResult(default_params)]
        return fake_results, list(fake_results)

    monkeypatch.setattr(WalkForwardEngine, "_run_optuna_on_window", fake_optuna)

    wf_config = WFConfig(
        strategy_id=strategy_id,
        is_period_days=60,
        oos_period_days=30,
        warmup_bars=200,
    )
    base_template = {
        "fixed_params": {"dateFilter": False},
        "risk_per_trade_pct": 2.0,
        "contract_size": 0.01,
        "commission_rate": 0.0005,
    }
    engine = WalkForwardEngine(wf_config, base_template, {})

    result, _study_id = engine.run_wf_optimization(df)

    assert result.total_windows >= 2
    assert result.stitched_oos is not None
    assert result.windows
