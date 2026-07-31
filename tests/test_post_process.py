from pathlib import Path
from copy import deepcopy
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.post_process import (
    PostProcessConfig,
    _ft_worker_entry,
    annotate_ft_threshold,
    calculate_comparison_metrics,
    calculate_ft_dates,
    calculate_period_dates,
    calculate_profit_degradation,
    filter_ft_passed_results,
    ft_result_meets_threshold,
    normalize_ft_reject_action,
    run_forward_test,
)
from core.backtest_engine import StrategyResult, TradeRecord


def test_calculate_ft_dates_basic():
    start = pd.Timestamp("2025-05-01", tz="UTC")
    end = pd.Timestamp("2025-09-01", tz="UTC")
    is_end, ft_start, ft_end, is_days, ft_days = calculate_ft_dates(start, end, 30)
    assert ft_end == end
    assert ft_start == end - pd.Timedelta(days=30)
    assert is_end == ft_start
    assert is_days == (is_end - start).days
    assert ft_days == 30


def test_calculate_ft_dates_invalid():
    start = pd.Timestamp("2025-05-01", tz="UTC")
    end = pd.Timestamp("2025-05-10", tz="UTC")
    try:
        calculate_ft_dates(start, end, 10)
    except ValueError as exc:
        assert "FT period" in str(exc)
    else:
        raise AssertionError("Expected ValueError for FT period >= range")


def test_calculate_profit_degradation_annualized():
    is_profit = 10.0
    ft_profit = 5.0
    ratio = calculate_profit_degradation(is_profit, ft_profit, 100, 50)
    assert abs(ratio - 1.0) < 1e-6


def test_calculate_period_dates_oos_only():
    start = pd.Timestamp("2025-05-01", tz="UTC")
    end = pd.Timestamp("2025-11-20", tz="UTC")
    result = calculate_period_dates(
        start,
        end,
        ft_enabled=False,
        oos_enabled=True,
        oos_period_days=30,
    )
    assert result["oos_end"] == end
    assert result["oos_start"] == end - pd.Timedelta(days=30)
    assert result["is_end"] == result["oos_start"]
    assert result["is_days"] == (result["is_end"] - start).days


def test_calculate_period_dates_ft_and_oos():
    start = pd.Timestamp("2025-05-01", tz="UTC")
    end = pd.Timestamp("2025-11-20", tz="UTC")
    result = calculate_period_dates(
        start,
        end,
        ft_enabled=True,
        ft_period_days=15,
        oos_enabled=True,
        oos_period_days=30,
    )
    assert result["oos_start"] == end - pd.Timedelta(days=30)
    assert result["ft_end"] == result["oos_start"]
    assert result["ft_start"] == result["ft_end"] - pd.Timedelta(days=15)
    assert result["is_end"] == result["ft_start"]


def test_calculate_comparison_metrics():
    is_metrics = {
        "net_profit_pct": 20.0,
        "max_drawdown_pct": 5.0,
        "romad": 4.0,
        "sharpe_ratio": 1.5,
        "profit_factor": 1.8,
    }
    ft_metrics = {
        "net_profit_pct": 10.0,
        "max_drawdown_pct": 7.0,
        "romad": 2.0,
        "sharpe_ratio": 1.0,
        "profit_factor": 1.2,
    }
    comparison = calculate_comparison_metrics(is_metrics, ft_metrics, 100, 50)
    assert comparison["max_dd_change"] == 2.0
    assert comparison["romad_change"] == -2.0
    assert comparison["sharpe_change"] == -0.5
    assert comparison["pf_change"] == pytest.approx(-0.6)


def test_ft_threshold_helpers_support_signed_thresholds():
    results = [
        {"trial_number": 1, "ft_net_profit_pct": -4.9},
        {"trial_number": 2, "ft_net_profit_pct": -5.1},
        {"trial_number": 3, "ft_net_profit_pct": 6.0},
    ]

    assert ft_result_meets_threshold(results[0], -5.0) is True
    assert ft_result_meets_threshold(results[1], -5.0) is False
    assert ft_result_meets_threshold(results[2], 5.0) is True

    annotated = annotate_ft_threshold(results, -5.0)
    assert [item["ft_passes_threshold"] for item in annotated] == [True, False, True]
    assert [item["trial_number"] for item in filter_ft_passed_results(annotated)] == [1, 3]


def test_normalize_ft_reject_action_accepts_ui_labels():
    assert normalize_ft_reject_action("Cooldown + Re-optimize") == "cooldown_reoptimize"
    assert normalize_ft_reject_action("no_trade") == "no_trade"


def test_forward_test_strips_candidate_runtime_without_mutation(monkeypatch):
    captured = {}

    class FakePool:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def starmap(self, _worker, worker_args):
            captured["worker_args"] = worker_args
            return []

    class FakeContext:
        @staticmethod
        def Pool(*, processes):
            captured["processes"] = processes
            return FakePool()

    monkeypatch.setattr("core.post_process.mp.get_context", lambda _method: FakeContext())
    original = {
        "maLength": 21,
        "dateFilter": True,
        "start": "hostile-start",
        "end": "hostile-end",
        "warmupBars": 9999,
    }
    candidate = SimpleNamespace(
        optuna_trial_number=7,
        params=deepcopy(original),
        net_profit_pct=1.0,
        max_drawdown_pct=1.0,
        total_trades=1,
        win_rate=100.0,
        max_consecutive_losses=0,
        sharpe_ratio=1.0,
        romad=1.0,
        profit_factor=1.0,
    )

    assert run_forward_test(
        csv_path="unused.csv",
        strategy_id="s06_r_trend_v02_b2",
        optuna_results=[candidate],
        config=PostProcessConfig(enabled=True, top_k=1, warmup_bars=20),
        is_period_days=30,
        ft_period_days=5,
        ft_start_date="2025-01-01",
        ft_end_date="2025-01-05",
        n_workers=1,
    ) == []
    task = captured["worker_args"][0][2]
    assert task["params"] == {"maLength": 21}
    assert captured["worker_args"][0][3:5] == ("2025-01-01", "2025-01-05")
    assert candidate.params == original


def test_forward_test_worker_projects_aligned_inclusive_bounds(monkeypatch):
    index = pd.date_range("2025-01-01", "2025-01-03 23:00", freq="h", tz="UTC")
    df = pd.DataFrame(
        {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0},
        index=index,
    )
    captured = {}

    class FakeStrategy:
        @staticmethod
        def run(df_slice, params, trade_start_idx):
            captured.update(params=params, end=df_slice.index[-1], trade_start_idx=trade_start_idx)
            final_bar = df_slice.index[-1]
            return StrategyResult(
                trades=[TradeRecord(entry_time=final_bar, exit_time=final_bar)],
                equity_curve=[100.0] * len(df_slice),
                balance_curve=[100.0] * len(df_slice),
                timestamps=list(df_slice.index),
            )

    monkeypatch.setattr("core.backtest_engine.load_data", lambda _path: df)
    monkeypatch.setattr("strategies.get_strategy", lambda _strategy_id: FakeStrategy)
    payload = _ft_worker_entry(
        "unused.csv",
        "s06_r_trend_v02_b2",
        {
            "trial_number": 1,
            "source_rank": 1,
            "params": {"maLength": 21},
            "is_metrics": {},
        },
        "2025-01-02",
        "2025-01-03",
        3,
        30,
        2,
    )

    assert payload is not None
    assert captured["params"]["dateFilter"] is True
    assert captured["params"]["start"] == pd.Timestamp("2025-01-02", tz="UTC")
    assert captured["params"]["end"] == pd.Timestamp("2025-01-03 23:00", tz="UTC")
    assert captured["end"] == captured["params"]["end"]
