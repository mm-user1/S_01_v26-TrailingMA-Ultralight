from types import SimpleNamespace

import optuna
import pandas as pd
import pytest

from core.grid_engine import (
    GRID_FAST_ONLY_OBJECTIVES,
    GRID_SUPPORTED_FAST_OBJECTIVES,
    GRID_SUPPORTED_SLOW_OBJECTIVES,
    rank_grid_results,
)
from core.optuna_engine import (
    OBJECTIVE_DIRECTIONS,
    OptimizationResult,
    OptunaConfig,
    OptunaOptimizer,
    _result_from_trial,
)


def _result(*, daily, monthly=1.0, trial=1):
    return OptimizationResult(
        params={"trial": trial},
        net_profit_pct=float(trial),
        max_drawdown_pct=1.0,
        total_trades=1,
        sharpe_ratio=monthly,
        sharpe_daily=daily,
        sharpe_daily_observations=None if daily is None else 3,
        sharpe_daily_active_days=None if daily is None else 2,
        optuna_trial_number=trial,
    )


def test_daily_sharpe_objective_authorities_and_grid_fast_only_contract():
    assert OBJECTIVE_DIRECTIONS["sharpe_daily"] == "maximize"
    assert GRID_FAST_ONLY_OBJECTIVES == {"sharpe_daily"}
    assert "sharpe_daily" in GRID_SUPPORTED_FAST_OBJECTIVES
    assert "sharpe_daily" not in GRID_SUPPORTED_SLOW_OBJECTIVES


@pytest.mark.parametrize(
    "objectives, expected",
    [
        (["net_profit_pct"], False),
        (["sharpe_daily"], True),
        (["net_profit_pct", "sharpe_daily"], True),
    ],
)
def test_optuna_derives_daily_request_from_objectives(monkeypatch, objectives, expected):
    captured = []

    def fake_run(args):
        captured.append(args)
        return _result(daily=1.25 if args[-1] else None)

    monkeypatch.setattr("core.optuna_engine._run_single_combination", fake_run)
    optimizer = OptunaOptimizer(
        SimpleNamespace(score_config=None),
        OptunaConfig(objectives=objectives, primary_objective=objectives[0]),
    )
    optimizer.df = pd.DataFrame({"close": [1.0]})
    optimizer.strategy_class = object

    result = optimizer._evaluate_parameters({"x": 1})

    assert captured[0][-1] is expected
    assert (result.sharpe_daily is not None) is expected


def test_optuna_trial_reconstruction_preserves_daily_ratio_and_integer_diagnostics():
    trial = optuna.trial.create_trial(
        state=optuna.trial.TrialState.COMPLETE,
        value=2.5,
        user_attrs={
            "merlin.params": {"x": 1},
            "merlin.objective_values": [2.5],
            "merlin.all_metrics": {
                "sharpe_daily": 2.5,
                "sharpe_daily_observations": 7,
                "sharpe_daily_active_days": 0,
            },
        },
    )

    result = _result_from_trial(trial)

    assert result.sharpe_daily == 2.5
    assert result.sharpe_daily_observations == 7
    assert result.sharpe_daily_active_days == 0
    assert isinstance(result.sharpe_daily_observations, int)
    assert isinstance(result.sharpe_daily_active_days, int)


@pytest.mark.parametrize("invalid", [1.5, "7", True, -1])
def test_optuna_trial_reconstruction_rejects_malformed_daily_diagnostics(invalid):
    trial = optuna.trial.create_trial(
        state=optuna.trial.TrialState.COMPLETE,
        value=1.0,
        user_attrs={
            "merlin.all_metrics": {
                "sharpe_daily_observations": invalid,
                "sharpe_daily_active_days": 0,
            }
        },
    )

    with pytest.raises(ValueError, match="sharpe_daily_observations"):
        _result_from_trial(trial)


def test_optuna_unavailable_daily_objective_is_not_sanitized_to_zero():
    optimizer = OptunaOptimizer(
        SimpleNamespace(score_config=None),
        OptunaConfig(
            objectives=["sharpe_daily"],
            sanitize_enabled=True,
            sanitize_trades_threshold=10,
        ),
    )

    values, sanitized, returned, should_fail = optimizer._prepare_objective_values(
        {"sharpe_daily": None, "total_trades": 0}
    )

    assert values == [None]
    assert sanitized == []
    assert should_fail is True
    assert pd.isna(returned)


def test_grid_daily_ranking_filters_partial_population_and_never_substitutes_zero():
    ranked = rank_grid_results(
        [_result(daily=None, trial=1), _result(daily=-0.5, trial=2), _result(daily=1.5, trial=3)],
        objectives=["sharpe_daily"],
        primary_objective=None,
        constraints=[],
    )

    assert [item.optuna_trial_number for item in ranked] == [3, 2]
    assert [item.objective_values for item in ranked] == [[1.5], [-0.5]]


def test_grid_daily_monthly_and_other_objective_ranking():
    ranked = rank_grid_results(
        [_result(daily=0.5, monthly=2.0, trial=1), _result(daily=1.5, monthly=1.0, trial=2)],
        objectives=["sharpe_daily", "sharpe_ratio", "net_profit_pct"],
        primary_objective="sharpe_daily",
        constraints=[],
    )

    assert ranked[0].optuna_trial_number == 2
    assert ranked[0].objective_values == [1.5, 1.0, 2.0]


def test_grid_all_unavailable_daily_has_actionable_daily_only_hint():
    with pytest.raises(ValueError) as exc_info:
        rank_grid_results(
            [_result(daily=None, trial=1)],
            objectives=["sharpe_daily"],
            primary_objective=None,
            constraints=[],
        )

    message = str(exc_info.value)
    assert "no candidates with usable objective values" in message
    assert "Daily Sharpe is unavailable" in message
    assert "Sharpe is undefined" not in message
