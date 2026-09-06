import math

import pytest


from core.optuna_engine import (
    ConstraintSpec,
    OptimizationConfig,
    OptunaConfig,
    OptunaOptimizer,
    evaluate_constraints,
)


def _make_optimizer(objectives, sanitize_enabled=True, sanitize_trades_threshold=0):
    base_config = OptimizationConfig(
        csv_file="dummy.csv",
        strategy_id="s01_trailing_ma",
        enabled_params={},
        param_ranges={},
        param_types={},
        fixed_params={},
    )
    optuna_config = OptunaConfig(
        objectives=objectives,
        sanitize_enabled=sanitize_enabled,
        sanitize_trades_threshold=sanitize_trades_threshold,
    )
    return OptunaOptimizer(base_config, optuna_config)


def test_sanitize_enabled_trades_zero_sanitizes_metrics():
    optimizer = _make_optimizer(
        ["sharpe_ratio", "sortino_ratio", "sqn", "profit_factor"],
        sanitize_enabled=True,
        sanitize_trades_threshold=0,
    )
    metrics = {
        "total_trades": 0,
        "sharpe_ratio": float("nan"),
        "sortino_ratio": float("inf"),
        "sqn": None,
        "profit_factor": None,
    }
    values, _, objective_return, failed = optimizer._prepare_objective_values(metrics)
    assert failed is False
    assert values == [0.0, 0.0, 0.0, 0.0]
    assert objective_return == tuple(values)


def test_sanitize_enabled_trades_one_fails_non_finite():
    optimizer = _make_optimizer(
        ["sharpe_ratio"],
        sanitize_enabled=True,
        sanitize_trades_threshold=0,
    )
    metrics = {"total_trades": 1, "sharpe_ratio": float("nan")}
    values, _, objective_return, failed = optimizer._prepare_objective_values(metrics)
    assert failed is True
    assert values == [None]
    assert math.isnan(objective_return)


def test_sanitize_disabled_trades_zero_fails():
    optimizer = _make_optimizer(
        ["sharpe_ratio", "profit_factor"],
        sanitize_enabled=False,
        sanitize_trades_threshold=0,
    )
    metrics = {"total_trades": 0, "sharpe_ratio": float("nan"), "profit_factor": None}
    _, _, objective_return, failed = optimizer._prepare_objective_values(metrics)
    assert failed is True
    assert isinstance(objective_return, tuple)
    assert all(math.isnan(value) for value in objective_return)


def test_profit_factor_inf_always_fails():
    optimizer = _make_optimizer(
        ["profit_factor"],
        sanitize_enabled=True,
        sanitize_trades_threshold=10,
    )
    metrics = {"total_trades": 0, "profit_factor": float("inf")}
    _, _, objective_return, failed = optimizer._prepare_objective_values(metrics)
    assert failed is True
    assert math.isnan(objective_return)


def test_multi_objective_profit_factor_inf_returns_nan_tuple():
    optimizer = _make_optimizer(
        ["net_profit_pct", "profit_factor"],
        sanitize_enabled=True,
        sanitize_trades_threshold=10,
    )
    metrics = {"total_trades": 10, "net_profit_pct": 5.0, "profit_factor": float("inf")}
    _, _, objective_return, failed = optimizer._prepare_objective_values(metrics)
    assert failed is True
    assert isinstance(objective_return, tuple)
    assert len(objective_return) == 2
    assert all(math.isnan(value) for value in objective_return)


def test_profit_factor_inf_not_objective_does_not_fail_constraints_only():
    optimizer = _make_optimizer(
        ["net_profit_pct"],
        sanitize_enabled=True,
        sanitize_trades_threshold=10,
    )
    metrics = {"total_trades": 10, "net_profit_pct": 5.0, "profit_factor": float("inf")}
    _, _, objective_return, failed = optimizer._prepare_objective_values(metrics)
    assert failed is False
    assert objective_return == 5.0

    constraints = [ConstraintSpec(metric="profit_factor", threshold=1.2, enabled=True)]
    violations = evaluate_constraints(metrics, constraints)
    assert len(violations) == 1
    assert violations[0] > 0
