from __future__ import annotations

import json
import math
from dataclasses import asdict, fields

import pandas as pd
import pytest

from core.engine_v2 import runner as runner_module
from core.engine_v2.compiled_kernel import compiled_batch_available
from core.grid_v2 import (
    GridV2Settings,
    GridV2StrategyHooks,
    build_grid_v2_plan,
    execute_grid_v2_candidates,
)
from core.metrics import AdvancedMetrics, BasicMetrics
from strategies.s03_reversal_v11_regime_er_b2 import strategy as s03_strategy
from strategies.s03_reversal_v11_regime_er_b2.strategy import load_config as load_s03_config
from strategies.s06_r_trend_v02_b2 import strategy as s06_strategy
from strategies.s06_r_trend_v02_b2.strategy import load_config as load_s06_config

from s03_regime_er_test_helpers import (
    REFERENCE_A as S03_REFERENCE,
    merged_reference_params as s03_params,
    prepared_reference_dataset as s03_dataset,
    run_reference as run_s03_reference,
)
from s06_b2_test_helpers import (
    merged_reference_params as s06_params,
    prepared_reference_dataset as s06_dataset,
    run_reference as run_s06_reference,
)


S06_REFERENCE = "reference_b_trend_bracket"


SELECTED_METRIC_OWNERS = {
    "net_profit_pct": (BasicMetrics, "net_profit_pct"),
    "max_drawdown_pct": (BasicMetrics, "max_drawdown_pct"),
    "romad": (AdvancedMetrics, "romad"),
    "profit_factor": (AdvancedMetrics, "profit_factor"),
    "win_rate_pct": (BasicMetrics, "win_rate"),
    "total_trades": (BasicMetrics, "total_trades"),
    "winning_trades": (BasicMetrics, "winning_trades"),
    "losing_trades": (BasicMetrics, "losing_trades"),
    "gross_profit": (BasicMetrics, "gross_profit"),
    "gross_loss": (BasicMetrics, "gross_loss"),
    "max_consecutive_losses": (BasicMetrics, "max_consecutive_losses"),
    "sharpe_ratio": (AdvancedMetrics, "sharpe_ratio"),
    "sortino_ratio": (AdvancedMetrics, "sortino_ratio"),
    "sqn": (AdvancedMetrics, "sqn"),
    "ulcer_index": (AdvancedMetrics, "ulcer_index"),
    "consistency_score": (AdvancedMetrics, "consistency_score"),
}


def _assert_metric_equal(actual, expected) -> None:
    if expected is None:
        assert actual is None
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected)
    else:
        assert actual == expected


@pytest.mark.parametrize(
    "run_reference",
    [
        pytest.param(lambda: run_s06_reference(S06_REFERENCE), id="position_bracket"),
        pytest.param(lambda: run_s03_reference(S03_REFERENCE), id="signal_reversal"),
    ],
)
def test_runner_retains_exact_typed_snapshots_from_one_enrichment(monkeypatch, run_reference):
    original = runner_module.metrics.enrich_strategy_result
    returned = []

    def counted_enrichment(*args, **kwargs):
        snapshots = original(*args, **kwargs)
        returned.append(snapshots)
        return snapshots

    monkeypatch.setattr(runner_module.metrics, "enrich_strategy_result", counted_enrichment)

    run = run_reference()

    assert len(returned) == 1
    assert run.basic_metrics is returned[0][0]
    assert run.advanced_metrics is returned[0][1]
    assert isinstance(run.basic_metrics, BasicMetrics)
    assert isinstance(run.advanced_metrics, AdvancedMetrics)
    assert run.basic_metrics.win_rate == pytest.approx(
        run.basic_metrics.winning_trades / run.basic_metrics.total_trades * 100.0
    )
    assert run.advanced_metrics.sortino_ratio is not None
    for name in (
        "net_profit_pct",
        "max_drawdown_pct",
        "total_trades",
        "winning_trades",
        "losing_trades",
        "gross_profit",
        "gross_loss",
    ):
        _assert_metric_equal(getattr(run.strategy_result, name), getattr(run.basic_metrics, name))
    for name in (
        "romad",
        "profit_factor",
        "sharpe_ratio",
        "sqn",
        "ulcer_index",
        "consistency_score",
    ):
        _assert_metric_equal(getattr(run.strategy_result, name), getattr(run.advanced_metrics, name))


def _execute_one_selected(
    *,
    strategy_module,
    config,
    params,
    prepared_data,
    prefer_compiled: bool,
):
    df, trade_start_idx = prepared_data
    hooks = GridV2StrategyHooks.from_strategy(strategy_module)
    plan = build_grid_v2_plan(
        config,
        GridV2Settings(
            enabled_axes=(),
            top_n=1,
            slow_enrich_selected=True,
            prefer_compiled=prefer_compiled,
        ),
        base_params=params,
    )
    result = execute_grid_v2_candidates(plan, df, trade_start_idx, hooks, candidate_indices=(0,))
    candidate = plan.candidate_for_index(result.selected[0].row.candidate_id - 1)
    normalized = hooks.normalize_params(dict(candidate.params))
    direct = runner_module.run_v2_strategy(
        data=hooks.build_execution_data(df, normalized),
        profile=plan.profile,
        params=normalized,
        trade_start_idx=trade_start_idx,
    )
    return result, direct


@pytest.mark.parametrize("prefer_compiled", [False, True], ids=["reference", "compiled"])
def test_position_selected_payload_uses_complete_typed_metrics(prefer_compiled):
    if prefer_compiled and not compiled_batch_available():
        pytest.skip("Compiled Grid V2 is unavailable in this process.")
    result, direct = _execute_one_selected(
        strategy_module=s06_strategy,
        config=load_s06_config(),
        params=s06_params(S06_REFERENCE),
        prepared_data=s06_dataset(),
        prefer_compiled=prefer_compiled,
    )

    selected = result.selected[0]
    assert set(selected.metrics) == set(SELECTED_METRIC_OWNERS)
    for key, (owner, field_name) in SELECTED_METRIC_OWNERS.items():
        snapshot = direct.basic_metrics if owner is BasicMetrics else direct.advanced_metrics
        _assert_metric_equal(selected.metrics[key], getattr(snapshot, field_name))
    assert selected.row.candidate_id == result.rows[0].candidate_id
    assert selected.row.status == result.rows[0].status == "ok"
    expected_guardrails = asdict(direct.guardrail_summary)
    assert set(selected.guardrail_summary) == set(expected_guardrails)
    for key, expected in expected_guardrails.items():
        if isinstance(expected, float) and not math.isfinite(expected):
            assert selected.guardrail_summary[key] == str(expected)
        else:
            _assert_metric_equal(selected.guardrail_summary[key], expected)
    json.dumps(selected.metrics, allow_nan=False)


def test_signal_reversal_selected_payload_transports_finite_sortino():
    result, direct = _execute_one_selected(
        strategy_module=s03_strategy,
        config=load_s03_config(),
        params=s03_params(S03_REFERENCE),
        prepared_data=s03_dataset(),
        prefer_compiled=False,
    )

    selected = result.selected[0]
    assert selected.metrics["win_rate_pct"] == pytest.approx(direct.basic_metrics.win_rate)
    assert math.isfinite(selected.metrics["sortino_ratio"])
    assert selected.metrics["sortino_ratio"] == pytest.approx(direct.advanced_metrics.sortino_ratio)


def test_selected_payload_preserves_undefined_sortino():
    end = pd.Timestamp("2025-10-01T00:00:00Z")
    result, direct = _execute_one_selected(
        strategy_module=s06_strategy,
        config=load_s06_config(),
        params=s06_params(S06_REFERENCE, {"end": "2025-10-01T00:00:00Z"}),
        prepared_data=s06_dataset(end=end),
        prefer_compiled=False,
    )

    assert direct.basic_metrics.total_trades == 24
    assert direct.advanced_metrics.sharpe_ratio is not None
    assert direct.advanced_metrics.sortino_ratio is None
    assert result.selected[0].metrics["sortino_ratio"] is None


def test_selected_metric_keys_have_declared_typed_owners():
    declared = {
        BasicMetrics: {item.name for item in fields(BasicMetrics)},
        AdvancedMetrics: {item.name for item in fields(AdvancedMetrics)},
    }

    assert set(SELECTED_METRIC_OWNERS) == {
        "net_profit_pct",
        "max_drawdown_pct",
        "romad",
        "profit_factor",
        "win_rate_pct",
        "total_trades",
        "winning_trades",
        "losing_trades",
        "gross_profit",
        "gross_loss",
        "max_consecutive_losses",
        "sharpe_ratio",
        "sortino_ratio",
        "sqn",
        "ulcer_index",
        "consistency_score",
    }
    assert all(field_name in declared[owner] for owner, field_name in SELECTED_METRIC_OWNERS.values())
