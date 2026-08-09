"""Runner and real-fixture evidence for opt-in Daily Sharpe."""

from __future__ import annotations

from dataclasses import fields
import math

import numpy as np
import pandas as pd
import pytest

from core import metrics
from core.engine_v2 import runner as runner_module
from core.engine_v2.runner import run_v2_strategy
from core.metrics import AdvancedMetrics, _advanced_metric_view
from strategies.s03_reversal_v11_regime_er_b2 import strategy as s03_strategy
from strategies.s06_r_trend_v02_b2 import strategy as s06_strategy
from strategies.s06_r_trend_v02_regime_trendlines_b2 import strategy as s06_regime_strategy

from s03_regime_er_test_helpers import (
    REFERENCE_A,
    merged_reference_params as s03_params,
    prepared_reference_dataset as s03_dataset,
)
from s06_b2_test_helpers import (
    merged_reference_params as s06_params,
    prepared_reference_dataset as s06_dataset,
    trade_signature,
)
from s06_regime_tl_test_helpers import (
    REFERENCE_B as S06_REGIME_REFERENCE,
    merged_reference_params as s06_regime_params,
    prepared_reference_dataset as s06_regime_dataset,
)


S06_REFERENCE = "reference_b_trend_bracket"
DAILY_FIELDS = {
    "sharpe_daily",
    "sharpe_daily_observations",
    "sharpe_daily_active_days",
}


def _run_s06(enabled: bool):
    params = s06_params(S06_REFERENCE)
    prepared, trade_start_idx = s06_dataset()
    return run_v2_strategy(
        data=s06_strategy.build_v2_execution_data(prepared, params),
        profile=s06_strategy.load_profile(),
        params=params,
        trade_start_idx=trade_start_idx,
        compute_sharpe_daily=enabled,
    )


def _run_s03(enabled: bool):
    params = s03_params(REFERENCE_A)
    prepared, trade_start_idx = s03_dataset()
    return run_v2_strategy(
        data=s03_strategy.build_v2_execution_data(prepared, params),
        profile=s03_strategy.load_profile(),
        params=params,
        trade_start_idx=trade_start_idx,
        compute_sharpe_daily=enabled,
    )


def _run_s06_regime(enabled: bool):
    params = s06_regime_params(S06_REGIME_REFERENCE)
    prepared, trade_start_idx = s06_regime_dataset()
    return run_v2_strategy(
        data=s06_regime_strategy.build_v2_execution_data(prepared, params),
        profile=s06_regime_strategy.load_profile(),
        params=params,
        trade_start_idx=trade_start_idx,
        compute_sharpe_daily=enabled,
    )


def _literal_daily_oracle(equity, timestamps, anchor):
    closes = []
    active_day = None
    last_equity = None
    for raw_equity, raw_timestamp in zip(equity, timestamps):
        timestamp = pd.Timestamp(raw_timestamp)
        timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
        if active_day is not None and timestamp.date() != active_day:
            closes.append(float(last_equity))
        active_day = timestamp.date()
        last_equity = raw_equity
    closes.append(float(last_equity))

    opening = float(anchor)
    returns = []
    for close in closes:
        returns.append(close / opening - 1.0)
        opening = close
    return np.asarray(returns, dtype=float)


def _assert_old_snapshots_identical(disabled, enabled):
    old_advanced_fields = [item.name for item in fields(AdvancedMetrics) if item.name not in DAILY_FIELDS]
    assert disabled.basic_metrics == enabled.basic_metrics
    for name in old_advanced_fields:
        assert getattr(disabled.advanced_metrics, name) == getattr(enabled.advanced_metrics, name)
    assert disabled.strategy_result.to_dict() == enabled.strategy_result.to_dict()
    assert disabled.strategy_result.equity_curve == enabled.strategy_result.equity_curve
    assert disabled.strategy_result.balance_curve == enabled.strategy_result.balance_curve
    assert trade_signature(disabled.strategy_result) == trade_signature(enabled.strategy_result)
    assert disabled.guardrail_summary == enabled.guardrail_summary
    for item in fields(disabled.standing_state):
        left = getattr(disabled.standing_state, item.name)
        right = getattr(enabled.standing_state, item.name)
        if isinstance(left, float) and math.isnan(left):
            assert isinstance(right, float) and math.isnan(right)
        else:
            assert left == right


@pytest.mark.parametrize(
    "run_family",
    [_run_s06, _run_s06_regime, _run_s03],
    ids=["position_bracket", "position_regime_trendlines", "signal_reversal"],
)
def test_enabled_real_runner_matches_independent_oracle_and_preserves_existing_output(run_family):
    disabled = run_family(False)
    enabled = run_family(True)
    _assert_old_snapshots_identical(disabled, enabled)

    view = _advanced_metric_view(enabled.strategy_result)
    oracle_returns = _literal_daily_oracle(
        view.equity_observations,
        view.timestamps,
        view.initial_equity,
    )
    oracle_sharpe = metrics._calculate_sharpe_daily_value(oracle_returns, 0.02)

    assert disabled.advanced_metrics.sharpe_daily is None
    assert disabled.advanced_metrics.sharpe_daily_observations is None
    assert disabled.advanced_metrics.sharpe_daily_active_days is None
    assert enabled.advanced_metrics.sharpe_daily == pytest.approx(oracle_sharpe)
    assert enabled.advanced_metrics.sharpe_daily_observations == oracle_returns.size
    assert enabled.advanced_metrics.sharpe_daily_active_days == int(
        np.count_nonzero(np.abs(oracle_returns) > 1e-12)
    )
    assert not any(hasattr(enabled.strategy_result, name) for name in DAILY_FIELDS)


def test_enabled_runner_uses_exactly_one_enrichment_and_returns_its_typed_snapshot(monkeypatch):
    original = runner_module.metrics.enrich_strategy_result
    returned = []

    def counted(*args, **kwargs):
        snapshot = original(*args, **kwargs)
        returned.append((snapshot, kwargs))
        return snapshot

    monkeypatch.setattr(runner_module.metrics, "enrich_strategy_result", counted)
    run = _run_s06(True)

    assert len(returned) == 1
    assert returned[0][1]["compute_sharpe_daily"] is True
    assert run.basic_metrics is returned[0][0][0]
    assert run.advanced_metrics is returned[0][0][1]
    assert set(DAILY_FIELDS) <= set(run.advanced_metrics.to_dict())


def test_disabled_runner_never_invokes_daily_aggregation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled runner constructed UTC daily returns")

    monkeypatch.setattr(metrics, "_calculate_daily_returns", forbidden)
    run = _run_s03(False)

    assert run.advanced_metrics.sharpe_daily is None
    assert run.advanced_metrics.sharpe_daily_observations is None
    assert run.advanced_metrics.sharpe_daily_active_days is None
