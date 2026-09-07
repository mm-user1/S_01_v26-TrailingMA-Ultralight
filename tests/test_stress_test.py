import json

import numpy as np

from core.post_process import (
    StressTestConfig,
    StressTestResult,
    StressTestStatus,
    calculate_retention_metrics,
    generate_perturbations,
    run_stress_test,
)
from core.storage import generate_study_id, get_db_connection, load_study_from_db, save_stress_test_results


class MockTrial:
    def __init__(self, trial_number: int, params: dict):
        self.optuna_trial_number = trial_number
        self.params = params


def _seed_study_with_trial() -> str:
    study_id = generate_study_id()
    study_name = f"ST_{study_id[:8]}"
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO studies (study_id, study_name, strategy_id, optimization_mode)
            VALUES (?, ?, ?, ?)
            """,
            (study_id, study_name, "s01_trailing_ma", "optuna"),
        )
        conn.execute(
            """
            INSERT INTO trials (study_id, trial_number, params_json)
            VALUES (?, ?, ?)
            """,
            (study_id, 1, json.dumps({"maLength": 50, "closeCountLong": 7})),
        )
        conn.commit()
    return study_id


def test_generate_perturbations_basic():
    base_params = {"maLength": 250, "closeCountLong": 7}
    config_json = {
        "parameters": {
            "maLength": {"type": "int", "optimize": {"enabled": True, "step": 25, "min": 25, "max": 500}},
            "closeCountLong": {"type": "int", "optimize": {"enabled": True, "step": 1, "min": 1, "max": 20}},
            "maType": {"type": "select", "optimize": {"enabled": True}},
        }
    }

    perturbations = generate_perturbations(base_params, config_json)

    assert len(perturbations) == 4
    ma_perturbations = [p for p in perturbations if p["perturbed_param"] == "maLength"]
    assert len(ma_perturbations) == 2
    assert any(p["perturbed_value"] == 225 for p in ma_perturbations)
    assert any(p["perturbed_value"] == 275 for p in ma_perturbations)


def test_retention_calculation_with_quantile_method():
    base_metrics = {"net_profit_pct": 50.0, "romad": 2.0}
    perturbation_results = [
        {"net_profit_pct": 48.0, "romad": 1.9, "perturbed_param": "maLength"},
        {"net_profit_pct": 45.0, "romad": 1.8, "perturbed_param": "maLength"},
        {"net_profit_pct": 52.0, "romad": 2.1, "perturbed_param": "stopLongX"},
        {"net_profit_pct": 30.0, "romad": 1.2, "perturbed_param": "stopLongX"},
        {"net_profit_pct": 51.0, "romad": 2.0, "perturbed_param": "closeCount"},
    ]

    metrics = calculate_retention_metrics(base_metrics, perturbation_results, failure_threshold=0.7)

    assert metrics.status == StressTestStatus.OK
    assert metrics.total_perturbations == 5

    expected_ratios = np.array([48 / 50, 45 / 50, 52 / 50, 30 / 50, 51 / 50])
    expected_5pct = np.quantile(expected_ratios, 0.05, method="linear")
    expected_median = np.quantile(expected_ratios, 0.50, method="linear")

    assert abs(metrics.profit_lower_tail - expected_5pct) < 0.001
    assert abs(metrics.profit_median - expected_median) < 0.001


def test_invalid_romad_option_a():
    base_metrics = {"net_profit_pct": 50.0, "romad": None}
    perturbation_results = [
        {"net_profit_pct": 45.0, "romad": 1.5, "perturbed_param": "p1"},
        {"net_profit_pct": 48.0, "romad": 1.8, "perturbed_param": "p2"},
        {"net_profit_pct": 40.0, "romad": 1.2, "perturbed_param": "p3"},
        {"net_profit_pct": 52.0, "romad": 2.0, "perturbed_param": "p4"},
    ]

    metrics = calculate_retention_metrics(base_metrics, perturbation_results, failure_threshold=0.7)

    assert metrics.status == StressTestStatus.OK
    assert metrics.profit_retention is not None
    assert metrics.romad_retention is None
    assert metrics.romad_failure_rate is None
    assert metrics.romad_failure_count == 0
    assert metrics.combined_failure_rate == metrics.profit_failure_rate
    assert metrics.combined_failure_count == metrics.profit_failure_count


def test_insufficient_data_status():
    base_metrics = {"net_profit_pct": 50.0, "romad": 2.0}
    perturbation_results = [
        {"net_profit_pct": 48.0, "romad": 1.9, "perturbed_param": "p1"},
        {"net_profit_pct": 45.0, "romad": 1.8, "perturbed_param": "p2"},
    ]

    metrics = calculate_retention_metrics(base_metrics, perturbation_results)

    assert metrics.status == StressTestStatus.INSUFFICIENT_DATA
    assert metrics.profit_retention is not None
    assert metrics.total_perturbations == 2


def test_aggregation_with_zero_values():
    base_metrics = {"net_profit_pct": 50.0, "romad": 2.0}
    perturbation_results = [
        {"net_profit_pct": 0.5, "romad": 0.1, "perturbed_param": "p1"},
        {"net_profit_pct": 0.5, "romad": 0.1, "perturbed_param": "p2"},
        {"net_profit_pct": 0.5, "romad": 0.1, "perturbed_param": "p3"},
        {"net_profit_pct": 0.5, "romad": 0.1, "perturbed_param": "p4"},
    ]

    metrics = calculate_retention_metrics(base_metrics, perturbation_results)

    assert metrics.profit_retention is not None
    assert metrics.profit_retention < 0.1


def test_retention_can_exceed_one():
    base_metrics = {"net_profit_pct": 50.0, "romad": 2.0}
    perturbation_results = [
        {"net_profit_pct": 60.0, "romad": 2.5, "perturbed_param": "p1"},
        {"net_profit_pct": 55.0, "romad": 2.2, "perturbed_param": "p2"},
        {"net_profit_pct": 58.0, "romad": 2.3, "perturbed_param": "p3"},
        {"net_profit_pct": 62.0, "romad": 2.6, "perturbed_param": "p4"},
    ]

    metrics = calculate_retention_metrics(base_metrics, perturbation_results)

    assert metrics.profit_worst > 1.0
    assert metrics.profit_median > 1.0
    assert metrics.profit_retention > 1.0


def test_bad_base_profit():
    base_metrics = {"net_profit_pct": -10.0, "romad": -0.5}
    perturbation_results = [
        {"net_profit_pct": 5.0, "romad": 0.2, "perturbed_param": "p1"},
    ]

    metrics = calculate_retention_metrics(base_metrics, perturbation_results)

    assert metrics.status == StressTestStatus.SKIPPED_BAD_BASE
    assert metrics.profit_retention is None
    assert metrics.romad_retention is None
    assert metrics.combined_failure_rate == 1.0


def test_all_perturbations_failed_is_insufficient_data():
    base_metrics = {"net_profit_pct": 50.0, "romad": 2.0}
    perturbation_results = []
    total_generated = 10

    metrics = calculate_retention_metrics(
        base_metrics,
        perturbation_results,
        failure_threshold=0.7,
        total_perturbations_generated=total_generated,
    )

    assert metrics.status == StressTestStatus.INSUFFICIENT_DATA
    assert metrics.status != StressTestStatus.SKIPPED_NO_PARAMS
    assert metrics.total_perturbations == 10
    assert metrics.profit_failure_count == 10
    assert metrics.combined_failure_rate == 1.0


def test_n_valid_zero_uses_n_generated():
    base_metrics = {"net_profit_pct": 50.0, "romad": 2.0}

    perturbation_results = [
        {"net_profit_pct": None, "romad": 1.5, "perturbed_param": f"p{i}"}
        for i in range(15)
    ]
    total_generated = 20

    metrics = calculate_retention_metrics(
        base_metrics,
        perturbation_results,
        failure_threshold=0.7,
        total_perturbations_generated=total_generated,
    )

    assert metrics.status == StressTestStatus.INSUFFICIENT_DATA
    assert metrics.total_perturbations == 20
    assert metrics.profit_failure_count == 20
    assert metrics.combined_failure_count == 20


def test_bounds_checking():
    base_params = {"maLength": 25}
    config_json = {
        "parameters": {
            "maLength": {"type": "int", "optimize": {"enabled": True, "step": 25, "min": 25, "max": 500}}
        }
    }

    perturbations = generate_perturbations(base_params, config_json)

    assert len(perturbations) == 1
    assert perturbations[0]["perturbed_value"] == 50


def test_percentile_not_always_min():
    ratios = [0.1] + [0.9] * 19

    p5 = np.quantile(ratios, 0.05, method="linear")

    assert p5 > 0.1, f"5th percentile ({p5}) should not equal min (0.1)"


def test_stress_test_workflow(monkeypatch):
    import core.post_process as pp

    base_metrics = {
        "net_profit_pct": 50.0,
        "max_drawdown_pct": 10.0,
        "total_trades": 20,
        "win_rate": 55.0,
        "sharpe_ratio": 1.2,
        "romad": 2.0,
        "profit_factor": 1.6,
    }
    perturbation_results = [
        {"net_profit_pct": 48.0, "romad": 1.9, "perturbed_param": "maLength"},
        {"net_profit_pct": 45.0, "romad": 1.8, "perturbed_param": "maLength"},
        {"net_profit_pct": 52.0, "romad": 2.1, "perturbed_param": "stopLongX"},
        {"net_profit_pct": 30.0, "romad": 1.2, "perturbed_param": "stopLongX"},
    ]

    monkeypatch.setattr(pp, "_run_is_backtest", lambda *args, **kwargs: base_metrics)
    monkeypatch.setattr(pp, "run_perturbations_parallel", lambda *args, **kwargs: perturbation_results)

    candidates = [
        MockTrial(trial_number=1, params={"maLength": 50, "stopLongX": 2.0}),
        MockTrial(trial_number=2, params={"maLength": 60, "stopLongX": 2.1}),
    ]
    config_json = {
        "parameters": {
            "maLength": {"type": "int", "optimize": {"enabled": True, "step": 10, "min": 10, "max": 100}},
            "stopLongX": {"type": "float", "optimize": {"enabled": True, "step": 0.1, "min": 0.5, "max": 5}},
        }
    }
    config = StressTestConfig(enabled=True, top_k=2, failure_threshold=0.7)

    results, summary = run_stress_test(
        csv_path="test_data.csv",
        strategy_id="s01_trailing_ma",
        source_results=candidates,
        config=config,
        is_start_date="2025-05-01",
        is_end_date="2025-05-10",
        fixed_params={},
        config_json=config_json,
        n_workers=1,
    )

    assert len(results) == 2
    assert all(r.st_rank is not None for r in results)
    assert summary.get("avg_profit_retention") is not None

    valid_results = [r for r in results if r.profit_retention is not None]
    if len(valid_results) >= 2:
        assert valid_results[0].profit_retention >= valid_results[1].profit_retention


def test_database_save_load():
    study_id = _seed_study_with_trial()
    st_results = [
        StressTestResult(
            trial_number=1,
            source_rank=1,
            status="ok",
            base_net_profit_pct=50.0,
            base_max_drawdown_pct=10.0,
            base_romad=2.0,
            base_sharpe_ratio=1.2,
            profit_retention=0.9,
            romad_retention=0.8,
            profit_worst=0.7,
            profit_lower_tail=0.75,
            profit_median=0.95,
            romad_worst=0.6,
            romad_lower_tail=0.7,
            romad_median=0.9,
            profit_failure_rate=0.1,
            romad_failure_rate=0.2,
            combined_failure_rate=0.15,
            profit_failure_count=1,
            romad_failure_count=2,
            combined_failure_count=2,
            total_perturbations=10,
            failure_threshold=0.7,
            param_worst_ratios={"maLength": 0.7},
            most_sensitive_param="maLength",
            st_rank=1,
            rank_change=0,
        )
    ]
    st_summary = {
        "avg_profit_retention": 0.9,
        "avg_romad_retention": 0.8,
        "avg_combined_failure_rate": 0.15,
        "total_perturbations_run": 10,
        "candidates_skipped_bad_base": 0,
        "candidates_skipped_no_params": 0,
        "candidates_insufficient_data": 0,
    }
    config = StressTestConfig(enabled=True, top_k=1, failure_threshold=0.7, sort_metric="profit_retention")

    save_stress_test_results(study_id, st_results, st_summary, config)

    data = load_study_from_db(study_id)

    assert data["study"]["st_enabled"] == 1
    st_trials = [t for t in data["trials"] if t.get("st_rank") is not None]
    assert len(st_trials) == len(st_results)
    assert st_trials[0]["st_rank"] == 1
    assert "profit_retention" in st_trials[0]
