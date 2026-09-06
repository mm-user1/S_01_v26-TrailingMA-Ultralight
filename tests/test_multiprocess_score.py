"""
Integration test for multi-process composite score optimization.
"""
import io
import json
from pathlib import Path

import pytest


from core import storage  # noqa: E402
from core.optuna_engine import (  # noqa: E402
    DEFAULT_SCORE_CONFIG,
    NSGAIISampler,
    OptimizationConfig,
    OptunaConfig,
    OptunaOptimizer,
    SamplerConfig,
    run_optimization,
    run_optuna_optimization,
)
from strategies import get_strategy_config, list_strategies  # noqa: E402

DATA_PATH = (
    Path(__file__).parent.parent
    / "data"
    / "raw"
    / "OKX_LINKUSDT.P, 15 2025.05.01-2025.11.20.csv"
)


def _minimal_optimization_config(strategy_id, *, mode="optuna"):
    config = OptimizationConfig(
        csv_file=str(DATA_PATH),
        strategy_id=strategy_id,
        enabled_params={},
        param_ranges={},
        param_types={},
        fixed_params={},
        worker_processes=1,
        optimization_mode=mode,
        objectives=["net_profit_pct"],
        primary_objective="net_profit_pct",
    )
    config.optuna_n_trials = 1
    config.optuna_enable_pruning = False
    return config


@pytest.mark.parametrize(
    "strategy_id",
    [
        item["id"]
        for item in list_strategies()
        if str(get_strategy_config(item["id"]).get("engine", "v1")).strip().lower()
        == "v2"
    ],
)
def test_every_registered_v2_strategy_is_blocked_at_direct_optuna_boundary(
    monkeypatch, strategy_id
):
    import core.optuna_engine as optuna_engine

    monkeypatch.setattr(
        optuna_engine,
        "OptunaOptimizer",
        lambda *_args, **_kwargs: pytest.fail("OptunaOptimizer must not be constructed"),
    )
    with pytest.raises(ValueError, match="Backtester V2 is Grid-only"):
        run_optuna_optimization(
            _minimal_optimization_config(strategy_id),
            OptunaConfig(objectives=["net_profit_pct"], n_trials=1),
        )


def test_v2_generic_optuna_dispatch_is_blocked_before_optuna_configuration(monkeypatch):
    import core.optuna_engine as optuna_engine

    monkeypatch.setattr(
        optuna_engine,
        "run_optuna_optimization",
        lambda *_args, **_kwargs: pytest.fail("Optuna dispatch must not start"),
    )
    with pytest.raises(ValueError, match="Optuna is unsupported"):
        run_optimization(_minimal_optimization_config("s06_r_trend_v02_b2"))


def test_v1_public_optuna_dispatch_still_executes_actual_optimizer():
    config = OptimizationConfig(
        csv_file=str(DATA_PATH),
        strategy_id="s01_trailing_ma",
        enabled_params={"maType": True},
        param_ranges={},
        param_types={"maType": "select"},
        fixed_params={
            "maType_options": ["EMA"],
            "closeCountLong": 2,
            "closeCountShort": 2,
        },
        worker_processes=1,
        optimization_mode="optuna",
        objectives=["net_profit_pct"],
        primary_objective="net_profit_pct",
    )
    config.optuna_n_trials = 1
    config.optuna_enable_pruning = False
    config.sampler_type = "random"

    results, study_id = run_optimization(config)

    assert len(results) == 1
    assert study_id


@pytest.mark.slow
class TestMultiProcessScore:
    """Test multi-process optimization with composite scoring enabled."""

    @pytest.fixture
    def base_config(self):
        score_config = DEFAULT_SCORE_CONFIG.copy()
        score_config["enabled_metrics"] = {
            "romad": True,
            "sharpe": True,
            "pf": True,
            "ulcer": True,
            "sqn": True,
            "consistency": True,
        }
        score_config["weights"] = {
            "romad": 0.25,
            "sharpe": 0.20,
            "pf": 0.20,
            "ulcer": 0.15,
            "sqn": 0.10,
            "consistency": 0.10,
        }
        score_config["invert_metrics"] = {"ulcer": True}
        return OptimizationConfig(
            csv_file=str(DATA_PATH),
            strategy_id="s01_trailing_ma",
            enabled_params={"maLength": True},
            param_ranges={"maLength": (10, 50, 10)},
            param_types={"maLength": "int"},
            fixed_params={
                "maType": "EMA",
                "closeCountLong": 2,
                "closeCountShort": 2,
            },
            worker_processes=2,
            score_config=score_config,
        )

    def test_multiprocess_uses_minmax(self, base_config):
        """Multi-process mode should use minmax normalization."""
        optuna_config = OptunaConfig(
            objectives=["net_profit_pct"],
            budget_mode="trials",
            n_trials=5,
        )

        optimizer = OptunaOptimizer(base_config, optuna_config)
        results = optimizer.optimize()

        assert all("maLength" in r.params for r in results)
        assert any(r.score > 0 for r in results)

    def test_single_and_multi_produce_same_scores(self, base_config):
        """Single-process and multi-process should produce same scores for same params."""
        optuna_config = OptunaConfig(
            objectives=["net_profit_pct"],
            budget_mode="trials",
            n_trials=3,
        )

        base_config.worker_processes = 1
        optimizer_single = OptunaOptimizer(base_config, optuna_config)
        results_single = optimizer_single.optimize()

        base_config.worker_processes = 2
        optimizer_multi = OptunaOptimizer(base_config, optuna_config)
        results_multi = optimizer_multi.optimize()

        for r_single in results_single:
            for r_multi in results_multi:
                if r_single.params == r_multi.params:
                    assert r_single.score == pytest.approx(r_multi.score, rel=0.01)

    def test_multiprocess_accepts_in_memory_csv_without_journal_files(self, base_config):
        """Multiprocess mode should keep Optuna runtime state in RAM and avoid journal files."""
        before_entries = {path.name for path in storage.JOURNAL_DIR.iterdir()}
        base_config.csv_file = io.StringIO(DATA_PATH.read_text(encoding="utf-8"))
        optuna_config = OptunaConfig(
            objectives=["net_profit_pct"],
            budget_mode="trials",
            n_trials=4,
        )

        optimizer = OptunaOptimizer(base_config, optuna_config)
        results = optimizer.optimize()

        after_entries = {path.name for path in storage.JOURNAL_DIR.iterdir()}
        assert after_entries == before_entries
        assert isinstance(results, list)

    def test_single_process_ignores_save_study_without_creating_sqlite(self, base_config, monkeypatch, tmp_path):
        """Deprecated raw Optuna persistence should not create optuna_study.db."""
        base_config.worker_processes = 1
        optuna_config = OptunaConfig(
            objectives=["net_profit_pct"],
            budget_mode="trials",
            n_trials=3,
            save_study=True,
        )

        # tmp_path is an absolute pytest temp directory outside the repository;
        # monkeypatch.chdir restores the original CWD even on assertion failure.
        sqlite_path = tmp_path / "optuna_study.db"
        monkeypatch.chdir(tmp_path)
        optimizer = OptunaOptimizer(base_config, optuna_config)
        optimizer.optimize()

        assert not sqlite_path.exists()

    def test_multiprocess_worker_failure_raises_runtime_error(self, base_config):
        """Worker crashes or setup failures must fail the study instead of silently continuing."""
        base_config.csv_file = str(DATA_PATH.with_name("missing_input.csv"))
        optuna_config = OptunaConfig(
            objectives=["net_profit_pct"],
            budget_mode="trials",
            n_trials=2,
        )

        optimizer = OptunaOptimizer(base_config, optuna_config)
        with pytest.raises(RuntimeError, match="Multi-process .* optimisation failed"):
            optimizer.optimize()

    def test_nsga_multiprocess_small_space_stops_without_duplicate_results(self):
        """Centralized NSGA ask/tell should stop at unique combinations instead of repeating them."""
        base_config = OptimizationConfig(
            csv_file=str(DATA_PATH),
            strategy_id="s01_trailing_ma",
            enabled_params={"maType": True},
            param_ranges={},
            param_types={"maType": "select"},
            fixed_params={
                "maType_options": ["EMA", "SMA"],
                "closeCountLong": 2,
                "closeCountShort": 2,
            },
            worker_processes=2,
        )
        optuna_config = OptunaConfig(
            objectives=["net_profit_pct", "max_drawdown_pct"],
            sampler_config=SamplerConfig(sampler_type="nsga2", population_size=4),
            budget_mode="trials",
            n_trials=4,
            coverage_mode=False,
        )

        optimizer = OptunaOptimizer(base_config, optuna_config)
        results = optimizer.optimize()
        signatures = {json.dumps(result.params, sort_keys=True) for result in results}
        summary = getattr(base_config, "optuna_summary", {})

        assert len(results) == 2
        assert len(signatures) == 2
        assert summary["completed_trials"] == 2
        assert summary["total_trials"] == 2
        assert optimizer._duplicate_skipped_count >= 0

    def test_tpe_multiprocess_small_space_stops_without_duplicate_results(self):
        """Centralized TPE ask/tell should stop at unique combinations instead of repeating them."""
        base_config = OptimizationConfig(
            csv_file=str(DATA_PATH),
            strategy_id="s01_trailing_ma",
            enabled_params={"maType": True},
            param_ranges={},
            param_types={"maType": "select"},
            fixed_params={
                "maType_options": ["EMA", "SMA"],
                "closeCountLong": 2,
                "closeCountShort": 2,
            },
            worker_processes=2,
        )
        optuna_config = OptunaConfig(
            objectives=["net_profit_pct"],
            sampler_config=SamplerConfig(sampler_type="tpe", n_startup_trials=0),
            budget_mode="trials",
            n_trials=4,
            coverage_mode=False,
        )

        optimizer = OptunaOptimizer(base_config, optuna_config)
        results = optimizer.optimize()
        signatures = {json.dumps(result.params, sort_keys=True) for result in results}
        summary = getattr(base_config, "optuna_summary", {})

        assert len(results) == 2
        assert len(signatures) == 2
        assert summary["completed_trials"] == 2
        assert summary["total_trials"] == 2
        assert optimizer._duplicate_skipped_count >= 0

    def test_nsga_multiprocess_preserves_coverage_trials_as_generation_zero(self):
        """NSGA coverage mode should keep deterministic startup trials and generation metadata."""
        base_config = OptimizationConfig(
            csv_file=str(DATA_PATH),
            strategy_id="s01_trailing_ma",
            enabled_params={"maType": True},
            param_ranges={},
            param_types={"maType": "select"},
            fixed_params={
                "maType_options": ["EMA", "SMA"],
                "closeCountLong": 2,
                "closeCountShort": 2,
            },
            worker_processes=2,
        )
        optuna_config = OptunaConfig(
            objectives=["net_profit_pct", "max_drawdown_pct"],
            sampler_config=SamplerConfig(sampler_type="nsga2", population_size=4),
            budget_mode="trials",
            n_trials=2,
            warmup_trials=2,
            coverage_mode=True,
        )

        optimizer = OptunaOptimizer(base_config, optuna_config)
        results = optimizer.optimize()
        effective_trials = [
            trial
            for trial in optimizer.study.trials
            if not bool((trial.user_attrs or {}).get("merlin.duplicate_skipped", False))
        ]

        assert {result.params["maType"] for result in results} == {"EMA", "SMA"}
        assert len(effective_trials) == 2
        assert all(trial.system_attrs.get(NSGAIISampler._GENERATION_KEY) == 0 for trial in effective_trials)
