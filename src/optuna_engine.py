"""Optuna-based Bayesian optimization engine for S_01 TrailingMA."""
from __future__ import annotations

import logging
import multiprocessing as mp
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional

import optuna
from optuna.pruners import MedianPruner, PercentilePruner, PatientPruner
from optuna.samplers import RandomSampler, TPESampler
from optuna.trial import TrialState
import pandas as pd

from backtest_engine import load_data
from optimizer_engine import (
    DEFAULT_SCORE_CONFIG,
    OptimizationResult,
    _init_worker,
    _prepare_dataset_for_optimization,
    _simulate_combination,
    generate_parameter_grid,
    calculate_score,
)

logger = logging.getLogger(__name__)


@dataclass
class OptunaConfig:
    """Configuration parameters that control Optuna optimisation."""

    target: str = "score"
    budget_mode: str = "trials"  # "trials", "time", or "convergence"
    n_trials: int = 500
    time_limit: int = 3600  # seconds
    convergence_patience: int = 50
    enable_pruning: bool = True
    sampler: str = "tpe"  # "tpe" or "random"
    pruner: str = "median"  # "median", "percentile", "patient", "none"
    warmup_trials: int = 20
    save_study: bool = False
    study_name: Optional[str] = None


class OptunaOptimizer:
    """Optuna-based optimiser that reuses the grid search simulation engine."""

    def __init__(
        self,
        base_config,
        optuna_config: OptunaConfig,
        strategy_class,
        param_definitions: Dict[str, Dict[str, Any]],
    ) -> None:
        self.base_config = base_config
        self.optuna_config = optuna_config
        self.strategy_class = strategy_class
        self.param_definitions = param_definitions
        self.pool: Optional[mp.pool.Pool] = None
        self.trial_results: List[OptimizationResult] = []
        self.best_value: float = float("-inf")
        self.trials_without_improvement: int = 0
        self.start_time: Optional[float] = None
        self.pruned_trials: int = 0
        self.study: Optional[optuna.Study] = None
        self.pruner: Optional[optuna.pruners.BasePruner] = None

    # ------------------------------------------------------------------
    # Search space handling
    # ------------------------------------------------------------------
    def _build_search_space(self) -> Dict[str, Dict[str, Any]]:
        """Construct the Optuna search space from the optimiser configuration."""

        space: Dict[str, Dict[str, Any]] = {}

        for param_name, param_def in self.param_definitions.items():
            if not self.base_config.enabled_params.get(param_name, False):
                continue

            param_type = param_def.get("type")
            if param_type == "bool":
                space[param_name] = {"type": "categorical", "choices": [True, False]}
                continue

            if param_name not in self.base_config.param_ranges:
                raise ValueError(f"Missing range for parameter '{param_name}'.")

            start, stop, step = self.base_config.param_ranges[param_name]
            low = min(float(start), float(stop))
            high = max(float(start), float(stop))
            step_value = abs(float(step)) if step else 0.0

            if param_type == "int":
                spec: Dict[str, Any] = {
                    "type": "int",
                    "low": int(round(low)),
                    "high": int(round(high)),
                    "step": max(1, int(round(step_value))) if step_value else 1,
                }
            elif param_type == "float":
                spec = {
                    "type": "float",
                    "low": float(low),
                    "high": float(high),
                }
                if step_value:
                    spec["step"] = float(step_value)
            elif param_type == "categorical":
                if isinstance(self.base_config.param_ranges[param_name], (list, tuple)):
                    choices = list(self.base_config.param_ranges[param_name])
                else:
                    choices = param_def.get("choices", [param_def.get("default")])
                spec = {"type": "categorical", "choices": choices}
            else:
                raise ValueError(f"Unsupported parameter type: {param_type}")

            space[param_name] = spec

        return space

    # ------------------------------------------------------------------
    # Sampler / pruner factories
    # ------------------------------------------------------------------
    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        if self.optuna_config.sampler == "random":
            return RandomSampler()
        return TPESampler(
            n_startup_trials=max(0, int(self.optuna_config.warmup_trials)),
            multivariate=True,
            constant_liar=True,
        )

    def _create_pruner(self) -> Optional[optuna.pruners.BasePruner]:
        if not self.optuna_config.enable_pruning or self.optuna_config.pruner == "none":
            return None
        if self.optuna_config.pruner == "percentile":
            return PercentilePruner(
                percentile=25.0,
                n_startup_trials=max(0, int(self.optuna_config.warmup_trials)),
            )
        if self.optuna_config.pruner == "patient":
            return PatientPruner(
                wrapped_pruner=MedianPruner(
                    n_startup_trials=max(0, int(self.optuna_config.warmup_trials))
                ),
                patience=3,
            )
        return MedianPruner(
            n_startup_trials=max(0, int(self.optuna_config.warmup_trials))
        )

    # ------------------------------------------------------------------
    # Worker pool initialisation
    # ------------------------------------------------------------------
    def _setup_worker_pool(self, df: pd.DataFrame) -> None:
        for name, definition in self.param_definitions.items():
            if not self.base_config.enabled_params.get(name, False) and name not in self.base_config.fixed_params:
                self.base_config.fixed_params[name] = definition.get("default")

        combinations = generate_parameter_grid(self.base_config, self.param_definitions)
        cache_requirements = self.strategy_class.get_cache_requirements(combinations)
        df_prepared, trade_start_idx = _prepare_dataset_for_optimization(
            df, self.base_config, cache_requirements
        )

        worker_config = replace(self.base_config, csv_file=None)

        pool_args = (
            df_prepared,
            cache_requirements,
            self.strategy_class.STRATEGY_ID,
            worker_config,
            trade_start_idx,
        )

        processes = min(32, max(1, int(self.base_config.worker_processes)))
        self.pool = mp.Pool(processes=processes, initializer=_init_worker, initargs=pool_args)

    # ------------------------------------------------------------------
    # Objective evaluation
    # ------------------------------------------------------------------
    def _evaluate_parameters(self, params_dict: Dict[str, Any]) -> OptimizationResult:
        if self.pool is None:
            raise RuntimeError("Worker pool is not initialised.")
        return self.pool.apply(_simulate_combination, (params_dict,))

    def _prepare_trial_parameters(self, trial: optuna.Trial, search_space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        params_dict: Dict[str, Any] = {}

        for key, spec in search_space.items():
            p_type = spec["type"]
            if p_type == "int":
                params_dict[key] = trial.suggest_int(
                    key,
                    int(spec["low"]),
                    int(spec["high"]),
                    step=int(spec.get("step", 1)),
                )
            elif p_type == "float":
                if spec.get("log"):
                    params_dict[key] = trial.suggest_float(
                        key,
                        float(spec["low"]),
                        float(spec["high"]),
                        log=True,
                    )
                else:
                    step = spec.get("step")
                    if step:
                        params_dict[key] = trial.suggest_float(
                            key,
                            float(spec["low"]),
                            float(spec["high"]),
                            step=float(step),
                        )
                    else:
                        params_dict[key] = trial.suggest_float(
                            key,
                            float(spec["low"]),
                            float(spec["high"]),
                        )
            elif p_type == "categorical":
                params_dict[key] = trial.suggest_categorical(key, list(spec["choices"]))

        for name, definition in self.param_definitions.items():
            if self.base_config.enabled_params.get(name, False):
                continue
            value = self.base_config.fixed_params.get(name, definition.get("default"))
            params_dict[name] = value

        if self.base_config.lock_trail_types:
            trail_type = params_dict.get("trailMaLongType") or params_dict.get("trail_ma_long_type")
            if trail_type is not None:
                params_dict["trailMaShortType"] = trail_type
                params_dict["trail_ma_short_type"] = trail_type

        return params_dict

    def _objective(self, trial: optuna.Trial, search_space: Dict[str, Dict[str, Any]]) -> float:
        params_dict = self._prepare_trial_parameters(trial, search_space)

        result = self._evaluate_parameters(params_dict)

        if self.base_config.filter_min_profit and (
            result.net_profit_pct < float(self.base_config.min_profit_threshold)
        ):
            self.pruned_trials += 1
            raise optuna.TrialPruned("Below minimum profit threshold")

        # For composite score target, calculate score dynamically based on all results so far
        score_config = self.base_config.score_config or DEFAULT_SCORE_CONFIG
        objective_value: float

        if self.optuna_config.target == "score":
            # Add current result to accumulated results
            temp_results = self.trial_results + [result]
            # Calculate scores for all results using percentile ranking
            scored_results = calculate_score(temp_results, score_config)
            # Get the score of the current (last) result
            if scored_results:
                result = scored_results[-1]
                objective_value = float(result.score)
            else:
                objective_value = 0.0
        elif self.optuna_config.target == "net_profit":
            objective_value = float(result.net_profit_pct)
        elif self.optuna_config.target == "romad":
            objective_value = float(result.romad or 0.0)
        elif self.optuna_config.target == "sharpe":
            objective_value = float(result.sharpe_ratio or 0.0)
        elif self.optuna_config.target == "max_drawdown":
            objective_value = -float(result.max_drawdown_pct)
        else:
            objective_value = float(result.romad or 0.0)

        # Check score threshold filter (only applies when score is calculated)
        if self.optuna_config.target == "score" and score_config.get("filter_enabled"):
            min_score = float(score_config.get("min_score_threshold", 0.0))
            if result.score < min_score:
                self.pruned_trials += 1
                raise optuna.TrialPruned("Below minimum score threshold")

        if self.pruner is not None:
            trial.report(objective_value, step=0)
            if trial.should_prune():
                self.pruned_trials += 1
                raise optuna.TrialPruned("Pruned by Optuna")

        self.trial_results.append(result)
        setattr(result, "optuna_trial_number", trial.number)
        setattr(result, "optuna_value", objective_value)

        if objective_value > self.best_value:
            self.best_value = objective_value
            self.trials_without_improvement = 0
        else:
            self.trials_without_improvement += 1

        return objective_value

    # ------------------------------------------------------------------
    # Main execution entrypoint
    # ------------------------------------------------------------------
    def optimize(self) -> List[OptimizationResult]:
        logger.info(
            "Starting Optuna optimisation: target=%s, budget_mode=%s",
            self.optuna_config.target,
            self.optuna_config.budget_mode,
        )

        self.start_time = time.time()
        self.trial_results = []
        self.best_value = float("-inf")
        self.trials_without_improvement = 0
        self.pruned_trials = 0

        df = load_data(self.base_config.csv_file)
        if hasattr(self.base_config.csv_file, "close") and not getattr(
            self.base_config.csv_file, "closed", True
        ):
            try:
                self.base_config.csv_file.close()
            except Exception:
                logger.debug("Failed to close csv_file handle after load_data")
        search_space = self._build_search_space()
        self._setup_worker_pool(df)

        sampler = self._create_sampler()
        self.pruner = self._create_pruner()

        storage = None
        if self.optuna_config.save_study:
            storage = optuna.storages.RDBStorage(
                url="sqlite:///optuna_study.db",
                engine_kwargs={"connect_args": {"timeout": 30}},
            )

        study_name = self.optuna_config.study_name or f"strategy_opt_{int(time.time())}"

        self.study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler,
            pruner=self.pruner,
            storage=storage,
            load_if_exists=self.optuna_config.save_study,
        )

        timeout = None
        n_trials = None
        callbacks = []

        if self.optuna_config.budget_mode == "time":
            timeout = max(60, int(self.optuna_config.time_limit))
        elif self.optuna_config.budget_mode == "trials":
            n_trials = max(1, int(self.optuna_config.n_trials))
        elif self.optuna_config.budget_mode == "convergence":
            n_trials = 10000

            def convergence_callback(study: optuna.Study, _trial: optuna.Trial) -> None:
                if self.trials_without_improvement >= int(self.optuna_config.convergence_patience):
                    study.stop()
                    logger.info(
                        "Stopping optimisation due to convergence threshold (patience=%s)",
                        self.optuna_config.convergence_patience,
                    )

            callbacks.append(convergence_callback)

        try:
            self.study.optimize(
                lambda trial: self._objective(trial, search_space),
                n_trials=n_trials,
                timeout=timeout,
                callbacks=callbacks or None,
                show_progress_bar=False,
            )
        except KeyboardInterrupt:
            logger.info("Optuna optimisation interrupted by user")
        finally:
            if self.pool is not None:
                self.pool.close()
                self.pool.join()
                self.pool = None
            self.pruner = None

        end_time = time.time()
        optimisation_time = end_time - (self.start_time or end_time)

        logger.info(
            "Optuna optimisation completed: trials=%s, best_value=%s, time=%.1fs",
            len(self.study.trials) if self.study else 0,
            getattr(self.study, "best_value", float("nan")),
            optimisation_time,
        )

        # Calculate scores for all results using percentile ranking
        score_config = self.base_config.score_config or DEFAULT_SCORE_CONFIG
        self.trial_results = calculate_score(self.trial_results, score_config)

        if self.study:
            completed_trials = sum(1 for trial in self.study.trials if trial.state == TrialState.COMPLETE)
            pruned_trials = sum(1 for trial in self.study.trials if trial.state == TrialState.PRUNED)
            best_trial_number = self.study.best_trial.number if self.study.best_trial else None
            best_value = self.study.best_value if completed_trials else None
        else:
            completed_trials = len(self.trial_results)
            pruned_trials = self.pruned_trials
            best_trial_number = None
            best_value = None

        summary = {
            "method": "Optuna",
            "target": self.optuna_config.target,
            "budget_mode": self.optuna_config.budget_mode,
            "total_trials": len(self.study.trials) if self.study else len(self.trial_results),
            "completed_trials": completed_trials,
            "pruned_trials": pruned_trials,
            "best_trial_number": best_trial_number,
            "best_value": best_value,
            "optimization_time_seconds": optimisation_time,
        }
        setattr(self.base_config, "optuna_summary", summary)

        if self.optuna_config.target == "max_drawdown":
            self.trial_results.sort(key=lambda item: float(item.max_drawdown_pct))
        elif self.optuna_config.target == "score":
            self.trial_results.sort(key=lambda item: float(item.score), reverse=True)
        elif self.optuna_config.target == "net_profit":
            self.trial_results.sort(key=lambda item: float(item.net_profit_pct), reverse=True)
        elif self.optuna_config.target == "romad":
            self.trial_results.sort(key=lambda item: float(item.romad or float("-inf")), reverse=True)
        elif self.optuna_config.target == "sharpe":
            self.trial_results.sort(
                key=lambda item: float(item.sharpe_ratio or float("-inf")), reverse=True
            )
        else:
            self.trial_results.sort(key=lambda item: float(item.score), reverse=True)

        return self.trial_results


def run_optuna_optimization(
    base_config, optuna_config: OptunaConfig, strategy_class
) -> List[OptimizationResult]:
    """Execute Optuna optimisation using the provided configuration."""

    param_definitions = strategy_class.get_param_definitions()
    optimizer = OptunaOptimizer(base_config, optuna_config, strategy_class, param_definitions)
    return optimizer.optimize()
