"""Optuna-based Bayesian optimization engine for S_01 TrailingMA."""
from __future__ import annotations

import logging
import multiprocessing as mp
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import optuna
from optuna.pruners import MedianPruner, PercentilePruner, PatientPruner
from optuna.samplers import RandomSampler, TPESampler
from optuna.trial import TrialState
import pandas as pd

from backtest_engine import load_data
from optimizer_engine import (
    DEFAULT_SCORE_CONFIG,
    OptimizationResult,
    PARAMETER_MAP,
    _generate_numeric_sequence,
    _init_worker,
    _parse_timestamp,
    _simulate_combination,
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

    def __init__(self, base_config, optuna_config: OptunaConfig) -> None:
        self.base_config = base_config
        self.optuna_config = optuna_config
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

        for frontend_name, (internal_name, is_int) in PARAMETER_MAP.items():
            if not self.base_config.enabled_params.get(frontend_name):
                continue

            if frontend_name not in self.base_config.param_ranges:
                raise ValueError(f"Missing range for parameter '{frontend_name}'.")

            start, stop, step = self.base_config.param_ranges[frontend_name]
            low = min(float(start), float(stop))
            high = max(float(start), float(stop))
            step_value = abs(float(step)) if step else 0.0

            if is_int:
                if low == high:
                    low = high = round(low)
                spec: Dict[str, Any] = {
                    "type": "int",
                    "low": int(round(low)),
                    "high": int(round(high)),
                }
                int_step = max(1, int(round(step_value))) if step_value else 1
                spec["step"] = int_step
            else:
                spec = {
                    "type": "float",
                    "low": float(low),
                    "high": float(high),
                }
                if step_value:
                    spec["step"] = float(step_value)
                if low > 0 and high / max(low, 1e-9) > 100:
                    spec["log"] = True

            space[internal_name] = spec

        trend_types = [ma.upper() for ma in self.base_config.ma_types_trend]
        trail_long_types = [ma.upper() for ma in self.base_config.ma_types_trail_long]
        trail_short_types = [ma.upper() for ma in self.base_config.ma_types_trail_short]

        if not trend_types or not trail_long_types or not trail_short_types:
            raise ValueError("At least one MA type must be selected in each group.")

        space["ma_type"] = {"type": "categorical", "choices": trend_types}

        if self.base_config.lock_trail_types:
            short_set = {ma.upper() for ma in trail_short_types}
            paired = [ma for ma in trail_long_types if ma in short_set]
            if not paired:
                raise ValueError(
                    "No overlapping trail MA types available when lock_trail_types is enabled."
                )
            space["trail_ma_long_type"] = {"type": "categorical", "choices": paired}
        else:
            space["trail_ma_long_type"] = {
                "type": "categorical",
                "choices": trail_long_types,
            }
            space["trail_ma_short_type"] = {
                "type": "categorical",
                "choices": trail_short_types,
            }

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
    def _collect_lengths(self, frontend_key: str) -> Iterable[int]:
        if self.base_config.enabled_params.get(frontend_key):
            start, stop, step = self.base_config.param_ranges[frontend_key]
            sequence = _generate_numeric_sequence(start, stop, step, True)
        else:
            value = self.base_config.fixed_params.get(frontend_key, 0)
            sequence = [value]
        return [int(round(val)) for val in sequence]

    def _setup_worker_pool(self, df: pd.DataFrame) -> None:
        from backtest_engine import prepare_dataset_with_warmup, StrategyParams

        ma_specs: Set[Tuple[str, int]] = set()

        trend_lengths = self._collect_lengths("maLength")
        trail_long_lengths = self._collect_lengths("trailLongLength")
        trail_short_lengths = self._collect_lengths("trailShortLength")

        if self.base_config.lock_trail_types:
            short_set = {ma.upper() for ma in self.base_config.ma_types_trail_short}
            trail_long_types = [ma for ma in self.base_config.ma_types_trail_long if ma in short_set]
            trail_short_types = trail_long_types
        else:
            trail_long_types = [ma.upper() for ma in self.base_config.ma_types_trail_long]
            trail_short_types = [ma.upper() for ma in self.base_config.ma_types_trail_short]

        for ma_type in [ma.upper() for ma in self.base_config.ma_types_trend]:
            for length in trend_lengths:
                ma_specs.add((ma_type, max(1, int(length))))

        for ma_type in trail_long_types:
            for length in trail_long_lengths:
                ma_specs.add((ma_type, max(0, int(length))))

        for ma_type in trail_short_types:
            for length in trail_short_lengths:
                ma_specs.add((ma_type, max(0, int(length))))

        if self.base_config.enabled_params.get("stopLongLP"):
            long_lp_values = {
                max(1, int(val))
                for val in _generate_numeric_sequence(
                    *self.base_config.param_ranges["stopLongLP"], True
                )
            }
        else:
            long_lp_values = {max(1, int(self.base_config.fixed_params.get("stopLongLP", 1)))}

        if self.base_config.enabled_params.get("stopShortLP"):
            short_lp_values = {
                max(1, int(val))
                for val in _generate_numeric_sequence(
                    *self.base_config.param_ranges["stopShortLP"], True
                )
            }
        else:
            short_lp_values = {max(1, int(self.base_config.fixed_params.get("stopShortLP", 1)))}

        use_date_filter = bool(self.base_config.fixed_params.get("dateFilter", False))
        start = _parse_timestamp(self.base_config.fixed_params.get("start"))
        end = _parse_timestamp(self.base_config.fixed_params.get("end"))

        # Prepare dataset with warmup if date filtering is enabled
        trade_start_idx = 0
        if use_date_filter and (start is not None or end is not None):
            # Find the maximum MA length from all possible values
            max_ma_length = max(
                max(trend_lengths, default=1),
                max(trail_long_lengths, default=0),
                max(trail_short_lengths, default=0)
            )

            # Create a dummy StrategyParams with max MA lengths for warmup calculation
            dummy_params = StrategyParams(
                use_backtester=True,
                use_date_filter=use_date_filter,
                start=start,
                end=end,
                ma_type="SMA",
                ma_length=max_ma_length,
                trail_ma_long_type="SMA",
                trail_ma_long_length=max_ma_length,
                trail_ma_short_type="SMA",
                trail_ma_short_length=max_ma_length,
                close_count_long=1,
                close_count_short=1,
                stop_long_atr=1.0,
                stop_long_rr=1.0,
                stop_long_lp=1,
                stop_short_atr=1.0,
                stop_short_rr=1.0,
                stop_short_lp=1,
                stop_long_max_pct=0.0,
                stop_short_max_pct=0.0,
                stop_long_max_days=0,
                stop_short_max_days=0,
                trail_rr_long=1.0,
                trail_rr_short=1.0,
                trail_ma_long_offset=0.0,
                trail_ma_short_offset=0.0,
                risk_per_trade_pct=self.base_config.risk_per_trade_pct,
                contract_size=self.base_config.contract_size,
                commission_rate=self.base_config.commission_rate,
                atr_period=self.base_config.atr_period
            )

            df, trade_start_idx = prepare_dataset_with_warmup(df, start, end, dummy_params)

        pool_args = (
            df,
            float(self.base_config.risk_per_trade_pct),
            float(self.base_config.contract_size),
            float(self.base_config.commission_rate),
            int(self.base_config.atr_period),
            list(ma_specs),
            list(long_lp_values),
            list(short_lp_values),
            use_date_filter,
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

        if self.base_config.lock_trail_types:
            trail_type = params_dict.get("trail_ma_long_type")
            if trail_type is not None:
                params_dict["trail_ma_short_type"] = trail_type

        for frontend_name, (internal_name, is_int) in PARAMETER_MAP.items():
            if self.base_config.enabled_params.get(frontend_name):
                continue
            value = self.base_config.fixed_params.get(frontend_name)
            if value is None:
                continue
            params_dict[internal_name] = int(round(float(value))) if is_int else float(value)

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


def run_optuna_optimization(base_config, optuna_config: OptunaConfig) -> List[OptimizationResult]:
    """Execute Optuna optimisation using the provided configuration."""

    optimizer = OptunaOptimizer(base_config, optuna_config)
    return optimizer.optimize()
