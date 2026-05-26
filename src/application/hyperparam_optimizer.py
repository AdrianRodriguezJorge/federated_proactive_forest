"""Hyperparameter Optimization using Optuna.

Bayesian optimization for Federated Proactive Forest aggregation strategies.
Wraps FLEXOrchestrator and RouletteOrchestrator to automatically find optimal
hyperparameters.
"""

from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import optuna
import numpy as np

from src.application.orchestrators import FLEXOrchestrator
from src.application.orchestrators.roulette_orchestrator import (
    RouletteOrchestrator,
)


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""

    strategy: str
    metric: str
    n_trials: int
    seed: int
    prune_trials: bool = True


class HyperparamOptimizer:
    """Wraps Orchestrators for Bayesian hyperparameter optimization.

    Supports all strategies S1-S7 and S8 (Roulette).
    """

    def __init__(
        self,
        dataset_split: Any,
        strategy: str,
        base_config: Dict[str, Any],
        search_space: Dict[str, Any],
        verbose: bool = False,
    ):
        """Initialize hyperparameter optimizer.

        Args:
            dataset_split (Any): Target dataset adapter split.
            strategy (str): Strategy name ('S1'-'S7', 'S8').
            base_config (Dict[str, Any]): Base configuration dict.
            search_space (Dict[str, Any]): Search space definition.
            verbose (bool): Enable verbose logging during optimization.

        Raises:
            ValueError: If strategy is invalid.
        """
        self.dataset_split = dataset_split
        self.strategy = strategy.upper()
        self.base_config = base_config
        self.search_space = search_space
        self.verbose = verbose
        self._trial_count = 0

        # Validate strategy
        valid_strategies = [
            "S1",
            "S2",
            "S3",
            "S4",
            "S5",
            "S6",
            "S7",
            "S8",
        ]
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"Invalid strategy '{self.strategy}'. Must be one of "
                f"{valid_strategies}"
            )

    def _sample_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample hyperparameters from search space using Optuna trial.

        Args:
            trial (optuna.Trial): Optuna trial object.

        Returns:
            Dict[str, Any]: Sampled hyperparameters.
        """
        params = {}

        for param_name, param_config in self.search_space.items():
            param_type = param_config.get("type", "float")

            if param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    low=param_config["low"],
                    high=param_config["high"],
                    step=param_config.get("step", 1),
                )
            elif param_type == "float":
                if param_config.get("log", False):
                    params[param_name] = trial.suggest_float(
                        param_name,
                        low=param_config["low"],
                        high=param_config["high"],
                        log=True,
                    )
                else:
                    params[param_name] = trial.suggest_float(
                        param_name,
                        low=param_config["low"],
                        high=param_config["high"],
                        step=param_config.get("step", None),
                    )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config["choices"]
                )

        return params

    def _build_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Build complete config dict by merging base config with sampled params.

        Args:
            params (Dict[str, Any]): Sampled hyperparameters.

        Returns:
            Dict[str, Any]: Complete configuration dict for Orchestrators.
        """
        config = copy.deepcopy(self.base_config)

        for param_path, value in params.items():
            if param_path == "f1_weight":
                config.setdefault("aggregation", {})["f1_weight"] = value
                config["aggregation"]["pcd_weight"] = 1.0 - value
            elif param_path == "local_weight":
                config.setdefault("prediction", {})["local_weight"] = value
                config["prediction"]["global_weight"] = 1.0 - value
            elif param_path == "use_weighted":
                config.setdefault("prediction", {})["use_weighted"] = value
            elif param_path == "t_max":
                config.setdefault("aggregation", {})["t_max"] = value
            elif param_path == "n_clients":
                config.setdefault("federation", {})["n_clients"] = value
            elif param_path == "n_estimators":
                config.setdefault("model", {})["n_estimators"] = value
            elif param_path == "alpha_pf":
                config.setdefault("model", {})["alpha"] = value
            elif param_path == "window_size":
                config.setdefault("aggregation", {})["window_size"] = value
            elif param_path == "max_rounds":
                config.setdefault("aggregation", {})["max_rounds"] = value
            elif param_path == "convergence_threshold":
                config.setdefault("aggregation", {})[
                    "convergence_threshold"
                ] = value
            elif param_path == "local_convergence":
                config.setdefault("model", {})[
                    "convergence_threshold"
                ] = value
            elif param_path == "local_roulette_weight":
                config.setdefault("aggregation", {})["local_roulette_weight"] = value
            elif param_path == "variant":
                config.setdefault("aggregation", {})["variant"] = value
            elif param_path == "dirichlet_alpha":
                config.setdefault("federation", {})["dirichlet_alpha"] = value
            elif param_path == "distribution":
                config.setdefault("federation", {})["distribution"] = value
            else:
                if "aggregation" in config:
                    config["aggregation"][param_path] = value
                elif "model" in config:
                    config["model"][param_path] = value

        return config

    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna to optimize.

        Args:
            trial (optuna.Trial): Optuna trial object.

        Returns:
            float: Metric value to optimize (higher is better).

        Raises:
            optuna.TrialPruned: If trial is pruned by MedianPruner.
        """
        self._trial_count += 1
        params = self._sample_params(trial)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🔬 TRIAL {self._trial_count}")
            print(f"{'='*60}")
            print(f"Params: {params}")

        config = self._build_config(params)
        config.setdefault("aggregation", {})["strategy"] = self.strategy

        seed = self.base_config.get("seed", 42)
        np.random.seed(seed)

        try:
            if self.strategy == "S8":
                orchestrator = RouletteOrchestrator(config)
            else:
                orchestrator = FLEXOrchestrator.from_config(config)

            orchestrator.setup_federation(self.dataset_split, seed=seed)
            results = orchestrator.run_federated_round()

            metric_name = self._get_metric_name()

            if metric_name == "macro_f1":
                metric_value = results.global_macro_f1
            elif metric_name == "accuracy":
                metric_value = results.global_accuracy
            elif metric_name == "hybrid_macro_f1":
                f1_scores = list(results.client_f1_scores.values())
                metric_value = np.mean(f1_scores) if f1_scores else 0.0
            else:
                metric_value = results.global_macro_f1

            trial.report(metric_value, step=0)

            if trial.should_prune():
                if self.verbose:
                    print(
                        f"✂️  Trial {self._trial_count} PRUNED "
                        f"(metric={metric_value:.4f})"
                    )
                raise optuna.TrialPruned()

            if self.verbose:
                print(
                    f"✅ Trial {self._trial_count} COMPLETED: "
                    f"{metric_name}={metric_value:.4f}"
                )
                print(
                    f"   Trees: {results.n_trees_global}, "
                    f"Accuracy: {results.global_accuracy:.4f}, "
                    f"Macro-F1: {results.global_macro_f1:.4f}"
                )

            return metric_value

        except optuna.TrialPruned:
            raise
        except Exception as e:
            if self.verbose:
                print(f"❌ Trial {self._trial_count} FAILED: {str(e)}")
                import traceback

                traceback.print_exc()
            raise optuna.TrialPruned()

    def _get_metric_name(self) -> str:
        """Get metric name for optimization."""
        return self.base_config.get("optimization_metric", "macro_f1")

    def optimize(
        self,
        n_trials: int = 50,
        metric: str = "macro_f1",
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        seed: int = 42,
    ) -> optuna.Study:
        """Run Bayesian hyperparameter optimization.

        Args:
            n_trials (int): Number of trials to run.
            metric (str): Metric to optimize.
            sampler (Optional[optuna.samplers.BaseSampler]): Optuna sampler.
            pruner (Optional[optuna.pruners.BasePruner]): Optuna pruner.
            seed (int): Stable random seed.

        Returns:
            optuna.Study: Completed Optuna study object.
        """
        self.base_config["optimization_metric"] = metric

        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)

        if pruner is None:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=0
            )

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=f"{self.strategy}_{metric}_{n_trials}trials",
        )

        print(f"\n{'='*60}")
        print(f"🚀 INICIANDO OPTIMIZACIÓN BAYESIANA")
        print(f"{'='*60}")
        print(f"   Estrategia: {self.strategy}")
        print(f"   Métrica: {metric}")
        print(f"   Trials: {n_trials}")
        print(f"   Sampler: {sampler.__class__.__name__}")
        print(f"   Pruner: {pruner.__class__.__name__}")
        print(f"{'='*60}\n")

        study.optimize(
            self._objective, n_trials=n_trials, show_progress_bar=True
        )

        print(f"\n{'='*60}")
        print(f"✅ OPTIMIZACIÓN COMPLETADA")
        print(f"{'='*60}")
        print(f"   Mejor trial: {study.best_trial.number}")
        print(f"   Mejor {metric}: {study.best_value:.4f}")
        print(f"\n   Mejores hiperparámetros:")
        for param, value in study.best_params.items():
            print(f"      {param}: {value}")
        print(f"{'='*60}\n")

        return study

    def get_default_params(self) -> Dict[str, Any]:
        """Get default hyperparameters from base config.

        Returns:
            Dict[str, Any]: Extracted default hyperparameters dict.
        """
        defaults = {}

        agg = self.base_config.get("aggregation", {})
        defaults["f1_weight"] = agg.get("f1_weight", 0.5)
        defaults["pcd_weight"] = agg.get("pcd_weight", 0.5)
        defaults["t_max"] = agg.get("t_max", 100)
        defaults["window_size"] = agg.get("window_size", 5)
        defaults["max_rounds"] = agg.get("max_rounds", 20)
        defaults["local_roulette_weight"] = agg.get("local_roulette_weight", 0.1)

        pred = self.base_config.get("prediction", {})
        defaults["local_weight"] = pred.get("local_weight", 0.4)
        defaults["global_weight"] = pred.get("global_weight", 0.6)
        defaults["use_weighted"] = pred.get("use_weighted", True)

        fed = self.base_config.get("federation", {})
        defaults["n_clients"] = fed.get("n_clients", 5)

        model = self.base_config.get("model", {})
        defaults["n_estimators"] = model.get("n_estimators", 100)
        defaults["alpha_pf"] = model.get("alpha", 0.1)
        defaults["local_convergence"] = model.get(
            "convergence_threshold", 0.002
        )

        return defaults
