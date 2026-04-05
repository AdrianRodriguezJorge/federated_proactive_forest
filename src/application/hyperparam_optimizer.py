"""Hyperparameter Optimization using Optuna.

Bayesian optimization for Federated Proactive Forest aggregation strategies.
Wraps FLEXOrchestrator to automatically find optimal hyperparameters.

Usage:
    optimizer = HyperparamOptimizer(
        dataset_split=dataset,
        strategy='S6',
        base_config=base_config,
        search_space=search_space
    )
    study = optimizer.optimize(n_trials=50, metric='macro_f1')
    print(study.best_params)
"""
from __future__ import annotations

import optuna
import numpy as np
from typing import Any, Dict, Optional
from dataclasses import dataclass

from src.application.orchestrators import FLEXOrchestrator


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    strategy: str
    metric: str
    n_trials: int
    seed: int
    prune_trials: bool = True


class HyperparamOptimizer:
    """
    Wraps FLEXOrchestrator for Bayesian hyperparameter optimization with Optuna.

    Supports all strategies S1-S7 and PW (Progressive Windows).

    Example:
        >>> optimizer = HyperparamOptimizer(
        ...     dataset_split=dataset,
        ...     strategy='S6',
        ...     base_config=base_config,
        ...     search_space=search_space
        ... )
        >>> study = optimizer.optimize(n_trials=50, metric='macro_f1')
        >>> print(f"Best params: {study.best_params}")
        >>> print(f"Best {metric}: {study.best_value:.4f}")
    """

    def __init__(
        self,
        dataset_split: Any,
        strategy: str,
        base_config: Dict[str, Any],
        search_space: Dict[str, Any],
        verbose: bool = False
    ):
        """
        Initialize hyperparameter optimizer.

        Args:
            dataset_split: DatasetSplit with X_train, y_train, X_test, y_test
            strategy: Strategy name ('S1'-'S7', 'PW')
            base_config: Base configuration dict (will be modified with sampled params)
            search_space: Search space definition for each hyperparameter
            verbose: Enable verbose logging during optimization
        """
        self.dataset_split = dataset_split
        self.strategy = strategy.upper()
        self.base_config = base_config
        self.search_space = search_space
        self.verbose = verbose
        self._trial_count = 0

        # Validate strategy
        valid_strategies = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'PW']
        if self.strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy '{self.strategy}'. Must be one of {valid_strategies}")

    def _sample_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Sample hyperparameters from search space using Optuna trial.

        Args:
            trial: Optuna trial object

        Returns:
            Dict with sampled hyperparameters
        """
        params = {}

        for param_name, param_config in self.search_space.items():
            param_type = param_config.get('type', 'float')

            if param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    low=param_config['low'],
                    high=param_config['high'],
                    step=param_config.get('step', 1)
                )
            elif param_type == 'float':
                if param_config.get('log', False):
                    params[param_name] = trial.suggest_float(
                        param_name,
                        low=param_config['low'],
                        high=param_config['high'],
                        log=True
                    )
                else:
                    params[param_name] = trial.suggest_float(
                        param_name,
                        low=param_config['low'],
                        high=param_config['high'],
                        step=param_config.get('step', None)
                    )
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )

        return params

    def _build_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build complete config dict by merging base config with sampled params.

        Args:
            params: Sampled hyperparameters

        Returns:
            Complete configuration dict for FLEXOrchestrator
        """
        import copy
        config = copy.deepcopy(self.base_config)

        # Apply sampled params to config
        for param_path, value in params.items():
            # Handle special cases where weights must sum to 1.0
            if param_path == 'f1_weight':
                # Set f1_weight and pcd_weight in aggregation
                # pcd_weight is automatically calculated as 1.0 - f1_weight
                # Used by S4, S7, and PW strategies
                config.setdefault('aggregation', {})['f1_weight'] = value
                config['aggregation']['pcd_weight'] = 1.0 - value
            elif param_path == 'local_weight':
                # Set local_weight and global_weight in prediction
                config.setdefault('prediction', {})['local_weight'] = value
                config['prediction']['global_weight'] = 1.0 - value
            elif param_path == 't_max':
                # Set t_max in aggregation
                config.setdefault('aggregation', {})['t_max'] = value
            elif param_path == 'n_clients':
                # Set n_clients in federation
                config.setdefault('federation', {})['n_clients'] = value
            elif param_path == 'n_estimators':
                # Set n_estimators in model
                config.setdefault('model', {})['n_estimators'] = value
            elif param_path == 'alpha_pf':
                # Set alpha in model
                config.setdefault('model', {})['alpha'] = value
            elif param_path == 'window_size':
                # Set window_size in aggregation (for PW)
                config.setdefault('aggregation', {})['window_size'] = value
            elif param_path == 'max_rounds':
                # Set max_rounds in aggregation (for PW)
                config.setdefault('aggregation', {})['max_rounds'] = value
            elif param_path == 'convergence_threshold':
                # Set convergence_threshold in aggregation
                config.setdefault('aggregation', {})['convergence_threshold'] = value
            else:
                # Generic: try to set in aggregation first, then model, then federation
                if 'aggregation' in config:
                    config['aggregation'][param_path] = value
                elif 'model' in config:
                    config['model'][param_path] = value

        return config

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna to optimize.

        Args:
            trial: Optuna trial object

        Returns:
            Metric value to optimize (higher is better)
        """
        self._trial_count += 1

        # Sample hyperparameters
        params = self._sample_params(trial)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🔬 TRIAL {self._trial_count}")
            print(f"{'='*60}")
            print(f"Params: {params}")

        # Build config
        config = self._build_config(params)

        # Ensure strategy is set correctly
        config.setdefault('aggregation', {})['strategy'] = self.strategy

        # Set seed for reproducibility
        seed = self.base_config.get('seed', 42)
        np.random.seed(seed)

        try:
            # Create and run orchestrator
            orchestrator = FLEXOrchestrator.from_config(config)
            orchestrator.setup_federation(self.dataset_split, seed=seed)
            results = orchestrator.run_federated_round()

            # Extract metric to optimize
            if self._get_metric_name() == 'macro_f1':
                metric_value = results.global_macro_f1
            elif self._get_metric_name() == 'accuracy':
                metric_value = results.global_accuracy
            elif self._get_metric_name() == 'hybrid_macro_f1':
                # Use best client hybrid F1
                client_f1_scores = {}
                for cid, preds in results.client_hybrid_predictions.items():
                    from sklearn.metrics import f1_score
                    f1 = f1_score(results.y_test, preds, average='macro', zero_division=0)
                    client_f1_scores[cid] = f1
                metric_value = np.mean(list(client_f1_scores.values())) if client_f1_scores else 0.0
            else:
                metric_value = results.global_macro_f1

            # Report intermediate result for pruning
            trial.report(metric_value, step=0)

            # Check if trial should be pruned
            if trial.should_prune():
                if self.verbose:
                    print(f"✂️  Trial {self._trial_count} PRUNED (metric={metric_value:.4f})")
                raise optuna.TrialPruned()

            if self.verbose:
                print(f"✅ Trial {self._trial_count} COMPLETED: {self._get_metric_name()}={metric_value:.4f}")
                print(f"   Trees: {results.n_trees_global}, "
                      f"Accuracy: {results.global_accuracy:.4f}, "
                      f"Macro-F1: {results.global_macro_f1:.4f}")

            return metric_value

        except Exception as e:
            if self.verbose:
                print(f"❌ Trial {self._trial_count} FAILED: {str(e)}")
            raise optuna.TrialPruned()

    def _get_metric_name(self) -> str:
        """Get metric name for optimization."""
        return self.base_config.get('optimization_metric', 'macro_f1')

    def optimize(
        self,
        n_trials: int = 50,
        metric: str = 'macro_f1',
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        seed: int = 42
    ) -> optuna.Study:
        """
        Run Bayesian hyperparameter optimization.

        Args:
            n_trials: Number of trials to run
            metric: Metric to optimize ('macro_f1', 'accuracy', 'hybrid_macro_f1')
            sampler: Optuna sampler (default: TPESampler)
            pruner: Optuna pruner (default: MedianPruner)
            seed: Random seed for reproducibility

        Returns:
            Optuna Study object with optimization results
        """
        # Update metric in config
        self.base_config['optimization_metric'] = metric

        # Create sampler
        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)

        # Create pruner
        if pruner is None:
            pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            study_name=f"{self.strategy}_{metric}_{n_trials}trials"
        )

        # Run optimization
        print(f"\n{'='*60}")
        print(f"🚀 INICIANDO OPTIMIZACIÓN BAYESIANA")
        print(f"{'='*60}")
        print(f"   Estrategia: {self.strategy}")
        print(f"   Métrica: {metric}")
        print(f"   Trials: {n_trials}")
        print(f"   Sampler: {sampler.__class__.__name__}")
        print(f"   Pruner: {pruner.__class__.__name__}")
        print(f"{'='*60}\n")

        study.optimize(self._objective, n_trials=n_trials, show_progress_bar=True)

        # Print results
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
        """Get default hyperparameters from base config."""
        defaults = {}

        agg = self.base_config.get('aggregation', {})
        defaults['f1_weight'] = agg.get('f1_weight', 0.5)
        defaults['pcd_weight'] = agg.get('pcd_weight', 0.5)
        defaults['t_max'] = agg.get('t_max', 100)

        pred = self.base_config.get('prediction', {})
        defaults['local_weight'] = pred.get('local_weight', 0.4)
        defaults['global_weight'] = pred.get('global_weight', 0.6)

        fed = self.base_config.get('federation', {})
        defaults['n_clients'] = fed.get('n_clients', 5)

        model = self.base_config.get('model', {})
        defaults['n_estimators'] = model.get('n_estimators', 100)
        defaults['alpha_pf'] = model.get('alpha', 0.1)

        return defaults
