"""Enhanced FLEXOrchestrator with full FLEX integration for Proactive Forest.

This module provides complete federated learning orchestration using FLEX Framework,
supporting both IID and Non-IID (Dirichlet) data distributions.
"""

from __future__ import annotations
import numpy as np
import warnings
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING, Union
from sklearn.model_selection import train_test_split

# Core FLEX imports
try:
    from flex.model import FlexModel
    from flex.data import Dataset, FedDataDistribution, FedDatasetConfig
except ImportError:
    # Fallback during refactoring if needed, though we expect FLEX to be installed
    FlexModel = Any
    Dataset = Any

from src.infrastructure.metrics.sklearn_metrics_service import SklearnMetricsService
from src.infrastructure.metrics.diversity_service import PredictionBasedDiversityService
from src.domain.services.label_service import SimpleLabelService

from src.domain.metrics.forest_evaluator import ForestEvaluator, ForestReport
from src.domain.model.proactive_forest import ProactiveForest
from src.domain.dataset.base_adapter import DatasetSplit
from src.domain.metadata.client_metadata import ClientMetadata
from src.application.orchestrators.result_consolidator import ResultConsolidator
from src.application.orchestrators.fed_data_distributor import FedDataDistributor

# PF FLEX primitives
from src.infrastructure.flex.flex_train_pf import (
    init_server_model_pf,
    train_pf,
    collect_clients_trees_pf,
)
from src.infrastructure.flex.flex_deploy_model_pf import (
    deploy_server_config_pf,
    deploy_server_model_pf,
)
from src.infrastructure.flex.flex_aggregate_pf import (
    aggregate_trees_pf,
    set_aggregated_trees_pf,
)
from src.infrastructure.flex.flex_evaluate_pf import (
    evaluate_global_pf_model,
    evaluate_global_pf_model_at_clients,
    evaluate_local_pf_model_at_clients,
)


from src.application.orchestrators.fl_results import FLResults


class FLEXOrchestrator:
    """Enhanced Federated Learning Orchestrator using native FLEX Framework constructs.
    
    This orchestrator handles the full lifecycle of a federated round, including:
    - Data distribution (IID/Non-IID).
    - Client initialization and local training.
    - Weight collection and global aggregation.
    - Model deployment and evaluation.
    """

    def __init__(self, 
                 config: Union[dict, Any], 
                 step_callback: Optional[Callable] = None,
                 use_flex_pool: bool = True):
        """Initializes the orchestrator.

        Args:
            config (Union[dict, Any]): Configuration dictionary or object containing hyperparameters.
            step_callback (Optional[Callable]): Function called at each step of the round for progress reporting.
            use_flex_pool (bool): Whether to use FLEX Framework's FlexPool for communication.
        """
        self.config = config if isinstance(config, dict) else config.dict()
        self.config_dict = self.config
        self.step_callback = step_callback or (lambda *a, **kw: None)
        
        self.dataset_split: Optional[DatasetSplit] = None
        self.federated_data = None
        self.use_flex_pool = use_flex_pool
        self.flex_pool = None
        
        self.metrics_svc = SklearnMetricsService()
        self.diversity_svc = PredictionBasedDiversityService()
        self.label_svc = SimpleLabelService()

        self._setup_logging()

    @classmethod
    def from_config(cls, 
                    config: Union[dict, Any], 
                    step_callback: Optional[Callable] = None,
                    use_flex_pool: bool = True) -> "FLEXOrchestrator":
        """Create an orchestrator instance from a configuration dict.
        Mirrors the previous API used in notebooks.
        """
        return cls(config, step_callback=step_callback, use_flex_pool=use_flex_pool)

    def _setup_logging(self):
        log_dir = 'results/logs'
        if not os.path.exists(log_dir): os.makedirs(log_dir)
        
        # Choose log filename based on strategy
        strategy = str(self.config.get('aggregation', {}).get('strategy', 'S1')).lower()
        log_filename = 'PW_federated_debug.log' if 'pw' in strategy else 'S1_S7_federated_debug.log'
        
        self.logger = logging.getLogger("FLEXOrchestrator")
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            fh = logging.FileHandler(os.path.join(log_dir, log_filename))
            fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(fh)

    def _get_config_value(self, *keys: str, default=None):
        val = self.config
        for key in keys:
            if isinstance(val, dict): val = val.get(key)
            else: return default
        return val if val is not None else default

    def _get_strategy_name(self) -> str:
        from src.domain.aggregation.aggregation_factory import AggregationFactory
        raw_strategy = self.config.get('strategy') or self.config.get('aggregation', {}).get('strategy', 'S1')
        return AggregationFactory.normalize_strategy_name(str(raw_strategy))

    def _init_flex_pool(self):
        """Internal helper to initialize FlexPool using the factory."""
        try:
            from src.infrastructure.flex.flex_pool_factory import FlexPoolFactory
            self.flex_pool = FlexPoolFactory.create_client_server_pool(
                federated_data=self.federated_data,
                init_model_func=init_server_model_pf,
                config=self.config  # Native injection via init_func kwargs
            )
            self.step_callback("FlexPool inicializado con éxito", 15)
        except Exception as e:
            self.flex_pool = None
            self.logger.error(f"Error al inicializar FlexPool: {e}")
            warnings.warn(f"FLEX integration downgraded: {e}")

    def setup_federation(self, dataset_split: DatasetSplit, seed: int = 42):
        """Setup federated data distribution using FedDataDistributor."""
        distributor = FedDataDistributor(self.config, self.use_flex_pool)
        self.dataset_split, self.federated_data = distributor.distribute(dataset_split, seed)
        
        self.label_svc.fit(self.dataset_split.class_names or self.dataset_split.get_all_labels())
        
        model_config = self.config.setdefault('model', {})
        if self.dataset_split.class_names:
            model_config['class_names'] = self.dataset_split.class_names

        if self.use_flex_pool:
            self._init_flex_pool()

        self.step_callback("Federación FLEX configurada", 10)

    def run_federated_round(self) -> FLResults:
        """
        Execute federated round using native FLEX orchestration.
        """
        if not self.federated_data or self.dataset_split is None:
            raise ValueError("Federation not setup.")

        if self.flex_pool is None:
            # This should not happen if use_flex_pool is True and FLEX is installed
            raise RuntimeError("FlexPool not initialized. Native integration required.")

        self.step_callback("Iniciando ronda federada nativa FLEX...", 5)

        # 1. DEPLOY configuration
        self.flex_pool.servers.map(deploy_server_config_pf, self.flex_pool.clients)

        # 2. TRAIN local models
        self.step_callback("Entrenamiento local (FLEX map)...", 20)
        self.flex_pool.clients.map(train_pf)

        # 3. COLLECT trees
        self.step_callback("Recolección de pesos (FLEX run)...", 45)
        self.flex_pool.aggregators.map(collect_clients_trees_pf, self.flex_pool.clients)

        # 4. AGGREGATE
        self.step_callback("Agregación global (FLEX aggregate)...", 60)
        strategy_name = self._get_strategy_name()
        n_estimators = self._get_config_value('model', 'n_estimators', default=100)
        t_max = self._get_config_value('aggregation', 't_max', default=n_estimators)

        X_val_server = self.dataset_split.X_val
        y_val_server = self.dataset_split.y_val
        
        # Validation data as FLEX Dataset for server eval primitives
        server_val_dataset = Dataset.from_array(X_val_server, y_val_server)

        agg_kwargs = {
            'server_config': self.config,
            'X_val': X_val_server,
            'y_val': y_val_server,
            't_max': t_max,
            'metrics_service': self.metrics_svc,
            'diversity_service': self.diversity_svc
        }
        self.flex_pool.aggregators.map(aggregate_trees_pf, **agg_kwargs)
        self.flex_pool.aggregators.map(set_aggregated_trees_pf, self.flex_pool.servers)


        # 5. DEPLOY global model
        self.flex_pool.servers.map(deploy_server_model_pf, self.flex_pool.clients)

        # 6. EVALUATE
        self.step_callback("Evaluación (FLEX evaluate)...", 80)
        # Server-side evaluation
        server_eval = self.flex_pool.servers.map(evaluate_global_pf_model, test_data=server_val_dataset)
        # Client-side evaluation
        client_eval = self.flex_pool.clients.map(evaluate_global_pf_model_at_clients)
        
        # 7. Collect results and build FLResults
        consolidator = ResultConsolidator(self.label_svc, self.config)
        return consolidator.consolidate(
            strategy_name=strategy_name,
            flex_pool=self.flex_pool,
            dataset_split=self.dataset_split,
            server_eval=server_eval
        )

__all__ = ['FLEXOrchestrator', 'FLResults']