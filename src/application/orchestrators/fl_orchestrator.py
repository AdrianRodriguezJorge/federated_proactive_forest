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

from src.domain.aggregation.aggregation_factory import AggregationFactory
from src.domain.metrics.forest_evaluator import ForestEvaluator, ForestReport
from src.domain.model.proactive_forest import ProactiveForest
from src.domain.dataset.base_adapter import DatasetSplit
from src.domain.metadata.client_metadata import ClientMetadata

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


@dataclass
class FLResults:
    strategy_id: str
    global_accuracy: float
    global_macro_f1: float
    n_trees_global: int
    client_ids: List[str]
    client_accuracies: Dict[str, float]
    client_f1_scores: Dict[str, float]
    client_metadata: Dict[str, ClientMetadata] = field(default_factory=dict)
    client_reports: Dict[str, Any] = field(default_factory=dict)
    selected_ids: Dict[str, List[int]] = field(default_factory=dict)
    all_tree_entries: List[Any] = field(default_factory=list)
    global_report: Any = None
    global_predictions: np.ndarray = None
    num_rounds: int = 1
    communication_cost: float = 0.0
    client_hybrid_predictions: Dict[str, np.ndarray] = field(default_factory=dict)
    y_test: np.ndarray = None
    class_names: List[str] = field(default_factory=list)
    client_hybrid_forest_sizes: Dict[str, int] = field(default_factory=dict)
    convergence_round: Optional[int] = None
    round_logs: List[Dict[str, Any]] = field(default_factory=list)
    hybrid_weights: Dict[str, float] = field(default_factory=lambda: {'local_weight': 0.4, 'global_weight': 0.6})


class FLEXOrchestrator:
    """
    Enhanced Federated Learning Orchestrator using native FLEX Framework constructs.
    """

    def __init__(self, 
                 config: Union[dict, Any], 
                 step_callback: Optional[Callable] = None,
                 use_flex_pool: bool = True):
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
    def from_config(cls, config: Union[dict, Any]) -> "FLEXOrchestrator":
        """Create an orchestrator instance from a configuration dict.
        Mirrors the previous API used in notebooks.
        """
        return cls(config)

    def _setup_logging(self):
        log_dir = 'logs'
        if not os.path.exists(log_dir): os.makedirs(log_dir)
        self.logger = logging.getLogger("FLEXOrchestrator")
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            fh = logging.FileHandler(os.path.join(log_dir, 'federated_debug.log'))
            fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(fh)

    def _get_config_value(self, *keys: str, default=None):
        val = self.config
        for key in keys:
            if isinstance(val, dict): val = val.get(key)
            else: return default
        return val if val is not None else default

    def _get_strategy_name(self) -> str:
        strategy = self.config.get('strategy') or self.config.get('aggregation', {}).get('strategy', 'S1')
        strategy_str = str(strategy).upper()
        if strategy_str in ['PW', 'PROGRESSIVE_WINDOWS'] or strategy_str.startswith('PW_'): return 'PW'
        if '_' in strategy_str: strategy_str = strategy_str.split('_')[0]
        return strategy_str

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
        """Setup federated data distribution using FLEX FedDataDistribution."""
        # Create Server Validation Split
        if len(dataset_split.X_train) > 20:
            try:
                X_train_fed, X_server_val, y_train_fed, y_server_val = train_test_split(
                    dataset_split.X_train, dataset_split.y_train, 
                    test_size=0.05, random_state=seed, stratify=dataset_split.y_train
                )
            except ValueError:
                X_train_fed, X_server_val, y_train_fed, y_server_val = train_test_split(
                    dataset_split.X_train, dataset_split.y_train, 
                    test_size=0.05, random_state=seed
                )
        else:
            X_train_fed, y_train_fed = dataset_split.X_train, dataset_split.y_train
            X_server_val, y_server_val = dataset_split.X_train, dataset_split.y_train

        dataset_split.X_val = X_server_val
        dataset_split.y_val = y_server_val
        dataset_split.X_train = X_train_fed  
        dataset_split.y_train = y_train_fed
        self.dataset_split = dataset_split
        
        self.label_svc.fit(dataset_split.class_names or dataset_split.get_all_labels())
        
        model_config = self.config.setdefault('model', {})
        if dataset_split.class_names:
            model_config['class_names'] = dataset_split.class_names

        # Create FLEX Dataset
        centralized_dataset = Dataset.from_array(X_array=X_train_fed, y_array=y_train_fed)

        n_clients = self._get_config_value('federation', 'n_clients') or self._get_config_value('n_clients', default=5)
        distribution_type = self._get_config_value('federation', 'distribution') or self._get_config_value('distribution', default='iid').lower()

        if distribution_type == 'iid':
            self.federated_data = FedDataDistribution.iid_distribution(centralized_dataset, n_nodes=n_clients)
        else:
            alpha = self._get_config_value('federation', 'dirichlet_alpha') or self._get_config_value('alpha', default=0.5)
            y_train = centralized_dataset.y_data.to_numpy()
            n_classes = len(np.unique(y_train))
            rng = np.random.RandomState(seed)
            weights_per_label = rng.dirichlet([alpha] * n_classes, size=n_clients)
            config = FedDatasetConfig(n_nodes=n_clients, weights_per_label=weights_per_label)
            self.federated_data = FedDataDistribution.from_config(centralized_dataset, config)

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
        return self._build_results(strategy_name, server_val_dataset, server_eval, client_eval)

    def _build_results(self, strategy_name: str, server_val_dataset: Any, server_eval: dict = None, client_eval: dict = None) -> FLResults:
        """Consolidate pool state into FLResults object."""
        server_id = "server"
        server_model = self.flex_pool._models[server_id]
        
        if server_eval and server_id in server_eval:
            global_acc = server_eval[server_id].get('accuracy', server_model.get('global_accuracy', 0.0))
            global_f1 = server_eval[server_id].get('macro_f1', server_model.get('global_f1', 0.0))
        else:
            global_acc = server_model.get('global_accuracy', 0.0)
            global_f1 = server_model.get('global_f1', 0.0)
            
        global_trees = server_model.get('trees', [])
        
        client_ids = list(self.flex_pool.clients.actor_ids)
        client_accuracies = {}
        client_f1_scores = {}
        client_metadata = {}
        client_hybrid_predictions = {}
        client_hybrid_forest_sizes = {}

        # For hybrid prediction logic (needs X_test)
        X_test, y_test = self.dataset_split.X_test, self.dataset_split.y_test
        y_test_numeric = self.label_svc.transform(y_test)
        # Fix: Sync class names with label_svc instead of potentially generic dataset_split classes
        class_names = self.label_svc.classes
        
        # We need a predictor for hybrid evaluation (not yet fully moved to a FLEX primitive 
        # to preserve the specific Proactive Forest logic exactly)
        from src.domain.prediction.hybrid_predictor import HybridPredictor
        local_weight = self.config.get('prediction', {}).get('local_weight', 0.4)
        predictor = HybridPredictor(
            local_weight=local_weight, 
            global_weight=1.0 - local_weight, 
            n_classes=len(class_names), 
            class_names=class_names,
            label_service=self.label_svc
        )

        # Extract info from aggregator state (stored in server_model during aggregation)
        # Note: In our aggregate_trees_from_pf refactor, we didn't store all_tree_entries 
        # as it's large, but we need it for hybrid mapping.
        # Actually, let's assume we need to re-extract it or we should have stored it.
        # For now, let's keep it simple or re-simulate if necessary.
        
        # Extract metadata and perform hybrid prediction
        for cid in client_ids:
            client_model = self.flex_pool._models[cid]
            client_accuracies[cid] = client_model.get('global_accuracy', 0.0)
            client_f1_scores[cid] = client_model.get('global_f1', 0.0)
            
            # Create ClientMetadata from client model state
            # This requires that train_pf filled 'metadata'
            # Extract metadata robustly (handle object or dict)
            meta_val = client_model.get('metadata')
            if isinstance(meta_val, ClientMetadata):
                meta = meta_val
            elif isinstance(meta_val, dict):
                meta = ClientMetadata(**meta_val)
            else:
                meta = ClientMetadata(
                    client_id=cid, 
                    n_trees=len(client_model.get('trees', [])), 
                    accuracy=client_model.get('local_accuracy', 0.0), 
                    macro_f1=client_model.get('local_f1', 0.0), 
                    pcd=0.0
                )
            
            # Retrieve selected_ids from server_model
            selected_ids_raw = server_model.get('selected_ids', {})
            selected_ids_dict = {str(k): v for k, v in selected_ids_raw.items()}
            
            # SYNC selection info back into metadata for UI consumption
            meta.selected_local_tree_ids = selected_ids_dict.get(str(cid), [])
            client_metadata[cid] = meta

            # Hybrid prediction
            local_trees = client_model.get('trees', [])
            # We need pure external trees. 
            # In a unified FLEX model, this is simpler if we have the tree list.
            external_global_trees = [t for t in global_trees if not any(t is lt for lt in local_trees)]
            
            hybrid_preds = predictor.predict(X_test, local_trees, external_global_trees)
            client_hybrid_predictions[cid] = hybrid_preds
            client_hybrid_forest_sizes[cid] = len(local_trees) + len(external_global_trees)

        # ── Global Model Full Evaluation ──────────────────────────────────────
        global_model = server_model.get('model')
        if global_model:
            global_preds = global_model.predict(X_test)
            try:
                global_pcd = float(global_model.diversity_measure(X_test, y_test, diversity='pcd'))
            except:
                global_pcd = 0.0
                
            global_report = ForestEvaluator.evaluate_from_predictions(
                global_preds, y_test_numeric, class_names, len(global_trees), pcd=global_pcd
            )
        else:
            # Fallback if model is missing
            global_report = ForestReport(
                accuracy=global_acc,
                macro_f1=global_f1,
                macro_precision=0.0,
                macro_recall=0.0,
                per_class_f1={cn: 0.0 for cn in class_names},
                per_class_prec={cn: 0.0 for cn in class_names},
                per_class_recall={cn: 0.0 for cn in class_names},
                confusion_matrix=np.zeros((len(class_names), len(class_names))),
                pcd=0.0,
                forest_size=len(global_trees),
                class_names=class_names,
                accuracy_ci=(0.0, 0.0),
                macro_f1_ci=(0.0, 0.0)
            )

        self.step_callback("Resultados consolidados", 100)
        
        return FLResults(
            strategy_id=strategy_name,
            global_accuracy=global_report.accuracy,
            global_macro_f1=global_report.macro_f1,
            n_trees_global=len(global_trees),
            client_ids=client_ids,
            client_accuracies=client_accuracies,
            client_f1_scores=client_f1_scores,
            client_metadata=client_metadata,
            global_report=global_report,
            client_hybrid_predictions=client_hybrid_predictions,
            y_test=y_test_numeric,
            class_names=class_names,
            client_hybrid_forest_sizes=client_hybrid_forest_sizes,
            convergence_round=server_model.get('convergence_round'),
            round_logs=server_model.get('round_logs', []),
            selected_ids=selected_ids_dict,
            all_tree_entries=server_model.get('all_tree_entries', [])
        )


__all__ = ['FLEXOrchestrator', 'FLResults']