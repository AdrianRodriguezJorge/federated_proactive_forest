"""Enhanced FLEXOrchestrator with full FLEX integration for Proactive Forest.

This module provides complete federated learning orchestration using FLEX Framework,
supporting both IID and Non-IID (Dirichlet) data distributions.

Features:
- IID and Non-IID Dirichlet distributions
- FlexPool with decorators for task orchestration
- Both Client-Server and P2P architectures
- Full integration with ProactiveForest algorithm
- Complete tree aggregation strategies
"""

from __future__ import annotations
import numpy as np
import warnings
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
from sklearn.model_selection import train_test_split
from src.infrastructure.metrics.sklearn_metrics_service import SklearnMetricsService
from src.infrastructure.metrics.diversity_service import PredictionBasedDiversityService
from src.domain.services.label_service import SimpleLabelService


# TYPE_CHECKING imports avoid runtime dependency while providing type hints
if TYPE_CHECKING:
    try:
        from flex.data import Dataset  # type: ignore
    except ImportError:
        Dataset = Any

from src.domain.aggregation.aggregation_factory import AggregationFactory
from src.domain.metrics.forest_evaluator import ForestEvaluator, ForestReport
from src.domain.model.proactive_forest import ProactiveForest
from src.domain.dataset.base_adapter import DatasetSplit
from src.domain.metadata.client_metadata import ClientMetadata
from src.infrastructure.flex.flex_train_pf import (
    init_server_model_pf,
    train_pf,
    collect_clients_trees_pf,
)
from src.infrastructure.flex.flex_deploy_model_pf import (
    deploy_server_config_pf,
    deploy_server_model_pf,
)
from src.infrastructure.flex.flex_collect_trees_pf import (
    aggregate_trees_from_pf,
    set_aggregated_trees_pf,
)
from src.infrastructure.flex.flex_evaluate_pf import (
    evaluate_global_pf_model,
    evaluate_global_pf_model_at_clients,
    evaluate_local_pf_model_at_clients,
)


@dataclass
class ClientReport:
    """Report metrics for a client after global model evaluation."""
    accuracy: float
    macro_f1: float


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
    communication_cost: float = 0.0  # Estimación de datos transferidos (MB)
    client_hybrid_predictions: Dict[str, np.ndarray] = field(default_factory=dict)
    y_test: np.ndarray = None
    class_names: List[str] = field(default_factory=list)
    client_hybrid_forest_sizes: Dict[str, int] = field(default_factory=dict)
    convergence_round: Optional[int] = None  # Ronda en la que se alcanzó convergencia (PW)
    hybrid_weights: Dict[str, float] = field(default_factory=lambda: {'local_weight': 0.4, 'global_weight': 0.6})  # Pesos usados en inferencia híbrida


class FLEXOrchestrator:
    """
    Enhanced Federated Learning Orchestrator using FLEX Framework.
    
    Supports:
    - IID and Non-IID (Dirichlet) data distributions
    - Both Client-Server and P2P architectures
    - Multiple aggregation strategies
    - Full tree-based federation
    
    Usage:
        config = {
            'n_clients': 5,
            'distribution': 'noniid_dirichlet',  # or 'iid'
            'alpha': 0.5,  # Dirichlet parameter (lower = more heterogeneous)
            'strategy': 'S1',  # Aggregation strategy
            'n_estimators': 100,
            'alpha_pf': 0.1,  # Proactive Forest parameter
            'architecture': 'client_server',  # or 'p2p'
        }
        
        orchestrator = FLEXOrchestrator.from_config(config)
        orchestrator.setup_federation(dataset_split)
        results = orchestrator.run_federated_round()
    """

    def __init__(self, 
                 config: dict, 
                 step_callback: Optional[Callable] = None,
                 use_flex_pool: bool = True):
        """
        Initialize orchestrator.
        
        Args:
            config: Configuration dict with federation parameters
            step_callback: Optional callback for progress tracking
            use_flex_pool: Whether to use FlexPool with decorators (if FLEX available)
        """
        self.config = config
        self.step_callback = step_callback or (lambda *a, **kw: None)
        self.dataset_split = None
        self.federated_data = None
        self.client_partitions = {}
        self.use_flex_pool = use_flex_pool
        self.flex_pool = None
        
        # Communication tracking
        self._communication_rounds = 0
        self._data_transferred = 0.0

        # Services
        self.metrics_svc = SklearnMetricsService()
        self.diversity_svc = PredictionBasedDiversityService()
        self.label_svc = SimpleLabelService()

        # Setup persistent logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging to both file and console."""
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_file = os.path.join(log_dir, 'federated_debug.log')
        
        # Configure logger
        self.logger = logging.getLogger("FLEXOrchestrator")
        self.logger.setLevel(logging.DEBUG)
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            # File handler
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)
            
            # Formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            
            self.logger.addHandler(fh)
            
        self.logger.info("FLEXOrchestrator initialized with strategy: %s", self._get_strategy_name())

    @classmethod
    def from_config(cls, config: dict, step_callback: Optional[Callable] = None) -> "FLEXOrchestrator":
        """Create orchestrator from configuration dict."""
        return cls(config, step_callback)

    def _get_strategy_name(self) -> str:
        """
        Extract and normalize strategy name from config.

        Searches in multiple locations:
        - config['strategy'] (flat structure)
        - config['aggregation']['strategy'] (nested structure from Streamlit)

        Normalizes: 's7_perclient_f1_pcd' → 'S7', 'S1' → 'S1', 'pw' → 'PW'
        """
        # Try flat structure first
        strategy = self.config.get('strategy')

        # Try nested structure (from Streamlit)
        if not strategy:
            strategy = self.config.get('aggregation', {}).get('strategy')

        # Default fallback
        if not strategy:
            strategy = 'S1'

        # Normalize: extract S1-S7 from format like "s7_perclient_f1_pcd"
        strategy_str = str(strategy).upper()

        # Special case: PW (Progressive Windows)
        if strategy_str == 'PW' or strategy_str == 'PROGRESSIVE_WINDOWS' or strategy_str.startswith('PW_'):
            return 'PW'

        if '_' in strategy_str:
            # Extract first part: "S7_..." → "S7"
            strategy_str = strategy_str.split('_')[0]

        return strategy_str

    def setup_federation(self, dataset_split: DatasetSplit, seed: int = 42):
        """
        Setup federated data distribution using FLEX Framework.
        
        Supports:
        - IID distribution: uniform random splitting
        - Non-IID Dirichlet: controlled heterogeneity
        
        Args:
            dataset_split: DatasetSplit with X_train, y_train, X_test, y_test
            seed: Random seed for reproducibility
        """
        try:
            from flex.data import Dataset
        except ImportError:
            raise ImportError("FLEX not installed. Run: pip install flex-framework")

        # Create Server Validation Split (5% instead of 10% to preserve client data)
        # Ensure we have at least some samples for val
        if len(dataset_split.X_train) > 20:
            from sklearn.model_selection import train_test_split
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
        
        # Fit label service once with all possible labels
        self.label_svc.fit(dataset_split.get_all_labels())

        # Create FLEX Dataset
        centralized_dataset = Dataset.from_array(
            X_array=X_train_fed,
            y_array=y_train_fed
        )

        # Setup distribution based on configuration
        n_clients = self._get_config_value('federation', 'n_clients') or self._get_config_value('n_clients', default=5)
        distribution_type = self._get_config_value('federation', 'distribution') or self._get_config_value('distribution', default='iid')

        if distribution_type.lower() == 'iid':
            self._setup_iid_distribution(centralized_dataset, n_clients, seed)
        elif distribution_type.lower() in ['noniid', 'noniid_dirichlet']:
            alpha = self._get_config_value('federation', 'dirichlet_alpha') or self._get_config_value('alpha', default=0.5)
            self._setup_noniid_dirichlet_distribution(centralized_dataset, n_clients, alpha, seed)
        else:
            raise ValueError(f"Unknown distribution type: {distribution_type}")

        self.step_callback("Configuración de federación completada", 10)

    def _get_config_value(self, *keys: str, default=None):
        """
        Get value from config, searching in nested paths.
        
        Example:
            _get_config_value('federation', 'n_clients') 
            searches config['federation']['n_clients']
            
            _get_config_value('n_clients')
            searches config['n_clients']
        """
        val = self.config
        for key in keys:
            if isinstance(val, dict):
                val = val.get(key)
            else:
                return default
        return val if val is not None else default

    def _setup_iid_distribution(self, dataset: Any, n_clients: int, seed: int):
        """Setup IID distribution using FLEX."""
        try:
            from flex.data import FedDataDistribution
        except ImportError:
            raise ImportError("FLEX not installed")

        self.federated_data = FedDataDistribution.iid_distribution(
            centralized_data=dataset,
            n_nodes=n_clients
        )
        
        self.client_partitions = self._convert_fed_data_to_partitions()

    def _setup_noniid_dirichlet_distribution(self, 
                                           dataset: Any, 
                                           n_clients: int, 
                                           alpha: float,
                                           seed: int):
        """
        Setup Non-IID Dirichlet distribution using FLEX.
        
        Uses Dirichlet distribution to create controlled heterogeneous data.
        Lower alpha = more heterogeneous.
        
        Args:
            dataset: Centralized dataset
            n_clients: Number of clients
            alpha: Dirichlet concentration parameter
            seed: Random seed
        """
        try:
            from flex.data import FedDataDistribution, FedDatasetConfig
        except ImportError:
            raise ImportError("FLEX not installed")

        # Get class distribution
        y_train = dataset.y_data.to_numpy()
        n_classes = len(np.unique(y_train))

        # Generate Dirichlet-based weights per label
        rng = np.random.RandomState(seed)
        weights_per_label = rng.dirichlet(
            [alpha] * n_classes,
            size=n_clients
        )

        # Create config with Dirichlet weights
        config = FedDatasetConfig(
            n_nodes=n_clients,
            weights_per_label=weights_per_label  # ← Key: Dirichlet-based heterogeneity
        )

        # Create distribution from config
        self.federated_data = FedDataDistribution.from_config(dataset, config)
        self.client_partitions = self._convert_fed_data_to_partitions()

        # Ensure FlexPool is initialized if requested, otherwise keep None.
        if self.use_flex_pool:
            try:
                from src.infrastructure.flex.flex_pool_factory import FlexPoolFactory

                # init_model_func is a simple placeholder; in a full FLEX pipeline this should
                # return a fresh model or function to instantiate per actor.
                self.flex_pool = FlexPoolFactory.create_client_server_pool(
                    federated_data=self.federated_data,
                    init_model_func=lambda: None
                )
                self.step_callback("FlexPool inicializado", 15)
            except ImportError:
                self.flex_pool = None
                warnings.warn("FLEX no está instalado, self.flex_pool queda en None.")
            except Exception as e:
                self.flex_pool = None
                warnings.warn(f"No se pudo inicializar FlexPool: {e}")

    def _convert_fed_data_to_partitions(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Convert FLEX FedDataDistribution to simple client partitions dict."""
        partitions = {}
        for node_id, node_data in self.federated_data.items():
            client_id = f"client_{node_id}"
            X_client = node_data.X_data.to_numpy()
            y_client = node_data.y_data.to_numpy() if node_data.y_data is not None else None
            partitions[client_id] = (X_client, y_client)
        return partitions

    def run_federated_round(self) -> FLResults:
        """
        Execute complete federated learning round.
        
        Steps:
        1. Train local forests on each client
        2. Collect trees from all clients
        3. Aggregate using selected strategy
        4. Evaluate global model
        
        Returns:
            FLResults with complete round metrics
        """
        if not self.client_partitions or self.dataset_split is None:
            raise ValueError("Federation not set up. Call setup_federation first.")

        # ── STEP 1: TRAIN local forests on each client ────────────────────────
        self.step_callback("Entrenando bosques locales en clientes...", 15)
        client_forests, client_metadata = self._train_local_forests()
        client_ids = list(client_forests.keys())

        # Log: árboles entrenados por cliente
        print("\n" + "=" * 60)
        print("🌳 ÁRBOLES ENTRENADOS POR CLIENTE")
        print("=" * 60)
        for cid in client_ids:
            n_trees = len(client_forests[cid].get_trees())
            print(f"  {cid}: {n_trees} árboles entrenados")
        total_entrenados = sum(len(pf.get_trees()) for pf in client_forests.values())
        print(f"  TOTAL: {total_entrenados} árboles")

        # ── STEP 2: COLLECT trees from all clients ────────────────────────────
        self.step_callback("Recolectando árboles de clientes...", 40)
        client_trees = {cid: pf.get_trees() for cid, pf in client_forests.items()}

        # Estimate communication cost (MB per tree)
        trees_per_client = [len(trees) for trees in client_trees.values()]
        avg_tree_size_kb = 0.1  # Approximate for small trees
        self._data_transferred = sum(trees_per_client) * avg_tree_size_kb / 1024

        # Prepare FLEX server/client models
        server_flex_model = {
            'config': self.config,
            'model': None,
            'trees': [],
        }

        # Perform aggregation
        metrics_svc = SklearnMetricsService()
        
        # Deploy model config to server model (FLEX)
        deploy_server_config_pf(server_flex_model, self.config)

        clients_flex_models = {}
        for cid, pf in client_forests.items():
            clients_flex_models[cid] = {
                'model': pf,
                'trees': pf.get_trees(),
                'X_test': self.dataset_split.X_test,
                'y_test': self.dataset_split.y_test,
            }

        for client_flex_model in clients_flex_models.values():
            deploy_server_config_pf(server_flex_model, client_flex_model)
            deploy_server_model_pf(server_flex_model, client_flex_model)

        collect_clients_trees_pf(server_flex_model, clients_flex_models)

        # Store client metadata for aggregation
        server_flex_model['client_metadata'] = client_metadata

        # ── STEP 3: AGGREGATE using selected strategy ──────────────────────────
        self.step_callback("Agregando bosques...", 60)
        strategy_name = self._get_strategy_name()

        # Get n_estimators from config to limit the number of trees selected
        n_estimators = self._get_config_value('model', 'n_estimators') or self._get_config_value('n_estimators', default=100)
        t_max = self._get_config_value('aggregation', 't_max', default=n_estimators)  # T_MAX por defecto = n_estimators

        # Get validation data for Progressive Forest (S2-S7) and PW
        # Now using the SERVER VALIDATION set instead of the global test set.
        X_val_server = self.dataset_split.X_val if self.dataset_split.X_val is not None else self.dataset_split.X_test
        y_val_server = self.dataset_split.y_val if self.dataset_split.y_val is not None else self.dataset_split.y_test

        # Build kwargs based on strategy type
        aggregate_kwargs = {}
        if strategy_name in ['S2', 'S3', 'S4']:
            aggregate_kwargs['X_val'] = X_val_server
            aggregate_kwargs['y_val'] = y_val_server
            aggregate_kwargs['max_trees'] = n_estimators
            aggregate_kwargs['t_max'] = t_max
        elif strategy_name in ['S5', 'S6', 'S7']:
            aggregate_kwargs['X_val'] = X_val_server
            aggregate_kwargs['y_val'] = y_val_server
            aggregate_kwargs['max_trees_per_client'] = n_estimators
            aggregate_kwargs['t_max'] = t_max
        elif strategy_name == 'PW':
            # Progressive Windows uses different parameters
            aggregate_kwargs['X_val'] = X_val_server
            aggregate_kwargs['y_val'] = y_val_server
            aggregate_kwargs['t_max'] = t_max
            aggregate_kwargs['window_size'] = self._get_config_value('aggregation', 'window_size', default=5)
            aggregate_kwargs['max_rounds'] = self._get_config_value('aggregation', 'max_rounds', default=20)
            aggregate_kwargs['class_names'] = self.dataset_split.class_names
            # f1_weight will be set in the combined block below

        # S4, S7, and PW use f1_weight/pcd_weight combination
        if strategy_name in ['S4', 'S7', 'PW']:
            aggregate_kwargs['f1_weight'] = self._get_config_value('aggregation', 'f1_weight', default=0.5)
            # pcd_weight is automatically calculated as 1.0 - f1_weight in hyperparam_optimizer
            # but we set it here for backward compatibility
            f1_weight = aggregate_kwargs['f1_weight']
            aggregate_kwargs['pcd_weight'] = self._get_config_value('aggregation', 'pcd_weight', default=1.0 - f1_weight)

        # Perform aggregation
        # Execute aggregation primitive (S1-S7 or PW)
        aggregate_kwargs['metrics_service'] = self.metrics_svc
        aggregate_kwargs['diversity_service'] = self.diversity_svc
        
        aggregate_trees_from_pf(server_flex_model, **aggregate_kwargs)

        set_aggregated_trees_pf(server_flex_model)

        global_trees = server_flex_model.get('trees', [])
        selected_ids = server_flex_model.get('selected_indices', {})
        all_tree_entries = server_flex_model.get('all_tree_entries', [])

        # Log: árboles seleccionados por cliente
        print("\n" + "=" * 60)
        print(f"📊 ÁRBOLES SELECCIONADOS (Estrategia: {strategy_name})")
        print("=" * 60)
        for cid, indices in selected_ids.items():
            n_selected = len(indices)
            n_total = len(client_trees[cid])
            print(f"  {cid}: {n_selected}/{n_total} árboles seleccionados")

        # Log: tamaño del bosque global
        print("\n" + "=" * 60)
        print(f"🌲 BOSQUE GLOBAL")
        print("=" * 60)
        print(f"  Tamaño final: {len(global_trees)} árboles")

        # Log: tamaño final de cada cliente (bosque global + árboles locales)
        # IMPORTANT: Show sizes without duplicates (each client doesn't use its own trees from global)
        print("\n" + "=" * 60)
        print("📈 TAMAÑO FINAL POR CLIENTE (sin duplicados)")
        print("=" * 60)
        global_tree_sources = [e.client_id for e in all_tree_entries]
        for cid in client_ids:
            n_local = len(client_trees[cid])
            n_external_global = sum(1 for source_cid in global_tree_sources if source_cid != cid)
            total = n_local + n_external_global
            n_from_this_client = sum(1 for source_cid in global_tree_sources if source_cid == cid)
            print(f"  {cid}: {n_local} locales + {n_external_global} globales externos ({n_from_this_client} propios excluidos) = {total} árboles")
        print("=" * 60 + "\n")

        # ── STEP 4: EVALUATE on test set ───────────────────────────────────────
        self.step_callback("Evaluando modelo global...", 85)
        server_model = server_flex_model.get('model')
        if server_model is None:
            global_forest = ProactiveForest.from_trees(global_trees, class_names=self.dataset_split.class_names)
        else:
            global_forest = server_model

        # Evaluate using FLEX primitives if possible
        # Convert test set to combined data for evaluate_global_pf_model (optional)
        # since evaluate_global_pf_model expects data with labels in last col
        # we keep traditional local evaluation path below.

        X_test, y_test = self.dataset_split.X_test, self.dataset_split.y_test
        class_names = self.dataset_split.class_names

        # Get global predictions in numeric format (class indices)
        global_predictions_raw = global_forest.predict(X_test)

        # Convert predictions to numeric format if they are strings
        if len(global_predictions_raw) > 0 and isinstance(global_predictions_raw[0], str):
            class_to_idx = {cn: idx for idx, cn in enumerate(class_names)}
            global_predictions = np.array([class_to_idx[pred] for pred in global_predictions_raw])
        else:
            # Ensure numeric format (could be int or float)
            global_predictions = np.asarray(global_predictions_raw, dtype=np.int64)

        for client_flex_model in clients_flex_models.values():
            evaluate_global_pf_model_at_clients(client_flex_model)
            evaluate_local_pf_model_at_clients(client_flex_model)

        # Standardize labels to indices for evaluation
        y_test_numeric = self.label_svc.transform(y_test)

        global_report = ForestEvaluator.evaluate(
            global_forest, X_test, y_test_numeric, class_names, metrics_svc=self.metrics_svc
        )
        self.logger.info("Global model accuracy: %.4f, F1: %.4f", global_report.accuracy, global_report.macro_f1)


        # Perform hybrid prediction on clients using local and global trees
        # IMPORTANT: Avoid duplicates - each client should not use global trees that came from itself
        client_hybrid_predictions = {}

        # Get local_weight from strategy instance (PW) or fallback to config
        strategy_instance = server_flex_model.get('strategy_instance') if strategy_name == 'PW' else None

        if strategy_instance and hasattr(strategy_instance, 'local_weight'):
            local_weight = strategy_instance.local_weight
            global_weight = 1.0 - local_weight
        else:
            local_weight = self.config.get('prediction', {}).get('local_weight', 0.4)
            global_weight = self.config.get('prediction', {}).get('global_weight', 0.6)
        
        n_classes = len(class_names)
        from src.domain.prediction.hybrid_predictor import HybridPredictor
        predictor = HybridPredictor(
            local_weight=local_weight, 
            global_weight=global_weight, 
            n_classes=n_classes, 
            class_names=class_names,
            label_service=self.label_svc
        )
        client_hybrid_forest_sizes = {}
        
        # Build a mapping: global_index -> client_id (to know which client each global tree came from)
        global_tree_sources = [e.client_id for e in all_tree_entries]
        
        for cid, pf in client_forests.items():
            local_trees = pf.get_trees()
            # FIX: Use all_tree_entries to guarantee correct mapping
            # all_tree_entries contains ONLY the selected global trees in correct order
            external_global_trees = [
                entry.tree for entry in all_tree_entries 
                if entry.client_id != cid
            ]
            
            hybrid_preds = predictor.predict(X_test, local_trees, external_global_trees)
            client_hybrid_predictions[cid] = hybrid_preds
            
            # Forest size = local trees + external global trees (no duplicates)
            client_hybrid_forest_sizes[cid] = len(local_trees) + len(external_global_trees)
            
            # Log debug stats
            stats = predictor.get_debug_stats()
            self.logger.debug("Client %s Hybrid Stats: Local trees: %d, External Global: %d", 
                             cid, len(local_trees), len(external_global_trees))
            self.logger.debug("Client %s Vote distribution: %s", cid, stats)

        self.step_callback("Ronda completada", 100)

        # Calculate convergence round for PW strategy
        convergence_round = None
        if strategy_instance and hasattr(strategy_instance, 'convergence_round'):
            convergence_round = strategy_instance.convergence_round

        return FLResults(
            strategy_id=strategy_name,
            global_accuracy=global_report.accuracy,
            global_macro_f1=global_report.macro_f1,
            n_trees_global=len(global_trees),
            client_ids=client_ids,
            client_accuracies={cid: m.accuracy for cid, m in client_metadata.items()},
            client_f1_scores={cid: m.macro_f1 for cid, m in client_metadata.items()},
            client_metadata=client_metadata,
            client_reports={},  # Removed client reports as per user request
            selected_ids=selected_ids,
            all_tree_entries=all_tree_entries,
            global_report=global_report,
            global_predictions=global_predictions,
            num_rounds=1,
            communication_cost=self._data_transferred,
            client_hybrid_predictions=client_hybrid_predictions,
            y_test=y_test_numeric, # Return numeric y_test for consistency with hybrid_preds
            class_names=class_names,
            client_hybrid_forest_sizes=client_hybrid_forest_sizes,
            convergence_round=convergence_round,
            hybrid_weights={'local_weight': local_weight, 'global_weight': global_weight},
        )

    def _train_local_forests(self) -> Tuple[Dict[str, ProactiveForest], Dict[str, ClientMetadata]]:
        """
        Train ProactiveForest locally on each client's data.
        
        Returns:
            Tuple of (client_forests, client_metadata)
        """
        client_forests = {}
        client_metadata = {}

        n_estimators = self._get_config_value('model', 'n_estimators') or self._get_config_value('n_estimators', default=100)
        alpha_pf = self._get_config_value('model', 'alpha') or self._get_config_value('alpha_pf', default=0.1)
        seed = self.config.get('seed', 42)

        for client_id, (X_client, y_client) in self.client_partitions.items():
            # FIX: No longer using 'if len(X_client) < 10' to duplication. Always split.
            try:
                # Try stratified split first
                X_train_local, X_val_local, y_train_local, y_val_local = train_test_split(
                    X_client, y_client, test_size=0.2, random_state=seed, stratify=y_client
                )
            except (ValueError, TypeError):
                # Fallback to simple split if stratification is impossible or dataset is too small
                try:
                    X_train_local, X_val_local, y_train_local, y_val_local = train_test_split(
                        X_client, y_client, test_size=0.2, random_state=seed
                    )
                except ValueError:
                    # Extreme fallback: if only 1 sample exists, use it for both but with a warning 
                    # (this is rare in FL but prevents crashes)
                    import warnings
                    warnings.warn(f"Client {client_id} has insufficient data for split. Data leakage will occur.")
                    X_train_local, X_val_local = X_client, X_client
                    y_train_local, y_val_local = y_client, y_client

            # Train ProactiveForest on local train split
            pf = ProactiveForest(
                n_estimators=n_estimators,
                alpha=alpha_pf,
                verbose=self._get_config_value('verbose', default=False),
                class_names=self.dataset_split.class_names
            )
            pf.fit(X_train_local, y_train_local, X_val=X_val_local, y_val=y_val_local)

            # Evaluate each tree individually on local validation set (Real Tree Ranking)
            tree_metrics = []

            # Ensure y_val_local is numeric for comparison
            y_val_local_eval = self.label_svc.transform(y_val_local)
            
            # Get all trees from the fitted forest
            local_trees = pf.get_trees()
            
            # Batch prediction: Iterate trees but optimize inside
            for tree in local_trees:
                # Vectorized tree predict
                y_pred_tree_raw = tree.predict(X_val_local)
                
                # Transform labels (centralized)
                y_pred_tree = self.label_svc.transform(y_pred_tree_raw)

                # Accuracy and F1 (optimized by reducing repetitive work if possible)
                acc_t = self.metrics_svc.accuracy_score(y_val_local_eval, y_pred_tree)
                f1_t = self.metrics_svc.f1_score(y_val_local_eval, y_pred_tree, average='macro')
                
                tree_metrics.append({
                    'accuracy': float(acc_t), 
                    'macro_f1': float(f1_t)
                })

            # Forest-level metrics for this client (on local validation)
            local_report = ForestEvaluator.evaluate(
                pf,
                X_val_local,
                y_val_local_eval,
                self.dataset_split.class_names,
                metrics_svc=self.metrics_svc
            )

            acc = float(local_report.accuracy)
            f1 = float(local_report.macro_f1)
            pcd = float(local_report.pcd) if local_report.pcd is not None else 0.0

            client_forests[client_id] = pf
            client_metadata[client_id] = ClientMetadata(
                client_id=client_id,
                n_trees=len(pf.get_trees()),
                accuracy=acc,
                macro_f1=f1,
                pcd=pcd,
                tree_metrics=tree_metrics
            )

        return client_forests, client_metadata


__all__ = ['FLEXOrchestrator', 'FLResults']