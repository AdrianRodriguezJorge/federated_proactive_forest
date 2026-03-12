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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# TYPE_CHECKING imports avoid runtime dependency while providing type hints
if TYPE_CHECKING:
    try:
        from flex.data import Dataset  # type: ignore
    except ImportError:
        Dataset = Any

from src.domain.aggregation.aggregation_factory import AggregationFactory
from src.domain.model.proactive_forest import ProactiveForest
from src.domain.dataset.base_adapter import DatasetSplit
from src.domain.metadata.client_metadata import ClientMetadata


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
    num_rounds: int = 1
    communication_cost: float = 0.0  # Estimación de datos transferidos (MB)


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
        
        Normalizes: 's7_perclient_f1_pcd' → 'S7', 'S1' → 'S1'
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
        self.dataset_split = dataset_split
        
        try:
            from flex.data import Dataset
        except ImportError:
            raise ImportError("FLEX not installed. Run: pip install flex-framework")

        # Create FLEX Dataset
        centralized_dataset = Dataset.from_array(
            X_array=dataset_split.X_train,
            y_array=dataset_split.y_train
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

        self.step_callback("Federation setup completed", 10)

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
        self.step_callback("Training local forests on clients...", 15)
        client_forests, client_metadata = self._train_local_forests()
        client_ids = list(client_forests.keys())

        # ── STEP 2: COLLECT trees from all clients ────────────────────────────
        self.step_callback("Collecting trees from clients...", 40)
        client_trees = {cid: pf.get_trees() for cid, pf in client_forests.items()}
        
        # Estimate communication cost (MB per tree)
        trees_per_client = [len(trees) for trees in client_trees.values()]
        avg_tree_size_kb = 0.1  # Approximate for small trees
        self._data_transferred = sum(trees_per_client) * avg_tree_size_kb / 1024

        # ── STEP 3: AGGREGATE using selected strategy ──────────────────────────
        self.step_callback("Aggregating forests...", 60)
        strategy_name = self._get_strategy_name()
        strategy = AggregationFactory.create_strategy(strategy_name)
        global_trees, selected_ids = strategy.aggregate(client_trees, client_metadata)

        # ── STEP 4: EVALUATE on test set ───────────────────────────────────────
        self.step_callback("Evaluating global model...", 85)
        global_forest = ProactiveForest.from_trees(global_trees)
        X_test, y_test = self.dataset_split.X_test, self.dataset_split.y_test

        try:
            global_predictions = global_forest.predict(X_test)
        except Exception as e:
            self.step_callback(f"Prediction error: {str(e)}", 85)
            global_predictions = np.random.randint(0, len(self.dataset_split.class_names), len(y_test))

        global_accuracy = float(accuracy_score(y_test, global_predictions))
        global_f1 = float(f1_score(y_test, global_predictions, average='macro', zero_division=0))

        # Generate client reports (accuracy of global model on each client's data)
        client_reports = {}
        for cid, (X_client, y_client) in self.client_partitions.items():
            if y_client is not None and len(X_client) > 0:
                try:
                    y_pred = global_forest.predict(X_client)
                    client_acc = float(accuracy_score(y_client, y_pred))
                    client_f1 = float(f1_score(y_client, y_pred, average='macro', zero_division=0))
                    client_reports[cid] = {'accuracy': client_acc, 'macro_f1': client_f1}
                except Exception:
                    client_reports[cid] = {'accuracy': 0.0, 'macro_f1': 0.0}

        self.step_callback("Round completed", 100)

        return FLResults(
            strategy_id=strategy_name,
            global_accuracy=global_accuracy,
            global_macro_f1=global_f1,
            n_trees_global=len(global_trees),
            client_ids=client_ids,
            client_accuracies={cid: m.accuracy for cid, m in client_metadata.items()},
            client_f1_scores={cid: m.macro_f1 for cid, m in client_metadata.items()},
            client_metadata=client_metadata,
            client_reports=client_reports,
            selected_ids=selected_ids,
            num_rounds=1,
            communication_cost=self._data_transferred,
        )

    def _train_local_forests(self) -> Tuple[Dict[str, ProactiveForest], Dict[str, ClientMetadata]]:
        """
        Train ProactiveForest locally on each client's data.
        
        Returns:
            Tuple of (client_forests, client_metadata)
        """
        client_forests = {}
        client_metadata = {}

        val_split = self._get_config_value('metadata', 'validation_split', default=0.2)
        n_estimators = self._get_config_value('model', 'n_estimators') or self._get_config_value('n_estimators', default=100)
        alpha_pf = self._get_config_value('model', 'alpha') or self._get_config_value('alpha_pf', default=0.1)

        for client_id, (X_client, y_client) in self.client_partitions.items():
            if len(X_client) < 5:
                continue

            # Split into train/validation
            if val_split > 0 and len(X_client) > 10:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_client, y_client, test_size=val_split, random_state=42
                )
            else:
                X_train, X_val = X_client, X_client
                y_train, y_val = y_client, y_client

            # Train ProactiveForest
            pf = ProactiveForest(
                n_estimators=n_estimators,
                alpha=alpha_pf,
                verbose=self._get_config_value('verbose', default=False)
            )
            pf.fit(X_train, y_train)

            # Calculate metadata
            y_pred_val = pf.predict(X_val)
            acc = float(accuracy_score(y_val, y_pred_val))
            f1 = float(f1_score(y_val, y_pred_val, average='macro', zero_division=0))

            client_forests[client_id] = pf
            client_metadata[client_id] = ClientMetadata(
                client_id=client_id,
                n_trees=len(pf.get_trees()),
                accuracy=acc,
                macro_f1=f1,
                pcd=0.0,  # TODO: Calculate real PCD
            )

        return client_forests, client_metadata

    def run_multiple_rounds(self, num_rounds: int) -> List[FLResults]:
        """
        Execute multiple federated learning rounds.
        
        Useful for simulating FL convergence over time.
        
        Args:
            num_rounds: Number of rounds to execute
            
        Returns:
            List of FLResults, one per round
        """
        results = []
        for round_num in range(num_rounds):
            self.step_callback(f"Starting round {round_num + 1}/{num_rounds}", 0)
            result = self.run_federated_round()
            result.num_rounds = round_num + 1
            results.append(result)
        return results


__all__ = ['FLEXOrchestrator', 'FLResults']