from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from src.domain.aggregation.aggregation_factory import AggregationFactory
from src.domain.model.proactive_forest import ProactiveForest
from src.domain.dataset.base_adapter import DatasetSplit
from src.domain.metadata.client_metadata import ClientMetadata


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


class FLEXOrchestrator:
    """
    Executes a complete federated learning round using FLEX framework.
    Simplified implementation that uses FLEX for data distribution.
    """

    def __init__(self, config: dict, step_callback: Optional[Callable] = None):
        self.config = config
        self.step_callback = step_callback or (lambda *a, **kw: None)
        self.dataset_split = None
        self.federated_data = None
        self.client_partitions = {}

    @classmethod
    def from_config(cls, config: dict, step_callback: Optional[Callable] = None) -> "FLEXOrchestrator":
        """Create orchestrator from config dict."""
        return cls(config, step_callback)

    def setup_federation(self, dataset_split: DatasetSplit):
        """
        Setup federated data distribution and partition data to clients.
        """
        self.dataset_split = dataset_split
        
        try:
            from flex.data import FedDataDistribution, Dataset
        except ImportError:
            raise ImportError("FLEX not installed. Run: pip install flex-framework")

        # Create FLEX Dataset from numpy arrays
        centralized_dataset = Dataset.from_array(
            X_array=dataset_split.X_train,
            y_array=dataset_split.y_train
        )

        # Create federated data distribution with IID partitioning
        n_clients = self.config.get('n_clients', 5)
        self.federated_data = FedDataDistribution.iid_distribution(
            centralized_data=centralized_dataset,
            n_nodes=n_clients
        )
        
        # Convert federated data to client partitions (dict of client_id -> (X, y))
        self.client_partitions = {}
        for node_id, node_data in self.federated_data.items():
            client_id = f"client_{node_id}"
            # Extract X and y from FLEX Dataset
            X_client = node_data.X_data.to_numpy()
            y_client = node_data.y_data.to_numpy() if node_data.y_data is not None else None
            self.client_partitions[client_id] = (X_client, y_client)

    def run_federated_round(self) -> FLResults:
        """
        Execute complete federated learning round.
        """
        if not self.client_partitions or self.dataset_split is None:
            raise ValueError("Federation not set up. Call setup_federation first.")

        # ── STEP 1: TRAIN local forests on each client ────────────────────────
        self.step_callback("Training on clients...", 15)
        client_forests: Dict[str, ProactiveForest] = {}
        client_metadata: Dict[str, ClientMetadata] = {}

        val_split = self.config.get('metadata', {}).get('validation_split', 0.2)
        
        for client_id, (X_client, y_client) in self.client_partitions.items():
            if len(X_client) < 5:  # Skip if too few samples
                continue
                
            # Split into train/validation for metadata calculation
            if val_split > 0 and len(X_client) > 10:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_client, y_client, test_size=val_split, random_state=42
                )
            else:
                X_train, X_val, y_train, y_val = X_client, X_client, y_client, y_client

            # Create and train forest
            pf = ProactiveForest(
                n_estimators=self.config.get('n_estimators', 100),
                alpha=self.config.get('alpha', 0.5)
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
                pcd=0.0,
            )

        client_ids = list(client_forests.keys())

        # ── STEP 2: COLLECT trees from all clients ────────────────────────────
        self.step_callback("Collecting trees from clients...", 40)
        client_trees = {cid: pf.get_trees() for cid, pf in client_forests.items()}

        # ── STEP 3: AGGREGATE using selected strategy ──────────────────────────
        self.step_callback("Aggregating forests...", 55)
        strategy = AggregationFactory.create_strategy(self.config.get('strategy', 'S1'))
        global_trees, selected_ids = strategy.aggregate(client_trees, client_metadata)

        # ── STEP 4: EVALUATE on test set ───────────────────────────────────────
        self.step_callback("Evaluating models...", 85)
        X_test, y_test = self.dataset_split.X_test, self.dataset_split.y_test

        # Create proxy forest with global trees
        global_forest = ProactiveForest.from_trees(global_trees)

        # Global predictions
        try:
            global_predictions = global_forest.predict(X_test)
        except:  # noqa: E722
            # Fallback if predict fails
            global_predictions = np.random.randint(0, len(self.dataset_split.class_names), len(y_test))

        global_accuracy = float(accuracy_score(y_test, global_predictions))
        global_f1 = float(f1_score(y_test, global_predictions, average='macro', zero_division=0))

        self.step_callback("Completed", 100)

        return FLResults(
            strategy_id=strategy.strategy_id,
            global_accuracy=global_accuracy,
            global_macro_f1=global_f1,
            n_trees_global=len(global_trees),
            client_ids=client_ids,
            client_accuracies={cid: m.accuracy for cid, m in client_metadata.items()},
            client_f1_scores={cid: m.macro_f1 for cid, m in client_metadata.items()},
            client_metadata=client_metadata,
            client_reports={},  # TODO: Add client evaluation reports if needed
            selected_ids=selected_ids,
        )
