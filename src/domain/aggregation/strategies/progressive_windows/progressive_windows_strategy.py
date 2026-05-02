"""Progressive Windows Strategy.

Implementation of the Progressive Windows aggregation strategy
for federated Proactive Forest.
"""
from __future__ import annotations
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass, field

from ...base_strategy import IAggregationStrategy
from ...tree_ranker import TreeEntry
from ....model.cpf_implementation.estimator import ProactiveForestClassifier
from ....model.cpf_implementation.tree import DecisionTree
from src.domain.metrics.metrics_service import IMetricsService, IDiversityService
from src.domain.services.label_service import SimpleLabelService


@dataclass
class WindowReport:
    """Report for a window of trees from a client."""
    client_id: str
    window_trees: List[Any]
    window_f1_scores: List[float]
    window_pcd_scores: List[float]
    accuracy: float
    macro_f1: float


@dataclass
class ProgressiveWindowsResult:
    """Result of Progressive Windows selection process."""
    global_trees: List[Any] = field(default_factory=list)
    selected_ids: Dict[str, List[int]] = field(default_factory=dict)
    all_tree_entries: List[TreeEntry] = field(default_factory=list)
    rounds_completed: int = 0
    convergence_round: Optional[int] = None
    final_accuracy: float = 0.0
    final_macro_f1: float = 0.0
    round_logs: List[Dict[str, Any]] = field(default_factory=list)


class ProgressiveWindowsStrategy(IAggregationStrategy):
    """
    Progressive Windows aggregation strategy.
    """

    DEFAULT_WINDOW_SIZE = 5  # W: trees per window
    DEFAULT_MAX_ROUNDS = 20  # Maximum rounds (R_MAX)
    DEFAULT_CONVERGENCE_THRESHOLD = 0.002  # Convergence threshold
    DEFAULT_F1_WEIGHT = 0.5  # Balance between F1 and Diversity (α)
    DEFAULT_LOCAL_WEIGHT = 0.5  # Hybrid prediction weight (local vs global)

    def __init__(self, 
                 metrics_service: Optional[IMetricsService] = None,
                 diversity_service: Optional[IDiversityService] = None,
                 window_size: int = DEFAULT_WINDOW_SIZE,
                 max_rounds: int = DEFAULT_MAX_ROUNDS,
                 convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD,
                 f1_weight: float = DEFAULT_F1_WEIGHT,
                 local_weight: float = DEFAULT_LOCAL_WEIGHT,
                 verbose: bool = False):
        self.metrics_svc = metrics_service
        self.diversity_svc = diversity_service
        self.window_size = window_size
        self.max_rounds = max_rounds
        self.convergence_threshold = convergence_threshold
        self.f1_weight = f1_weight
        self.local_weight = local_weight
        self.verbose = verbose

        # Track global forest state
        self._global_trees: List[Any] = []
        self._global_tree_sources: List[str] = []
        self.convergence_round: Optional[int] = None

    @property
    def strategy_id(self) -> str:
        return "PW"

    def aggregate(
        self,
        client_trees: Dict[str, List[Any]],
        client_metadata: Dict[str, Any],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        max_trees: Optional[int] = None,
        max_trees_per_client: Optional[int] = None,
        t_max: Optional[int] = None,
        local_weight: Optional[float] = None,
        **kwargs
    ) -> Tuple[List[Any], Dict[str, List[int]], List[Any], Optional[int], List[Dict]]:
        # Override defaults with kwargs
        self.f1_weight = kwargs.get('f1_weight', self.f1_weight)
        self.window_size = kwargs.get('window_size', self.window_size)
        self.max_rounds = kwargs.get('max_rounds', self.max_rounds)
        if local_weight is not None:
            self.local_weight = local_weight

        if t_max is not None:
            n_clients = len(client_trees)
            self.max_rounds = max(1, t_max // n_clients) if n_clients > 0 else self.max_rounds

        client_ids = list(client_trees.keys())
        n_clients = len(client_ids)
        if not client_ids:
            return [], {}, [], None, []

        result = ProgressiveWindowsResult()
        result.selected_ids = {cid: [] for cid in client_ids}
        self._global_trees = []
        self._global_tree_sources = []
        self.convergence_round = None

        all_tree_entries: List[TreeEntry] = []
        global_correct_counts = None
        n_val_samples = 0
        y_val_encoded = None
        
        if X_val is not None and y_val is not None:
            n_val_samples = X_val.shape[0]
            global_correct_counts = np.zeros(n_val_samples, dtype=int)
            class_names = kwargs.get('class_names', [])
            label_service = SimpleLabelService(class_names) if class_names else None
            y_val_encoded = label_service.transform(y_val) if label_service else y_val

        round_accuracies: List[float] = []
        stop_counter = 0
        previous_accuracy = None

        for round_num in range(self.max_rounds):
            rng = np.random.RandomState(seed=round_num)
            round_permutation = rng.permutation(client_ids).tolist()

            for client_idx, client_id in enumerate(round_permutation):
                client_all_trees = client_trees[client_id]
                window_start = round_num * self.window_size
                window_end = min(window_start + self.window_size, len(client_all_trees))

                if window_start >= len(client_all_trees):
                    continue

                window_trees = client_all_trees[window_start:window_end]
                client_meta = client_metadata.get(client_id, {})
                if hasattr(client_meta, 'to_dict'):
                    client_meta_dict = client_meta.to_dict()
                elif isinstance(client_meta, dict):
                    client_meta_dict = client_meta
                else:
                    client_meta_dict = {}

                tree_scores: List[Tuple[int, Any, float, float, float]] = []

                for local_idx, tree in enumerate(window_trees):
                    global_idx = window_start + local_idx
                    tree_metrics = client_meta_dict.get('tree_metrics', [])
                    if global_idx < len(tree_metrics):
                        tree_f1 = tree_metrics[global_idx].get('macro_f1', 0.5)
                    else:
                        tree_f1 = client_meta_dict.get('macro_f1', 0.5)

                    if X_val is not None and y_val_encoded is not None:
                        candidate_preds_raw = tree.predict(X_val)
                        candidate_preds_int = label_service.transform(candidate_preds_raw) if label_service else candidate_preds_raw
                        is_correct = (candidate_preds_int == y_val_encoded)
                        
                        candidate_correct_counts = global_correct_counts + is_correct.astype(int)
                        total_predictors_temp = len(self._global_trees) + 1
                        lower_bound = 0.1 * total_predictors_temp
                        upper_bound = 0.9 * total_predictors_temp
                        diverse_instances = np.sum((candidate_correct_counts >= lower_bound) & (candidate_correct_counts <= upper_bound))
                        diversity = diverse_instances / n_val_samples
                    else:
                        diversity = 1.0 # Default if no validation data

                    effective_f1_weight = self.f1_weight if len(self._global_trees) > 0 else 1.0
                    pcd_weight = 1.0 - effective_f1_weight
                    score = effective_f1_weight * tree_f1 + pcd_weight * diversity
                    tree_scores.append((global_idx, tree, score, tree_f1, diversity))

                    all_tree_entries.append(TreeEntry(
                        tree=tree, client_id=client_id, tree_local_id=global_idx,
                        accuracy=client_meta_dict.get('accuracy', 0.0), macro_f1=tree_f1, pcd=diversity
                    ))

                if not tree_scores:
                    continue

                best_local_idx, best_tree, best_score, best_f1, best_diversity = max(tree_scores, key=lambda x: x[2])
                self._global_trees.append(best_tree)
                self._global_tree_sources.append(client_id)
                result.global_trees.append(best_tree)
                result.selected_ids[client_id].append(best_local_idx)

                if X_val is not None and y_val_encoded is not None:
                    best_preds_raw = best_tree.predict(X_val)
                    best_preds_int = label_service.transform(best_preds_raw) if label_service else best_preds_raw
                    global_correct_counts += (best_preds_int == y_val_encoded).astype(int)

            result.rounds_completed = round_num + 1

            if X_val is not None and y_val is not None and len(self._global_trees) > 0:
                metrics_svc = self.metrics_svc or kwargs.get('metrics_service')
                preds_raw = self._predict_forest_batch(self._global_trees, X_val)
                preds = label_service.transform(preds_raw) if label_service else preds_raw
                current_accuracy = float(metrics_svc.accuracy_score(y_val_encoded, preds)) if metrics_svc else float(np.mean(preds == y_val_encoded))
                current_f1 = float(metrics_svc.f1_score(y_val_encoded, preds, average='macro')) if metrics_svc else 0.0
                
                round_accuracies.append(current_accuracy)
                result.round_logs.append({
                    'episode': round_num + 1,
                    'n_trees': len(self._global_trees),
                    'accuracy': current_accuracy,
                    'macro_f1': current_f1
                })

                if previous_accuracy is not None:
                    accuracy_improvement = current_accuracy - previous_accuracy
                    if accuracy_improvement <= self.convergence_threshold:
                        stop_counter += 1
                        if stop_counter >= 2:
                            result.convergence_round = round_num + 1
                            self.convergence_round = round_num + 1
                            break
                    else:
                        stop_counter = 0

                previous_accuracy = current_accuracy
                result.final_accuracy = current_accuracy

        if self.convergence_round is None:
            self.convergence_round = result.rounds_completed
        if X_val is not None and y_val is not None:
            result.final_macro_f1 = self._evaluate_forest_f1(self._global_trees, X_val, y_val_encoded, metrics_svc=metrics_svc, label_service=label_service, **kwargs)

        return result.global_trees, result.selected_ids, all_tree_entries, result.convergence_round, result.round_logs

    def _predict_forest_batch(self, trees: List[Any], X: np.ndarray) -> np.ndarray:
        from src.domain.prediction.voting import calculate_mode
        n_samples = X.shape[0]
        # Use object dtype to handle mixed string/numeric predictions before mode
        all_predictions = np.empty((n_samples, len(trees)), dtype=object)
        for j, tree in enumerate(trees):
            all_predictions[:, j] = tree.predict(X)
        return calculate_mode(all_predictions, axis=1)

    def _evaluate_forest_f1(self, trees: List[Any], X: np.ndarray, y_encoded: np.ndarray, 
                            metrics_svc: Optional[IMetricsService] = None, 
                            label_service: Optional[SimpleLabelService] = None, **kwargs) -> float:
        if not trees: return 0.0
        preds_raw = self._predict_forest_batch(trees, X)
        preds = label_service.transform(preds_raw) if label_service else preds_raw
        if metrics_svc:
            return float(metrics_svc.f1_score(y_encoded, preds, average='macro'))
        return float(np.mean(preds == y_encoded))

    def get_global_trees(self) -> List[Any]:
        return self._global_trees

    def get_global_tree_sources(self) -> List[str]:
        return self._global_tree_sources
