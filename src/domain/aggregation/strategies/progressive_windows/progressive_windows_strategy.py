"""Progressive Windows Strategy.

Implementation of the Progressive Windows (PW) aggregation strategy
for federated Proactive Forest.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import f1_score

from src.domain.aggregation.base_strategy import IAggregationStrategy
from src.domain.aggregation.tree_ranker import TreeEntry
from src.domain.metrics.metrics_service import IDiversityService, IMetricsService
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
    """Progressive Windows aggregation strategy."""

    DEFAULT_WINDOW_SIZE = 5  # W: trees per window
    DEFAULT_MAX_ROUNDS = 20  # Maximum rounds (R_MAX)
    DEFAULT_CONVERGENCE_THRESHOLD = 0.002  # Convergence threshold
    DEFAULT_F1_WEIGHT = 0.5  # Balance between F1 and Diversity (α)
    DEFAULT_LOCAL_WEIGHT = 0.5  # Hybrid prediction weight (local vs global)

    def __init__(
        self,
        metrics_service: Optional[IMetricsService] = None,
        diversity_service: Optional[IDiversityService] = None,
        window_size: int = DEFAULT_WINDOW_SIZE,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD,
        f1_weight: float = DEFAULT_F1_WEIGHT,
        local_weight: float = DEFAULT_LOCAL_WEIGHT,
        verbose: bool = False,
    ):
        """Initializes ProgressiveWindowsStrategy.

        Args:
            metrics_service (Optional[IMetricsService]): Performance metrics.
            diversity_service (Optional[IDiversityService]): Diversity metrics.
            window_size (int): Windows size (W).
            max_rounds (int): Maximum rounds limit.
            convergence_threshold (float): Improvement threshold.
            f1_weight (float): Alpha weight for F1.
            local_weight (float): Hybrid local weight.
            verbose (bool): Print debug logs.
        """
        super().__init__(
            metrics_service=metrics_service,
            diversity_service=diversity_service,
        )
        self.window_size = window_size
        self.max_rounds = max_rounds
        self.convergence_threshold = convergence_threshold
        self.f1_weight = f1_weight
        self.local_weight = local_weight
        self.verbose = verbose

        self._global_trees: List[Any] = []
        self._global_tree_sources: List[str] = []
        self.convergence_round: Optional[int] = None

    @property
    def strategy_id(self) -> str:
        """Get unique strategy ID.

        Returns:
            str: "PW"
        """
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
        **kwargs: Any,
    ) -> Tuple[
        List[Any],
        Dict[str, List[int]],
        List[Any],
        Optional[int],
        List[Dict[str, Any]],
    ]:
        """Aggregate candidate windows progressively under round-robin permutation.

        Args:
            client_trees (Dict[str, List[Any]]): Client new windows.
            client_metadata (Dict[str, Any]): Client metadata dictionary.
            X_val (Optional[np.ndarray]): Validation features.
            y_val (Optional[np.ndarray]): Validation labels.
            max_trees (Optional[int]): Total max trees fallback.
            max_trees_per_client (Optional[int]): Max trees per client.
            t_max (Optional[int]): Maximum rounds limit (t_max).
            local_weight (Optional[float]): Local weight override.
            **kwargs (Any): Additional parameters.

        Returns:
            Tuple: A tuple containing:
                - List[Any]: Progressively built global trees.
                - Dict[str, List[int]]: Selected client indices.
                - List[Any]: All tree entries.
                - Optional[int]: Convergence round.
                - List[Dict[str, Any]]: Performance logs.

        Raises:
            ValueError: If validation data is missing.
        """
        self.f1_weight = kwargs.get("f1_weight", self.f1_weight)
        self.window_size = kwargs.get("window_size", self.window_size)
        self.max_rounds = kwargs.get("max_rounds", self.max_rounds)
        self.convergence_threshold = kwargs.get(
            "global_convergence_threshold", self.convergence_threshold
        )
        if local_weight is not None:
            self.local_weight = local_weight

        if t_max is not None:
            n_clients = len(client_trees)
            if n_clients > 0:
                self.max_rounds = max(1, t_max // n_clients)

        client_ids = list(client_trees.keys())
        n_clients = len(client_ids)
        if not client_ids:
            return [], {}, [], None, []

        result = ProgressiveWindowsResult()
        result.selected_ids = {cid: [] for cid in client_ids}
        self.convergence_round = None

        all_tree_entries: List[TreeEntry] = []
        global_correct_counts = None
        y_val_encoded = None

        if X_val is None or y_val is None:
            raise ValueError(
                "CRITICAL: Server-side validation dataset (X_val, y_val) "
                "is required and mandatory for Progressive Windows Strategy."
            )

        n_val_samples = X_val.shape[0]
        global_correct_counts = np.zeros(n_val_samples, dtype=int)
        class_names = kwargs.get("class_names", [])
        label_service = SimpleLabelService(class_names) if class_names else None
        if label_service:
            y_val_encoded = label_service.transform(y_val)
        else:
            y_val_encoded = y_val

        current_global_trees = kwargs.get("current_global_trees", [])
        current_round = kwargs.get("current_round", 0)

        self._global_trees = list(current_global_trees)
        result.global_trees = list(current_global_trees)
        self._global_tree_sources = []

        # Reconstruct global_correct_counts based on existing selected trees
        for tree in self._global_trees:
            best_preds_raw = tree.predict(X_val)
            if label_service:
                best_preds_int = label_service.transform(best_preds_raw)
            else:
                best_preds_int = best_preds_raw
            global_correct_counts += (best_preds_int == y_val_encoded).astype(
                int
            )

        rng = np.random.RandomState(seed=current_round)
        round_permutation = rng.permutation(client_ids).tolist()

        for client_idx, client_id in enumerate(round_permutation):
            window_trees = client_trees[client_id]
            tree_scores: List[Tuple[int, Any, float, float, float]] = []

            for local_idx, tree in enumerate(window_trees):
                global_idx = local_idx
                cand_preds_raw = tree.predict(X_val)
                if label_service:
                    cand_preds_norm = label_service.transform(cand_preds_raw)
                else:
                    cand_preds_norm = cand_preds_raw

                tree_f1 = float(
                    f1_score(
                        y_val_encoded,
                        cand_preds_norm,
                        average="macro",
                        zero_division=0,
                    )
                )

                if self.diversity_svc is not None:
                    diversity = self.diversity_svc.calculate_marginal_pcd(
                        candidate_predictions=cand_preds_norm,
                        current_hits_per_sample=global_correct_counts,
                        n_existing_trees=len(self._global_trees),
                        y_true=y_val_encoded,
                    )
                else:
                    diversity = 1.0

                effective_f1_weight = (
                    self.f1_weight if len(self._global_trees) > 0 else 1.0
                )
                pcd_weight = 1.0 - effective_f1_weight
                score = (
                    effective_f1_weight * tree_f1 + pcd_weight * diversity
                )
                tree_scores.append(
                    (global_idx, tree, score, tree_f1, diversity)
                )

                all_tree_entries.append(
                    TreeEntry(
                        tree=tree,
                        client_id=client_id,
                        tree_local_id=global_idx,
                        accuracy=0.0,
                        macro_f1=tree_f1,
                        pcd=diversity,
                    )
                )

            if not tree_scores:
                continue

            (
                best_local_idx,
                best_tree,
                best_score,
                best_f1,
                best_diversity,
            ) = max(tree_scores, key=lambda x: x[2])
            self._global_trees.append(best_tree)
            self._global_tree_sources.append(client_id)
            result.global_trees.append(best_tree)
            result.selected_ids[client_id].append(best_local_idx)

            best_preds_raw = best_tree.predict(X_val)
            if label_service:
                best_preds_int = label_service.transform(best_preds_raw)
            else:
                best_preds_int = best_preds_raw
            global_correct_counts += (best_preds_int == y_val_encoded).astype(
                int
            )

        result.rounds_completed = current_round + 1

        if len(self._global_trees) > 0:
            metrics_svc = self.metrics_svc or kwargs.get("metrics_service")
            preds_raw = self._predict_forest_batch(self._global_trees, X_val)
            if label_service:
                preds = label_service.transform(preds_raw)
            else:
                preds = preds_raw

            if metrics_svc:
                current_accuracy = float(
                    metrics_svc.accuracy_score(y_val_encoded, preds)
                )
                current_f1 = float(
                    metrics_svc.f1_score(
                        y_val_encoded, preds, average="macro"
                    )
                )
            else:
                current_accuracy = float(np.mean(preds == y_val_encoded))
                current_f1 = 0.0

            result.round_logs.append(
                {
                    "episode": current_round + 1,
                    "n_trees": len(self._global_trees),
                    "accuracy": current_accuracy,
                    "macro_f1": current_f1,
                }
            )
            result.final_accuracy = current_accuracy
            result.convergence_round = current_round + 1

        if self.convergence_round is None:
            self.convergence_round = result.rounds_completed

        metrics_svc = self.metrics_svc or kwargs.get("metrics_service")
        result.final_macro_f1 = self._evaluate_forest_f1(
            self._global_trees,
            X_val,
            y_val_encoded,
            metrics_svc=metrics_svc,
            label_service=label_service,
            **kwargs,
        )

        return (
            result.global_trees,
            result.selected_ids,
            all_tree_entries,
            result.convergence_round,
            result.round_logs,
        )

    def _predict_forest_batch(
        self, trees: List[Any], X: np.ndarray
    ) -> np.ndarray:
        """Consolidates predictions across the forest using modes."""
        from src.domain.prediction.voting import calculate_mode

        n_samples = X.shape[0]
        all_predictions = np.empty((n_samples, len(trees)), dtype=object)
        for j, tree in enumerate(trees):
            all_predictions[:, j] = tree.predict(X)
        return calculate_mode(all_predictions, axis=1)

    def _evaluate_forest_f1(
        self,
        trees: List[Any],
        X: np.ndarray,
        y_encoded: np.ndarray,
        metrics_svc: Optional[IMetricsService] = None,
        label_service: Optional[SimpleLabelService] = None,
        **kwargs: Any,
    ) -> float:
        """Helper to get Macro-F1 score of the forest ensemble."""
        if not trees:
            return 0.0
        preds_raw = self._predict_forest_batch(trees, X)
        if label_service:
            preds = label_service.transform(preds_raw)
        else:
            preds = preds_raw
        if metrics_svc:
            return float(
                metrics_svc.f1_score(y_encoded, preds, average="macro")
            )
        return float(np.mean(preds == y_encoded))

    def get_global_trees(self) -> List[Any]:
        """Return global trees.

        Returns:
            List[Any]: Trees.
        """
        return self._global_trees

    def get_global_tree_sources(self) -> List[str]:
        """Return sources of trees.

        Returns:
            List[str]: Client IDs.
        """
        return self._global_tree_sources
