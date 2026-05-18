"""Weighted hybrid prediction logic wrapper.

Combines predictions from local and global models using a weighted voting scheme:
score = local_weight * local_votes + global_weight * global_votes.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from src.domain.model.hybrid_forest import HybridForest


class HybridPredictor:
    """Weighted hybrid prediction coordinator.

    Manages creation and execution of the HybridForest domain model under
    specified client/server weighting.
    """

    def __init__(
        self,
        local_weight: float = 0.4,
        global_weight: float = 0.6,
        n_classes: int = 2,
        class_names: Optional[List[str]] = None,
        label_service: Optional[Any] = None,
        use_weighted: bool = True,
    ):
        """Initializes the Hybrid Predictor.

        Args:
            local_weight (float): Local estimators combined voting weight.
            global_weight (float): Global server-federated voting weight.
            n_classes (int): Class dimension count.
            class_names (Optional[List[str]]): Target classes name mapping.
            label_service (Optional[Any]): String encoder transform service.
            use_weighted (bool): Whether to apply custom weights or simple
                unweighted proportions.

        Raises:
            AssertionError: If custom weights do not sum up to 1.0.
        """
        assert abs(local_weight + global_weight - 1.0) < 1e-6, (
            "local_weight + global_weight must sum to 1.0"
        )
        self.lw = local_weight
        self.gw = global_weight
        self.n_classes = n_classes
        self.class_names = class_names or [str(i) for i in range(n_classes)]
        self.label_svc = label_service
        self.use_weighted = use_weighted

    def predict(
        self, X: np.ndarray, local_trees: List[Any], global_trees: List[Any]
    ) -> np.ndarray:
        """Perform prediction using the HybridForest domain model.

        Args:
            X (np.ndarray): Data features matrix.
            local_trees (List[Any]): Local estimators.
            global_trees (List[Any]): Global server-federated estimators.

        Returns:
            np.ndarray: Predicted label indices.
        """
        forest = self.create_forest(local_trees, global_trees)
        return forest.predict(X)

    def create_forest(
        self, local_trees: List[Any], global_trees: List[Any]
    ) -> HybridForest:
        """Factory method to create a HybridForest instance.

        Args:
            local_trees (List[Any]): Local estimators list.
            global_trees (List[Any]): Global federated estimators list.

        Returns:
            HybridForest: Configuration-matched hybrid ensemble model.
        """
        if self.use_weighted:
            lw, gw = self.lw, self.gw
        else:
            total = len(local_trees) + len(global_trees)
            if total > 0:
                lw = len(local_trees) / total
                gw = len(global_trees) / total
            else:
                lw, gw = 0.0, 0.0

        return HybridForest(
            local_trees=local_trees,
            global_trees=global_trees,
            local_weight=lw,
            global_weight=gw,
            n_classes=self.n_classes,
            class_names=self.class_names,
            label_service=self.label_svc,
        )

    def get_debug_stats(self) -> Dict[str, Any]:
        """Get predictor debug stats.

        Returns:
            Dict[str, Any]: Empty debug status dict.
        """
        return {}
