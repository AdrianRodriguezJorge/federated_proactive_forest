"""
Inferencia híbrida ponderada: voto mayoritario entre árboles locales y globales.
Usa directamente los DecisionTree del paquete proactive_forest.
"""
import numpy as np
from typing import Any, List


class HybridPredictor:
    """Weighted hybrid prediction logic.
    
    Combines predictions from local and global models using a weighted voting scheme:
    score = local_weight * local_votes + global_weight * global_votes.
    """

    def __init__(self, local_weight: float = 0.4, global_weight: float = 0.6,
                 n_classes: int = 2, class_names: List[str] = None,
                 label_service: Any = None,
                 use_weighted: bool = True):
        assert abs(local_weight + global_weight - 1.0) < 1e-6, \
            "local_weight + global_weight debe ser 1.0"
        self.lw = local_weight
        self.gw = global_weight
        self.n_classes = n_classes
        self.class_names = class_names or [str(i) for i in range(n_classes)]
        self.label_svc = label_service
        self.use_weighted = use_weighted

    def predict(self, X: np.ndarray,
                local_trees: List[Any],
                global_trees: List[Any]) -> np.ndarray:
        """
        Perform prediction using the HybridForest domain model.
        """
        forest = self.create_forest(local_trees, global_trees)
        return forest.predict(X)

    def create_forest(self, local_trees: List[Any], global_trees: List[Any]) -> Any:
        """
        Factory method to create a HybridForest instance with current configuration.
        """
        from src.domain.model.hybrid_forest import HybridForest
        
        # Determine effective weights based on use_weighted flag
        if self.use_weighted:
            lw, gw = self.lw, self.gw
        else:
            total = len(local_trees) + len(global_trees)
            if total > 0:
                lw, gw = len(local_trees) / total, len(global_trees) / total
            else:
                lw, gw = 0.0, 0.0

        return HybridForest(
            local_trees=local_trees,
            global_trees=global_trees,
            local_weight=lw,
            global_weight=gw,
            n_classes=self.n_classes,
            class_names=self.class_names,
            label_service=self.label_svc
        )

    def get_debug_stats(self) -> dict:
        # Note: Debug stats might need to be moved to HybridForest if still required
        return {}
