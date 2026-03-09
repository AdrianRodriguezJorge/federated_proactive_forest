"""
Inferencia híbrida ponderada: voto mayoritario entre árboles locales y globales.
Usa directamente los DecisionTree del paquete proactive_forest.
"""
import numpy as np
from typing import Any, List


class HybridPredictor:
    """
    Predicción combinada ponderada: local_weight·local + global_weight·global.
    local_weight + global_weight debe ser 1.0.
    """

    def __init__(self, local_weight: float = 0.4, global_weight: float = 0.6,
                 n_classes: int = 2):
        assert abs(local_weight + global_weight - 1.0) < 1e-6, \
            "local_weight + global_weight debe ser 1.0"
        self.lw = local_weight
        self.gw = global_weight
        self.n_classes = n_classes

    def predict(self, X: np.ndarray,
                local_trees: List[Any],
                global_trees: List[Any]) -> np.ndarray:
        n_samples = X.shape[0]
        combined = np.zeros((n_samples, self.n_classes))

        def accumulate(trees, weight):
            if not trees:
                return
            w_per_tree = weight / len(trees)
            for tree in trees:
                for i in range(n_samples):
                    pred = tree.predict(X[i])
                    combined[i, pred] += w_per_tree

        accumulate(local_trees, self.lw)
        accumulate(global_trees, self.gw)
        return np.argmax(combined, axis=1)
