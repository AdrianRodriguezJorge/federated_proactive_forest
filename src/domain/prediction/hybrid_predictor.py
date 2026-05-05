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
                 label_service: Any = None):
        assert abs(local_weight + global_weight - 1.0) < 1e-6, \
            "local_weight + global_weight debe ser 1.0"
        self.lw = local_weight
        self.gw = global_weight
        self.n_classes = n_classes
        self.class_names = class_names or [str(i) for i in range(n_classes)]
        self.label_svc = label_service

    def predict(self, X: np.ndarray,
                local_trees: List[Any],
                global_trees: List[Any]) -> np.ndarray:
        n_samples = X.shape[0]
        # Probabilities matrix for each sample and class
        combined = np.zeros((n_samples, self.n_classes))
        
        # Track votes for debugging
        self._last_votes = {
            'local': np.zeros((n_samples, self.n_classes)),
            'global': np.zeros((n_samples, self.n_classes))
        }

        def accumulate_vectorized(trees, weight, source_key):
            if not trees:
                return
            w_per_tree = weight / len(trees)
            
            # No manual mapping needed, we use LabelService.transform for robustness

            
            for tree in trees:
                # Use batch prediction which returns array of predictions (labels)
                preds_raw = tree.predict(X)

                # Map labels to unified indices using LabelService if available
                if self.label_svc:
                    preds = self.label_svc.transform(preds_raw)
                else:
                    # Fallback if no service: try to convert to numeric directly
                    try:
                        preds = np.asarray(preds_raw, dtype=np.int64)
                    except (ValueError, TypeError):
                        # If strings and no service, we might have issues, but this shouldn't happen in our current architecture
                        preds = np.zeros(n_samples, dtype=np.int64)



                # Validate indices
                valid_mask = (preds >= 0) & (preds < self.n_classes)

                # Report mapping failures if any
                num_invalid = np.sum(~valid_mask)
                if num_invalid > 0:
                    import logging
                    logger = logging.getLogger("HybridPredictor")
                    logger.warning(f"Found {num_invalid} invalid predictions in tree from source {source_key}. "
                                   f"Sample of invalid values: {preds[~valid_mask][:5]}")

                # Vectorized accumulation
                for c in range(self.n_classes):
                    mask = (preds == c) & valid_mask
                    combined[mask, c] += w_per_tree
                    self._last_votes[source_key][mask, c] += w_per_tree

        accumulate_vectorized(local_trees, self.lw, 'local')
        accumulate_vectorized(global_trees, self.gw, 'global')
        return np.argmax(combined, axis=1)

    def get_debug_stats(self) -> dict:
        """Return statistics about the last prediction call."""
        if not hasattr(self, '_last_votes'):
            return {}
        
        return {
            'local_vote_sum': self._last_votes['local'].sum(axis=0).tolist(),
            'global_vote_sum': self._last_votes['global'].sum(axis=0).tolist(),
            'n_classes': self.n_classes,
            'class_names': self.class_names
        }
