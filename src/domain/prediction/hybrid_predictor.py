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
            
            # Map class names to indices once outside the loops
            # If label_svc is present, we use it for more robust mapping
            if self.label_svc:
                # We'll use the service's internal mapping if possible
                # SimpleLabelService uses _encoder
                class_to_idx = {str(name): i for name, i in self.label_svc._encoder.items()}
            else:
                class_to_idx = {str(name): i for i, name in enumerate(self.class_names)}
            
            for tree in trees:
                # Use batch prediction which returns array of predictions
                preds = tree.predict(X)

                # Normalize preds to integer indices
                # Trees store integer class indices in leaves (from np.argmax)
                # The tree.predict returns dtype=object array containing numpy integers
                
                # Check if predictions are strings/objects that need mapping
                if preds.dtype.kind in {'U', 'S'} or (preds.dtype == 'O' and isinstance(preds.flat[0] if len(preds) > 0 else None, str)):
                    # String predictions - need to map to indices
                    mapped_preds = []
                    for p in preds:
                        p_str = str(p)
                        idx = class_to_idx.get(p_str, -1)
                        # Handle potential float-as-string issues (e.g., "1.0" -> "1")
                        if idx == -1 and "." in p_str:
                            try:
                                p_int_str = str(int(float(p_str)))
                                idx = class_to_idx.get(p_int_str, -1)
                            except ValueError:
                                pass
                        mapped_preds.append(idx)
                    preds = np.array(mapped_preds, dtype=np.int64)
                else:
                    # Predictions are already numeric (numpy integers in object array)
                    # Convert to proper int64 array
                    try:
                        preds = np.array([int(p) for p in preds], dtype=np.int64)
                    except (ValueError, TypeError):
                        # Fallback: try direct conversion
                        preds = np.asarray(preds, dtype=np.int64)

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
