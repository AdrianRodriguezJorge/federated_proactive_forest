"""Tree metric extractor service.

Calculates individual tree performance scores (accuracy, macro F1, Percentage
Correct Diversity) against validation datasets prior to ensemble selection.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src.domain.metrics.metrics_service import IDiversityService


class TreeMetricExtractor:
    """Extracts or calculates validation metrics for individual trees.

    Used before ranking to determine each tree's accuracy, macro-F1, and PCD.
    """

    def __init__(
        self, diversity_service: Optional[IDiversityService] = None
    ):
        """Initializes TreeMetricExtractor.

        Args:
            diversity_service (Optional[IDiversityService]): Diversity service.
        """
        self.diversity_service = diversity_service

    def extract_metrics(
        self,
        client_trees: Dict[str, List[Any]],
        client_metadata: Dict[str, Any],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """Extract evaluation scores for candidate trees.

        Args:
            client_trees (Dict[str, List[Any]]): Client trees dict.
            client_metadata (Dict[str, Any]): Client metadata dict.
            X_val (Optional[np.ndarray]): Validation features.
            y_val (Optional[np.ndarray]): Validation labels.

        Returns:
            List[Dict[str, Any]]: Extracted stats for each tree.
        """
        raw_entries = []
        for cid, trees in client_trees.items():
            meta = client_metadata[cid]

            predictions_matrix = None
            if X_val is not None:
                predictions_matrix = np.empty(
                    (X_val.shape[0], len(trees)), dtype=object
                )
                for i, tree in enumerate(trees):
                    predictions_matrix[:, i] = tree.predict(X_val)

            for local_id, tree in enumerate(trees):
                # 1. Standardized Server-Side Evaluation
                if X_val is not None and y_val is not None:
                    preds = predictions_matrix[:, local_id]
                    preds_str = np.asarray(preds, dtype=str)
                    y_val_str = np.asarray(y_val, dtype=str)

                    tree_acc = float(accuracy_score(y_val_str, preds_str))
                    tree_f1 = float(
                        f1_score(
                            y_val_str,
                            preds_str,
                            average="macro",
                            zero_division=0,
                        )
                    )
                else:
                    tree_acc = 0.0
                    tree_f1 = 0.0

                # 2. Local PCD / Marginal PCD calculation
                tree_pcd = 0.0
                if (
                    predictions_matrix is not None
                    and X_val is not None
                    and y_val is not None
                ):
                    y_val_reshaped = np.asarray(y_val).reshape(-1, 1)
                    hits_matrix = predictions_matrix == y_val_reshaped
                    hits_per_sample = np.sum(hits_matrix, axis=1)

                    n_trees = predictions_matrix.shape[1]
                    lower = 0.1 * n_trees
                    upper = 0.9 * n_trees

                    diverse_samples = np.sum(
                        (hits_per_sample >= lower) & (hits_per_sample <= upper)
                    )
                    tree_pcd = float(diverse_samples / X_val.shape[0])

                raw_entries.append(
                    {
                        "tree": tree,
                        "client_id": cid,
                        "tree_local_id": local_id,
                        "accuracy": tree_acc,
                        "macro_f1": tree_f1,
                        "pcd": tree_pcd,
                    }
                )
        return raw_entries
