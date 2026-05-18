"""Hybrid Forest domain model.

Represents an ensemble of client-specific local trees combined with federated
global trees, supporting weighted classification and ensemble diversity scores.
"""

from typing import Any, List, Optional
import numpy as np


class HybridForest:
    """Domain model representing a hybrid ensemble of local and global trees.

    This class encapsulates the state of a client's final model after federation
    and provides methods for both inference and diversity evaluation.
    """

    def __init__(
        self,
        local_trees: List[Any],
        global_trees: List[Any],
        local_weight: float = 0.4,
        global_weight: float = 0.6,
        n_classes: int = 2,
        class_names: Optional[List[str]] = None,
        label_service: Optional[Any] = None,
    ):
        """Initializes the Hybrid Forest.

        Args:
            local_trees (List[Any]): Trees trained locally by the client.
            global_trees (List[Any]): Trees received from the server.
            local_weight (float): Weight assigned to local trees.
            global_weight (float): Weight assigned to global trees.
            n_classes (int): Number of target classes.
            class_names (Optional[List[str]]): Target class names.
            label_service (Optional[Any]): Service for label normalization.
        """
        self.local_trees = local_trees
        self.global_trees = global_trees
        self.lw = local_weight
        self.gw = global_weight
        self.n_classes = n_classes
        self.class_names = class_names or [str(i) for i in range(n_classes)]
        self.label_service = label_service

        self.all_trees = local_trees + global_trees

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Perform weighted majority voting inference.

        Args:
            X (np.ndarray): Data features matrix.

        Returns:
            np.ndarray: Predicted label indices.
        """
        n_samples = X.shape[0]
        combined_votes = np.zeros((n_samples, self.n_classes))

        # Calculate effective weight per tree type
        w_local = self.lw / len(self.local_trees) if self.local_trees else 0.0
        w_global = (
            self.gw / len(self.global_trees) if self.global_trees else 0.0
        )

        def accumulate(trees: List[Any], weight: float) -> None:
            if not trees:
                return
            for tree in trees:
                preds_raw = tree.predict(X)
                if self.label_service:
                    preds = self.label_service.transform(preds_raw)
                else:
                    preds = np.asarray(preds_raw, dtype=np.int64)

                # Vectorized vote accumulation
                for c in range(self.n_classes):
                    combined_votes[preds == c, c] += weight

        accumulate(self.local_trees, w_local)
        accumulate(self.global_trees, w_global)

        return np.argmax(combined_votes, axis=1)

    def diversity_measure(
        self, X: np.ndarray, y: np.ndarray, diversity: str = "pcd"
    ) -> float:
        """Calculate diversity metrics for the hybrid ensemble.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): True labels.
            diversity (str): Type of diversity measure ("pcd" supported).

        Returns:
            float: Diversity score.

        Raises:
            ValueError: If the diversity metric name is unsupported.
        """
        if diversity.lower() != "pcd":
            raise ValueError(
                f"Diversity measure '{diversity}' not implemented "
                f"for HybridForest."
            )

        n_trees = len(self.all_trees)
        if n_trees == 0:
            return 0.0

        n_samples = X.shape[0]

        # Normalize labels
        if self.label_service:
            y_true = self.label_service.transform(y)
        else:
            y_true = np.asarray(y, dtype=np.int64)

        # Vectorized hit counting across all trees
        all_preds = []
        for tree in self.all_trees:
            p_raw = tree.predict(X)
            if self.label_service:
                all_preds.append(self.label_service.transform(p_raw))
            else:
                all_preds.append(np.asarray(p_raw, dtype=np.int64))

        all_preds_matrix = np.array(all_preds)

        # Compare each tree's prediction with y_true
        hits_matrix = all_preds_matrix == y_true.reshape(1, -1)

        # Sum hits per sample
        hits_per_sample = np.sum(hits_matrix, axis=0)

        # PCD logic (Cepero 2023)
        lower = 0.1 * n_trees
        upper = 0.9 * n_trees
        diverse_samples = np.sum(
            (hits_per_sample >= lower) & (hits_per_sample <= upper)
        )

        return float(diverse_samples / n_samples)

    def get_trees(self) -> List[Any]:
        """Return the list of all trees.

        Returns:
            List[Any]: Combined local and global trees.
        """
        return self.all_trees
