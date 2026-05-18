"""Scikit-Learn RandomForestClassifier adapter wrapper.

Adapts standard scikit-learn random forests to comply with the project's
ABCForest interface, including custom diversity metrics calculation.
"""

from typing import Any, List, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.domain.model.base_forest import ABCForest


class RandomForestWrapper(ABCForest):
    """Wrapper for scikit-learn RandomForestClassifier.

    Adapts sklearn models to implement the ABCForest interface for fallback and
    comparative benchmarks.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 42,
        **kwargs: Any,
    ):
        """Initializes the Random Forest wrapper.

        Args:
            n_estimators (int): Number of tree estimators.
            random_state (int): Seed for reproducibility.
            **kwargs (Any): Additional scikit-learn parameters.
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.kwargs = kwargs
        self._rf: Optional[RandomForestClassifier] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the Random Forest estimator.

        Args:
            X (np.ndarray): Feature training matrix.
            y (np.ndarray): Target class labels array.
        """
        self._rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            **self.kwargs,
        )
        self._rf.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target labels for the input dataset.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted class labels.

        Raises:
            ValueError: If the estimator has not been fitted yet.
        """
        if self._rf is None:
            raise ValueError("Random Forest not fitted yet")
        return self._rf.predict(X)

    def get_trees(self) -> List[Any]:
        """Return the list of trained decision trees.

        Returns:
            List[Any]: Copy of inner decision tree estimators.
        """
        if self._rf is None:
            return []
        return self._rf.estimators_.copy()

    def diversity_measure(
        self, X: np.ndarray, y: np.ndarray, diversity: str = "pcd"
    ) -> float:
        """Calculate the ensemble Percentage Correct Diversity (PCD) score.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target labels.
            diversity (str): Name of diversity metric ("pcd").

        Returns:
            float: Calculated diversity score.
        """
        if diversity != "pcd":
            return 0.0

        trees = self.get_trees()
        if not trees:
            return 0.0

        n_samples = X.shape[0]
        n_trees = len(trees)

        # Collect predictions from all trees
        preds = np.empty((n_samples, n_trees), dtype=object)
        for i, tree in enumerate(trees):
            preds[:, i] = tree.predict(X)

        y_arr = np.asarray(y).reshape(-1, 1)
        hits = preds == y_arr
        hits_per_sample = np.sum(hits, axis=1)

        # PCD thresholds (10% - 90%)
        lower = 0.1 * n_trees
        upper = 0.9 * n_trees
        diverse = np.sum(
            (hits_per_sample >= lower) & (hits_per_sample <= upper)
        )

        return float(diverse / n_samples)

    @classmethod
    def from_trees(
        cls, trees: List[Any], class_names: Optional[List[str]] = None
    ) -> "RandomForestWrapper":
        """Create a Random Forest instance from a list of pre-trained trees.

        Args:
            trees (List[Any]): List of decision tree estimators.
            class_names (Optional[List[str]]): Target classes name mapping.

        Returns:
            RandomForestWrapper: Fitted wrapper instance.
        """
        instance = cls()
        instance._rf = RandomForestClassifier()
        instance._rf.estimators_ = trees.copy()
        return instance
