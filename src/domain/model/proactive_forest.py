"""Proactive Forest adapter class for Federated Learning.

Provides early stopping via validation splits, exploration rate adjustments, and
federated feature selection probability state loading/extraction.
"""

import warnings
from typing import Any, List, Optional
import numpy as np

from flextrees.utils import GlobalRandomForest
from .base_forest import ABCForest
from .cpf_implementation.estimator import ProactiveForestClassifier
from .progressive_forest import ComparativeProgressiveForest


class ProactiveForest(GlobalRandomForest, ABCForest):
    """Proactive Forest implementation for Federated Learning.

    Wraps ProactiveForestClassifier and ComparativeProgressiveForest
    to provide a high-level API for federated tree-based learning.
    It inherits from GlobalRandomForest to be compatible with flex-trees.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        alpha_pf: float = 0.1,
        random_state: int = 42,
        verbose: bool = False,
        class_names: Optional[List[str]] = None,
        local_convergence_threshold: float = 0.002,
    ):
        """Initializes the Proactive Forest model.

        Args:
            n_estimators (int): Number of trees to train in the forest.
            alpha_pf (float): Diversity rate parameter (Cepero parameter).
            random_state (int): Seed for reproducibility.
            verbose (bool): Whether to print training progress and logs.
            class_names (Optional[List[str]]): List of class names.
            local_convergence_threshold (float): Improvement threshold.
        """
        super().__init__(n_estimators=n_estimators)

        self.alpha_pf = alpha_pf
        self.random_state = random_state
        self.verbose = verbose
        self.class_names = class_names
        self.local_convergence_threshold = local_convergence_threshold
        self._is_fitted: bool = False

        self._classifier = ProactiveForestClassifier(
            n_estimators=n_estimators,
            alpha=alpha_pf,
            bootstrap=True,
            split_criterion="entropy",
            random_state=random_state,
        )
        self._cpf: Optional[ComparativeProgressiveForest] = None


        if class_names is not None and len(class_names) > 0:
            self.class_names = class_names
            self._classifier.classes_ = np.array(class_names)
            self._classifier._encoder_dict = {
                val: idx for idx, val in enumerate(class_names)
            }
            self._classifier._decoder_dict = {
                idx: val for idx, val in enumerate(class_names)
            }

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train the Proactive Forest using CPF with early stopping.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
            X_val (Optional[np.ndarray]): Validation features.
            y_val (Optional[np.ndarray]): Validation labels.
        """
        from ..services.label_service import SimpleLabelService

        label_svc = SimpleLabelService(self.class_names)

        if self.class_names is None or len(self.class_names) == 0:
            label_svc.fit(y)
            self.class_names = label_svc.classes

        y_labels = label_svc.inverse_transform(label_svc.transform(y))

        if not hasattr(self._classifier, "_encoder_dict"):
            self._classifier.classes_ = np.array(self.class_names)
            self._classifier._encoder_dict = {
                val: idx for idx, val in enumerate(self.class_names)
            }
            self._classifier._decoder_dict = {
                idx: val for idx, val in enumerate(self.class_names)
            }

        self._classifier._n_classes = len(self._classifier.classes_)

        if X_val is None or y_val is None:
            n_samples = len(X)
            if n_samples >= 10:
                indices = np.arange(n_samples)
                rng = np.random.default_rng(self.random_state)
                rng.shuffle(indices)

                split_idx = int(0.9 * n_samples)
                train_idx, val_idx = indices[:split_idx], indices[split_idx:]

                X_train, X_val = X[train_idx], X[val_idx]
                y_train = y_labels[train_idx]
                y_val_labels = y_labels[val_idx]
            else:
                warnings.warn(
                    "Dataset too small for internal validation split. "
                    "Data leakage may occur in early stopping."
                )
                X_train, X_val = X, X
                y_train = y_labels
                y_val_labels = y_labels
        else:
            X_train = X
            y_train = y_labels
            y_val_labels = label_svc.inverse_transform(
                label_svc.transform(y_val)
            )

        self._cpf = ComparativeProgressiveForest(
            self._classifier,
            verbose=self.verbose,
            local_convergence_threshold=self.local_convergence_threshold,
        )
        self._cpf.fit(X_train, y_train, X_val, y_val_labels)
        self._is_fitted = True
        self.estimators_ = self.get_trees()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained forest.

        Args:
            X (np.ndarray): Input features matrix.

        Returns:
            np.ndarray: Predicted class labels.

        Raises:
            ValueError: If the forest has not been fitted yet.
        """
        if not self._is_fitted or self._cpf is None:
            raise ValueError("Forest not fitted yet. Call fit() first.")

        y_pred = self._cpf.return_forest().predict(X)

        if isinstance(y_pred, np.ndarray) and np.issubdtype(
            y_pred.dtype, np.integer
        ):
            from ..services.label_service import SimpleLabelService

            label_svc = SimpleLabelService(self.class_names)
            y_pred = label_svc.inverse_transform(y_pred)

        return np.array(y_pred)

    def get_trees(self) -> List[Any]:
        """Return the list of trained decision trees.

        Returns:
            List[Any]: List of tree objects.

        Raises:
            ValueError: If the forest is not fitted yet.
        """
        if not self._is_fitted or self._cpf is None:
            raise ValueError("Forest not fitted yet.")
        forest = self._cpf.return_forest()
        return forest.get_trees()

    def diversity_measure(
        self, X: np.ndarray, y: np.ndarray, diversity: str = "pcd"
    ) -> float:
        """Calculate the diversity measure score of the forest.

        Args:
            X (np.ndarray): Input features matrix.
            y (np.ndarray): Target class labels array.
            diversity (str): Type of diversity ("pcd").

        Returns:
            float: Diversity score.

        Raises:
            ValueError: If the forest has not been fitted yet.
        """
        if not self._is_fitted or self._cpf is None:
            raise ValueError("Forest not fitted yet.")
        forest = self._cpf.return_forest()
        return forest.diversity_measure(X, y, diversity)

    @classmethod
    def from_trees(
        cls, trees: List[Any], class_names: Optional[List[str]] = None
    ) -> "ProactiveForest":
        """Create a forest instance from a list of trees.

        Args:
            trees (List[Any]): List of tree objects.
            class_names (Optional[List[str]]): List of class names.

        Returns:
            ProactiveForest: A fitted forest instance.
        """
        instance = cls(class_names=class_names)

        if trees:
            n_features = trees[0].n_features
            n_classes = len(class_names) if class_names else 1
        else:
            n_features = 0
            n_classes = 1

        dummy_classifier = ProactiveForestClassifier(
            n_estimators=len(trees), alpha=0.1
        )
        dummy_classifier._n_features = n_features
        dummy_classifier._n_classes = n_classes
        if class_names:
            dummy_classifier.classes_ = np.array(class_names)
            dummy_classifier._encoder_dict = {
                val: idx for idx, val in enumerate(class_names)
            }
            dummy_classifier._decoder_dict = {
                idx: val for idx, val in enumerate(class_names)
            }
        dummy_classifier.set_trees(trees)

        instance._classifier = dummy_classifier
        instance._is_fitted = True
        instance.estimators_ = trees
        instance._cpf = ComparativeProgressiveForest(dummy_classifier)

        return instance

    def get_feature_probabilities(self) -> np.ndarray:
        """Return the current feature probability vector (roulette state).

        Returns:
            np.ndarray: Probability vector of shape (n_features,).
        """
        probs = self._classifier.get_feature_probabilities()
        return np.array(probs, dtype=np.float64)

    def set_feature_probabilities(self, probabilities: np.ndarray) -> None:
        """Inject a new feature probability vector (federated roulette).

        Args:
            probabilities (np.ndarray): 1-D array of shape (n_features,)
                summing to 1.0.
        """
        probs = np.asarray(probabilities, dtype=np.float64)
        self._classifier.set_feature_probabilities(probs.tolist())

    @property
    def n_features(self) -> int:
        """Number of features the classifier was fitted on.

        Returns:
            int: Dimension count.
        """
        return getattr(self._classifier, "_n_features", 0) or 0