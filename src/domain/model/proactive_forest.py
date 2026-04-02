from typing import List, Any, Optional
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from .base_forest import ABCForest
from .cpf_implementation.estimator import ProactiveForestClassifier
from .progressive_forest import ComparativeProgressiveForest


class ProactiveForest(ABCForest):
    """
    Proactive Forest implementation for Federated Learning.
    """

    def __init__(self, n_estimators: int = 100, alpha: float = 0.1, random_state: int = 42, verbose: bool = False, class_names: Optional[List[str]] = None):
        """
        Args:
            n_estimators: Number of trees in the forest
            alpha: Diversity rate for feature probability adjustment (Cepero parameter)
            random_state: Random seed
            verbose: Whether to print CPF training logs
            class_names: List of all possible class names (for consistent encoding)
        """
        self.n_estimators = n_estimators
        self.alpha = alpha
        self.random_state = random_state
        self.verbose = verbose
        self.class_names = class_names
        self._is_fitted = False

        # Create the internal ProactiveForestClassifier using CPF
        self._classifier = ProactiveForestClassifier(
            n_estimators=n_estimators,
            alpha=alpha,
            bootstrap=True,
            split_criterion='entropy'
        )
        self._cpf = None

        # Set encoder if class_names provided
        if class_names is not None and len(class_names) > 0:
            self._classifier._encoder = LabelEncoder()
            self._classifier._encoder.classes_ = np.array(class_names)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Proactive Forest using CPF algorithm with early stopping.
        
        Args:
            X: Training features
            y: Training labels
        """
        y_arr = np.asarray(y)

        # Convert indices to class names if class_names is provided.
        if self.class_names is not None and np.issubdtype(y_arr.dtype, np.integer):
            if np.any((y_arr < 0) | (y_arr >= len(self.class_names))):
                raise ValueError("y contains index values outside class_names range")
            y_labels = np.array([self.class_names[int(v)] for v in y_arr], dtype=object)
        else:
            y_labels = y_arr

        # Configure encoder on the classifier for consistent global class mapping.
        if self._classifier._encoder is None:
            self._classifier._encoder = LabelEncoder()
            if self.class_names is not None and len(self.class_names) > 0:
                self._classifier._encoder.classes_ = np.array(self.class_names)
            else:
                self._classifier._encoder.fit(y_labels)
        else:
            if self.class_names is not None and len(self.class_names) > 0:
                self._classifier._encoder.classes_ = np.array(self.class_names)

        self._classifier._n_classes = len(self._classifier._encoder.classes_)

        # Split for early stopping (80-20)
        if len(X) > 30:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_labels, test_size=0.2, random_state=self.random_state
            )
        else:
            X_train, X_val = X, X
            y_train, y_val = y_labels, y_labels

        # Use Comparative Progressive Forest with early stopping
        self._cpf = ComparativeProgressiveForest(self._classifier, verbose=self.verbose)
        self._cpf.fit(X_train, y_train, X_val, y_val)
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained forest."""
        if not self._is_fitted or self._cpf is None:
            raise ValueError("Forest not fitted yet. Call fit() first.")

        # El clasificador interno puede devolver directamente etiquetas ya decodificadas
        y_pred = self._cpf.return_forest().predict(X)

        # Si el resultado está en formato entero y hay un encoder disponible, decodificar
        if self._classifier._encoder is not None:
            if isinstance(y_pred, np.ndarray) and np.issubdtype(y_pred.dtype, np.integer):
                y_pred = self._classifier._encoder.inverse_transform(y_pred)
            elif isinstance(y_pred, (list, np.ndarray)) and len(y_pred) > 0 and isinstance(y_pred[0], (int, np.integer)):
                y_pred = self._classifier._encoder.inverse_transform(np.array(y_pred, dtype=int))
            # Si ya son strings, el resultado está listo y no requiere inverse_transform

        return np.array(y_pred)

    def get_trees(self) -> List[Any]:
        """Return the list of trained trees."""
        if not self._is_fitted or self._cpf is None:
            raise ValueError("Forest not fitted yet.")
        forest = self._cpf.return_forest()
        return forest.get_trees()

    def diversity_measure(self, X, y, diversity='pcd'):
        """Calculate diversity measure of the forest."""
        if not self._is_fitted or self._cpf is None:
            raise ValueError("Forest not fitted yet.")
        forest = self._cpf.return_forest()
        return forest.diversity_measure(X, y, diversity)

    @classmethod
    def from_trees(cls, trees: List[Any], class_names: List[str] = None) -> 'ProactiveForest':
        """Create a forest instance from a list of trees."""
        instance = cls()

        # Infer n_features from the trees (assuming all trees have the same n_features)
        if trees:
            n_features = trees[0].n_features
            n_classes = len(class_names) if class_names else 1
        else:
            n_features = 0
            n_classes = 1

        # Create a dummy classifier with the trees
        dummy_classifier = ProactiveForestClassifier(n_estimators=len(trees), alpha=0.1)
        dummy_classifier._n_features = n_features
        dummy_classifier._n_classes = n_classes
        if class_names:
            dummy_classifier._encoder = LabelEncoder()
            dummy_classifier._encoder.classes_ = np.array(class_names)
        dummy_classifier.set_trees(trees)

        # Mark as fitted
        instance._classifier = dummy_classifier
        instance._is_fitted = True

        # Wrap with empty CPF (not used for prediction)
        instance._cpf = ComparativeProgressiveForest(dummy_classifier)

        return instance