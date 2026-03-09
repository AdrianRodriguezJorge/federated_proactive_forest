from typing import Tuple
import numpy as np
from sklearn.metrics import accuracy_score
from .proactive_forest import ProactiveForest

class ProgressiveForest:
    """
    Progressive Forest with early stopping based on convergence.
    Implements the Comparative Progressive Forest (CPF) algorithm.
    """

    CONVERGENCE_THRESHOLD = 0.002
    INITIAL_EPISODE_SIZE = 5

    def __init__(self, forest: ProactiveForest, verbose: bool = False):
        self.forest = forest
        self.verbose = verbose

    def fit_with_early_stopping(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray) -> 'ProgressiveForest':
        """
        Train the forest with early stopping based on validation accuracy convergence.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Self for method chaining
        """
        # Simplified implementation - in full version, implement the episode-based training
        # For now, just fit the forest normally
        self.forest.fit(X_train, y_train)
        return self

    def get_forest(self) -> ProactiveForest:
        """Return the trained forest."""
        return self.forest