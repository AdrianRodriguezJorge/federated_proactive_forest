"""Abstract base forest definitions for federated proactive forest models.

Specifies the interface that all ensemble classifier variants (ProactiveForest,
ProgressiveForest, HybridForest, etc.) must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional
import numpy as np


class ABCForest(ABC):
    """Abstract Base Class for Forest models in Federated Learning.

    Provides the common interface that all forest implementations must follow
    for fitting, predicting, and tree serialization/deserialization.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the forest on the given dataset.

        Args:
            X (np.ndarray): Input feature training matrix.
            y (np.ndarray): Target class labels array.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on the given dataset.

        Args:
            X (np.ndarray): Input feature evaluation matrix.

        Returns:
            np.ndarray: Predicted class labels.
        """
        pass

    @abstractmethod
    def get_trees(self) -> List[Any]:
        """Return the list of trained decision trees.

        Returns:
            List[Any]: List of DecisionTree instances.
        """
        pass

    @classmethod
    @abstractmethod
    def from_trees(
        cls, trees: List[Any], class_names: Optional[List[str]] = None
    ) -> "ABCForest":
        """Create a forest instance from a list of pre-trained trees.

        Args:
            trees (List[Any]): List of DecisionTree instances.
            class_names (Optional[List[str]]): Target class names index.

        Returns:
            ABCForest: Initialized forest model instance.
        """
        pass