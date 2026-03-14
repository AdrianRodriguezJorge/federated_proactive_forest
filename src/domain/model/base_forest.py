from abc import ABC, abstractmethod
from typing import List, Any
import numpy as np
from sklearn.base import BaseEstimator

class ABCForest(ABC):
    """
    Abstract Base Class for Forest models in Federated Learning.
    Provides the interface that all forest implementations must follow.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the forest on the given data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on the given data."""
        pass

    @abstractmethod
    def get_trees(self) -> List[Any]:
        """Return the list of trained trees."""
        pass

    @classmethod
    @abstractmethod
    def from_trees(cls, trees: List[Any], class_names: List[str] = None) -> 'ABCForest':
        """Create a forest instance from a list of trees."""
        pass