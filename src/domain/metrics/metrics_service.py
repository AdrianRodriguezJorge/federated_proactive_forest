from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

class IMetricsService(ABC):
    """
    Interface for calculating machine learning metrics.
    Decouples the domain layer from specific libraries like scikit-learn.
    """

    @abstractmethod
    def accuracy_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

    @abstractmethod
    def f1_score(self, y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> Any:
        # returns float if average is not None, else np.ndarray
        pass

    @abstractmethod
    def confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, labels: List[Any] = None) -> np.ndarray:
        pass

    @abstractmethod
    def precision_score(self, y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> Any:
        pass

    @abstractmethod
    def recall_score(self, y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> Any:
        pass


class IDiversityService(ABC):
    """
    Interface for calculating diversity measures in ensembles.
    Standardizes PCD and other diversity metrics.
    """

    @abstractmethod
    def calculate_pcd(self, predictions_matrix: np.ndarray) -> float:
        """
        Calculate Pairwise Classifier Disagreement (PCD).
        
        Args:
            predictions_matrix: Matrix of shape (n_samples, n_classifiers)
                               containing class labels or indices.
        
        Returns:
            Mean pairwise disagreement score in range [0, 1].
        """
        pass
