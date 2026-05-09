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
    def f1_score(self, y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro', labels: List[Any] = None) -> Any:
        pass

    @abstractmethod
    def confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, labels: List[Any] = None) -> np.ndarray:
        pass

    @abstractmethod
    def precision_score(self, y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro', labels: List[Any] = None) -> Any:
        pass

    @abstractmethod
    def recall_score(self, y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro', labels: List[Any] = None) -> Any:
        pass


class IDiversityService(ABC):
    """
    Interface for calculating diversity measures in ensembles.
    Standardizes PCD and other diversity metrics.
    """

    @abstractmethod
    def calculate_pcd(self, predictions_matrix: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculate Percentage Correct Diversity (PCD) based on Cepero (2023).
        """
        pass

    @abstractmethod
    def calculate_marginal_pcd(self, candidate_predictions: np.ndarray, 
                               current_hits_per_sample: np.ndarray,
                               n_existing_trees: int,
                               y_true: np.ndarray) -> float:
        """
        Calculate the potential PCD if a candidate were added to the current ensemble.
        
        Args:
            candidate_predictions: Array of (n_samples,) with candidate predictions.
            current_hits_per_sample: Array of (n_samples,) with hit counts per sample.
            n_existing_trees: Number of trees currently in the ensemble.
            y_true: Ground truth labels.
        
        Returns:
            Potential PCD score.
        """
        pass
