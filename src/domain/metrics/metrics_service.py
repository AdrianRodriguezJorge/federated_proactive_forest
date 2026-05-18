"""Interfaces for machine learning evaluation and diversity services.

Decouples the domain layer from specific evaluation libraries (such as
scikit-learn) and standardizes calculations for accuracy, F1, precision,
recall, confusion matrix, and Percentage Correct Diversity (PCD).
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional
import numpy as np


class IMetricsService(ABC):
    """Interface for calculating machine learning evaluation metrics."""

    @abstractmethod
    def accuracy_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate classification accuracy.

        Args:
            y_true (np.ndarray): 1D array of ground truth labels.
            y_pred (np.ndarray): 1D array of predicted labels.

        Returns:
            float: Accuracy score (proportion of correct predictions).
        """
        pass

    @abstractmethod
    def f1_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = "macro",
        labels: Optional[List[Any]] = None,
    ) -> Any:
        """Calculate the F1 classification metric.

        Args:
            y_true (np.ndarray): 1D array of ground truth labels.
            y_pred (np.ndarray): 1D array of predicted labels.
            average (str): Metric aggregation strategy ('macro', 'micro',
                'weighted'). Defaults to "macro".
            labels (Optional[List[Any]]): Set of labels to include when
                calculating average. Defaults to None.

        Returns:
            Any: Calculated F1 score (float or array of floats).
        """
        pass

    @abstractmethod
    def confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[Any]] = None,
    ) -> np.ndarray:
        """Calculate classification confusion matrix.

        Args:
            y_true (np.ndarray): 1D array of ground truth labels.
            y_pred (np.ndarray): 1D array of predicted labels.
            labels (Optional[List[Any]]): Set of labels to index the matrix.
                Defaults to None.

        Returns:
            np.ndarray: 2D confusion matrix array.
        """
        pass

    @abstractmethod
    def precision_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = "macro",
        labels: Optional[List[Any]] = None,
    ) -> Any:
        """Calculate the precision classification metric.

        Args:
            y_true (np.ndarray): 1D array of ground truth labels.
            y_pred (np.ndarray): 1D array of predicted labels.
            average (str): Aggregation strategy ('macro', 'micro', 'weighted').
                Defaults to "macro".
            labels (Optional[List[Any]]): Set of labels to calculate.
                Defaults to None.

        Returns:
            Any: Precision score.
        """
        pass

    @abstractmethod
    def recall_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = "macro",
        labels: Optional[List[Any]] = None,
    ) -> Any:
        """Calculate the recall classification metric.

        Args:
            y_true (np.ndarray): 1D array of ground truth labels.
            y_pred (np.ndarray): 1D array of predicted labels.
            average (str): Aggregation strategy ('macro', 'micro', 'weighted').
                Defaults to "macro".
            labels (Optional[List[Any]]): Set of labels to calculate.
                Defaults to None.

        Returns:
            Any: Recall score.
        """
        pass


class IDiversityService(ABC):
    """Interface for calculating diversity measures in ensembles.

    Standardizes Percentage Correct Diversity (PCD) and other diversity
    metrics.
    """

    @abstractmethod
    def calculate_pcd(
        self, predictions_matrix: np.ndarray, y_true: np.ndarray
    ) -> float:
        """Calculate Percentage Correct Diversity (PCD).

        PCD quantifies diversity based on the proportion of predictions
        that correct the ensemble's collective incorrect voting (Cepero 2023).

        Args:
            predictions_matrix (np.ndarray): 2D array of shape
                (n_estimators, n_samples) with tree-level predictions.
            y_true (np.ndarray): 1D array of shape (n_samples,) with true
                target labels.

        Returns:
            float: PCD diversity score.
        """
        pass

    @abstractmethod
    def calculate_marginal_pcd(
        self,
        candidate_predictions: np.ndarray,
        current_hits_per_sample: np.ndarray,
        n_existing_trees: int,
        y_true: np.ndarray,
    ) -> float:
        """Calculate the potential PCD if a candidate were added.

        Enables fast incremental evaluation of candidate trees during
        aggregation without recomputing full prediction matrices.

        Args:
            candidate_predictions (np.ndarray): Array of shape (n_samples,)
                with candidate predictions.
            current_hits_per_sample (np.ndarray): Array of shape (n_samples,)
                with hits per sample in the current ensemble.
            n_existing_trees (int): Number of trees currently in the ensemble.
            y_true (np.ndarray): Ground truth target labels.

        Returns:
            float: Potential PCD score if candidate is aggregated.
        """
        pass
