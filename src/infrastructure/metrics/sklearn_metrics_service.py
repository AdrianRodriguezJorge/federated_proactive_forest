"""Scikit-learn implementation of IMetricsService.

Bridges the metrics port using scikit-learn classification evaluation
utilities, with zero-division handling.
"""

from typing import Any, List, Optional
import numpy as np
from sklearn import metrics

from src.domain.metrics.metrics_service import IMetricsService


class SklearnMetricsService(IMetricsService):
    """Scikit-learn based machine learning metrics calculator."""

    def accuracy_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate classification accuracy.

        Args:
            y_true (np.ndarray): 1D array of ground truth labels.
            y_pred (np.ndarray): 1D array of predicted labels.

        Returns:
            float: Accuracy score.
        """
        return float(metrics.accuracy_score(y_true, y_pred))

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
            labels (Optional[List[Any]]): Set of labels to calculate.
                Defaults to None.

        Returns:
            Any: F1 score (float or array of floats).
        """
        res = metrics.f1_score(
            y_true,
            y_pred,
            average=average,
            labels=labels,
            zero_division=0,
        )
        return float(res) if average is not None else res

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
        return metrics.confusion_matrix(y_true, y_pred, labels=labels)

    def precision_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = "macro",
        labels: Optional[List[Any]] = None,
    ) -> Any:
        """Calculate precision metric with zero-division safety.

        Args:
            y_true (np.ndarray): 1D array of ground truth labels.
            y_pred (np.ndarray): 1D array of predicted labels.
            average (str): Aggregation strategy. Defaults to "macro".
            labels (Optional[List[Any]]): Set of labels to calculate.
                Defaults to None.

        Returns:
            Any: Precision score.
        """
        res = metrics.precision_score(
            y_true,
            y_pred,
            average=average,
            labels=labels,
            zero_division=0,
        )
        return float(res) if average is not None else res

    def recall_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = "macro",
        labels: Optional[List[Any]] = None,
    ) -> Any:
        """Calculate recall metric with zero-division safety.

        Args:
            y_true (np.ndarray): 1D array of ground truth labels.
            y_pred (np.ndarray): 1D array of predicted labels.
            average (str): Aggregation strategy. Defaults to "macro".
            labels (Optional[List[Any]]): Set of labels to calculate.
                Defaults to None.

        Returns:
            Any: Recall score.
        """
        res = metrics.recall_score(
            y_true,
            y_pred,
            average=average,
            labels=labels,
            zero_division=0,
        )
        return float(res) if average is not None else res
