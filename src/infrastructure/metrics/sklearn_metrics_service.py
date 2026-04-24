from typing import List, Any
import numpy as np
from sklearn import metrics
from src.domain.metrics.metrics_service import IMetricsService

class SklearnMetricsService(IMetricsService):
    """
    Implementation of IMetricsService using scikit-learn.
    This belongs to the infrastructure layer.
    """

    def accuracy_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(metrics.accuracy_score(y_true, y_pred))

    def f1_score(self, y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro', labels: List[Any] = None) -> Any:
        res = metrics.f1_score(y_true, y_pred, average=average, labels=labels, zero_division=0)
        return float(res) if average is not None else res

    def confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, labels: List[Any] = None) -> np.ndarray:
        return metrics.confusion_matrix(y_true, y_pred, labels=labels)

    def precision_score(self, y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro', labels: List[Any] = None) -> Any:
        res = metrics.precision_score(y_true, y_pred, average=average, labels=labels, zero_division=0)
        return float(res) if average is not None else res

    def recall_score(self, y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro', labels: List[Any] = None) -> Any:
        res = metrics.recall_score(y_true, y_pred, average=average, labels=labels, zero_division=0)
        return float(res) if average is not None else res
