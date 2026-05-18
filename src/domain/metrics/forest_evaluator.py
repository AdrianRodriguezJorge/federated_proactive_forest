"""Evaluation reporting and metric calculations for Proactive Forests.

Provides structural models for evaluation reports and evaluation logic capable
of computing standard metrics, bootstrapped confidence intervals, and diversity
measures under target class alignments.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .metrics_service import IMetricsService


@dataclass
class ForestReport:
    """Structure storing evaluation metrics for a Proactive Forest model.

    Attributes:
        accuracy (float): Overall prediction accuracy.
        macro_f1 (float): Unweighted macro F1-score across all classes.
        macro_precision (float): Unweighted macro precision.
        macro_recall (float): Unweighted macro recall.
        per_class_f1 (Dict[str, float]): Dictionary mapping class names to
            their specific F1-score.
        per_class_prec (Dict[str, float]): Dictionary mapping class names to
            their specific precision.
        per_class_recall (Dict[str, float]): Dictionary mapping class names to
            their specific recall.
        confusion_matrix (np.ndarray): 2D array representation of prediction
            hits and misses.
        pcd (float): Percentage Correct Diversity (PCD) score.
        forest_size (int): Total number of trees in the evaluated forest.
        class_names (List[str]): List of target class labels.
        accuracy_ci (Tuple[float, float]): Bootstrap confidence interval for
            accuracy. Defaults to (0.0, 0.0).
        macro_f1_ci (Tuple[float, float]): Bootstrap confidence interval for
            macro F1. Defaults to (0.0, 0.0).
    """

    accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    per_class_f1: Dict[str, float]
    per_class_prec: Dict[str, float]
    per_class_recall: Dict[str, float]
    confusion_matrix: np.ndarray
    pcd: float
    forest_size: int
    class_names: List[str]
    accuracy_ci: Tuple[float, float] = (0.0, 0.0)
    macro_f1_ci: Tuple[float, float] = (0.0, 0.0)

    @property
    def report(self) -> str:
        """Generate a detailed classification report string.

        Returns:
            str: Preformatted classification report from scikit-learn.
        """
        # Create dummy y_true and y_pred from confusion matrix for report
        cm = self.confusion_matrix
        y_true = []
        y_pred = []
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                count = int(cm[i, j])
                y_true.extend([self.class_names[i]] * count)
                y_pred.extend([self.class_names[j]] * count)

        if len(y_true) == 0:
            return "No predictions available for classification report."

        return classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            zero_division=0,
        )


class ForestEvaluator:
    """Evaluates a trained ProactiveForestClassifier or DecisionForest."""

    @staticmethod
    def evaluate(
        forest,
        X: np.ndarray,
        y: np.ndarray,
        class_names: List[str],
        metrics_svc: Optional[IMetricsService] = None,
        n_bootstrap: int = 0,
    ) -> ForestReport:
        """Evaluates a trained forest model.

        Converts raw inputs to normalized numpy representations using
        centralized LabelServices to avoid any data leakage.

        Args:
            forest: Trained forest classifier instance.
            X (np.ndarray): Input feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Ground truth labels.
            class_names (List[str]): Target class names.
            metrics_svc (Optional[IMetricsService]): Implementation of
                IMetricsService. Required.
            n_bootstrap (int): Number of bootstrap iterations to compute
                confidence intervals. Set to 0 to disable. Defaults to 0.

        Returns:
            ForestReport: Computed metrics and bootstrap intervals.

        Raises:
            ValueError: If metrics_svc is not provided.
        """
        # If no metrics service provided, we can't perform evaluation
        if metrics_svc is None:
            raise ValueError(
                "An implementation of IMetricsService must be "
                "provided for evaluation."
            )

        y_pred = forest.predict(X)

        # Convert to numpy array to handle pandas Series
        if hasattr(y, "values"):
            y = y.values
        if hasattr(y_pred, "values"):
            y_pred = y_pred.values

        # Ensure y and y_pred are normalized using project standards
        from ..services.label_service import SimpleLabelService

        label_svc = SimpleLabelService(class_names)

        # If class_names is empty, skip label conversion and use raw values
        if not class_names:
            y_norm = np.array(y)
            y_pred_norm = np.array(y_pred)
        else:
            # Convert raw labels (strings or indices) to project class names
            y_indices = label_svc.transform(y)
            y_pred_indices = label_svc.transform(y_pred)

            # Finally use the string representation for metrics
            classes_arr = np.array(label_svc.classes)
            y_norm = (
                classes_arr[y_indices] if len(y_indices) > 0 else np.array([])
            )
            y_pred_norm = (
                classes_arr[y_pred_indices]
                if len(y_pred_indices) > 0
                else np.array([])
            )

        # Use normalized variables for metric calculations
        y = y_norm
        y_pred = y_pred_norm

        # Use class_names as labels only if they are provided
        labels = class_names if class_names else None

        per_f1 = metrics_svc.f1_score(
            y, y_pred, average=None, labels=labels
        )
        per_prec = metrics_svc.precision_score(
            y, y_pred, average=None, labels=labels
        )
        per_rec = metrics_svc.recall_score(
            y, y_pred, average=None, labels=labels
        )

        # PCD usando el método nativo del bosque
        try:
            pcd = float(forest.diversity_measure(X, y, diversity="pcd"))
        except (AttributeError, TypeError, ValueError):
            pcd = 0.0

        accuracy_ci = ForestEvaluator._compute_bootstrap_ci(
            y,
            y_pred,
            lambda yt, yp: metrics_svc.accuracy_score(yt, yp),
            n_bootstrap=n_bootstrap,
        )
        macro_f1_ci = ForestEvaluator._compute_bootstrap_ci(
            y,
            y_pred,
            lambda yt, yp: metrics_svc.f1_score(yt, yp, average="macro"),
            n_bootstrap=n_bootstrap,
        )

        return ForestReport(
            accuracy=metrics_svc.accuracy_score(y, y_pred),
            macro_f1=metrics_svc.f1_score(
                y, y_pred, average="macro", labels=labels
            ),
            macro_precision=metrics_svc.precision_score(
                y, y_pred, average="macro", labels=labels
            ),
            macro_recall=metrics_svc.recall_score(
                y, y_pred, average="macro", labels=labels
            ),
            per_class_f1={cn: float(v) for cn, v in zip(class_names, per_f1)},
            per_class_prec={
                cn: float(v) for cn, v in zip(class_names, per_prec)
            },
            per_class_recall={
                cn: float(v) for cn, v in zip(class_names, per_rec)
            },
            confusion_matrix=metrics_svc.confusion_matrix(
                y, y_pred, labels=labels
            ),
            pcd=pcd,
            forest_size=len(forest.get_trees()),
            class_names=class_names,
            accuracy_ci=accuracy_ci,
            macro_f1_ci=macro_f1_ci,
        )

    @staticmethod
    def _compute_bootstrap_ci(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_func: Any,
        n_bootstrap: int = 0,
        alpha: float = 0.05,
    ) -> Tuple[float, float]:
        """Helper to compute bootstrap confidence intervals.

        Args:
            y_true (np.ndarray): 1D array of true labels.
            y_pred (np.ndarray): 1D array of predicted labels.
            metric_func (Any): Callable calculating target metric.
            n_bootstrap (int): Bootstrap iteration count. Defaults to 0.
            alpha (float): Significance level for percentile limits.
                Defaults to 0.05.

        Returns:
            Tuple[float, float]: Lower and upper confidence limits.
        """
        if n_bootstrap <= 1:
            return (0.0, 0.0)

        rng = np.random.RandomState(42)
        scores = []
        for _ in range(n_bootstrap):
            indices = rng.choice(len(y_true), size=len(y_true), replace=True)
            y_true_bs = y_true[indices]
            y_pred_bs = y_pred[indices]
            scores.append(metric_func(y_true_bs, y_pred_bs))
        lower = np.percentile(scores, 100 * alpha / 2)
        upper = np.percentile(scores, 100 * (1 - alpha / 2))
        return (float(lower), float(upper))

    @staticmethod
    def evaluate_from_predictions(
        y_pred: np.ndarray,
        y_true: np.ndarray,
        class_names: List[str],
        forest_size: int,
        pcd: float = 0.0,
        n_bootstrap: int = 0,
    ) -> ForestReport:
        """Evalúa métricas a partir de predicciones ya calculadas.

        Útil para realizar inferencias e integraciones híbridas sin necesidad
        de re-instanciar el clasificador del bosque.

        Args:
            y_pred (np.ndarray): Precomputed prediction array.
            y_true (np.ndarray): Ground truth labels.
            class_names (List[str]): Unique class label names.
            forest_size (int): Size of the forest.
            pcd (float): Percentage Correct Diversity. Defaults to 0.0.
            n_bootstrap (int): Bootstrap iterations count. Defaults to 0.

        Returns:
            ForestReport: Computed metrics report.
        """
        # Convert to numpy array to handle pandas Series
        if hasattr(y_true, "values"):
            y_true = y_true.values
        if hasattr(y_pred, "values"):
            y_pred = y_pred.values

        # Use centralized LabelService for normalization
        from ..services.label_service import SimpleLabelService

        label_svc = SimpleLabelService(class_names)

        y_true_indices = label_svc.transform(y_true)
        y_pred_indices = label_svc.transform(y_pred)

        classes_arr = np.array(label_svc.classes)
        y_true = (
            classes_arr[y_true_indices]
            if len(y_true_indices) > 0
            else np.array([])
        )
        y_pred = (
            classes_arr[y_pred_indices]
            if len(y_pred_indices) > 0
            else np.array([])
        )

        labels = class_names

        per_f1 = f1_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
        per_prec = precision_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
        per_rec = recall_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )

        accuracy_ci = ForestEvaluator._compute_bootstrap_ci(
            y_true,
            y_pred,
            lambda yt, yp: accuracy_score(yt, yp),
            n_bootstrap=n_bootstrap,
        )
        macro_f1_ci = ForestEvaluator._compute_bootstrap_ci(
            y_true,
            y_pred,
            lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
            n_bootstrap=n_bootstrap,
        )

        return ForestReport(
            accuracy=float(accuracy_score(y_true, y_pred)),
            macro_f1=float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            macro_precision=float(
                precision_score(
                    y_true, y_pred, average="macro", zero_division=0
                )
            ),
            macro_recall=float(
                recall_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            per_class_f1={cn: float(v) for cn, v in zip(class_names, per_f1)},
            per_class_prec={
                cn: float(v) for cn, v in zip(class_names, per_prec)
            },
            per_class_recall={
                cn: float(v) for cn, v in zip(class_names, per_rec)
            },
            confusion_matrix=confusion_matrix(y_true, y_pred, labels=labels),
            pcd=pcd,
            forest_size=forest_size,
            class_names=class_names,
            accuracy_ci=accuracy_ci,
            macro_f1_ci=macro_f1_ci,
        )
