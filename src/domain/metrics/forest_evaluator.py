"""
Cálculo de todas las métricas para el panel Streamlit.
Usa el ProactiveForestClassifier original directamente.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              precision_score, recall_score, f1_score, classification_report)


@dataclass
class ForestReport:
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

    @property
    def report(self) -> str:
        """Generate a detailed classification report string."""
        # Create dummy y_true and y_pred from confusion matrix for classification_report
        # This is a bit hacky, but since we don't store the original predictions,
        # we reconstruct them from the confusion matrix
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
        
        return classification_report(y_true, y_pred, target_names=self.class_names, zero_division=0)


class ForestEvaluator:
    """Evalúa un ProactiveForestClassifier (o DecisionForestClassifier) ya entrenado."""

    @staticmethod
    def evaluate(forest, X: np.ndarray, y: np.ndarray,
                 class_names: List[str]) -> ForestReport:
        y_pred = forest.predict(X)

        # Convert to numpy array to handle pandas Series with non-default index
        if hasattr(y, 'values'):
            y = y.values
        if hasattr(y_pred, 'values'):
            y_pred = y_pred.values

        # Ensure y and y_pred are strings for consistency
        if len(y) > 0 and isinstance(y[0], (int, np.integer)):
            y = np.array([class_names[i] for i in y])
        if len(y_pred) > 0 and isinstance(y_pred[0], (int, np.integer)):
            y_pred = np.array([class_names[i] for i in y_pred])
        
        labels = class_names  # Usar nombres de clases como labels

        per_f1   = f1_score(y, y_pred, labels=labels, average=None, zero_division=0)
        per_prec = precision_score(y, y_pred, labels=labels, average=None, zero_division=0)
        per_rec  = recall_score(y, y_pred, labels=labels, average=None, zero_division=0)

        # PCD usando el método nativo del bosque
        try:
            pcd = float(forest.diversity_measure(X, y, diversity='pcd'))
        except Exception:
            pcd = 0.0

        return ForestReport(
            accuracy=float(accuracy_score(y, y_pred)),
            macro_f1=float(f1_score(y, y_pred, average='macro', zero_division=0)),
            macro_precision=float(precision_score(y, y_pred, average='macro', zero_division=0)),
            macro_recall=float(recall_score(y, y_pred, average='macro', zero_division=0)),
            per_class_f1={cn: float(v) for cn, v in zip(class_names, per_f1)},
            per_class_prec={cn: float(v) for cn, v in zip(class_names, per_prec)},
            per_class_recall={cn: float(v) for cn, v in zip(class_names, per_rec)},
            confusion_matrix=confusion_matrix(y, y_pred, labels=labels),
            pcd=pcd,
            forest_size=len(forest.get_trees()),
            class_names=class_names,
        )

    @staticmethod
    def evaluate_from_predictions(y_pred: np.ndarray, y_true: np.ndarray,
                                  class_names: List[str], forest_size: int,
                                  pcd: float = 0.0) -> ForestReport:
        """Evalúa métricas a partir de predicciones ya calculadas (útil para inferencia híbrida)."""

        # Convert to numpy array to handle pandas Series with non-default index
        if hasattr(y_true, 'values'):
            y_true = y_true.values
        if hasattr(y_pred, 'values'):
            y_pred = y_pred.values

        # Ensure y and y_pred are strings for consistency
        if len(y_true) > 0 and isinstance(y_true[0], (int, np.integer)):
            y_true = np.array([class_names[i] for i in y_true])
        if len(y_pred) > 0 and isinstance(y_pred[0], (int, np.integer)):
            y_pred = np.array([class_names[i] for i in y_pred])
        
        labels = class_names

        per_f1   = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        per_prec = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        per_rec  = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

        return ForestReport(
            accuracy=float(accuracy_score(y_true, y_pred)),
            macro_f1=float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            macro_precision=float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
            macro_recall=float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
            per_class_f1={cn: float(v) for cn, v in zip(class_names, per_f1)},
            per_class_prec={cn: float(v) for cn, v in zip(class_names, per_prec)},
            per_class_recall={cn: float(v) for cn, v in zip(class_names, per_rec)},
            confusion_matrix=confusion_matrix(y_true, y_pred, labels=labels),
            pcd=pcd,
            forest_size=forest_size,
            class_names=class_names,
        )
