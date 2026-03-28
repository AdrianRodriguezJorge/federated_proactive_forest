"""CPF Implementation — Comparative Progressive Forest (Cepero 2023), adaptado para FL."""
from .estimator import ProactiveForestClassifier, DecisionForestClassifier
from ..progressive_forest import ComparativeProgressiveForest

__all__ = ['ProactiveForestClassifier', 'DecisionForestClassifier', 'ComparativeProgressiveForest']
