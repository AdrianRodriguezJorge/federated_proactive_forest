"""
Legacy re-export for backward compatibility.
The actual implementation has been moved to src/domain/model/cpf_implementation/
"""
from src.domain.model.cpf_implementation import ProactiveForestClassifier, DecisionForestClassifier, ComparativeProgressiveForest

__all__ = ['ProactiveForestClassifier', 'DecisionForestClassifier', 'ComparativeProgressiveForest']
