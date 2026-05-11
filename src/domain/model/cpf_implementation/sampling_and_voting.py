"""
Módulo para generación de conjuntos de entrenamiento y votación en ensembles.
"""
from abc import ABC, abstractmethod
import numpy as np
from .utils import get_instances


class SetGenerator(ABC):
    def __init__(self, n_instances):
        self._n_instances = n_instances
        self._set_ids = None

    @abstractmethod
    def training_ids(self):
        pass

    @abstractmethod
    def oob_ids(self):
        pass

    def clear(self):
        self._set_ids = None


class SimpleSet(SetGenerator):
    def training_ids(self):
        if self._set_ids is None:
            self._set_ids = np.array(range(self._n_instances))
        return self._set_ids

    def oob_ids(self):
        return np.array([])


class BaggingSet(SetGenerator):
    def training_ids(self):
        if self._set_ids is None:
            self._set_ids = np.random.choice(self._n_instances, replace=True, size=self._n_instances)
        return self._set_ids

    def oob_ids(self):
        return [i for i in range(self._n_instances) if i not in self._set_ids]


class ProbabilitySet(SetGenerator):
    def training_ids(self, prob):
        if self._set_ids is None:
            indices = list(range(self._n_instances))
            self._set_ids = get_instances(indices, self._n_instances, prob)
        return self._set_ids

    def oob_ids(self):
        return [i for i in range(self._n_instances) if i not in self._set_ids]


class WeightingVoter(ABC):
    def __init__(self, predictors, n_classes):
        self._predictors = predictors
        self._n_classes = n_classes

    @abstractmethod
    def predict(self, x):
        pass

    def predict_proba(self, x, indexs):
        results = np.zeros(len(indexs))
        for model in self._predictors:
            pred_proba = model.predict_proba(x, indexs)
            results += pred_proba
        return (results / len(self._predictors)).tolist()


class MajorityVoter(WeightingVoter):
    def predict(self, x):
        results = np.zeros(self._n_classes)
        for model in self._predictors:
            results[model.predict(x)] += 1
        return np.argmax(results)


class PerformanceWeightingVoter(WeightingVoter):
    def predict(self, x):
        weights = np.array([model.weight for model in self._predictors])
        sum_weights = np.sum(weights)
        if sum_weights == 0:
            weights = np.ones(len(weights)) / len(weights)
        else:
            weights = weights / sum_weights
            
        results = {}
        for model, w in zip(self._predictors, weights):
            pred = model.predict(x)
            if pred not in results:
                results[pred] = 0
            results[pred] += w
        return max(results, key=results.get)


class SoftPerformanceWeightingVoter(WeightingVoter):
    """
    Weighted Soft Voting: Multiplies each tree's probabilities by its performance weight.
    This provides better ensemble decisions than hard voting.
    """
    def predict(self, x):
        weights = np.array([model.weight for model in self._predictors])
        sum_weights = np.sum(weights)
        if sum_weights == 0:
            weights = np.ones(len(weights)) / len(weights)
        else:
            weights = weights / sum_weights
            
        # Accumulate weighted probabilities
        # results shape: (n_classes,)
        results = np.zeros(self._n_classes)
        class_indices = list(range(self._n_classes))
        for model, w in zip(self._predictors, weights):
            results += np.array(model.predict_proba(x, class_indices)) * w
            
        return np.argmax(results)


class DistributionSummationVoter(WeightingVoter):
    def predict(self, x):
        results = np.zeros(self._n_classes)
        class_indices = list(range(self._n_classes))
        for model in self._predictors:
            results += np.array(model.predict_proba(x, class_indices))
        return np.argmax(results)
