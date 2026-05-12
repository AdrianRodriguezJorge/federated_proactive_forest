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
        # OPT-3: Convert to set for O(1) lookups instead of O(n) per element on numpy array
        bag_set = set(self._set_ids)
        return [i for i in range(self._n_instances) if i not in bag_set]


class ProbabilitySet(SetGenerator):
    def training_ids(self, prob):
        if self._set_ids is None:
            indices = list(range(self._n_instances))
            self._set_ids = get_instances(indices, self._n_instances, prob)
        return self._set_ids

    def oob_ids(self):
        # OPT-12: Same O(n²) → O(n) fix as BaggingSet
        bag_set = set(self._set_ids)
        return [i for i in range(self._n_instances) if i not in bag_set]


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
    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            results = np.zeros(self._n_classes)
            for model in self._predictors:
                results[model.predict(X)] += 1
            return np.argmax(results)
        
        # Batch mode
        n_samples = X.shape[0]
        votes = np.zeros((n_samples, self._n_classes))
        for model in self._predictors:
            preds = model.predict(X).astype(int)
            # Use advanced indexing to increment votes
            votes[np.arange(n_samples), preds] += 1
        return np.argmax(votes, axis=1)


class PerformanceWeightingVoter(WeightingVoter):
    def predict(self, X):
        X = np.asarray(X)
        weights = np.array([model.weight for model in self._predictors])
        sum_weights = np.sum(weights)
        if sum_weights == 0:
            weights = np.ones(len(weights)) / len(weights)
        else:
            weights = weights / sum_weights
            
        if X.ndim == 1:
            results = {}
            for model, w in zip(self._predictors, weights):
                pred = model.predict(X)
                results[pred] = results.get(pred, 0) + w
            return max(results, key=results.get)

        # Batch mode
        n_samples = X.shape[0]
        weighted_votes = np.zeros((n_samples, self._n_classes))
        for model, w in zip(self._predictors, weights):
            preds = model.predict(X).astype(int)
            weighted_votes[np.arange(n_samples), preds] += w
        return np.argmax(weighted_votes, axis=1)


class SoftPerformanceWeightingVoter(WeightingVoter):
    """
    Weighted Soft Voting: Multiplies each tree's probabilities by its performance weight.
    This provides better ensemble decisions than hard voting.
    """
    def predict(self, X):
        X = np.asarray(X)
        weights = np.array([model.weight for model in self._predictors])
        sum_weights = np.sum(weights)
        if sum_weights == 0:
            weights = np.ones(len(weights)) / len(weights)
        else:
            weights = weights / sum_weights
            
        class_indices = list(range(self._n_classes))
        
        if X.ndim == 1:
            results = np.zeros(self._n_classes)
            for model, w in zip(self._predictors, weights):
                results += np.array(model.predict_proba(X, class_indices)) * w
            return np.argmax(results)

        # Batch mode: tree.predict_proba is NOT vectorized yet, 
        # so we loop over samples but use tree results efficiently.
        # Note: Ideally tree.predict_proba should be vectorized too.
        n_samples = X.shape[0]
        accumulated_probs = np.zeros((n_samples, self._n_classes))
        for model, w in zip(self._predictors, weights):
            # model is a DecisionTree. We still have to loop samples for predict_proba
            # but we do it inside here to avoid re-calculating weights.
            for i in range(n_samples):
                accumulated_probs[i] += np.array(model.predict_proba(X[i], class_indices)) * w
                
        return np.argmax(accumulated_probs, axis=1)


class DistributionSummationVoter(WeightingVoter):
    def predict(self, x):
        results = np.zeros(self._n_classes)
        class_indices = list(range(self._n_classes))
        for model in self._predictors:
            results += np.array(model.predict_proba(x, class_indices))
        return np.argmax(results)
