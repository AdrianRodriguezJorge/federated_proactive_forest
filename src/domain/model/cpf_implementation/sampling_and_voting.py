"""Dataset partitioning generators and ensemble voting strategies.

Provides training subset generation (Bagging, Probability-based sampling) and
predictions integration strategies (Majority voting, Performance weighting, Soft
voting).
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional
import numpy as np

from .utils import get_instances


class SetGenerator(ABC):
    """Abstract class for training index subset generators."""

    def __init__(self, n_instances: int):
        """Initializes generator.

        Args:
            n_instances (int): Number of total samples.
        """
        self._n_instances = n_instances
        self._set_ids = None

    @abstractmethod
    def training_ids(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Generate indices for training.

        Returns:
            np.ndarray: Array of selected sample indices.
        """
        pass

    @abstractmethod
    def oob_ids(self) -> List[int]:
        """Generate out-of-bag (OOB) indices.

        Returns:
            List[int]: Out-of-bag sample indices.
        """
        pass

    def clear(self) -> None:
        """Reset cached indices state."""
        self._set_ids = None


class SimpleSet(SetGenerator):
    """Simple sequential dataset index generator (no bagging)."""

    def training_ids(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Returns standard linear sequential indices.

        Returns:
            np.ndarray: Index array.
        """
        if self._set_ids is None:
            self._set_ids = np.array(range(self._n_instances))
        return self._set_ids

    def oob_ids(self) -> np.ndarray:
        """Return empty OOB indices.

        Returns:
            np.ndarray: Empty array.
        """
        return np.array([])


class BaggingSet(SetGenerator):
    """Bootstrap bagging index generator with replacement."""

    def training_ids(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Generates bootstrap training indices.

        Returns:
            np.ndarray: Array of selected indices.
        """
        if self._set_ids is None:
            self._set_ids = np.random.choice(
                self._n_instances, replace=True, size=self._n_instances
            )
        return self._set_ids

    def oob_ids(self) -> List[int]:
        """Find non-selected bootstrap out-of-bag indices in O(n).

        Returns:
            List[int]: Out-of-bag sample indices.
        """
        bag_set = set(self._set_ids)
        return [i for i in range(self._n_instances) if i not in bag_set]


class ProbabilitySet(SetGenerator):
    """Weighted probability-based training index generator."""

    def training_ids(self, prob: List[float]) -> np.ndarray:
        """Generate indices weighted by probability list.

        Args:
            prob (List[float]): Sampling weights.

        Returns:
            np.ndarray: Sampled indices.
        """
        if self._set_ids is None:
            indices = list(range(self._n_instances))
            self._set_ids = get_instances(indices, self._n_instances, prob)
        return self._set_ids

    def oob_ids(self) -> List[int]:
        """Find non-selected out-of-bag indices in O(n).

        Returns:
            List[int]: Out-of-bag indices.
        """
        bag_set = set(self._set_ids)
        return [i for i in range(self._n_instances) if i not in bag_set]


class WeightingVoter(ABC):
    """Abstract baseline class for weighted voter implementations."""

    def __init__(self, predictors: List[Any], n_classes: int):
        """Initialize weighting voter.

        Args:
            predictors (List[Any]): List of base models.
            n_classes (int): Class dimension count.
        """
        self._predictors = predictors
        self._n_classes = n_classes

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Consolidate predictions of models.

        Args:
            x (np.ndarray): Data matrix.

        Returns:
            np.ndarray: Ensemble label index predictions.
        """
        pass

    def predict_proba(self, x: np.ndarray, indexs: List[int]) -> List[float]:
        """Calculate class distribution probabilities over all estimators.

        Args:
            x (np.ndarray): Single sample array.
            indexs (List[int]): Class index targets.

        Returns:
            List[float]: Normalized combined probabilities list.
        """
        results = np.zeros(len(indexs))
        for model in self._predictors:
            pred_proba = model.predict_proba(x, indexs)
            results += pred_proba
        return (results / len(self._predictors)).tolist()


class MajorityVoter(WeightingVoter):
    """Simple plurality majority vote classification consolidator."""

    def predict(self, X: Any) -> np.ndarray:
        """Perform hard majority voting.

        Args:
            X (Any): Data matrix or single sample.

        Returns:
            np.ndarray: Predicted label index.
        """
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
            votes[np.arange(n_samples), preds] += 1
        return np.argmax(votes, axis=1)


class PerformanceWeightingVoter(WeightingVoter):
    """Performance Weighting consolidator."""

    def predict(self, X: Any) -> np.ndarray:
        """Perform weighted prediction based on OOB performance.

        Args:
            X (Any): Data matrix or single sample.

        Returns:
            np.ndarray: Predicted label indices.
        """
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
    """Weighted Soft Voting strategy for ensemble classification.

    Multiplies each tree's probability distribution by its OOB performance
    weight to obtain a combined weighted soft classification.
    """

    def predict(self, X: Any) -> np.ndarray:
        """Predict labels using performance-weighted class probabilities.

        Args:
            X (Any): Input features matrix.

        Returns:
            np.ndarray: Predicted label indices.
        """
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
                probs = np.array(model.predict_proba(X, class_indices))
                results += probs * w
            return np.argmax(results)

        # Batch mode: loop over samples but calculate weights efficiently
        n_samples = X.shape[0]
        accumulated_probs = np.zeros((n_samples, self._n_classes))
        for model, w in zip(self._predictors, weights):
            for i in range(n_samples):
                probs = np.array(model.predict_proba(X[i], class_indices))
                accumulated_probs[i] += probs * w

        return np.argmax(accumulated_probs, axis=1)


class DistributionSummationVoter(WeightingVoter):
    """Simple class probability summation voter."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Sum model probability predictions and select max index.

        Args:
            x (np.ndarray): Single sample features array.

        Returns:
            np.ndarray: Chosen label index.
        """
        results = np.zeros(self._n_classes)
        class_indices = list(range(self._n_classes))
        for model in self._predictors:
            results += np.array(model.predict_proba(x, class_indices))
        return np.argmax(results)
