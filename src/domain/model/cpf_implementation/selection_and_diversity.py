"""Feature subset selection and ensemble diversity measures.

Provides diversity metrics (Percentage Correct Diversity, Q-Statistic) and
feature subset selectors (All, Log2-sized subset, Probability-weighted sample).
"""

from abc import ABC, abstractmethod
import math
from typing import Any, List, Optional
import numpy as np


class DiversityMeasure(ABC):
    """Abstract baseline class for ensemble diversity measurement."""

    @abstractmethod
    def get_measure(
        self, predictors: List[Any], X: np.ndarray, y: np.ndarray
    ) -> float:
        """Calculate the diversity score of an ensemble on a dataset.

        Args:
            predictors (List[Any]): List of base model estimators.
            X (np.ndarray): Input features matrix.
            y (np.ndarray): Target labels array.

        Returns:
            float: Calculated diversity score.
        """
        pass


class PercentageCorrectDiversity(DiversityMeasure):
    """Percentage Correct Diversity (PCD) metric for ensemble diversity."""

    def get_measure(
        self, predictors: List[Any], X: np.ndarray, y: np.ndarray
    ) -> float:
        """Calculate the Percentage Correct Diversity (PCD) score.

        Args:
            predictors (List[Any]): List of base estimators.
            X (np.ndarray): Feature matrix.
            y (np.ndarray): True labels.

        Returns:
            float: PCD diversity score.
        """
        if not predictors:
            return 0.0

        n_instances = X.shape[0]
        n_predictors = len(predictors)

        # Vectorized prediction across all estimators
        all_preds = np.array([p.predict(X) for p in predictors])

        # Compare with true labels (broadcasting)
        correct_mask = all_preds == y

        # Count correct predictors per instance
        n_corrects = np.sum(correct_mask, axis=0)

        # Apply PCD thresholds
        lower_bound = 0.1 * n_predictors
        upper_bound = 0.9 * n_predictors

        tally = np.sum(
            (n_corrects >= lower_bound) & (n_corrects <= upper_bound)
        )

        return tally / n_instances


class QStatisticDiversity(DiversityMeasure):
    """Yule's Q-statistic diversity measure for pair-wise disagreement."""

    def get_measure(
        self, predictors: List[Any], X: np.ndarray, y: np.ndarray
    ) -> float:
        """Calculate the average pairwise Q-statistic score.

        Args:
            predictors (List[Any]): List of base estimators.
            X (np.ndarray): Feature matrix.
            y (np.ndarray): True labels.

        Returns:
            float: Average pairwise Q-statistic diversity score.
        """
        if not predictors or len(predictors) < 2:
            return 0.0

        n_predictors = len(predictors)

        # Matrix of successes
        successes = np.array(
            [(p.predict(X) == y).astype(int) for p in predictors]
        )
        failures = 1 - successes

        q_total = 0.0
        for i in range(0, n_predictors - 1):
            for j in range(i + 1, n_predictors):
                # Calculate contingency counts using dot products
                n11 = np.sum(successes[i] * successes[j])
                n00 = np.sum(failures[i] * failures[j])
                n10 = np.sum(successes[i] * failures[j])
                n01 = np.sum(failures[i] * successes[j])

                # Laplace smoothing to prevent division by zero
                n11 = n11 if n11 > 0 else 1
                n00 = n00 if n00 > 0 else 1
                n10 = n10 if n10 > 0 else 1
                n01 = n01 if n01 > 0 else 1

                same = n11 * n00
                diff = n10 * n01
                q_total += (same - diff) / (same + diff)

        return 2 * q_total / (n_predictors * (n_predictors - 1))


class FeatureSelection(ABC):
    """Abstract base class for sub-space feature selection strategies."""

    @abstractmethod
    def get_features(
        self, n_features: int, prob: Optional[List[float]] = None
    ) -> List[int]:
        """Select a subset of feature indices.

        Args:
            n_features (int): Number of total features available.
            prob (Optional[List[float]]): Array of selection probabilities.

        Returns:
            List[int]: Selected feature indices.
        """
        pass


class AllFeatureSelection(FeatureSelection):
    """Features selector selecting all available feature indices."""

    @property
    def name(self) -> str:
        """Return selector name.

        Returns:
            str: "all"
        """
        return "all"

    def get_features(
        self, n_features: int, prob: Optional[List[float]] = None
    ) -> List[int]:
        """Select all features.

        Args:
            n_features (int): Feature count.
            prob (Optional[List[float]]): Ignored.

        Returns:
            List[int]: All feature indices.
        """
        return list(range(n_features))


class LogFeatureSelection(FeatureSelection):
    """Feature selector using log2(n_features) + 1 sized random sample."""

    @property
    def name(self) -> str:
        """Return selector name.

        Returns:
            str: "log"
        """
        return "log"

    def get_features(
        self, n_features: int, prob: Optional[List[float]] = None
    ) -> np.ndarray:
        """Select a log-sized random subset of features.

        Args:
            n_features (int): Total features.
            prob (Optional[List[float]]): Option probabilities.

        Returns:
            np.ndarray: Selected feature indices.
        """
        sample_size = int(math.floor(math.log2(n_features)) + 1)
        population = list(range(n_features))
        selected = np.random.choice(
            population, replace=False, size=sample_size, p=prob
        )
        return selected


class ProbFeatureSelection(FeatureSelection):
    """Feature selector drawing n_features sample with replacement."""

    @property
    def name(self) -> str:
        """Return selector name.

        Returns:
            str: "prob"
        """
        return "prob"

    def get_features(
        self, n_features: int, prob: Optional[List[float]] = None
    ) -> np.ndarray:
        """Select a sample of size n_features with replacement.

        Args:
            n_features (int): Total features.
            prob (Optional[List[float]]): Probabilities.

        Returns:
            np.ndarray: Unique selected feature indices.
        """
        sample_size = n_features
        population = list(range(n_features))
        selected = np.random.choice(
            population, replace=True, size=sample_size, p=prob
        )
        return np.unique(selected)


def resolve_feature_selection(name: str) -> FeatureSelection:
    """Resolve a feature selection selector by identifier name.

    Args:
        name (str): Identifier name ("all", "log", "prob").

    Returns:
        FeatureSelection: Selector instance.

    Raises:
        ValueError: If name is not recognized.
    """
    if name == "all":
        return AllFeatureSelection()
    elif name == "log":
        return LogFeatureSelection()
    elif name == "prob":
        return ProbFeatureSelection()
    else:
        raise ValueError(f"Unknown feature selection criterion {name}")
