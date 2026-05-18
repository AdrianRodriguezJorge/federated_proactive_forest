"""Management of feature selection probabilities in Proactive Forest.

Tracks feature importance over iterations to dynamically adjust selection weights
in consecutive trees, balancing exploration and exploitation.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional
import numpy as np


class ProbabilityLedger(ABC):
    """Abstract base class tracking feature selection probabilities."""

    def __init__(
        self,
        probabilities: Optional[List[float]],
        n_features: int,
        alpha: float,
    ):
        """Initializes selection probability state.

        Args:
            probabilities (Optional[List[float]]): Predefined feature weights.
                If None, defaults to uniform probabilities.
            n_features (int): Number of feature dimensions.
            alpha (float): Scaling factor (diversity rate).

        Raises:
            ValueError: If specifications are mismatched or invalid.
        """
        if probabilities is None:
            if n_features is not None and n_features > 0:
                initial_p = 1.0 / n_features
                self._probabilities = np.full(n_features, initial_p)
            else:
                raise ValueError(
                    "Cannot initialize ledger without a positive "
                    "number of features."
                )
        else:
            if len(probabilities) == n_features:
                self._probabilities = np.array(probabilities, dtype=float)
            else:
                raise ValueError(
                    "Number of features must match the length "
                    "of the list of probabilities."
                )
        self._n_features = n_features
        self._alpha = alpha
        super().__init__()

    @abstractmethod
    def update_probabilities(self, new_tree: Any, rate: float) -> None:
        """Update ledger weights dynamically based on a new estimator.

        Args:
            new_tree (Any): The newly constructed tree.
            rate (float): Normalized index rate of progress (e.g. i / total).
        """
        pass

    def _normalize(self) -> None:
        """Normalize probability weights to sum to 1.0."""
        total = np.sum(self._probabilities)
        if total > 0:
            self._probabilities /= total

    @property
    def probabilities(self) -> List[float]:
        """Get probabilities as a list of floats.

        Returns:
            List[float]: Normalised select probabilities.
        """
        return self._probabilities.tolist()

    @probabilities.setter
    def probabilities(self, probabilities: List[float]) -> None:
        """Set probabilities array.

        Args:
            probabilities (List[float]): Target weight array.
        """
        self._probabilities = np.array(probabilities, dtype=float)

    @property
    def n_features(self) -> int:
        """Get feature dimension size.

        Returns:
            int: Dimension size.
        """
        return self._n_features

    @property
    def alpha(self) -> float:
        """Get the diversity alpha scaling factor.

        Returns:
            float: Scaling factor alpha.
        """
        return self._alpha


class FIProbabilityLedger(ProbabilityLedger):
    """Exploration ledger scaling feature importances over iterations.

    Formula:
        p_new = p_old * (1 - cumulative_FI * alpha * rate)
    """

    def __init__(
        self,
        probabilities: Optional[List[float]],
        n_features: int,
        alpha: float = 0.1,
    ):
        """Initializes Feature Importance-based probability ledger."""
        self._feature_importances = np.zeros(n_features)
        self._n_trees = 0
        super().__init__(probabilities, n_features, alpha)

    def update_probabilities(self, new_tree: Any, rate: float) -> None:
        """Update probability weights based on tree feature importance.

        Args:
            new_tree (Any): New decision tree instance.
            rate (float): Progress index of the forest training.
        """
        self._feature_importances += new_tree.feature_importances()
        self._n_trees += 1
        self._probabilities = self._probabilities * (
            1
            - (self._feature_importances / self._n_trees) * self._alpha * rate
        )
        self._normalize()
