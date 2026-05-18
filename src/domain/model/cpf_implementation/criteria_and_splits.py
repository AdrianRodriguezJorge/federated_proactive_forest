"""Split criteria and partitioning logic for decision trees.

Provides numerical and categorical node splitting criteria (Gini, Entropy) and
split candidate selection strategies (Best, Random) for Proactive Forest trees.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import numpy as np

from . import utils


class SplitCriterion(ABC):
    """Abstract base class for node impurity splitting criteria."""

    @abstractmethod
    def impurity(self, x: np.ndarray) -> float:
        """Calculate impurity of a given array of target labels.

        Args:
            x (np.ndarray): 1D array of target labels.

        Returns:
            float: Calculated impurity score.
        """
        pass


class GiniCriterion(SplitCriterion):
    """Gini impurity metric for multiclass split evaluation."""

    @property
    def name(self) -> str:
        """Return name of the criterion.

        Returns:
            str: "gini"
        """
        return "gini"

    def impurity(self, x: np.ndarray) -> float:
        """Compute Gini impurity.

        Args:
            x (np.ndarray): Target labels.

        Returns:
            float: Gini impurity score.
        """
        if len(x) == 0:
            return 0.0
        counts = np.bincount(x)
        prob = counts / float(len(x))
        return 1.0 - np.sum(prob * prob)

    def impurity_vectorized(
        self, counts: np.ndarray, total: np.ndarray
    ) -> np.ndarray:
        """Calculates Gini for a matrix of counts.

        Args:
            counts (np.ndarray): Matrix of shape (n_thresholds, n_classes).
            total (np.ndarray): Array of shape (n_thresholds,).

        Returns:
            np.ndarray: Array of Gini scores for each threshold.
        """
        # Avoid division by zero
        total_safe = np.where(total > 0, total, 1)
        probs = counts / total_safe[:, np.newaxis]
        return 1.0 - np.sum(probs * probs, axis=1)


class EntropyCriterion(SplitCriterion):
    """Information gain entropy metric for split evaluation."""

    @property
    def name(self) -> str:
        """Return name of the criterion.

        Returns:
            str: "entropy"
        """
        return "entropy"

    def impurity(self, x: np.ndarray) -> float:
        """Compute entropy impurity.

        Args:
            x (np.ndarray): Target labels.

        Returns:
            float: Entropy impurity score.
        """
        if len(x) == 0:
            return 0.0
        counts = np.bincount(x)
        prob = counts / float(len(x))
        prob = prob[prob > 0]
        return -np.sum(prob * np.log2(prob))

    def impurity_vectorized(
        self, counts: np.ndarray, total: np.ndarray
    ) -> np.ndarray:
        """Calculates entropy for a matrix of counts.

        Args:
            counts (np.ndarray): Matrix of shape (n_thresholds, n_classes).
            total (np.ndarray): Array of shape (n_thresholds,).

        Returns:
            np.ndarray: Array of entropy scores for each threshold.
        """
        # Avoid division by zero
        total_safe = np.where(total > 0, total, 1)
        probs = counts / total_safe[:, np.newaxis]
        # Mask zero probabilities for log
        log_probs = np.zeros_like(probs)
        mask = probs > 0
        log_probs[mask] = np.log2(probs[mask])
        return -np.sum(probs * log_probs, axis=1)


def resolve_split_criterion(name: str) -> SplitCriterion:
    """Resolve a split criterion by name.

    Args:
        name (str): Identifier name ("gini" or "entropy").

    Returns:
        SplitCriterion: Evaluator object.

    Raises:
        ValueError: If name is not recognized.
    """
    if name == "gini":
        return GiniCriterion()
    elif name == "entropy":
        return EntropyCriterion()
    else:
        raise ValueError(f"Unknown criterion {name}")


def compute_split_values(x: np.ndarray) -> np.ndarray:
    """Compute potential split candidates for a feature.

    Args:
        x (np.ndarray): 1D feature column values.

    Returns:
        np.ndarray: Potential split values.
    """
    if utils.categorical_data(x):
        return np.unique(x)
    else:
        uniques = np.unique(x)
        max_bins = 100
        if len(uniques) > max_bins:
            uniques = np.unique(
                np.quantile(x, np.linspace(0, 1, max_bins))
            )
        return np.array(
            [
                (uniques[i] + uniques[i + 1]) / 2
                for i in range(len(uniques) - 1)
            ]
        )


def compute_split_info(
    split_criterion: SplitCriterion,
    X: np.ndarray,
    y: np.ndarray,
    feature_id: int,
    split_value: Union[int, float],
    n_leaf_min: int,
    impurity_y: Optional[float] = None,
) -> Optional[Tuple[float, int, Union[int, float]]]:
    """Evaluate split suitability and compute information gain.

    Args:
        split_criterion (SplitCriterion): Evaluation criterion.
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target labels.
        feature_id (int): Feature index being split.
        split_value (Union[int, float]): Candidate split threshold.
        n_leaf_min (int): Minimum samples required in a leaf.
        impurity_y (Optional[float]): Base node impurity. Defaults to None.

    Returns:
        Optional[Tuple[float, int, Union[int, float]]]: Information gain,
            feature_id, and split value if split is valid; otherwise None.
    """
    y_left, y_right = split_target(X, y, feature_id, split_value)
    n_left, n_right = len(y_left), len(y_right)
    n_min = np.min([n_left, n_right])
    if n_min == 0 or n_min < n_leaf_min:
        return None
    if impurity_y is None:
        impurity_y = split_criterion.impurity(y)
    gain = compute_split_gain(
        split_criterion, y, y_left, y_right, impurity_y
    )
    return gain, feature_id, split_value


def split_target(
    X: np.ndarray, y: np.ndarray, feature_id: int, value: Union[int, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Split the target labels based on feature splitting.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target labels.
        feature_id (int): Feature index.
        value (Union[int, float]): Split value threshold.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Left and right target splits.
    """
    is_categorical = utils.categorical_data(X[:, feature_id])
    if is_categorical:
        return split_categorical_target(X, y, feature_id, value)
    else:
        return split_numerical_target(X, y, feature_id, value)


def split_categorical_target(
    X: np.ndarray, y: np.ndarray, feature_id: int, value: Any
) -> Tuple[np.ndarray, np.ndarray]:
    """Split the target labels for a categorical feature.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target labels.
        feature_id (int): Feature index.
        value (Any): Target category value.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Left (equal to value) and right splits.
    """
    mask = X[:, feature_id] == value
    return y[mask], y[~mask]


def split_numerical_target(
    X: np.ndarray, y: np.ndarray, feature_id: int, value: Union[int, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Split the target labels for a numerical feature.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target labels.
        feature_id (int): Feature index.
        value (Union[int, float]): Split threshold.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Left (<= value) and right splits.
    """
    mask = X[:, feature_id] <= value
    return y[mask], y[~mask]


def split_categorical_data(
    X: np.ndarray, y: np.ndarray, feature_id: int, value: Any
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the feature matrix and target labels for a categorical feature.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target labels.
        feature_id (int): Feature index.
        value (Any): Split category value.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Left features,
            right features, left targets, right targets.
    """
    mask = X[:, feature_id] == value
    return X[mask], X[~mask], y[mask], y[~mask]


def split_numerical_data(
    X: np.ndarray, y: np.ndarray, feature_id: int, value: Union[int, float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the feature matrix and target labels for a numerical feature.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target labels.
        feature_id (int): Feature index.
        value (Union[int, float]): Split threshold.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Left features,
            right features, left targets, right targets.
    """
    mask = X[:, feature_id] <= value
    return X[mask], X[~mask], y[mask], y[~mask]


def compute_split_gain(
    split_criterion: SplitCriterion,
    y: np.ndarray,
    y_left: np.ndarray,
    y_right: np.ndarray,
    impurity_y: Optional[float] = None,
) -> float:
    """Calculate the information gain resulting from a split.

    Args:
        split_criterion (SplitCriterion): Evaluation criterion.
        y (np.ndarray): Original target labels.
        y_left (np.ndarray): Left child target labels.
        y_right (np.ndarray): Right child target labels.
        impurity_y (Optional[float]): Base node impurity. Defaults to None.

    Returns:
        float: Computed split information gain.
    """
    if impurity_y is None:
        impurity_y = split_criterion.impurity(y)
    return (
        impurity_y
        - split_criterion.impurity(y_left) * len(y_left) / len(y)
        - split_criterion.impurity(y_right) * len(y_right) / len(y)
    )


class Split:
    """Representation of a node split candidate."""

    def __init__(self, feature_id: int, value: Union[int, float], gain: float):
        """Initialize the split.

        Args:
            feature_id (int): Feature index.
            value (Union[int, float]): Threshold or category value.
            gain (float): Information gain.
        """
        self.feature_id = feature_id
        self.value = value
        self.gain = gain


class SplitChooser(ABC):
    """Abstract selector for choosing split candidates."""

    @abstractmethod
    def get_split(self, splits: List[Split]) -> Optional[Split]:
        """Select a single split from a list of candidates.

        Args:
            splits (List[Split]): Candidate split objects.

        Returns:
            Optional[Split]: The chosen split if candidates exist; else None.
        """
        pass


class BestSplitChooser(SplitChooser):
    """Chooser that always selects the split with the maximum gain."""

    @property
    def name(self) -> str:
        """Return chooser name.

        Returns:
            str: "best"
        """
        return "best"

    def get_split(self, splits: List[Split]) -> Optional[Split]:
        """Select split with best gain.

        Args:
            splits (List[Split]): Candidates.

        Returns:
            Optional[Split]: Split with maximum gain.
        """
        best_split = None
        if len(splits) > 0:
            best_split = splits[0]
            for i in range(len(splits)):
                if splits[i].gain > best_split.gain:
                    best_split = splits[i]
        return best_split


class RandomSplitChooser(SplitChooser):
    """Chooser that selects a candidate split at random."""

    @property
    def name(self) -> str:
        """Return chooser name.

        Returns:
            str: "rand"
        """
        return "rand"

    def get_split(self, splits: List[Split]) -> Optional[Split]:
        """Select split at random.

        Args:
            splits (List[Split]): Candidates.

        Returns:
            Optional[Split]: Randomly chosen split.
        """
        split = None
        if len(splits) > 0:
            choice = np.random.randint(low=0, high=len(splits))
            split = splits[choice]
        return split


def resolve_split_selection(split_criterion: str) -> SplitChooser:
    """Resolve a split chooser by identifier.

    Args:
        split_criterion (str): Selector name ("best" or "rand").

    Returns:
        SplitChooser: Chooser instance.

    Raises:
        ValueError: If chooser is unrecognized.
    """
    if split_criterion == "best":
        return BestSplitChooser()
    elif split_criterion == "rand":
        return RandomSplitChooser()
    else:
        raise ValueError(
            f"{split_criterion} is not a recognizable split chooser."
        )
