"""Recursive decision tree builder.

Generates decision tree nodes using Gini/Entropy metrics and vectorized search
thresholds.
"""

from typing import Any, List, Optional
import numpy as np

from .criteria_and_splits import (
    Split,
    compute_split_info,
    compute_split_values,
    split_categorical_data,
    split_numerical_data,
)
from .tree import (
    DecisionForkCategorical,
    DecisionForkNumerical,
    DecisionLeaf,
    DecisionTree,
)
from . import utils


class TreeBuilder:
    """Builder class constructing decision trees recursively from datasets."""

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        split_criterion: Any = None,
        feature_selection: Any = None,
        feature_prob: Optional[List[float]] = None,
        min_gain_split: float = 0.0,
        split_chooser: Any = None,
    ):
        """Initialize TreeBuilder parameters."""
        self._n_classes = None
        if max_depth is None or max_depth > 0:
            self._max_depth = max_depth
        else:
            raise ValueError("The depth of the tree must be greater than 0.")
        if split_criterion is not None:
            self._split_criterion = split_criterion
        else:
            raise ValueError("The split criterion can not be None.")
        if split_chooser is not None:
            self._split_chooser = split_chooser
        else:
            raise ValueError("The split chooser can not be None.")
        if feature_selection is not None:
            self._feature_selection = feature_selection
        else:
            raise ValueError("The feature selection can not be None.")
        if min_samples_split is not None and min_samples_split > 1:
            self._min_samples_split = min_samples_split
        else:
            raise ValueError("The min_samples_split must be greater than 1.")
        if min_samples_leaf is not None and min_samples_leaf > 0:
            self._min_samples_leaf = min_samples_leaf
        else:
            raise ValueError("The min_samples_leaf must be greater than 0.")
        if min_gain_split is not None and min_gain_split >= 0:
            self._min_gain_split = min_gain_split
        else:
            raise ValueError(
                "The min_gain_split must be greater or equal than 0."
            )
        self._feature_prob = feature_prob

    @property
    def feature_prob(self) -> Optional[List[float]]:
        """Get selection probabilities list.

        Returns:
            Optional[List[float]]: Vector of features select probabilities.
        """
        return self._feature_prob

    @feature_prob.setter
    def feature_prob(self, feature_prob: Optional[List[float]]) -> None:
        """Set selection probabilities list.

        Args:
            feature_prob (Optional[List[float]]): Array of selection weights.
        """
        self._feature_prob = feature_prob

    def build_tree(
        self, X: np.ndarray, y: np.ndarray, n_classes: int
    ) -> DecisionTree:
        """Build decision tree classifier recursively.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Targets.
            n_classes (int): Number of classes.

        Returns:
            DecisionTree: The fully built DecisionTree instance.
        """
        n_samples, n_features = X.shape
        if n_classes <= 0:
            raise ValueError("The number of classes must be greater than 0.")
        self._n_classes = n_classes
        if self._feature_prob is None:
            initial_prob = 1.0 / n_features
            self._feature_prob = [initial_prob for _ in range(n_features)]
        elif len(self._feature_prob) != n_features:
            raise ValueError(
                "The number of features does not match "
                "the given probabilities list."
            )
        tree = DecisionTree(n_features=n_features)
        tree.last_node_id = tree.root()
        self._build_tree_recursive(tree, tree.last_node_id, X, y, depth=1)
        return tree

    def _build_tree_recursive(
        self, tree: DecisionTree, cur_node: int, X: np.ndarray, y: np.ndarray, depth: int
    ) -> int:
        """Helper to recursively build the tree branches.

        Args:
            tree (DecisionTree): Main tree instance.
            cur_node (int): Index of the current parent node.
            X (np.ndarray): Segment feature values.
            y (np.ndarray): Segment targets.
            depth (int): Current depth level.

        Returns:
            int: Traversed node ID.
        """
        n_samples, n_features = X.shape
        leaf_reached = False
        if utils.all_instances_same_class(y):
            leaf_reached = True
        if n_samples < self._min_samples_split:
            leaf_reached = True
        if self._max_depth is not None and depth >= self._max_depth:
            leaf_reached = True
        best_split = None
        if not leaf_reached:
            best_split = self._find_split(X, y, n_features)
            if best_split is None or best_split.gain < self._min_gain_split:
                leaf_reached = True
        if leaf_reached:
            samples = utils.bin_count(y, length=self._n_classes)
            result = np.argmax(samples)
            tree.nodes.append(
                DecisionLeaf(samples=samples, depth=depth, result=result)
            )
        else:
            is_categorical = utils.categorical_data(
                X[:, best_split.feature_id]
            )
            samples = utils.bin_count(y, length=self._n_classes)
            if is_categorical:
                new_fork = DecisionForkCategorical(
                    samples=samples,
                    depth=depth,
                    feature_id=best_split.feature_id,
                    value=best_split.value,
                    gain=best_split.gain,
                )
                X_left, X_right, y_left, y_right = split_categorical_data(
                    X, y, best_split.feature_id, best_split.value
                )
            else:
                new_fork = DecisionForkNumerical(
                    samples=samples,
                    depth=depth,
                    feature_id=best_split.feature_id,
                    value=best_split.value,
                    gain=best_split.gain,
                )
                X_left, X_right, y_left, y_right = split_numerical_data(
                    X, y, best_split.feature_id, best_split.value
                )
            tree.nodes.append(new_fork)
            tree.last_node_id += 1
            node_to_split = tree.last_node_id
            new_branch = self._build_tree_recursive(
                tree, node_to_split, X_left, y_left, depth=depth + 1
            )
            tree.nodes[cur_node].left_branch = new_branch
            tree.last_node_id += 1
            node_to_split = tree.last_node_id
            new_branch = self._build_tree_recursive(
                tree, node_to_split, X_right, y_right, depth=depth + 1
            )
            tree.nodes[cur_node].right_branch = new_branch
        return cur_node

    def _find_split(
        self, X: np.ndarray, y: np.ndarray, n_features: int
    ) -> Optional[Split]:
        """Find the optimal split candidate across selected feature subset.

        Args:
            X (np.ndarray): Data feature matrix.
            y (np.ndarray): Target class labels.
            n_features (int): Total dimensions size.

        Returns:
            Optional[Split]: Best split candidate if found; else None.
        """
        best_overall_split = None
        features = self._feature_selection.get_features(
            n_features, self._feature_prob
        )
        impurity_y = self._split_criterion.impurity(y)

        for feature_id in features:
            x_f = X[:, feature_id]
            is_categorical = utils.categorical_data(x_f)

            if is_categorical:
                for split_value in compute_split_values(x_f):
                    split_info = compute_split_info(
                        self._split_criterion,
                        X,
                        y,
                        feature_id,
                        split_value,
                        self._min_samples_leaf,
                        impurity_y,
                    )
                    if split_info:
                        gain, f_id, val = split_info
                        if (
                            best_overall_split is None
                            or gain > best_overall_split.gain
                        ):
                            best_overall_split = Split(f_id, val, gain)
            else:
                # Vectorized search for numeric split thresholds
                sort_idx = np.argsort(x_f)
                x_sorted = x_f[sort_idx]
                y_sorted = y[sort_idx]

                diff_idx = np.where(x_sorted[:-1] != x_sorted[1:])[0]
                if len(diff_idx) == 0:
                    continue

                max_bins = 100
                if len(diff_idx) > max_bins:
                    step = len(diff_idx) // max_bins
                    diff_idx = diff_idx[::step][:max_bins]

                n_samples = len(y)
                y_one_hot = np.zeros((n_samples, self._n_classes))
                y_one_hot[np.arange(n_samples), y_sorted] = 1

                left_counts = np.cumsum(y_one_hot, axis=0)[diff_idx]
                right_counts = np.sum(y_one_hot, axis=0) - left_counts

                left_total = diff_idx + 1
                right_total = n_samples - left_total

                mask = (left_total >= self._min_samples_leaf) & (
                    right_total >= self._min_samples_leaf
                )
                if not np.any(mask):
                    continue

                left_counts, right_counts = (
                    left_counts[mask],
                    right_counts[mask],
                )
                left_total, right_total = (
                    left_total[mask],
                    right_total[mask],
                )
                diff_idx_masked = diff_idx[mask]

                if hasattr(self._split_criterion, "impurity_vectorized"):
                    left_impurity = (
                        self._split_criterion.impurity_vectorized(
                            left_counts, left_total
                        )
                    )
                    right_impurity = (
                        self._split_criterion.impurity_vectorized(
                            right_counts, right_total
                        )
                    )
                else:
                    left_impurity = np.array(
                        [
                            self._split_criterion.impurity(
                                y_sorted[: idx + 1]
                            )
                            for idx in diff_idx_masked
                        ]
                    )
                    right_impurity = np.array(
                        [
                            self._split_criterion.impurity(
                                y_sorted[idx + 1 :]
                            )
                            for idx in diff_idx_masked
                        ]
                    )

                gains = (
                    impurity_y
                    - (left_impurity * left_total / n_samples)
                    - (right_impurity * right_total / n_samples)
                )

                if len(gains) > 0:
                    best_idx = np.argmax(gains)
                    max_gain = gains[best_idx]

                    if (
                        best_overall_split is None
                        or max_gain > best_overall_split.gain
                    ):
                        val = (
                            x_sorted[diff_idx_masked[best_idx]]
                            + x_sorted[diff_idx_masked[best_idx] + 1]
                        ) / 2.0
                        best_overall_split = Split(feature_id, val, max_gain)

        return best_overall_split
