"""
Módulo para construir árboles de decisión de manera recursiva.
"""
import numpy as np
from . import utils
from .tree import (DecisionTree, DecisionLeaf,
                   DecisionForkCategorical, DecisionForkNumerical)
from .criteria_and_splits import (compute_split_info, split_categorical_data,
                                  split_numerical_data, Split, compute_split_values)


class TreeBuilder:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 split_criterion=None, feature_selection=None, feature_prob=None,
                 min_gain_split=0, split_chooser=None):
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
            raise ValueError("The min_gain_split must be greater or equal than 0.")
        self._feature_prob = feature_prob

    @property
    def feature_prob(self):
        return self._feature_prob

    @feature_prob.setter
    def feature_prob(self, feature_prob):
        self._feature_prob = feature_prob

    def build_tree(self, X, y, n_classes):
        n_samples, n_features = X.shape
        if n_classes <= 0:
            raise ValueError("The number of classes must be greater than 0.")
        self._n_classes = n_classes
        if self._feature_prob is None:
            initial_prob = 1 / n_features
            self._feature_prob = [initial_prob for _ in range(n_features)]
        elif len(self._feature_prob) != n_features:
            raise ValueError('The number of features does not match the given probabilities list.')
        tree = DecisionTree(n_features=n_features)
        tree.last_node_id = tree.root()
        self._build_tree_recursive(tree, tree.last_node_id, X, y, depth=1)
        return tree

    def _build_tree_recursive(self, tree, cur_node, X, y, depth):
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
            tree.nodes.append(DecisionLeaf(samples=samples, depth=depth, result=result))
        else:
            is_categorical = utils.categorical_data(X[:, best_split.feature_id])
            samples = utils.bin_count(y, length=self._n_classes)
            if is_categorical:
                new_fork = DecisionForkCategorical(samples=samples, depth=depth,
                                                   feature_id=best_split.feature_id,
                                                   value=best_split.value, gain=best_split.gain)
                X_left, X_right, y_left, y_right = split_categorical_data(X, y, best_split.feature_id, best_split.value)
            else:
                new_fork = DecisionForkNumerical(samples=samples, depth=depth,
                                                 feature_id=best_split.feature_id,
                                                 value=best_split.value, gain=best_split.gain)
                X_left, X_right, y_left, y_right = split_numerical_data(X, y, best_split.feature_id, best_split.value)
            tree.nodes.append(new_fork)
            tree.last_node_id += 1
            node_to_split = tree.last_node_id
            new_branch = self._build_tree_recursive(tree, node_to_split, X_left, y_left, depth=depth + 1)
            tree.nodes[cur_node].left_branch = new_branch
            tree.last_node_id += 1
            node_to_split = tree.last_node_id
            new_branch = self._build_tree_recursive(tree, node_to_split, X_right, y_right, depth=depth + 1)
            tree.nodes[cur_node].right_branch = new_branch
        return cur_node

    def _find_split(self, X, y, n_features):
        splits_info = []
        features = self._feature_selection.get_features(n_features, self._feature_prob)
        for feature_id in features:
            for split_value in compute_split_values(X[:, feature_id]):
                splits_info.append(
                    compute_split_info(self._split_criterion, X, y, feature_id, split_value, self._min_samples_leaf))
        splits = []
        for split_info in splits_info:
            if split_info is not None:
                gain, feature_id, split_value = split_info
                splits.append(Split(feature_id, value=split_value, gain=gain))
        return self._split_chooser.get_split(splits)
