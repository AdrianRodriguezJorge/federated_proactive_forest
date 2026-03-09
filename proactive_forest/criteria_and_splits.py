"""
Módulo para criterios de split y lógica de particionamiento en árboles de decisión.
"""
import numpy as np
from abc import ABC, abstractmethod
import proactive_forest.utils as utils


class SplitCriterion(ABC):
    @abstractmethod
    def impurity(self, x):
        pass


class GiniCriterion(SplitCriterion):
    @property
    def name(self):
        return 'gini'

    def impurity(self, x):
        if len(x) == 0:
            return 0.0
        counts = np.bincount(x)
        prob = counts / float(len(x))
        return 1.0 - np.sum(prob * prob)


class EntropyCriterion(SplitCriterion):
    @property
    def name(self):
        return 'entropy'

    def impurity(self, x):
        if len(x) == 0:
            return 0.0
        counts = np.bincount(x)
        prob = counts / float(len(x))
        return -sum(p * np.log2(p) for p in prob if p != 0)


def resolve_split_criterion(name):
    if name == 'gini':
        return GiniCriterion()
    elif name == 'entropy':
        return EntropyCriterion()
    else:
        raise ValueError('Unknown criterion {}'.format(name))


def compute_split_values(x):
    if utils.categorical_data(x):
        return np.unique(x)
    else:
        uniques = np.unique(x)
        return np.array([(uniques[i] + uniques[i + 1]) / 2 for i in range(len(uniques) - 1)])


def compute_split_info(split_criterion, X, y, feature_id, split_value, n_leaf_min):
    y_left, y_right = split_target(X, y, feature_id, split_value)
    n_left, n_right = len(y_left), len(y_right)
    n_min = np.min([n_left, n_right])
    if n_min == 0 or n_min < n_leaf_min:
        return None
    gain = compute_split_gain(split_criterion, y, y_left, y_right)
    return gain, feature_id, split_value


def split_target(X, y, feature_id, value):
    is_categorical = utils.categorical_data(X[:, feature_id])
    if is_categorical:
        return split_categorical_target(X, y, feature_id, value)
    else:
        return split_numerical_target(X, y, feature_id, value)


def split_categorical_target(X, y, feature_id, value):
    mask = X[:, feature_id] == value
    return y[mask], y[~mask]


def split_numerical_target(X, y, feature_id, value):
    mask = X[:, feature_id] <= value
    return y[mask], y[~mask]


def split_categorical_data(X, y, feature_id, value):
    mask = X[:, feature_id] == value
    return X[mask], X[~mask], y[mask], y[~mask]


def split_numerical_data(X, y, feature_id, value):
    mask = X[:, feature_id] <= value
    return X[mask], X[~mask], y[mask], y[~mask]


def compute_split_gain(split_criterion, y, y_left, y_right):
    return (split_criterion.impurity(y)
            - split_criterion.impurity(y_left) * len(y_left) / len(y)
            - split_criterion.impurity(y_right) * len(y_right) / len(y))


class Split:
    def __init__(self, feature_id, value, gain):
        self.feature_id = feature_id
        self.value = value
        self.gain = gain


class SplitChooser(ABC):
    @abstractmethod
    def get_split(self, splits):
        pass


class BestSplitChooser(SplitChooser):
    @property
    def name(self):
        return 'best'

    def get_split(self, splits):
        best_split = None
        if len(splits) > 0:
            best_split = splits[0]
            for i in range(len(splits)):
                if splits[i].gain > best_split.gain:
                    best_split = splits[i]
        return best_split


class RandomSplitChooser(SplitChooser):
    @property
    def name(self):
        return 'rand'

    def get_split(self, splits):
        split = None
        if len(splits) > 0:
            choice = np.random.randint(low=0, high=len(splits))
            split = splits[choice]
        return split


def resolve_split_selection(split_criterion):
    if split_criterion == 'best':
        return BestSplitChooser()
    elif split_criterion == 'rand':
        return RandomSplitChooser()
    else:
        raise ValueError("%s is not a recognizable split chooser." % split_criterion)
