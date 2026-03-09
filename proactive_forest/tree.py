"""
Módulo para la estructura de árboles de decisión.
"""
from abc import ABC, abstractmethod
import numpy as np


class DecisionTree:
    def __init__(self, n_features):
        self._n_features = n_features
        self._nodes = []
        self._last_node_id = None
        self._weight = 1

    @property
    def n_features(self):
        return self._n_features

    @n_features.setter
    def n_features(self, n_features):
        self._n_features = n_features

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, nodes):
        self._nodes = nodes

    @property
    def last_node_id(self):
        return self._last_node_id

    @last_node_id.setter
    def last_node_id(self, last_node_id):
        self._last_node_id = last_node_id

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, weight):
        self._weight = weight

    @staticmethod
    def root():
        return 0

    def predict(self, x):
        current_node = self.root()
        leaf_found = False
        prediction = None
        while not leaf_found:
            if isinstance(self._nodes[current_node], DecisionLeaf):
                leaf_found = True
                prediction = self._nodes[current_node].result
            else:
                current_node = self._nodes[current_node].result_branch(x)
        return prediction

    def predict_proba(self, x, indexs):
        current_node = self.root()
        leaf_found = False
        class_proba = None
        while not leaf_found:
            if isinstance(self._nodes[current_node], DecisionLeaf):
                leaf_found = True
                samp = []
                for i in indexs:
                    samp.append(self._nodes[current_node].samples[i])
                class_proba = [n + 1 for n in samp] / (np.sum(samp) + len(samp))
            else:
                current_node = self._nodes[current_node].result_branch(x)
        return class_proba.tolist()

    def feature_importances(self):
        importances = np.zeros(self._n_features)
        for node in self._nodes:
            if isinstance(node, DecisionFork):
                importances[node.feature_id] += node.gain * np.sum(node.samples) / np.sum(
                    self._nodes[self.root()].samples)
        normalizer = np.sum(importances)
        if normalizer > 0:
            importances /= normalizer
        return importances

    def total_nodes(self):
        return len(self._nodes)

    def total_splits(self):
        return sum(1 for n in self._nodes if isinstance(n, DecisionFork))

    def total_leaves(self):
        return sum(1 for n in self._nodes if isinstance(n, DecisionLeaf))


class DecisionNode(ABC):
    def __init__(self, samples, depth):
        self._samples = samples
        self._depth = depth
        super().__init__()

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, samples):
        self._samples = samples

    @property
    def depth(self):
        return self._depth

    @depth.setter
    def depth(self, depth):
        self._depth = depth


class DecisionFork(DecisionNode):
    def __init__(self, samples, depth, feature_id, gain, value):
        self._feature_id = feature_id
        self._gain = gain
        self._left_branch = None
        self._right_branch = None
        self._value = value
        super().__init__(samples, depth)

    @property
    def feature_id(self):
        return self._feature_id

    @feature_id.setter
    def feature_id(self, feature_id):
        self._feature_id = feature_id

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, gain):
        self._gain = gain

    @property
    def left_branch(self):
        return self._left_branch

    @left_branch.setter
    def left_branch(self, left_branch):
        self._left_branch = left_branch

    @property
    def right_branch(self):
        return self._right_branch

    @right_branch.setter
    def right_branch(self, right_branch):
        self._right_branch = right_branch

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @abstractmethod
    def result_branch(self, x):
        pass


class DecisionForkNumerical(DecisionFork):
    def result_branch(self, x):
        if x[self.feature_id] <= self.value:
            return self.left_branch
        else:
            return self.right_branch


class DecisionForkCategorical(DecisionFork):
    def result_branch(self, x):
        if x[self.feature_id] == self.value:
            return self.left_branch
        else:
            return self.right_branch


class DecisionLeaf(DecisionNode):
    def __init__(self, samples, depth, result):
        super().__init__(samples, depth)
        self._result = result

    @property
    def result(self):
        return self._result

    @result.setter
    def result(self, result):
        self._result = result
