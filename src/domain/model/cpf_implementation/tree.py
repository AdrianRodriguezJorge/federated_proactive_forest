"""Decision tree model representation and inference traversal.

Provides Node, Fork (Numerical, Categorical), and Leaf objects representing the
decision tree nodes, and tree traversal for batch/single sample classification.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union
import numpy as np


class DecisionNode(ABC):
    """Abstract base class representing a single node in a decision tree."""

    def __init__(self, samples: np.ndarray, depth: int):
        """Initialize decision node.

        Args:
            samples (np.ndarray): Target class count distribution or samples list.
            depth (int): The recursive depth level of the node.
        """
        self._samples = samples
        self._depth = depth
        super().__init__()

    @property
    def samples(self) -> np.ndarray:
        """Get sample distribution.

        Returns:
            np.ndarray: Count distribution array.
        """
        return self._samples

    @samples.setter
    def samples(self, samples: np.ndarray) -> None:
        """Set sample distribution.

        Args:
            samples (np.ndarray): Count distribution.
        """
        self._samples = samples

    @property
    def depth(self) -> int:
        """Get the tree depth.

        Returns:
            int: Tree depth.
        """
        return self._depth

    @depth.setter
    def depth(self, depth: int) -> None:
        """Set the tree depth.

        Args:
            depth (int): Tree depth.
        """
        self._depth = depth


class DecisionFork(DecisionNode):
    """Represents an internal decision tree splitting fork."""

    def __init__(
        self,
        samples: np.ndarray,
        depth: int,
        feature_id: int,
        gain: float,
        value: Union[int, float],
    ):
        """Initialize DecisionFork.

        Args:
            samples (np.ndarray): Node class counts.
            depth (int): Depth level.
            feature_id (int): Splitting feature index.
            gain (float): Information gain.
            value (Union[int, float]): Split threshold or category.
        """
        self._feature_id = feature_id
        self._gain = gain
        self._left_branch = None
        self._right_branch = None
        self._value = value
        super().__init__(samples, depth)

    @property
    def feature_id(self) -> int:
        """Get split feature index.

        Returns:
            int: Feature index.
        """
        return self._feature_id

    @feature_id.setter
    def feature_id(self, feature_id: int) -> None:
        """Set split feature index.

        Args:
            feature_id (int): Feature index.
        """
        self._feature_id = feature_id

    @property
    def gain(self) -> float:
        """Get split gain.

        Returns:
            float: Information gain score.
        """
        return self._gain

    @gain.setter
    def gain(self, gain: float) -> None:
        """Set split gain.

        Args:
            gain (float): Information gain score.
        """
        self._gain = gain

    @property
    def left_branch(self) -> Optional[int]:
        """Get left branch node ID.

        Returns:
            Optional[int]: Left node index.
        """
        return self._left_branch

    @left_branch.setter
    def left_branch(self, left_branch: Optional[int]) -> None:
        """Set left branch node ID.

        Args:
            left_branch (Optional[int]): Left node ID.
        """
        self._left_branch = left_branch

    @property
    def right_branch(self) -> Optional[int]:
        """Get right branch node ID.

        Returns:
            Optional[int]: Right node index.
        """
        return self._right_branch

    @right_branch.setter
    def right_branch(self, right_branch: Optional[int]) -> None:
        """Set right branch node ID.

        Args:
            right_branch (Optional[int]): Right node ID.
        """
        self._right_branch = right_branch

    @property
    def value(self) -> Union[int, float]:
        """Get splitting threshold or category value.

        Returns:
            Union[int, float]: Value.
        """
        return self._value

    @value.setter
    def value(self, value: Union[int, float]) -> None:
        """Set splitting value.

        Args:
            value (Union[int, float]): Split value.
        """
        self._value = value

    @abstractmethod
    def result_branch(self, x: np.ndarray) -> int:
        """Evaluate single sample feature value and select branch ID.

        Args:
            x (np.ndarray): 1D sample features array.

        Returns:
            int: Traversed branch child node ID.
        """
        pass


class DecisionForkNumerical(DecisionFork):
    """Splitting node for continuous numerical attributes."""

    def result_branch(self, x: np.ndarray) -> int:
        """Traverse numeric branch threshold <= value.

        Args:
            x (np.ndarray): 1D sample.

        Returns:
            int: Child node index.
        """
        if x[self.feature_id] <= self.value:
            return self.left_branch
        else:
            return self.right_branch


class DecisionForkCategorical(DecisionFork):
    """Splitting node for categorical attributes."""

    def result_branch(self, x: np.ndarray) -> int:
        """Traverse categorical branch feature == value.

        Args:
            x (np.ndarray): 1D sample.

        Returns:
            int: Child node index.
        """
        if x[self.feature_id] == self.value:
            return self.left_branch
        else:
            return self.right_branch


class DecisionLeaf(DecisionNode):
    """Leaf node containing the final target label result class."""

    def __init__(self, samples: np.ndarray, depth: int, result: Any):
        """Initialize DecisionLeaf.

        Args:
            samples (np.ndarray): Node class count distribution.
            depth (int): Depth level.
            result (Any): Node predicted class label.
        """
        super().__init__(samples, depth)
        self._result = result

    @property
    def result(self) -> Any:
        """Get leaf predicted label.

        Returns:
            Any: Predicted class label.
        """
        return self._result

    @result.setter
    def result(self, result: Any) -> None:
        """Set leaf predicted label.

        Args:
            result (Any): Predicted class label.
        """
        self._result = result


class DecisionTree:
    """Decision Tree data structure storing nodes list."""

    def __init__(self, n_features: int):
        """Initializes decision tree structure.

        Args:
            n_features (int): Number of input feature dimensions.
        """
        self._n_features = n_features
        self._nodes = []
        self._last_node_id = None
        self._weight = 1.0

    @property
    def n_features(self) -> int:
        """Get feature count.

        Returns:
            int: Feature dimensions size.
        """
        return self._n_features

    @n_features.setter
    def n_features(self, n_features: int) -> None:
        """Set feature count.

        Args:
            n_features (int): Feature dimensions size.
        """
        self._n_features = n_features

    @property
    def nodes(self) -> List[DecisionNode]:
        """Get nodes list.

        Returns:
            List[DecisionNode]: Sequential nodes list.
        """
        return self._nodes

    @nodes.setter
    def nodes(self, nodes: List[DecisionNode]) -> None:
        """Set nodes list.

        Args:
            nodes (List[DecisionNode]): Nodes list.
        """
        self._nodes = nodes

    @property
    def last_node_id(self) -> Optional[int]:
        """Get the last registered node ID.

        Returns:
            Optional[int]: The last node index.
        """
        return self._last_node_id

    @last_node_id.setter
    def last_node_id(self, last_node_id: Optional[int]) -> None:
        """Set the last node ID.

        Args:
            last_node_id (Optional[int]): Last node ID.
        """
        self._last_node_id = last_node_id

    @property
    def weight(self) -> float:
        """Get current OOB estimator performance weight.

        Returns:
            float: Performance weight.
        """
        return self._weight

    @weight.setter
    def weight(self, weight: float) -> None:
        """Set performance weight.

        Args:
            weight (float): Weight value.
        """
        self._weight = weight

    @staticmethod
    def root() -> int:
        """Get index of the root node.

        Returns:
            int: 0
        """
        return 0

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict classes for a dataset.

        Args:
            x (np.ndarray): Feature matrix (1D or 2D).

        Returns:
            np.ndarray: Predicted class labels.
        """
        x = np.asarray(x)
        if x.ndim == 1:
            return self._predict_single(x)

        predictions = np.zeros(x.shape[0], dtype=object)
        indices = np.arange(x.shape[0])

        def push_samples(node_id: int, idx: np.ndarray) -> None:
            if len(idx) == 0:
                return
            node = self._nodes[node_id]
            if isinstance(node, DecisionLeaf):
                predictions[idx] = node.result
            else:
                if isinstance(node, DecisionForkNumerical):
                    left_mask = x[idx, node.feature_id] <= node.value
                else:  # Categorical
                    left_mask = x[idx, node.feature_id] == node.value

                push_samples(node.left_branch, idx[left_mask])
                push_samples(node.right_branch, idx[~left_mask])

        push_samples(self.root(), indices)
        return predictions

    def _predict_single(self, x: np.ndarray) -> Any:
        """Predict labels for a single sample index.

        Args:
            x (np.ndarray): Single sample 1D array.

        Returns:
            Any: Predicted label.
        """
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

    def predict_proba(self, x: np.ndarray, indexs: List[int]) -> List[float]:
        """Predict class distribution probability for a single sample.

        Args:
            x (np.ndarray): Sample 1D array.
            indexs (List[int]): Class index targets.

        Returns:
            List[float]: Probability distribution list.
        """
        current_node = self.root()
        leaf_found = False
        class_proba = None
        while not leaf_found:
            if isinstance(self._nodes[current_node], DecisionLeaf):
                leaf_found = True
                samp = []
                for i in indexs:
                    if i < len(self._nodes[current_node].samples):
                        samp.append(self._nodes[current_node].samples[i])
                    else:
                        samp.append(0)
                class_proba = [n + 1 for n in samp] / (
                    np.sum(samp) + len(samp)
                )
            else:
                current_node = self._nodes[current_node].result_branch(x)
        return class_proba.tolist()

    def feature_importances(self) -> np.ndarray:
        """Calculate calculated feature importances for this tree.

        Returns:
            np.ndarray: Importances array.
        """
        importances = np.zeros(self._n_features)
        root_samples_sum = np.sum(self._nodes[self.root()].samples)

        for node in self._nodes:
            if isinstance(node, DecisionFork):
                node_samples_sum = np.sum(node.samples)
                importances[node.feature_id] += (
                    node.gain * node_samples_sum / root_samples_sum
                )

        normalizer = np.sum(importances)
        if normalizer > 0:
            importances /= normalizer
        return importances

    def total_nodes(self) -> int:
        """Return total nodes size.

        Returns:
            int: Node count.
        """
        return len(self._nodes)

    def total_splits(self) -> int:
        """Return total splits size.

        Returns:
            int: Split fork count.
        """
        return sum(1 for n in self._nodes if isinstance(n, DecisionFork))

    def total_leaves(self) -> int:
        """Return total leaves size.

        Returns:
            int: Leaf count.
        """
        return sum(1 for n in self._nodes if isinstance(n, DecisionLeaf))
