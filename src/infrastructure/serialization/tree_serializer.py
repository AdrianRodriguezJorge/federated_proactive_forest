"""Tree serialization utilities."""
import pickle
from typing import Any, List
import joblib


class TreeSerializer:
    """
    Utilities for serializing and deserializing decision trees.
    """

    @staticmethod
    def serialize_tree(tree: Any) -> bytes:
        """
        Serialize a single tree to bytes.

        Args:
            tree: Decision tree object

        Returns:
            Serialized tree as bytes
        """
        return pickle.dumps(tree)

    @staticmethod
    def deserialize_tree(data: bytes) -> Any:
        """
        Deserialize a tree from bytes.

        Args:
            data: Serialized tree data

        Returns:
            Deserialized tree object
        """
        return pickle.loads(data)

    @staticmethod
    def serialize_forest(trees: List[Any]) -> bytes:
        """
        Serialize a list of trees.

        Args:
            trees: List of tree objects

        Returns:
            Serialized forest as bytes
        """
        return joblib.dumps(trees)

    @staticmethod
    def deserialize_forest(data: bytes) -> List[Any]:
        """
        Deserialize a forest from bytes.

        Args:
            data: Serialized forest data

        Returns:
            List of deserialized trees
        """
        return joblib.loads(data)