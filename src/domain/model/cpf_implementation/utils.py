"""Utility functions for decision tree and ensemble model operations.

Includes array value verification, categorical datatype detection, class
distribution bin counting, and weighted random sample drawing.
"""

from typing import Any, List, Optional
import numpy as np


def all_instances_same_class(x: np.ndarray) -> bool:
    """Check if all targets in an array belong to the same class.

    Args:
        x (np.ndarray): 1D array of target labels.

    Returns:
        bool: True if only one class exists; else False.
    """
    return len(np.unique(x)) == 1


def categorical_data(x: np.ndarray) -> bool:
    """Check if an array contains categorical string data.

    Args:
        x (np.ndarray): 1D array of values.

    Returns:
        bool: True if the first item is a string; else False.
    """
    return isinstance(x[0], str)


def bin_count(x: np.ndarray, length: int) -> List[int]:
    """Calculate target class bin counts up to a specified size limit.

    Args:
        x (np.ndarray): 1D array of integer class labels.
        length (int): Dimension of target classes.

    Returns:
        List[int]: Bin count of each class index.
    """
    return np.bincount(x, minlength=length).tolist()


def count_classes(x: np.ndarray) -> int:
    """Count the total number of unique classes present in an array.

    Args:
        x (np.ndarray): 1D target labels array.

    Returns:
        int: Total number of unique labels.
    """
    return len(np.unique(x))


def check_positive_array(x: List[float]) -> bool:
    """Check if all values in an array are strictly positive.

    Args:
        x (List[float]): Input list of float values.

    Returns:
        bool: True if all elements > 0; else False.
    """
    array = np.array(x)
    return all(array > 0)


def check_array_sum_one(x: List[float]) -> bool:
    """Check if the sum of all values in an array is equal to exactly 1.0.

    Args:
        x (List[float]): Input list of float values.

    Returns:
        bool: True if sum == 1.0; else False.
    """
    array = np.array(x)
    return sum(array) == 1.0


def get_instances(
    features_id: List[Any], sample_size: int, probabilities: List[float]
) -> Optional[List[Any]]:
    """Draw a weighted random sample of items with replacement.

    Args:
        features_id (List[Any]): List of candidate items to sample from.
        sample_size (int): Total number of items to draw.
        probabilities (List[float]): Selection probabilities.

    Returns:
        Optional[List[Any]]: Drawn samples list if inputs are valid; else None.
    """
    if (
        len(features_id) == len(probabilities)
        and len(features_id) > 0
        and sample_size > 0
    ):
        probs = np.array(probabilities, dtype=np.float64)
        probs = probs / probs.sum()  # Normalize to ensure sum == 1.0
        sampled = np.random.choice(
            features_id, size=sample_size, replace=True, p=probs
        )
        return list(sampled)
    return None
