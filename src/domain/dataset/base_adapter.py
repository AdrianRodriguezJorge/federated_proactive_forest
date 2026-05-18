"""Hexagonal port for datasets.

The federated learning domain layer should remain independent of specific
data frameworks (like pandas) or path resolutions. This module defines
ports and transfer models for abstracting data access.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class DatasetSplit:
    """Standardized representation of a train/test/validation split.

    Attributes:
        X_train (np.ndarray): 2D array of training features.
        X_test (np.ndarray): 2D array of testing features.
        y_train (np.ndarray): 1D array of training target labels.
        y_test (np.ndarray): 1D array of testing target labels.
        feature_names (List[str]): List of column names representing features.
        class_names (List[str]): List of unique target class names.
        dataset_name (str): Label identifying the source dataset.
        X_val (Optional[np.ndarray]): Optional 2D array of validation features.
            Defaults to None.
        y_val (Optional[np.ndarray]): Optional 1D array of validation labels.
            Defaults to None.
    """

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    class_names: List[str]
    dataset_name: str
    X_val: Optional[np.ndarray] = None
    y_val: Optional[np.ndarray] = None

    def get_all_labels(self) -> np.ndarray:
        """Returns all unique labels present across splits.

        Consolidates train, test, and validation labels to extract a complete
        global array of unique target values.

        Returns:
            np.ndarray: Sorted unique labels present in this split.
        """
        labels = [self.y_train, self.y_test]
        if self.y_val is not None:
            labels.append(self.y_val)
        return np.unique(np.concatenate(labels))


class IDatasetAdapter(ABC):
    """Hexagonal port interface for data loading adapters.

    Implementing this interface ensures compatibility with the entire
    orchestration pipeline. The adapter handles raw reading, scaling,
    target encoding, and split division.
    """

    @abstractmethod
    def load(self) -> DatasetSplit:
        """Load and preprocess the dataset.

        Returns:
            DatasetSplit: The loaded and formatted train/test splits.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the dataset.

        Returns:
            str: Identifier name of the dataset.
        """
        pass

    @property
    @abstractmethod
    def n_classes(self) -> int:
        """Return the number of unique target classes.

        Returns:
            int: Class count for the target variable.
        """
        pass
