"""Unified label encoding and decoding service.

Provides interfaces and implementations to map raw target classes to
consistent numeric indices and back.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class ILabelService(ABC):
    """Interface for unified label encoding and decoding.

    Ensures consistent class-to-index mapping across the system.
    """

    @abstractmethod
    def fit(self, labels: Any) -> None:
        """Fit the label encoder to the unique set of labels.

        Args:
            labels (Any): A single label or an array/list of labels.
        """
        pass

    @abstractmethod
    def transform(self, labels: Any) -> np.ndarray:
        """Transform labels to numeric indices using the fitted encoder.

        Args:
            labels (Any): A single label or sequence of labels.

        Returns:
            np.ndarray: Numeric indices mapping to the fitted classes.
        """
        pass

    @abstractmethod
    def inverse_transform(self, indices: np.ndarray) -> np.ndarray:
        """Convert numeric indices back to original labels.

        Args:
            indices (np.ndarray): Array of numeric indices to convert.

        Returns:
            np.ndarray: Original labels corresponding to the indices.
        """
        pass

    @property
    @abstractmethod
    def classes(self) -> List[Any]:
        """Return the list of original class names.

        Returns:
            List[Any]: List of unique class labels.
        """
        pass


class SimpleLabelService(ILabelService):
    """Concrete implementation of ILabelService with robust type handling.

    Supports string labels and integer indices.
    If integers are provided, they are validated against the fitted classes.
    Invalid or out-of-range labels default to class 0 to prevent crashes.
    """

    def __init__(self, class_names: Optional[List[Any]] = None):
        """Initializes the label service.

        Args:
            class_names (Optional[List[Any]]): Predefined list of class names.
                If provided, the service is fitted immediately.
        """
        self._classes: List[Any] = []
        self._encoder: Dict[Any, int] = {}
        self._decoder: Dict[int, Any] = {}
        if class_names is not None:
            self.fit(class_names)

    def fit(self, labels: Any) -> None:
        """Fit the encoder with the unique set of labels.

        Handles single labels, lists, numpy arrays, or pandas series.
        Prevents overwriting if the encoder already has string class names and
        receives integer indexes.

        Args:
            labels (Any): Single label, list, array, or Series to fit.
        """
        if labels is None:
            return

        # Ensure we have a flat array/list
        is_flat_seq = (
            isinstance(labels, (list, np.ndarray, pd.Series))
            or hasattr(labels, "__array__")
            or (
                hasattr(labels, "__len__")
                and not isinstance(labels, (str, bytes, dict))
            )
        )
        if is_flat_seq:
            labels_list = labels
        else:
            labels_list = [labels]

        if len(labels_list) == 0:
            return

        # STICKY PROTECTION:
        # If we already have non-numeric class names, and we receive numeric
        # data, do NOT overwrite our mapping. We assume the numbers are the
        # Indices we already know.
        has_names = len(self._classes) > 0 and any(
            not str(c).isdigit() for c in self._classes
        )
        all_numeric_input = all(
            isinstance(lab, (int, np.integer))
            or (isinstance(lab, str) and lab.isdigit())
            for lab in labels_list
        )

        if has_names and all_numeric_input:
            # We already have names, and input is numeric. Stay as we are.
            return

        unique_labels = np.unique([str(lab) for lab in labels_list])
        self._classes = sorted(list(unique_labels))
        self._encoder = {label: i for i, label in enumerate(self._classes)}
        self._decoder = {i: label for i, label in enumerate(self._classes)}

    def transform(self, labels: Any) -> np.ndarray:
        """Transform labels to numeric indices using the fitted encoder.

        Handles:
        - List/Array of strings: ['class_A', 'class_B'] -> [0, 1]
        - List/Array of ints: [0, 1, 2] -> [0, 1, 2] (validated)
        - Single string or int

        Args:
            labels (Any): Input labels to encode.

        Returns:
            np.ndarray: Encoded numeric indices as a 1D NumPy array.
        """
        if labels is None:
            return np.array([], dtype=np.int64)

        if isinstance(labels, pd.Series):
            labels_list = labels.values
        elif (
            isinstance(labels, (list, np.ndarray))
            or hasattr(labels, "__array__")
            or (
                hasattr(labels, "__len__")
                and not isinstance(labels, (str, bytes, dict))
            )
        ):
            labels_list = labels
        else:
            labels_list = [labels]

        if len(labels_list) == 0:
            return np.array([], dtype=np.int64)

        n_classes = len(self._classes)
        if n_classes == 0:
            # Fallback for unexpected empty state
            return np.zeros(len(labels_list), dtype=np.int64)

        # --- OPTIMIZATION: FAST PATH ---
        # If input is already an integer numpy array (e.g. from tree.predict)
        if isinstance(labels_list, np.ndarray) and np.issubdtype(
            labels_list.dtype, np.integer
        ):
            # Clip invalid indices to 0 to prevent crashes
            safe_labels = np.where(
                (labels_list >= 0) & (labels_list < n_classes), labels_list, 0
            )
            return safe_labels.astype(np.int64)

        result = []
        for lab in labels_list:
            # Type 1: Label is already an integer (index)
            if isinstance(lab, (int, np.integer)):
                val = int(lab)
                if 0 <= val < n_classes:
                    result.append(val)
                else:
                    # Log warning but don't crash, default to 0
                    # (this replaces the clipping in ForestEvaluator)
                    result.append(0)
            else:
                # Type 2: Label is a string/other
                label_str = str(lab)
                if label_str in self._encoder:
                    result.append(self._encoder[label_str])
                elif label_str.isdigit():
                    # Fallback for numeric strings (e.g., '2')
                    idx = int(label_str)
                    if 0 <= idx < n_classes:
                        result.append(idx)
                    else:
                        result.append(0)
                else:
                    # Final fallback to class 0
                    result.append(0)

        return np.array(result, dtype=np.int64)

    def inverse_transform(self, indices: np.ndarray) -> np.ndarray:
        """Convert numeric indices back to original labels.

        Args:
            indices (np.ndarray): 1D array of encoded integer indices.

        Returns:
            np.ndarray: 1D array of decoded original labels.
        """
        if indices is None:
            return np.array([])

        if isinstance(indices, (list, np.ndarray, pd.Series, np.integer)):
            if isinstance(indices, (int, np.integer)):
                indices_list = [indices]
            else:
                indices_list = indices
        else:
            indices_list = [indices]

        n_classes = len(self._classes)
        if n_classes == 0:
            return np.array([str(i) for i in indices_list])

        return np.array(
            [self._decoder.get(int(i), self._classes[0]) for i in indices_list]
        )

    @property
    def classes(self) -> List[Any]:
        """Return the list of original class names.

        Returns:
            List[Any]: List of unique class labels fitted by encoder.
        """
        return self._classes
