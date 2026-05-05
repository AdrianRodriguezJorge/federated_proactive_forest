from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional, Union
import numpy as np
import pandas as pd

class ILabelService(ABC):
    """
    Interface for unified label encoding and decoding.
    Ensures consistent class-to-index mapping across the system.
    """

    @abstractmethod
    def fit(self, labels: Any) -> None:
        pass

    @abstractmethod
    def transform(self, labels: Any) -> np.ndarray:
        pass

    @abstractmethod
    def inverse_transform(self, indices: np.ndarray) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def classes(self) -> List[Any]:
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
            class_names (Optional[List[Any]]): Predefined list of class names. If provided, the service is fitted immediately.
        """
        self._classes: List[Any] = []
        self._encoder: Dict[Any, int] = {}
        self._decoder: Dict[int, Any] = {}
        if class_names is not None:
            self.fit(class_names)

    def fit(self, labels: Any) -> None:
        """Fit the encoder with the unique set of labels."""
        if labels is None:
            return
            
        # Ensure we have a flat array/list
        if isinstance(labels, (list, np.ndarray, pd.Series)):
            labels_list = labels
        else:
            labels_list = [labels]
            
        if len(labels_list) == 0:
            return

        # STICKY PROTECTION: 
        # If we already have non-numeric class names, and we receive numeric data,
        # do NOT overwrite our mapping. We assume the numbers are the Indices we already know.
        has_names = len(self._classes) > 0 and any(not str(c).isdigit() for c in self._classes)
        all_numeric_input = all(isinstance(lab, (int, np.integer)) or (isinstance(lab, str) and lab.isdigit()) for lab in labels_list)
        
        if has_names and all_numeric_input:
            # We already have names, and input is numeric. Stay as we are.
            return

        unique_labels = np.unique([str(lab) for lab in labels_list])
        self._classes = sorted(list(unique_labels))
        self._encoder = {label: i for i, label in enumerate(self._classes)}
        self._decoder = {i: label for i, label in enumerate(self._classes)}

    def transform(self, labels: Any) -> np.ndarray:
        """
        Transform labels to numeric indices using the fitted encoder.
        
        Handles:
        - List/Array of strings: ['class_A', 'class_B'] -> [0, 1]
        - List/Array of ints: [0, 1, 2] -> [0, 1, 2] (validated)
        - Single string or int
        """
        if labels is None:
            return np.array([], dtype=np.int64)

        if isinstance(labels, (list, np.ndarray, pd.Series)):
            labels_list = labels
        else:
            labels_list = [labels]
            
        if len(labels_list) == 0:
            return np.array([], dtype=np.int64)

        n_classes = len(self._classes)
        if n_classes == 0:
            # Fallback for unexpected empty state
            return np.zeros(len(labels_list), dtype=np.int64)

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
                    # Fallback for numeric strings (e.g., '2') if we have class names
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
        """Convert numeric indices back to original labels."""
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

        return np.array([self._decoder.get(int(i), self._classes[0]) for i in indices_list])

    @property
    def classes(self) -> List[Any]:
        """Return the list of original class names."""
        return self._classes
