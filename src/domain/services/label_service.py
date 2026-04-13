from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional
import numpy as np

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
    """
    Concrete implementation of ILabelService.
    """

    def __init__(self, class_names: Optional[List[Any]] = None):
        self._classes = []
        self._encoder = {}
        self._decoder = {}
        if class_names is not None:
            self.fit(class_names)

    def fit(self, labels: Any) -> None:
        unique_labels = np.unique(labels)
        self._classes = sorted(list(unique_labels))
        self._encoder = {label: i for i, label in enumerate(self._classes)}
        self._decoder = {i: label for i, label in enumerate(self._classes)}

    def transform(self, labels: Any) -> np.ndarray:
        """Transform labels to numeric indices.
        
        Handles both string labels (e.g., 'hAd') and integer indices (e.g., 0, 1, 2).
        If labels are already integers within the valid range, returns them as-is.
        """
        if isinstance(labels, (list, np.ndarray)):
            labels_list = list(labels) if isinstance(labels, np.ndarray) else labels
        else:
            labels_list = [labels]
        
        # Check if first element is already an integer
        if len(labels_list) > 0:
            first_elem = labels_list[0]
            # If it's already an integer (or numpy integer)
            if isinstance(first_elem, (int, np.integer)):
                # Validate it's within range
                if isinstance(first_elem, (int, np.integer)):
                    # Already encoded, just validate and return
                    result = np.array([int(x) for x in labels_list], dtype=np.int64)
                    # Optional: validate range
                    if np.any(result < 0) or np.any(result >= len(self._classes)):
                        # Invalid indices, try to handle gracefully
                        import warnings
                        warnings.warn(f"Label indices out of range: {result[:5]}")
                    return result
        
        # Otherwise, treat as string labels and encode
        return np.array([self._encoder.get(label, 0) for label in labels_list])

    def inverse_transform(self, indices: np.ndarray) -> np.ndarray:
        if isinstance(indices, (list, np.ndarray)):
            return np.array([self._decoder.get(i, self._classes[0]) for i in indices])
        return np.array([self._decoder.get(indices, self._classes[0])])

    @property
    def classes(self) -> List[Any]:
        return self._classes
