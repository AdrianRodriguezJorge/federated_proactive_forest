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
        if isinstance(labels, (list, np.ndarray)):
            return np.array([self._encoder.get(label, 0) for label in labels])
        return np.array([self._encoder.get(labels, 0)])

    def inverse_transform(self, indices: np.ndarray) -> np.ndarray:
        if isinstance(indices, (list, np.ndarray)):
            return np.array([self._decoder.get(i, self._classes[0]) for i in indices])
        return np.array([self._decoder.get(indices, self._classes[0])])

    @property
    def classes(self) -> List[Any]:
        return self._classes
