from typing import List, Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .base_forest import ABCForest

class RandomForestWrapper(ABCForest):
    """
    Wrapper for scikit-learn RandomForestClassifier to implement ABCForest interface.
    Used as a fallback or for comparison purposes.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42, **kwargs):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.kwargs = kwargs
        self._rf: RandomForestClassifier = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the Random Forest."""
        self._rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            **self.kwargs
        )
        self._rf.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self._rf is None:
            raise ValueError("Random Forest not fitted yet")
        return self._rf.predict(X)

    def get_trees(self) -> List[Any]:
        """Return the list of trained trees."""
        if self._rf is None:
            return []
        return self._rf.estimators_.copy()

    @classmethod
    def from_trees(cls, trees: List[Any], class_names: List[str] = None) -> 'RandomForestWrapper':
        """Create a Random Forest from a list of trees."""
        instance = cls()
        instance._rf = RandomForestClassifier()
        instance._rf.estimators_ = trees.copy()
        return instance