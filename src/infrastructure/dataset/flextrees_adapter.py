"""Adapter for datasets from flex-trees library."""
import numpy as np
from typing import List
from src.domain.dataset.base_adapter import IDatasetAdapter, DatasetSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class FlexTreesAdapter(IDatasetAdapter):
    """
    Adapter for datasets available in flex-trees library.
    Supports ILDP, Adult, Car Evaluation, etc.
    """

    def __init__(self, dataset_name: str, test_size: float = 0.2, random_state: int = 42):
        self.dataset_name = dataset_name
        self.test_size = test_size
        self.random_state = random_state
        self._label_encoder = LabelEncoder()
        self._feature_names: List[str] = []
        self._class_names: List[str] = []

    @property
    def name(self) -> str:
        return self.dataset_name

    @property
    def n_classes(self) -> int:
        return len(self._class_names)

    def load(self) -> DatasetSplit:
        """Load dataset from flex-trees."""
        try:
            from flextrees.datasets.tabular_datasets import get_dataset
        except ImportError:
            raise ImportError("flex-trees not installed. Run: pip install flex-trees")

        # Load dataset using flex-trees
        if self.dataset_name.lower() == 'ildp':
            from flextrees.datasets.tabular_datasets import ildp
            X, y = ildp(ret_feature_names=False, categorical=False)
            self._feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        else:
            # For other datasets, would need to implement
            raise NotImplementedError(f"Dataset {self.dataset_name} not implemented yet")

        # Encode labels
        y_encoded = self._label_encoder.fit_transform(y)
        self._class_names = list(self._label_encoder.classes_)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_encoded
        )

        return DatasetSplit(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=self._feature_names,
            class_names=self._class_names,
            dataset_name=self.dataset_name
        )