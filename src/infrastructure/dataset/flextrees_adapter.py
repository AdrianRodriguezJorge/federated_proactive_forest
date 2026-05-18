"""Adapter for datasets from flex-trees library.

NOTE: This adapter is currently unused in the main FL workflows.
It is kept as a placeholder for future use if flex-trees datasets are needed.
"""

from typing import List
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.domain.dataset.base_adapter import DatasetSplit, IDatasetAdapter


class FlexTreesAdapter(IDatasetAdapter):
    """Adapter for datasets available in flex-trees library.

    Supports ILDP, Adult, Car Evaluation, etc.
    """

    def __init__(
        self,
        dataset_name: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """Initializes the flex-trees adapter.

        Args:
            dataset_name (str): Identifier of the dataset to retrieve.
            test_size (float): Proportion of data reserved for testing.
                Defaults to 0.2.
            random_state (int): Seed for split reproducibility. Defaults to 42.
        """
        self.dataset_name = dataset_name
        self.test_size = test_size
        self.random_state = random_state
        self._label_encoder = LabelEncoder()
        self._feature_names: List[str] = []
        self._class_names: List[str] = []

    @property
    def name(self) -> str:
        """Return the name of the dataset.

        Returns:
            str: Dataset name.
        """
        return self.dataset_name

    @property
    def n_classes(self) -> int:
        """Return the number of unique target classes.

        Returns:
            int: Target class count.
        """
        return len(self._class_names)

    def load(self) -> DatasetSplit:
        """Load dataset from flex-trees.

        Returns:
            DatasetSplit: Formatted train and test splits with metadata.

        Raises:
            ImportError: If flex-trees library is not installed.
            NotImplementedError: If the requested dataset is not supported.
        """
        try:
            from flextrees.datasets.tabular_datasets import get_dataset
        except ImportError:
            raise ImportError(
                "flex-trees not installed. Run: pip install flex-trees"
            )

        # Load dataset using flex-trees
        if self.dataset_name.lower() == "ildp":
            from flextrees.datasets.tabular_datasets import ildp

            X, y = ildp(ret_feature_names=False, categorical=False)
            self._feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        else:
            # For other datasets, would need to implement
            raise NotImplementedError(
                f"Dataset {self.dataset_name} not implemented yet"
            )

        # Mantener y como strings
        self._label_encoder.fit(y)  # Fit para obtener classes
        self._class_names = [str(c) for c in self._label_encoder.classes_]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,  # y como strings
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        # Convert y to strings explicitly
        y_train = np.array([str(label) for label in y_train])
        y_test = np.array([str(label) for label in y_test])

        return DatasetSplit(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=self._feature_names,
            class_names=self._class_names,
            dataset_name=self.dataset_name,
        )