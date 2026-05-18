"""Tabular CSV dataset adapter.

Implements the IDatasetAdapter interface to load, preprocess, encode,
and split numerical and categorical tabular datasets from CSV files.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OrdinalEncoder,
    StandardScaler,
)

from src.domain.dataset.base_adapter import DatasetSplit, IDatasetAdapter


class GenericCsvAdapter(IDatasetAdapter):
    """Plug-and-play adapter to load and preprocess tabular CSV datasets.

    Enforces Enfoque B (Atomic Triple Split): 70% Train, 15% Val, 15% Test.
    Fits all preprocessing components (scalers, encoders) ONLY on the
    training set to avoid any data leakage.
    """

    def __init__(
        self,
        name: str,
        train_path: str,
        target_column: str,
        test_path: Optional[str] = None,
        categorical_features: Optional[List[str]] = None,
        columns_to_drop: Optional[List[str]] = None,
        scale: bool = True,
        scaler_type: str = "standard",
        test_size: float = 0.15,
        seed: int = 42,
        sep: str = ",",
    ):
        """Initializes the CSV data adapter.

        Args:
            name (str): Unique identifier for this dataset.
            train_path (str): File path to the training CSV file.
            target_column (str): Name of the target variable/column.
            test_path (Optional[str]): Optional path to a test CSV file.
                Defaults to None.
            categorical_features (Optional[List[str]]): Column names of
                known categorical features. Defaults to None.
            columns_to_drop (Optional[List[str]]): Columns to drop from
                the dataset. Defaults to None.
            scale (bool): Whether to scale numeric features. Defaults to True.
            scaler_type (str): Type of scaler to use ('standard' or 'minmax').
                Defaults to "standard".
            test_size (float): Portion of the data reserved for the test set.
                Defaults to 0.15.
            seed (int): Random seed for split reproducibility. Defaults to 42.
            sep (str): Delimiter separating CSV values. Defaults to ",".
        """
        self._name = name
        self.train_path = train_path
        self.test_path = test_path
        self.target_column = target_column
        self.categorical_features = categorical_features or []
        self.columns_to_drop = columns_to_drop or []
        self.scale = scale
        self.scaler_type = scaler_type
        self.test_size = test_size
        self.seed = seed
        self.sep = sep
        self._class_names_: List[str] = []

    @property
    def name(self) -> str:
        """Return the name of the dataset.

        Returns:
            str: Dataset name.
        """
        return self._name

    @property
    def n_classes(self) -> int:
        """Return the number of unique target classes.

        Returns:
            int: Target class count.
        """
        return len(self._class_names_)

    def load(self) -> DatasetSplit:
        """Load datasets from CSV paths and prepare splits.

        Performs a 3-way split: 70% Train, 15% Val, 15% Test. All scaler and
        encoder fitting occurs exclusively on the training subset.

        Returns:
            DatasetSplit: Formatted train, val, and test splits with metadata.

        Raises:
            RuntimeError: If files cannot be read.
            KeyError: If target column is missing.
            ValueError: If feature conversions or scaling fail.
        """
        try:
            full_df = pd.read_csv(self.train_path, sep=self.sep)

            # 1. Atomic 3-Way Split (70/15/15)
            # First split: Separate Test (15%) from the rest (85%)
            train_val_df, test_df = train_test_split(
                full_df,
                test_size=self.test_size,  # 0.15
                random_state=self.seed,
                stratify=full_df[self.target_column],
            )

            # Second split: Separate Val (15% total) from Train (70% total)
            # Relative size: 0.15 / 0.85 approx 0.1765
            val_relative_size = 0.15 / (1.0 - self.test_size)

            train_df, val_df = train_test_split(
                train_val_df,
                test_size=val_relative_size,
                random_state=self.seed,
                stratify=train_val_df[self.target_column],
            )

        except Exception as e:
            raise RuntimeError(
                f"Error loading CSV from {self.train_path}: {e}"
            )

        # Drop specified columns
        cols_to_drop = [
            c for c in self.columns_to_drop if c in train_df.columns
        ]
        if cols_to_drop:
            train_df = train_df.drop(columns=cols_to_drop)
            val_df = val_df.drop(columns=cols_to_drop)
            test_df = test_df.drop(columns=cols_to_drop)

        if self.target_column not in train_df.columns:
            raise KeyError(
                f"Target column '{self.target_column}' not found in dataset."
            )

        feat_cols = [c for c in train_df.columns if c != self.target_column]

        # -- Categorical Encoding (Ordinal) --
        detected_cat = []
        for col in feat_cols:
            if col not in self.categorical_features:
                col_type = train_df[col].dtype
                if col_type == "object" or col_type.name == "category":
                    detected_cat.append(col)

        all_cat_cols = list(
            dict.fromkeys(self.categorical_features + detected_cat)
        )

        if all_cat_cols:
            # RIGOR ENFOQUE B: fit ONLY on train
            enc = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            train_df[all_cat_cols] = enc.fit_transform(
                train_df[all_cat_cols].astype(str)
            )
            val_df[all_cat_cols] = enc.transform(
                val_df[all_cat_cols].astype(str)
            )
            test_df[all_cat_cols] = enc.transform(
                test_df[all_cat_cols].astype(str)
            )

        # -- Label Encoding --
        le = LabelEncoder()
        # RIGOR ENFOQUE B: Fit ONLY on training labels
        le.fit(train_df[self.target_column].astype(str))
        self._class_names_ = [str(c) for c in le.classes_]

        y_train = train_df[self.target_column].astype(str).values
        y_val = val_df[self.target_column].astype(str).values
        y_test = test_df[self.target_column].astype(str).values

        # -- Feature Validation and Conversion --
        try:
            X_train = train_df[feat_cols].values.astype(np.float64)
            X_val = val_df[feat_cols].values.astype(np.float64)
            X_test = test_df[feat_cols].values.astype(np.float64)
        except ValueError as e:
            non_numeric = [
                col
                for col in feat_cols
                if not np.issubdtype(train_df[col].dtype, np.number)
            ]
            raise ValueError(
                f"Feature conversion failed. Non-numeric columns not "
                f"marked as categorical: {non_numeric}. Error: {e}"
            )

        # -- Scaling --
        num_cols_idx = [
            feat_cols.index(c) for c in feat_cols if c not in all_cat_cols
        ]
        if self.scale and num_cols_idx:
            # RIGOR ENFOQUE B: Fit ONLY on train statistics
            scaler = (
                StandardScaler()
                if self.scaler_type == "standard"
                else MinMaxScaler()
            )
            X_train[:, num_cols_idx] = scaler.fit_transform(
                X_train[:, num_cols_idx]
            )
            X_val[:, num_cols_idx] = scaler.transform(X_val[:, num_cols_idx])
            X_test[:, num_cols_idx] = scaler.transform(X_test[:, num_cols_idx])

        return DatasetSplit(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            feature_names=feat_cols,
            class_names=self._class_names_,
            dataset_name=self._name,
        )
