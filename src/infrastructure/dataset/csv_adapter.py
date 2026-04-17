"""Adaptador genérico para cualquier CSV tabular."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from src.domain.dataset.base_adapter import IDatasetAdapter, DatasetSplit
from typing import List, Optional


class GenericCsvAdapter(IDatasetAdapter):
    """
    Adaptador plug-and-play para cualquier CSV.
    Si se proveen train_path y test_path, los usa directamente.
    Si sólo hay train_path, hace train_test_split con test_size.
    """

    def __init__(self, name: str, train_path: str, target_column: str,
                 test_path: Optional[str] = None,
                 categorical_features: Optional[List[str]] = None,
                 columns_to_drop: Optional[List[str]] = None,
                 scale: bool = True, scaler_type: str = "standard",
                 test_size: float = 0.2, seed: int = 42, sep: str = ","):
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
        self._class_names_: list = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_classes(self) -> int:
        return len(self._class_names_)

    def load(self) -> DatasetSplit:
        """
        Load datasets from CSV paths and prepare for federated training.
        Includes categorical encoding, label encoding, and feature scaling.
        """
        try:
            train_df = pd.read_csv(self.train_path, sep=self.sep)
            if self.test_path:
                test_df = pd.read_csv(self.test_path, sep=self.sep)
            else:
                train_df, test_df = train_test_split(
                    train_df, test_size=self.test_size, random_state=self.seed
                )
        except Exception as e:
            raise RuntimeError(f"Error loading CSV from {self.train_path}: {e}")

        # Drop specified columns (e.g., metadata columns that shouldn't be features)
        cols_to_drop = [c for c in self.columns_to_drop if c in train_df.columns]
        if cols_to_drop:
            train_df = train_df.drop(columns=cols_to_drop)
            test_df = test_df.drop(columns=cols_to_drop)

        if self.target_column not in train_df.columns:
            raise KeyError(f"Target column '{self.target_column}' not found in dataset.")

        feat_cols = [c for c in train_df.columns if c != self.target_column]

        # Auto-detect string/object columns in BOTH train and test sets
        detected_cat = []
        for col in feat_cols:
            if col not in self.categorical_features:
                is_obj_train = train_df[col].dtype == "object" or train_df[col].dtype.name == "category"
                is_obj_test = test_df[col].dtype == "object" or test_df[col].dtype.name == "category"
                if is_obj_train or is_obj_test:
                    detected_cat.append(col)

        # Combine explicit + detected categorical columns
        all_cat_cols = list(dict.fromkeys(self.categorical_features + detected_cat))

        if all_cat_cols:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            train_df[all_cat_cols] = enc.fit_transform(train_df[all_cat_cols].astype(str))
            test_df[all_cat_cols]  = enc.transform(test_df[all_cat_cols].astype(str))

        # Handle target column
        le = LabelEncoder()
        # Ensure target is string for consistent encoding
        y_train_raw = train_df[self.target_column].astype(str).values
        y_test_raw = test_df[self.target_column].astype(str).values
        
        # Fit on BOTH train and test to guarantee all classes are known
        le.fit(np.concatenate([y_train_raw, y_test_raw]))
        self._class_names_ = [str(c) for c in le.classes_]
        
        y_train = y_train_raw
        y_test = y_test_raw

        # Feature validation and conversion
        try:
            X_train = train_df[feat_cols].values.astype(np.float64)
            X_test  = test_df[feat_cols].values.astype(np.float64)
        except ValueError as e:
            # If conversion fails, identify which columns are non-numeric
            non_numeric = []
            for col in feat_cols:
                try:
                    train_df[col].values.astype(np.float64)
                except ValueError:
                    non_numeric.append(col)
            raise ValueError(f"Feature conversion to float failed. Columns containing non-numeric data without being marked as categorical: {non_numeric}. Original error: {e}")

        if self.scale:
            scaler = StandardScaler() if self.scaler_type == "standard" else MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

        return DatasetSplit(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            feature_names=feat_cols,
            class_names=self._class_names_,
            dataset_name=self._name,
        )
