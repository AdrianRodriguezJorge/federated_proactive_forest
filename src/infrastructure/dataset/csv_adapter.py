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
                 scale: bool = True, scaler_type: str = "standard",
                 test_size: float = 0.2, seed: int = 42):
        self._name = name
        self.train_path = train_path
        self.test_path = test_path
        self.target_column = target_column
        self.categorical_features = categorical_features or []
        self.scale = scale
        self.scaler_type = scaler_type
        self.test_size = test_size
        self.seed = seed
        self._class_names_: list = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_classes(self) -> int:
        return len(self._class_names_)

    def load(self) -> DatasetSplit:
        train_df = pd.read_csv(self.train_path)
        if self.test_path:
            test_df = pd.read_csv(self.test_path)
        else:
            train_df, test_df = train_test_split(train_df, test_size=self.test_size,
                                                 random_state=self.seed)

        feat_cols = [c for c in train_df.columns if c != self.target_column]

        if self.categorical_features:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            train_df[self.categorical_features] = enc.fit_transform(train_df[self.categorical_features])
            test_df[self.categorical_features]  = enc.transform(test_df[self.categorical_features])

        le = LabelEncoder()
        y_train = le.fit_transform(train_df[self.target_column].values)
        y_test  = le.transform(test_df[self.target_column].values)
        self._class_names_ = [str(c) for c in le.classes_]

        X_train = train_df[feat_cols].values.astype(np.float64)
        X_test  = test_df[feat_cols].values.astype(np.float64)

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
