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
                 test_size: float = 0.15, seed: int = 42, sep: str = ","):
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
        Load datasets from CSV paths and prepare for federated training using a 3-way split.
        Implements Enfoque B (Atomic Triple Split): 70% Train, 15% Val, 15% Test.
        All preprocessing fitting occurs ONLY on the training set.
        """
        try:
            full_df = pd.read_csv(self.train_path, sep=self.sep)
            
            # 1. Atomic 3-Way Split (70/15/15)
            # First split: Separate Test (15%) from the rest (85%)
            train_val_df, test_df = train_test_split(
                full_df, 
                test_size=self.test_size, # 0.15
                random_state=self.seed,
                stratify=full_df[self.target_column]
            )
            
            # Second split: Separate Val (15% total) from Train (70% total)
            # We need 15/85 = 17.647% of the train_val_df
            val_relative_size = 0.15 / (1.0 - self.test_size) # 0.15 / 0.85 approx 0.1765
            
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=val_relative_size,
                random_state=self.seed,
                stratify=train_val_df[self.target_column]
            )
            
        except Exception as e:
            raise RuntimeError(f"Error loading CSV from {self.train_path}: {e}")

        # Drop specified columns
        cols_to_drop = [c for c in self.columns_to_drop if c in train_df.columns]
        if cols_to_drop:
            train_df = train_df.drop(columns=cols_to_drop)
            val_df = val_df.drop(columns=cols_to_drop)
            test_df = test_df.drop(columns=cols_to_drop)

        if self.target_column not in train_df.columns:
            raise KeyError(f"Target column '{self.target_column}' not found in dataset.")

        feat_cols = [c for c in train_df.columns if c != self.target_column]

        # ── Categorical Encoding (Ordinal) ───────────────────────────────────
        detected_cat = []
        for col in feat_cols:
            if col not in self.categorical_features:
                if train_df[col].dtype == "object" or train_df[col].dtype.name == "category":
                    detected_cat.append(col)

        all_cat_cols = list(dict.fromkeys(self.categorical_features + detected_cat))

        if all_cat_cols:
            # RIGOR ENFOQUE B: fit ONLY on train
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            train_df[all_cat_cols] = enc.fit_transform(train_df[all_cat_cols].astype(str))
            val_df[all_cat_cols]   = enc.transform(val_df[all_cat_cols].astype(str))
            test_df[all_cat_cols]  = enc.transform(test_df[all_cat_cols].astype(str))

        # ── Label Encoding ───────────────────────────────────────────────────
        le = LabelEncoder()
        # RIGOR ENFOQUE B: Fit ONLY on training labels
        le.fit(train_df[self.target_column].astype(str))
        self._class_names_ = [str(c) for c in le.classes_]
        
        y_train = train_df[self.target_column].astype(str).values
        y_val   = val_df[self.target_column].astype(str).values
        y_test  = test_df[self.target_column].astype(str).values

        # ── Feature Validation and Conversion ───────────────────────────────
        try:
            X_train = train_df[feat_cols].values.astype(np.float64)
            X_val   = val_df[feat_cols].values.astype(np.float64)
            X_test  = test_df[feat_cols].values.astype(np.float64)
        except ValueError as e:
            non_numeric = [col for col in feat_cols if not np.issubdtype(train_df[col].dtype, np.number)]
            raise ValueError(f"Feature conversion failed. Non-numeric columns not marked as categorical: {non_numeric}. Error: {e}")

        # ── Scaling ──────────────────────────────────────────────────────────
        if self.scale:
            # RIGOR ENFOQUE B: Fit ONLY on train statistics
            scaler = StandardScaler() if self.scaler_type == "standard" else MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_val   = scaler.transform(X_val)
            X_test  = scaler.transform(X_test)

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
