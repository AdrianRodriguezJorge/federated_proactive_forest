"""Adaptador para el dataset Iris."""
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import List

from .base_adapter import IDatasetAdapter, DatasetSplit


class IrisAdapter(IDatasetAdapter):
    """Adaptador para el dataset Iris de sklearn."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def load(self) -> DatasetSplit:
        # Cargar datos de iris
        iris = load_iris()
        X, y = iris.data, iris.target

        # Dividir en train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # Escalar características
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return DatasetSplit(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=iris.feature_names,
            class_names=iris.target_names.tolist(),
            dataset_name="Iris"
        )

    @property
    def name(self) -> str:
        return "Iris"

    @property
    def n_classes(self) -> int:
        return 3