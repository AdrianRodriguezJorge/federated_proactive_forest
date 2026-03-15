"""Adaptador genérico para datasets en formato CSV.

NOTE: Esta implementación es legacy. El proyecto usa ahora los adaptadores
bajo `src.infrastructure.dataset` (ej. `src.infrastructure.dataset.csv_adapter`).
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from pathlib import Path

from .base_adapter import IDatasetAdapter, DatasetSplit


class GenericCsvAdapter(IDatasetAdapter):
    """Adaptador genérico para datasets en formato CSV."""

    def __init__(self,
                 file_path: str,
                 target_column: str,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 separator: str = ',',
                 scale_features: bool = True):
        self.file_path = Path(file_path)
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.separator = separator
        self.scale_features = scale_features
        self._n_classes = None
        self._feature_names = None
        self._class_names = None

    def load(self) -> DatasetSplit:
        # Importar pandas aquí para evitar problemas de importación
        import pandas as pd

        if not self.file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {self.file_path}")

        # Cargar datos
        df = pd.read_csv(self.file_path, sep=self.separator)

        if self.target_column not in df.columns:
            raise ValueError(f"Columna objetivo '{self.target_column}' no encontrada en el CSV")

        # Separar características y objetivo
        X = df.drop(columns=[self.target_column]).values
        y = df[self.target_column].values

        # Codificar etiquetas si son strings
        if y.dtype == object or isinstance(y[0], str):
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            self._class_names = label_encoder.classes_.tolist()
        else:
            self._class_names = sorted(list(set(y.astype(str))))

        # Dividir en train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # Escalar características si se solicita
        if self.scale_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Nombres de características
        feature_cols = [col for col in df.columns if col != self.target_column]
        self._feature_names = feature_cols

        # Calcular número de clases
        self._n_classes = len(np.unique(y))

        return DatasetSplit(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=self._feature_names,
            class_names=self._class_names,
            dataset_name=self.file_path.stem
        )

    @property
    def name(self) -> str:
        return self.file_path.stem

    @property
    def n_classes(self) -> int:
        if self._n_classes is None:
            # Cargar datos para calcular n_classes
            self.load()
        return self._n_classes