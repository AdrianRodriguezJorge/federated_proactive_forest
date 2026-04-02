"""Adaptador para Iris Dataset (150 muestras, 4 características, 3 clases)."""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from src.domain.dataset.base_adapter import IDatasetAdapter, DatasetSplit


class IrisAdapter(IDatasetAdapter):
    """Adaptador de Iris usando sklearn.datasets.load_iris.

    Este adaptador **no depende de un CSV local**, y evita duplicar el dataset
    dentro del repositorio.

    La clase mantiene la misma interfaz que antes (acepta un parámetro
    `data_path` por compatibilidad) pero lo ignora internamente.
    """

    FEAT_COLS = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]

    def __init__(self, data_path: str = None, scale: bool = True,
                 scaler_type: str = "standard", train_test_split_ratio: float = 0.7):
        self.scale = scale
        self.scaler_type = scaler_type
        self.train_test_split_ratio = train_test_split_ratio

        if scale:
            self._scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        else:
            self._scaler = None

        self._class_names_ = []

    @property
    def name(self) -> str:
        return "iris"

    @property
    def n_classes(self) -> int:
        return len(self._class_names_)

    def load(self) -> DatasetSplit:
        """Carga y retorna el split para Iris."""
        iris = load_iris()
        X, y = iris.data, iris.target

        # Guardar nombres de clases antes del split
        self._class_names_ = iris.target_names.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=1.0 - self.train_test_split_ratio,
            stratify=y,
            random_state=42
        )

        # Convert y values to class names (strings)
        y_train = np.array([self._class_names_[i] for i in y_train])
        y_test = np.array([self._class_names_[i] for i in y_test])

        if self._scaler:
            X_train = self._scaler.fit_transform(X_train)
            X_test = self._scaler.transform(X_test)

        return DatasetSplit(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=self.FEAT_COLS,
            class_names=self._class_names_,
            dataset_name=self.name
        )
