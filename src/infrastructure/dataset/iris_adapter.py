"""Adaptador para Iris Dataset (150 muestras, 4 características, 3 clases)."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from src.domain.dataset.base_adapter import IDatasetAdapter, DatasetSplit
from sklearn.model_selection import train_test_split


class IrisAdapter(IDatasetAdapter):
    """
    Iris Dataset Adapter
    
    Características:
    - 4 características numéricas: sepal_length, sepal_width, petal_length, petal_width
    - 3 clases: Iris-setosa, Iris-versicolor, Iris-virginica
    - 150 muestras totales (típicamente 70% train, 30% test)
    - SIN características categóricas
    
    Nota: Iris es pequeño, por lo que no necesita train/test paths separados.
          Se hace 70/30 split automáticamente.
    """
    
    COLUMNS = [
        "sepallength", "sepalwidth", "petallength", "petalwidth", "class"
    ]
    FEAT_COLS = ["sepallength", "sepalwidth", "petallength", "petalwidth"]
    # Clases conocidas de Iris (para asegurar que el encoder conoce TODAS las clases)
    KNOWN_CLASSES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    
    def __init__(self, data_path: str, scale: bool = True, 
                 scaler_type: str = "standard", train_test_split_ratio: float = 0.7):
        """
        Args:
            data_path: Ruta al archivo iris.csv
            scale: Si normalizar características
            scaler_type: "standard" (media=0, var=1) o "minmax" (0-1)
            train_test_split_ratio: Proporción train (default 70-30)
        """
        self.data_path = data_path
        self.scale = scale
        self.scaler_type = scaler_type
        self.train_test_split_ratio = train_test_split_ratio
        
        if scale:
            self._scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        else:
            self._scaler = None
        
        self._label_encoder = LabelEncoder()
        self._class_names_ = []
    
    @property
    def name(self) -> str:
        """Identificador del dataset."""
        return "iris"
    
    @property
    def n_classes(self) -> int:
        """Número de clases."""
        return len(self._class_names_)
    
    def load(self) -> DatasetSplit:
        """
        Carga, prepara y divide el dataset Iris.
        
        Returns:
            DatasetSplit con X_train, X_test, y_train, y_test, class_names
        """
        # 1. Leer CSV (tiene headers)
        df = pd.read_csv(self.data_path)
        
        # 2. Separar features y target
        X = df[self.FEAT_COLS].values.astype(np.float64)
        y_raw = df["class"].values
        
        # 3. Mantener etiquetas como strings (el CPF las codifica internamente)
        # Asegurar que todas las clases conocidas estén presentes
        y = y_raw
        self._class_names_ = sorted(list(set(y_raw)))  # clases únicas en orden
        
        # 4. Train-test split (70-30)
        # Usar solo test_size para evitar problemas de redondeo con datasets pequeños
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=1.0 - self.train_test_split_ratio,
            stratify=y,  # Mantener proporciones de clases
            random_state=42
        )
        
        # 5. Escalar si está habilitado
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
