"""Puerto hexagonal para datasets — el dominio FL nunca importa pandas ni rutas."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class DatasetSplit:
    X_train: np.ndarray
    X_test:  np.ndarray
    y_train: np.ndarray
    y_test:  np.ndarray
    feature_names: List[str]
    class_names: List[str]
    dataset_name: str
    X_val: np.ndarray = None
    y_val: np.ndarray = None


class IDatasetAdapter(ABC):
    """
    Implementar este puerto = el dataset es compatible con todo el proyecto.
    El adaptador carga, codifica y escala. Retorna DatasetSplit listo para sklearn.
    """
    @abstractmethod
    def load(self) -> DatasetSplit: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def n_classes(self) -> int: ...
