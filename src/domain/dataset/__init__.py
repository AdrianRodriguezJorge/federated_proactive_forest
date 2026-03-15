"""Módulo de datasets del dominio FL.

Este paquete expone la interfaz (puerto) que deben implementar los adaptadores
de dataset. Las implementaciones concretas viven en
`src/infrastructure/dataset/`.
"""

from .base_adapter import IDatasetAdapter, DatasetSplit

__all__ = [
    'IDatasetAdapter',
    'DatasetSplit',
]