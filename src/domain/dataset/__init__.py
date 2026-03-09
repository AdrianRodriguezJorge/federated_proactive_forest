"""Módulo de datasets del dominio FL."""

from .base_adapter import IDatasetAdapter, DatasetSplit
from .iris_adapter import IrisAdapter
from .generic_csv_adapter import GenericCsvAdapter

__all__ = [
    'IDatasetAdapter',
    'DatasetSplit',
    'IrisAdapter',
    'GenericCsvAdapter'
]