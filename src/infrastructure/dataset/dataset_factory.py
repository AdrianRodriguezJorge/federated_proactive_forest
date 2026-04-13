from typing import Dict, Any, List, Optional
import os
from pathlib import Path

from .nslkdd_adapter import NslKddAdapter
from .iris_adapter import IrisAdapter
from .csv_adapter import GenericCsvAdapter
from ...domain.dataset.base_adapter import DatasetSplit

# Central Registry of Dataset Metadata
DATASET_METADATA = {
    "students_dropout": {
        "target_column": "Target",
        "sep": ";",
        "categorical_features": [
            "Marital status", "Application mode", "Application order", "Course",
            "Daytime/evening attendance", "Previous qualification", "Nacionality",
            "Mother's qualification", "Father's qualification", "Mother's occupation",
            "Father's occupation", "Displaced", "Educational special needs", "Debtor",
            "Tuition fees up to date", "Gender", "Scholarship holder", "International",
            "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
            "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
            "Curricular units 1st sem (without evaluations)", "Curricular units 2nd sem (credited)",
            "Curricular units 2nd sem (enrolled)", "Curricular units 2nd sem (evaluations)",
            "Curricular units 2nd sem (approved)", "Curricular units 2nd sem (without evaluations)"
        ]
    },
    "car": {
        "target_column": "class",
        "sep": ",",
        "categorical_features": ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
    },
    "nursery": {
        "target_column": "class",
        "sep": ",",
        "categorical_features": ["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health"]
    }
}

class DatasetFactory:
    """
    Factory for creating dataset adapters based on configuration.
    Ensures that data preprocessing, scaling, and splitting logic is consistent
    across all interfaces (CLI, Streamlit, Optimization).
    """

    @staticmethod
    def get_adapter_from_config(cfg: Dict[str, Any], project_root: Optional[Path] = None):
        """
        Creates an adapter based on the provider configuration dictionary.
        
        Args:
            cfg: Dataset configuration segment from a YAML/JSON config.
            project_root: Optional path to resolve relative file paths.
        """
        d = cfg
        dtype = d.get("type", "Custom CSV")
        
        # Resolve project root if not provided
        if project_root is None:
            # Assume we are in src/infrastructure/dataset/
            project_root = Path(__file__).resolve().parents[3]

        if dtype == "NSL-KDD":
            train_path = d.get("train_path") or str(project_root / "data" / "NSL-KDD_train.csv")
            test_path = d.get("test_path") or str(project_root / "data" / "NSL-KDD_test.csv")
            return NslKddAdapter(
                train_path=train_path,
                test_path=test_path,
                scale=d.get("scale", True),
                scaler_type=d.get("scaler_type", "standard")
            )

        if dtype == "Iris":
            return IrisAdapter(
                train_test_split_ratio=1.0 - d.get("test_size", 0.2),
                scale=d.get("scale", True),
                scaler_type=d.get("scaler_type", "standard"),
                seed=d.get("seed", 42)
            )

        # Handle Generic CSVs (Custom or Presets)
        preset_key = dtype.lower().replace(" ", "_").replace("-", "_")
        metadata = DATASET_METADATA.get(preset_key, {})
        
        # Merge config with metadata (config takes precedence)
        target_column = d.get("target_column") or metadata.get("target_column", "class")
        categorical_features = d.get("categorical_features") or metadata.get("categorical_features", [])
        sep = d.get("sep") or metadata.get("sep", ",")
        file_path = d.get("file_path") or d.get("train_path")
        
        # If relative, join with project root
        if file_path and not Path(file_path).is_absolute():
            file_path = str(project_root / file_path)

        return GenericCsvAdapter(
            name=dtype.lower().replace(" ", "_"),
            train_path=file_path,
            test_path=d.get("test_path") or None,
            target_column=target_column,
            test_size=d.get("test_size", 0.2),
            categorical_features=categorical_features,
            scale=d.get("scale", True),
            scaler_type=d.get("scaler_type", "standard"),
            seed=d.get("seed", 42),
            sep=sep
        )

    @staticmethod
    def load_from_config(cfg: Dict[str, Any], project_root: Optional[Path] = None) -> DatasetSplit:
        """Helper to create and load in one step."""
        adapter = DatasetFactory.get_adapter_from_config(cfg, project_root)
        return adapter.load()
