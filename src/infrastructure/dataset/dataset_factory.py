"""Factory for creating and loading dataset adapters.

Coordinates tabular dataset preset metadata registry and handles resolution
of absolute file paths from project root.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from src.domain.dataset.base_adapter import DatasetSplit
from .csv_adapter import GenericCsvAdapter

# Central Registry of Dataset Metadata
DATASET_METADATA = {
    "iris": {"target_column": "class", "sep": ",", "file_path": "data/iris.csv"},
    "car": {
        "target_column": "class",
        "sep": ",",
        "file_path": "data/car.csv",
        "categorical_features": [
            "buying",
            "maint",
            "doors",
            "persons",
            "lug_boot",
            "safety",
        ],
    },
    "nursery": {
        "target_column": "class",
        "sep": ",",
        "file_path": "data/nursery.csv",
        "categorical_features": [
            "parents",
            "has_nurs",
            "form",
            "children",
            "housing",
            "finance",
            "social",
            "health",
        ],
    },
    "vowel": {
        "target_column": "Class",
        "sep": ",",
        "file_path": "data/vowel.csv",
        "columns_to_drop": ["Train or Test", "Speaker Number", "Sex"],
    },
    "letter": {
        "target_column": "class",
        "sep": ",",
        "file_path": "data/letter.csv",
    },
    "optdigits": {
        "target_column": "class",
        "sep": ",",
        "file_path": "data/optdigits.csv",
    },
    "sonar": {
        "target_column": "Class",
        "sep": ",",
        "file_path": "data/sonar.csv",
    },
    "spambase": {
        "target_column": "class",
        "sep": ",",
        "file_path": "data/spambase.csv",
    },
    "glass": {
        "target_column": "Type",
        "sep": ",",
        "file_path": "data/glass.csv",
    },
    "molecular": {
        "target_column": "class",
        "sep": ",",
        "file_path": "data/molecular.csv",
        "columns_to_drop": ["instance"],
        "categorical_features": [
            "p-50", "p-49", "p-48", "p-47", "p-46", "p-45", "p-44", "p-43",
            "p-42", "p-41", "p-40", "p-39", "p-38", "p-37", "p-36", "p-35",
            "p-34", "p-33", "p-32", "p-31", "p-30", "p-29", "p-28", "p-27",
            "p-26", "p-25", "p-24", "p-23", "p-22", "p-21", "p-20", "p-19",
            "p-18", "p-17", "p-16", "p-15", "p-14", "p-13", "p-12", "p-11",
            "p-10", "p-9", "p-8", "p-7", "p-6", "p-5", "p-4", "p-3",
            "p-2", "p-1", "p1", "p2", "p3", "p4", "p5", "p6", "p7",
        ],
    },
    "pendigits": {
        "target_column": "class",
        "sep": ",",
        "file_path": "data/pendigits.csv",
    },
}


class DatasetFactory:
    """Factory for creating dataset adapters based on configuration.

    Ensures that data preprocessing, scaling, and splitting logic is consistent
    across all interfaces (CLI, Streamlit, Optimization).
    """

    @staticmethod
    def get_adapter_from_config(
        cfg: Dict[str, Any], project_root: Optional[Path] = None
    ) -> GenericCsvAdapter:
        """Creates an adapter based on the provider configuration dictionary.

        Args:
            cfg (Dict[str, Any]): Dataset configuration segment.
            project_root (Optional[Path]): Optional path to resolve relative
                file paths. If not specified, resolves to the project's root.

        Returns:
            GenericCsvAdapter: Instantiated data adapter.
        """
        d = cfg
        dtype = d.get("type", "Custom CSV")

        # Resolve project root if not provided
        if project_root is None:
            # Assume we are in src/infrastructure/dataset/
            project_root = Path(__file__).resolve().parents[3]

        # Handle Generic CSVs (Custom or Presets)
        preset_key = dtype.lower().replace(" ", "_").replace("-", "_")
        metadata = DATASET_METADATA.get(preset_key, {})

        # Merge config with metadata (config takes precedence)
        target_column = d.get("target_column") or metadata.get(
            "target_column", "class"
        )
        categorical_features = d.get("categorical_features") or metadata.get(
            "categorical_features", []
        )
        columns_to_drop = d.get("columns_to_drop") or metadata.get(
            "columns_to_drop", []
        )
        sep = d.get("sep") or metadata.get("sep", ",")
        file_path = (
            d.get("file_path")
            or d.get("train_path")
            or metadata.get("file_path")
        )

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
            columns_to_drop=columns_to_drop,
            scale=d.get("scale", True),
            scaler_type=d.get("scaler_type", "standard"),
            seed=d.get("seed", 42),
            sep=sep,
        )

    @staticmethod
    def load_from_config(
        cfg: Dict[str, Any], project_root: Optional[Path] = None
    ) -> DatasetSplit:
        """Helper to create and load in one step.

        Args:
            cfg (Dict[str, Any]): Dataset configuration segment.
            project_root (Optional[Path]): Optional path to resolve relative
                file paths.

        Returns:
            DatasetSplit: Preprocessed data splits.
        """
        adapter = DatasetFactory.get_adapter_from_config(cfg, project_root)
        return adapter.load()
