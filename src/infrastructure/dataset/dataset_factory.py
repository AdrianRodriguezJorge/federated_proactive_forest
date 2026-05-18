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
