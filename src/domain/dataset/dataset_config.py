from dataclasses import dataclass
from typing import List, Optional
import yaml

@dataclass
class DatasetConfig:
    """
    Configuration for a dataset adapter.
    Loaded from YAML configuration files.
    """
    name: str
    path: str
    target_column: str
    feature_columns: Optional[List[str]] = None
    categorical_columns: Optional[List[str]] = None
    numerical_columns: Optional[List[str]] = None
    test_size: float = 0.2
    random_state: int = 42
    scale_numerical: bool = True
    encode_categorical: bool = True

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'DatasetConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)