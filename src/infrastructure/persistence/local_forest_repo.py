"""Local forest repository for persistence."""
import os
from typing import List, Any, Optional
from src.domain.model.base_forest import ABCForest
from .tree_serializer import TreeSerializer


class LocalForestRepository:
    """
    Repository for storing and loading forests locally.
    """

    def __init__(self, storage_path: str = "./models"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)

    def save_forest(self, forest: ABCForest, name: str) -> None:
        """
        Save a forest to disk.

        Args:
            forest: Forest to save
            name: Name for the saved file
        """
        trees = forest.get_trees()
        data = TreeSerializer.serialize_forest(trees)

        filepath = os.path.join(self.storage_path, f"{name}.joblib")
        with open(filepath, 'wb') as f:
            f.write(data)

    def load_forest(self, name: str, forest_class: type) -> Optional[ABCForest]:
        """
        Load a forest from disk.

        Args:
            name: Name of the saved forest
            forest_class: Class to instantiate the forest with

        Returns:
            Loaded forest or None if not found
        """
        filepath = os.path.join(self.storage_path, f"{name}.joblib")
        if not os.path.exists(filepath):
            return None

        with open(filepath, 'rb') as f:
            data = f.read()

        trees = TreeSerializer.deserialize_forest(data)
        return forest_class.from_trees(trees)

    def list_saved_forests(self) -> List[str]:
        """
        List all saved forest names.

        Returns:
            List of forest names
        """
        if not os.path.exists(self.storage_path):
            return []

        files = os.listdir(self.storage_path)
        return [f.replace('.joblib', '') for f in files if f.endswith('.joblib')]