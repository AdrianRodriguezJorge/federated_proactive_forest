"""Local metadata repository."""
import json
import os
from typing import Dict, Any, Optional


class LocalMetadataRepository:
    """
    Repository for storing and loading metadata locally.
    """

    def __init__(self, storage_path: str = "./metadata"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)

    def save_metadata(self, metadata: Dict[str, Any], name: str) -> None:
        """
        Save metadata to disk.

        Args:
            metadata: Metadata dictionary
            name: Name for the saved file
        """
        filepath = os.path.join(self.storage_path, f"{name}.json")
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Load metadata from disk.

        Args:
            name: Name of the saved metadata

        Returns:
            Loaded metadata or None if not found
        """
        filepath = os.path.join(self.storage_path, f"{name}.json")
        if not os.path.exists(filepath):
            return None

        with open(filepath, 'r') as f:
            return json.load(f)

    def list_saved_metadata(self) -> list[str]:
        """
        List all saved metadata names.

        Returns:
            List of metadata names
        """
        if not os.path.exists(self.storage_path):
            return []

        files = os.listdir(self.storage_path)
        return [f.replace('.json', '') for f in files if f.endswith('.json')]