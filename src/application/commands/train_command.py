"""Command for training models on clients."""

from typing import Any, Callable, Dict
from src.domain.model.base_forest import ABCForest


class TrainCommand:
    """Command to train models on federated clients."""

    def __init__(self, forest_factory: Callable[[], ABCForest]):
        """Initializes TrainCommand.

        Args:
            forest_factory (Callable[[], ABCForest]): Callable to instantiate
                a new forest model.
        """
        self.forest_factory = forest_factory

    def execute_on_client(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train a forest on client data.

        Args:
            client_data (Dict[str, Any]): Dictionary containing 'X_train',
                'y_train', etc.

        Returns:
            Dict[str, Any]: Updated client data dictionary with trained model.
        """
        X_train = client_data["X_train"]
        y_train = client_data["y_train"]

        # Create and train forest
        forest = self.forest_factory()
        forest.fit(X_train, y_train)

        # Store trained model
        client_data["model"] = forest
        client_data["trees"] = forest.get_trees()

        return client_data