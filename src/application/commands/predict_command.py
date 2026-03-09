"""Command for making predictions."""
from typing import Dict, Any, List
import numpy as np

class PredictCommand:
    """
    Command to make predictions using trained models.
    """

    def execute_on_client(self, client_data: Dict[str, Any], X_test: np.ndarray) -> Dict[str, Any]:
        """
        Make predictions on test data.

        Args:
            client_data: Client data with trained model
            X_test: Test features

        Returns:
            Updated client data with predictions
        """
        model = client_data.get('model')
        if model is None:
            raise ValueError("No model found in client data")

        predictions = model.predict(X_test)
        client_data['predictions'] = predictions

        return client_data

    def execute_global(self, global_data: Dict[str, Any], X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions using global model.

        Args:
            global_data: Global model data
            X_test: Test features

        Returns:
            Global predictions
        """
        from src.domain.model.proactive_forest import ProactiveForest
        global_trees = global_data.get('global_trees', [])
        if not global_trees:
            raise ValueError("No global trees found")

        global_model = ProactiveForest.from_trees(global_trees)
        return global_model.predict(X_test)