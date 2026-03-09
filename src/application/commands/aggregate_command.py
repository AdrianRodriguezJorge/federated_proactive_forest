"""Command for aggregating models from clients."""
from typing import Dict, Any, List
from src.domain.aggregation.base_strategy import IAggregationStrategy

class AggregateCommand:
    """
    Command to aggregate models from multiple clients.
    """

    def __init__(self, strategy: IAggregationStrategy):
        self.strategy = strategy

    def execute(self, client_models: Dict[str, Any], client_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate client models into a global model.

        Args:
            client_models: Dictionary of client_id -> model data
            client_metadata: Dictionary of client_id -> metadata

        Returns:
            Global model data
        """
        # Extract trees from each client
        client_trees = {}
        for client_id, model_data in client_models.items():
            client_trees[client_id] = model_data.get('trees', [])

        # Aggregate using strategy
        global_trees, selected_indices = self.strategy.aggregate(client_trees, client_metadata)

        return {
            'global_trees': global_trees,
            'selected_indices': selected_indices
        }