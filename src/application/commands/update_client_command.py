"""Command for updating clients with global model."""
from typing import Dict, Any, List

class UpdateClientCommand:
    """
    Command to update clients with the global aggregated model.
    """

    def execute_on_client(self, client_data: Dict[str, Any], global_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update client with global model trees.

        Args:
            client_data: Client's current data
            global_data: Global model data

        Returns:
            Updated client data
        """
        client_id = client_data.get('client_id')
        selected_indices = global_data.get('selected_indices', {}).get(client_id, [])

        # Get selected trees from global model
        global_trees = global_data.get('global_trees', [])
        selected_trees = [global_trees[i] for i in selected_indices if i < len(global_trees)]

        # Update client's model with selected trees
        from src.domain.model.proactive_forest import ProactiveForest
        class_names = global_data.get('class_names', None)
        updated_forest = ProactiveForest.from_trees(selected_trees, class_names=class_names)
        client_data['model'] = updated_forest
        client_data['selected_trees'] = selected_trees

        return client_data