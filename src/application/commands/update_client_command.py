"""Command for updating clients with global model."""

from typing import Any, Dict


class UpdateClientCommand:
    """Command to update clients with the global aggregated model.

    Uses No-Repeat Merge: Concatenates local trees with global trees, excluding
    local trees already selected in aggregation.
    """

    def execute_on_client(
        self, client_data: Dict[str, Any], global_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update client with global model trees using No-Repeat Merge.

        Concatenates local trees with global trees, excluding selected local
        trees to avoid redundancy.

        Args:
            client_data (Dict[str, Any]): Client's current data.
            global_data (Dict[str, Any]): Global model data.

        Returns:
            Dict[str, Any]: Updated client data dictionary.

        Raises:
            ValueError: If no local model is found in client_data.
        """
        client_id = client_data.get("client_id")
        selected_indices = (
            global_data.get("selected_indices", {}).get(client_id, [])
        )
        global_trees = global_data.get("global_trees", [])
        all_tree_entries = global_data.get("all_tree_entries", [])

        # Calculate selected_local_ids: local indices of client's trees
        # selected in global
        selected_local_ids = []
        for idx in selected_indices:
            if idx < len(global_trees):
                tree = global_trees[idx]
                for entry in all_tree_entries:
                    if entry.tree is tree and entry.client_id == client_id:
                        selected_local_ids.append(entry.tree_local_id)
                        break

        # Get local trees
        local_model = client_data.get("model")
        if local_model is None:
            raise ValueError("No local model found in client data")
        local_trees = local_model.get_trees()

        # Perform No-Repeat Merge: surviving_local + global_trees
        from src.domain.update.client_updater import ClientUpdater

        merged_trees = ClientUpdater.merge(
            local_trees, global_trees, selected_local_ids
        )

        # Update client's model with merged trees
        from src.domain.model.proactive_forest import ProactiveForest

        class_names = global_data.get("class_names", None)
        updated_forest = ProactiveForest.from_trees(
            merged_trees, class_names=class_names
        )
        client_data["model"] = updated_forest
        client_data["merged_trees"] = merged_trees

        return client_data