"""Command for aggregating models from clients."""
from typing import Dict, Any, List, Optional
import numpy as np
from src.domain.aggregation.base_strategy import IAggregationStrategy

class AggregateCommand:
    """
    Command to aggregate models from multiple clients.
    """

    def __init__(self, strategy: IAggregationStrategy):
        self.strategy = strategy

    def execute(
        self,
        client_models: Dict[str, Any],
        client_metadata: Dict[str, Any],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        t_max: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Aggregate client models into a global model.

        Args:
            client_models: Dictionary of client_id -> model data
            client_metadata: Dictionary of client_id -> metadata
            X_val: Validation features for Progressive Forest (S2-S7)
            y_val: Validation labels for Progressive Forest (S2-S7)
            t_max: Maximum trees in global model (T_MAX en tesis)
            **kwargs: Strategy-specific parameters (f1_weight, pcd_weight, etc.)

        Returns:
            Global model data
        """
        # Extract trees from each client
        client_trees = {}
        for client_id, model_data in client_models.items():
            client_trees[client_id] = model_data.get('trees', [])

        # Aggregate using strategy
        result = self.strategy.aggregate(
            client_trees,
            client_metadata,
            X_val=X_val,
            y_val=y_val,
            t_max=t_max,
            **kwargs
        )
        
        # Unpack result (strategies now return 5 values)
        global_trees, selected_indices, all_tree_entries, conv_round, logs = result

        return {
            'global_trees': global_trees,
            'selected_indices': selected_indices,
            'all_tree_entries': all_tree_entries,
            'convergence_round': conv_round,
            'round_logs': logs
        }