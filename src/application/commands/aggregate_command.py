"""Command for aggregating models from clients."""

from typing import Any, Dict, Optional
import numpy as np

from src.domain.aggregation.base_strategy import IAggregationStrategy


class AggregateCommand:
    """Command to aggregate models from multiple clients."""

    def __init__(self, strategy: IAggregationStrategy):
        """Initializes AggregateCommand.

        Args:
            strategy (IAggregationStrategy): Selected aggregation strategy.
        """
        self.strategy = strategy

    def execute(
        self,
        client_models: Dict[str, Any],
        client_metadata: Dict[str, Any],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        t_max: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Aggregate client models into a global model.

        Args:
            client_models (Dict[str, Any]): Dictionary of client_id -> model.
            client_metadata (Dict[str, Any]): Dict of client_id -> metadata.
            X_val (Optional[np.ndarray]): Validation features.
            y_val (Optional[np.ndarray]): Validation labels.
            t_max (Optional[int]): Maximum trees in global model.
            **kwargs (Any): Strategy-specific parameters.

        Returns:
            Dict[str, Any]: Global model aggregation results package.
        """
        # Extract trees from each client
        client_trees = {}
        for client_id, model_data in client_models.items():
            client_trees[client_id] = model_data.get("trees", [])

        # Aggregate using strategy
        result = self.strategy.aggregate(
            client_trees,
            client_metadata,
            X_val=X_val,
            y_val=y_val,
            t_max=t_max,
            **kwargs,
        )

        # Unpack result (strategies return 5 values)
        (
            global_trees,
            selected_indices,
            all_tree_entries,
            conv_round,
            logs,
        ) = result

        return {
            "global_trees": global_trees,
            "selected_indices": selected_indices,
            "all_tree_entries": all_tree_entries,
            "convergence_round": conv_round,
            "round_logs": logs,
        }