from typing import Dict, List, Any, Tuple
from ..base_strategy import IAggregationStrategy

class S1SimplePoolStrategy(IAggregationStrategy):
    """
    Strategy S1: Simple Pool
    Simply pools all trees from all clients without any selection or ranking.
    """

    @property
    def strategy_id(self) -> str:
        return "S1"

    def aggregate(
        self,
        client_trees: Dict[str, List[Any]],
        client_metadata: Dict,
    ) -> Tuple[List[Any], Dict[str, List[int]]]:
        """
        Aggregate all trees from all clients.
        Returns all trees and selects all for each client.
        """
        global_trees = []
        selected_indices = {}

        for client_id, trees in client_trees.items():
            start_idx = len(global_trees)
            global_trees.extend(trees)
            # Select all trees for this client
            selected_indices[client_id] = list(range(start_idx, len(global_trees)))

        return global_trees, selected_indices