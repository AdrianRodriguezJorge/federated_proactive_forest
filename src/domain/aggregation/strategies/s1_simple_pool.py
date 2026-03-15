from typing import Dict, List, Any, Tuple
from ..base_strategy import IAggregationStrategy
from ..tree_ranker import TreeRanker


class S1SimplePoolStrategy(IAggregationStrategy):
    """Strategy S1: Simple Pool
    Simply pools all trees from all clients without any selection or ranking.
    """

    @property
    def strategy_id(self) -> str:
        return "S1"

    def aggregate(
        self,
        client_trees: Dict[str, List[Any]],
        client_metadata: Dict,
    ) -> Tuple[List[Any], Dict[str, List[int]], List[Any]]:
        """Aggregate all trees from all clients.

        Returns:
          - global_trees: trees pooled from all clients
          - selected_indices: all trees selected per client
          - all_entries: entries in the same order as global_trees
        """
        global_trees = []
        selected_indices = {}

        for client_id, trees in client_trees.items():
            global_trees.extend(trees)
            # Select all trees for this client (local indices)
            selected_indices[client_id] = list(range(len(trees)))

        # Provide entries in the same order for ranking display
        all_entries = TreeRanker.build_entries(client_trees, client_metadata)

        return global_trees, selected_indices, all_entries
