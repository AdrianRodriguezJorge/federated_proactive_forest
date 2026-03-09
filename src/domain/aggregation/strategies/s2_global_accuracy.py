from typing import Dict, List, Any, Tuple
import numpy as np
from ..base_strategy import IAggregationStrategy

class S2GlobalAccuracyStrategy(IAggregationStrategy):
    """
    Strategy S2: Global Accuracy
    Ranks all trees globally by accuracy and selects the best ones.
    """

    @property
    def strategy_id(self) -> str:
        return "S2"

    def aggregate(
        self,
        client_trees: Dict[str, List[Any]],
        client_metadata: Dict,
    ) -> Tuple[List[Any], Dict[str, List[int]]]:
        """
        Rank all trees by global accuracy and select top performers.
        """
        # This is a simplified implementation
        # In practice, would need to evaluate each tree's accuracy
        all_trees = []
        tree_info = []

        for client_id, trees in client_trees.items():
            for i, tree in enumerate(trees):
                all_trees.append(tree)
                tree_info.append({
                    'client_id': client_id,
                    'local_idx': i,
                    'global_idx': len(tree_info)
                })

        # For now, select all trees (simplified)
        selected_indices = {}
        for client_id in client_trees.keys():
            selected_indices[client_id] = list(range(len(all_trees)))

        return all_trees, selected_indices