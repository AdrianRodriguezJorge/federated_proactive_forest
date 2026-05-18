"""Strategy S1: Simple Pool aggregation strategy.

Pools all trees from all clients without any selective re-ranking or validation-
based progressive early stopping convergence checks.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..base_strategy import IAggregationStrategy
from ..tree_ranker import TreeRanker


class S1SimplePoolStrategy(IAggregationStrategy):
    """Strategy S1: Simple Pool.

    Simply pools all trees from all clients without any selection or ranking.
    """

    @property
    def strategy_id(self) -> str:
        """Get unique strategy ID.

        Returns:
            str: "S1"
        """
        return "S1"

    def aggregate(
        self,
        client_trees: Dict[str, List[Any]],
        client_metadata: Dict[str, Any],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        max_trees: Optional[int] = None,
        max_trees_per_client: Optional[int] = None,
        t_max: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[
        List[Any],
        Dict[str, List[int]],
        List[Any],
        Optional[int],
        List[Dict[str, Any]],
    ]:
        """Aggregate all trees from all clients.

        Args:
            client_trees (Dict[str, List[Any]]): Mapping from client ID to
                local trees list.
            client_metadata (Dict[str, Any]): Mapping from client ID to
                metadata dict.
            X_val (Optional[np.ndarray]): Validation features.
            y_val (Optional[np.ndarray]): Validation labels.
            max_trees (Optional[int]): Maximum total trees to select.
            max_trees_per_client (Optional[int]): Unused parameter.
            t_max (Optional[int]): Unused parameter.
            **kwargs (Any): Additional parameters.

        Returns:
            Tuple: A tuple containing:
                - List[Any]: Trees pooled from all clients.
                - Dict[str, List[int]]: Selected indices mapping.
                - List[Any]: TreeEntry candidates in pool order.
                - Optional[int]: None (no convergence early stopping).
                - List[Dict[str, Any]]: Empty logs list.
        """
        global_trees = []
        selected_indices = {}

        for client_id, trees in client_trees.items():
            global_trees.extend(trees)
            # Select all trees for this client (local indices)
            selected_indices[client_id] = list(range(len(trees)))

        # Provide entries in the same order for ranking display
        all_entries = TreeRanker.build_entries(client_trees, client_metadata)

        # Limit trees if max_trees is specified
        if max_trees is not None and len(global_trees) > max_trees:
            global_trees = global_trees[:max_trees]
            selected_entries = all_entries[:max_trees]

            # Recalculate selected_indices based on truncated global_trees
            selected_indices = {cid: [] for cid in client_trees.keys()}
            entry_idx = 0
            for client_id, trees in client_trees.items():
                for _ in trees:
                    if entry_idx < len(global_trees):
                        selected_indices[client_id].append(entry_idx)
                    entry_idx += 1

            return global_trees, selected_indices, selected_entries, None, []

        return global_trees, selected_indices, all_entries, None, []
