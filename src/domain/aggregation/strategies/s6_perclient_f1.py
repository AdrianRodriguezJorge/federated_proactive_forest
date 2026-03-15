from typing import Dict, List, Any, Tuple
from ..tree_ranker import TreeRanker, RankingCriterion, TreeEntry


class S6PerClientF1Strategy:
    """Strategy S6: Per-Client Macro-F1
    Each client selects its best trees by macro-F1 independently.
    """

    @property
    def strategy_id(self) -> str:
        return "S6"

    def aggregate(
        self,
        client_trees: Dict[str, List[Any]],
        client_metadata: Dict,
        max_trees_per_client: int = None,
    ) -> Tuple[List[Any], Dict[str, List[int]], List[TreeEntry]]:
        """Select best trees per client by macro-F1.

        Returns:
          - global_trees: concatenated selected trees from all clients
          - selected_ids: mapping of client -> global indices selected
          - all_entries: entries in the order they are appended to global_trees
        """
        global_trees = []
        selected_ids = {}
        all_entries: List[TreeEntry] = []

        for client_id, trees in client_trees.items():
            meta = client_metadata[client_id]

            # Create entries for this client's trees
            entries = []
            for local_idx, tree in enumerate(trees):
                entries.append(TreeEntry(
                    tree=tree,
                    client_id=client_id,
                    tree_local_id=local_idx,
                    accuracy=meta.accuracy,
                    macro_f1=meta.macro_f1,
                    pcd=meta.pcd,
                ))

            # Rank by macro-F1 within client
            ranker = TreeRanker(criterion=RankingCriterion.MACRO_F1)
            ranked_entries = ranker.rank(entries)

            # Select all (or limited) per client
            if max_trees_per_client is not None:
                ranked_entries = ranked_entries[:max_trees_per_client]

            # Add to global and track indices
            start_idx = len(global_trees)
            selected_ids[client_id] = []
            for entry in ranked_entries:
                global_trees.append(entry.tree)
                selected_ids[client_id].append(start_idx + len(selected_ids[client_id]))
                all_entries.append(entry)

        return global_trees, selected_ids, all_entries
