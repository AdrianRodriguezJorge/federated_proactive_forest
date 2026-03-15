from typing import Dict, List, Any, Tuple
from ..tree_ranker import TreeRanker, RankingCriterion, TreeEntry


class S7PerClientF1PCDStrategy:
    """Strategy S7: Per-Client F1 + PCD
    Each client ranks trees by a weighted combination of macro-F1 and PCD, and selects the best.
    """

    @property
    def strategy_id(self) -> str:
        return "S7"

    def aggregate(
        self,
        client_trees: Dict[str, List[Any]],
        client_metadata: Dict,
        f1_weight: float = 0.5,
        pcd_weight: float = 0.5,
        max_trees_per_client: int = None,
    ) -> Tuple[List[Any], Dict[str, List[int]], List[TreeEntry]]:
        """Select best trees per client by combined F1+PCD score.

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

            ranker = TreeRanker(RankingCriterion.F1_PCD, f1_weight=f1_weight, pcd_weight=pcd_weight)
            ranked_entries = ranker.rank(entries)

            if max_trees_per_client is not None:
                ranked_entries = ranked_entries[:max_trees_per_client]

            start_idx = len(global_trees)
            selected_ids[client_id] = []
            for entry in ranked_entries:
                global_trees.append(entry.tree)
                selected_ids[client_id].append(start_idx + len(selected_ids[client_id]))
                all_entries.append(entry)

        return global_trees, selected_ids, all_entries
