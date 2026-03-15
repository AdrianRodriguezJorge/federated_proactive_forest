from typing import Dict, List, Any, Tuple
from ..tree_ranker import TreeRanker, RankingCriterion, TreeEntry


class S4GlobalF1PCDStrategy:
    """Strategy S4: Global F1 + PCD
    Ranks all trees globally by weighted combination of macro-F1 and PCD.
    """

    @property
    def strategy_id(self) -> str:
        return "S4"

    def aggregate(
        self,
        client_trees: Dict[str, List[Any]],
        client_metadata: Dict,
        f1_weight: float = 0.5,
        pcd_weight: float = 0.5,
    ) -> Tuple[List[Any], Dict[str, List[int]], List[TreeEntry]]:
        """Rank all trees by combined F1+PCD score and select top performers.

        Returns:
          - global_trees: list of trees in ranked order
          - selected_ids: mapping of client -> global indices selected
          - all_entries: TreeEntry list in the rank order
        """
        entries = TreeRanker.build_entries(client_trees, client_metadata)

        # Rank by combined score
        ranker = TreeRanker(RankingCriterion.F1_PCD, f1_weight=f1_weight, pcd_weight=pcd_weight)
        ranked_entries = ranker.rank(entries)

        global_trees = [e.tree for e in ranked_entries]
        # Build selected_ids dictionary (local tree indices per client)
        selected_ids = {cid: [] for cid in client_trees.keys()}
        for entry in ranked_entries:
            selected_ids[entry.client_id].append(entry.tree_local_id)

        return global_trees, selected_ids, ranked_entries
