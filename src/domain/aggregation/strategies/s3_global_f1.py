from typing import Dict, List, Any, Tuple
from ..tree_ranker import TreeRanker, RankingCriterion, TreeEntry


class S3GlobalF1Strategy:
    """Strategy S3: Global F1
    Ranks all trees globally by macro-F1 and selects the best ones.
    """

    @property
    def strategy_id(self) -> str:
        return "S3"

    def aggregate(
        self,
        client_trees: Dict[str, List[Any]],
        client_metadata: Dict,
    ) -> Tuple[List[Any], Dict[str, List[int]], List[TreeEntry]]:
        """Rank all trees by macro-F1 and select top performers.

        Returns:
          - global_trees: list of trees in ranked order
          - selected_ids: mapping of client -> global indices selected
          - all_entries: TreeEntry list in the rank order
        """
        # Build entries for all trees
        entries = TreeRanker.build_entries(client_trees, client_metadata)

        # Rank by macro-F1
        ranker = TreeRanker(criterion=RankingCriterion.MACRO_F1)
        ranked_entries = ranker.rank(entries)

        global_trees = [e.tree for e in ranked_entries]
        # Build selected_ids dictionary (local tree indices per client)
        selected_ids = {cid: [] for cid in client_trees.keys()}
        for entry in ranked_entries:
            selected_ids[entry.client_id].append(entry.tree_local_id)

        return global_trees, selected_ids, ranked_entries
