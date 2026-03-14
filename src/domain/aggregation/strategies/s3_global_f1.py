from typing import Dict, List, Any, Tuple
from ..tree_ranker import TreeRanker, RankingCriterion, TreeEntry


class S3GlobalF1Strategy:
    """
    Strategy S3: Global F1
    Ranks all trees globally by F1 macro score and selects the best ones.
    """

    @property
    def strategy_id(self) -> str:
        return "S3"

    def aggregate(
        self,
        client_trees: Dict[str, List[Any]],
        client_metadata: Dict,
    ) -> Tuple[List[Any], Dict[str, List[int]]]:
        """
        Rank all trees by global F1 macro and select top performers.
        
        Returns:
          - List[Any]: Global trees (all trees ranked by F1)
          - Dict[str, List[int]]: {client_id: [global_indices_of_selected_trees]}
        """
        # Build entries for all trees
        entries = TreeRanker.build_entries(client_trees, client_metadata)
        
        # Rank by F1
        ranker = TreeRanker(criterion=RankingCriterion.MACRO_F1)
        ranked_entries = ranker.rank(entries)
        
        # All trees are selected (ranked)
        global_trees = [e.tree for e in ranked_entries]
        
        # Build selected_ids dictionary
        selected_ids = {cid: [] for cid in client_trees.keys()}
        for global_idx, entry in enumerate(ranked_entries):
            selected_ids[entry.client_id].append(global_idx)
        
        return global_trees, selected_ids
