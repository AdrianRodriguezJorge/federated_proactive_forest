from typing import Dict, List, Any, Tuple
from ..tree_ranker import TreeRanker, RankingCriterion, TreeEntry


class S2GlobalAccuracyStrategy:
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
        
        Returns:
          - List[Any]: Global trees (all trees ranked by accuracy)
          - Dict[str, List[int]]: {client_id: [global_indices_of_selected_trees]}
        """
        # Build entries for all trees
        entries = TreeRanker.build_entries(client_trees, client_metadata)
        
        # Rank by accuracy
        ranker = TreeRanker(criterion=RankingCriterion.ACCURACY)
        ranked_entries = ranker.rank(entries)
        
        # All trees are selected (ranked)
        global_trees = [e.tree for e in ranked_entries]
        
        # Build selected_ids dictionary
        selected_ids = {cid: [] for cid in client_trees.keys()}
        for global_idx, entry in enumerate(ranked_entries):
            selected_ids[entry.client_id].append(global_idx)
        
        return global_trees, selected_ids