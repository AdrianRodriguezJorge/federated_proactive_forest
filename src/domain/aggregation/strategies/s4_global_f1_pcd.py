from typing import Dict, List, Any, Tuple
from ..tree_ranker import TreeRanker, RankingCriterion, TreeEntry


class S4GlobalF1PCDStrategy:
    """
    Strategy S4: Global F1 + PCD (Pair Classifier Disagreement)
    Ranks all trees globally by combined F1 and diversity (PCD).
    Balances accuracy and diversity to ensure non-redundant tree selection.
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
    ) -> Tuple[List[Any], Dict[str, List[int]]]:
        """
        Rank all trees by combined F1 and PCD score.
        Score = f1_weight * F1 + pcd_weight * PCD
        
        Args:
            client_trees: {client_id: [trees...]}
            client_metadata: {client_id: ClientMetadata}
            f1_weight: Weight for F1 in ranking (default 0.5)
            pcd_weight: Weight for PCD diversity (default 0.5)
        
        Returns:
          - List[Any]: Global trees ranked by F1+PCD
          - Dict[str, List[int]]: {client_id: [global_indices_selected]}
        """
        # Build entries for all trees
        entries = TreeEntry.build_entries(client_trees, client_metadata)
        
        # Rank by combined F1+PCD
        ranker = TreeRanker(
            criterion=RankingCriterion.F1_PCD,
            f1_weight=f1_weight,
            pcd_weight=pcd_weight
        )
        ranked_entries = ranker.rank(entries)
        
        # All trees are selected (ranked)
        global_trees = [e.tree for e in ranked_entries]
        
        # Build selected_ids dictionary
        selected_ids = {cid: [] for cid in client_trees.keys()}
        for global_idx, entry in enumerate(ranked_entries):
            selected_ids[entry.client_id].append(global_idx)
        
        return global_trees, selected_ids
