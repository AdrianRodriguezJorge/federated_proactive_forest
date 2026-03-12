from typing import Dict, List, Any, Tuple
from ..tree_ranker import TreeRanker, RankingCriterion, TreeEntry


class S7PerClientF1PCDStrategy:
    """
    Strategy S7: Per-Client F1 + PCD
    Each client selects its best trees by combined F1 and diversity (PCD) independently.
    """

    @property
    def strategy_id(self) -> str:
        return "S7"

    def aggregate(
        self,
        client_trees: Dict[str, List[Any]],
        client_metadata: Dict,
        max_trees_per_client: int = None,
        f1_weight: float = 0.5,
        pcd_weight: float = 0.5,
    ) -> Tuple[List[Any], Dict[str, List[int]]]:
        """
        Each client selects its best trees by combined F1 and PCD diversity.
        Score = f1_weight * F1 + pcd_weight * PCD
        
        Args:
            client_trees: {client_id: [trees...]}
            client_metadata: {client_id: ClientMetadata}
            max_trees_per_client: Max trees to select per client (None = all)
            f1_weight: Weight for F1 in ranking (default 0.5)
            pcd_weight: Weight for PCD diversity (default 0.5)
        
        Returns:
          - List[Any]: Global trees (concatenation of selected trees per client)
          - Dict[str, List[int]]: {client_id: [global_indices_of_selected]}
        """
        global_trees = []
        selected_ids = {}
        
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
            
            # Rank by F1+PCD within client
            ranker = TreeRanker(
                criterion=RankingCriterion.F1_PCD,
                f1_weight=f1_weight,
                pcd_weight=pcd_weight
            )
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
        
        return global_trees, selected_ids
