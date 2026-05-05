import numpy as np
from typing import Dict, List, Any, Optional
from src.domain.metrics.metrics_service import IDiversityService

class TreeMetricExtractor:
    """
    Service responsible for extracting or calculating metrics (accuracy, macro_f1, pcd)
    for individual trees before ranking.
    """
    def __init__(self, diversity_service: Optional[IDiversityService] = None):
        self.diversity_service = diversity_service

    def extract_metrics(self, 
                        client_trees: Dict[str, List[Any]], 
                        client_metadata: Dict, 
                        X_val: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        Returns a list of dictionaries with extracted data for each tree, 
        ready to be converted to TreeEntry.
        """
        raw_entries = []
        for cid, trees in client_trees.items():
            meta = client_metadata[cid]
            tree_metrics = getattr(meta, 'tree_metrics', [])
            
            predictions_matrix = None
            if X_val is not None and self.diversity_service is not None:
                predictions_matrix = np.zeros((X_val.shape[0], len(trees)), dtype=int)
                for i, tree in enumerate(trees):
                    predictions_matrix[:, i] = tree.predict(X_val)

            for local_id, tree in enumerate(trees):
                tree_acc = tree_metrics[local_id]['accuracy'] if local_id < len(tree_metrics) else meta.accuracy
                tree_f1 = tree_metrics[local_id]['macro_f1'] if local_id < len(tree_metrics) else meta.macro_f1
                
                tree_pcd = meta.pcd
                if predictions_matrix is not None and self.diversity_service is not None:
                    n_others = predictions_matrix.shape[1] - 1
                    if n_others > 0:
                        other_preds = np.delete(predictions_matrix, local_id, axis=1)
                        disagreements = (predictions_matrix[:, [local_id]] != other_preds)
                        tree_pcd = float(np.mean(disagreements))

                raw_entries.append({
                    'tree': tree,
                    'client_id': cid,
                    'tree_local_id': local_id,
                    'accuracy': tree_acc,
                    'macro_f1': tree_f1,
                    'pcd': tree_pcd,
                })
        return raw_entries
