"""Lógica de ordenamiento de árboles reutilizada por S2–S7."""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np
from src.domain.metrics.metrics_service import IDiversityService


class RankingCriterion(str, Enum):
    ACCURACY = "accuracy"   # S2, S5
    MACRO_F1 = "macro_f1"   # S3, S6
    F1_PCD   = "f1_pcd"     # S4, S7


@dataclass
class TreeEntry:
    tree: Any               # DecisionTree del paquete proactive_forest
    client_id: str
    tree_local_id: int      # índice dentro del bosque del cliente
    accuracy: float         # accuracy del bosque que aportó este árbol
    macro_f1: float
    pcd: float              # PCD del bosque que aportó este árbol
    score: float = 0.0      # calculado por TreeRanker.rank()


class TreeRanker:
    def __init__(self, criterion: RankingCriterion,
                 f1_weight: float = 0.5, pcd_weight: float = 0.5,
                 diversity_service: Optional[IDiversityService] = None):
        self.criterion = criterion
        self.f1_w = f1_weight
        self.pcd_w = pcd_weight
        self.diversity_service = diversity_service

    def _score(self, entry: TreeEntry) -> float:
        if self.criterion == RankingCriterion.ACCURACY:
            return entry.accuracy
        if self.criterion == RankingCriterion.MACRO_F1:
            return entry.macro_f1
        return self.f1_w * entry.macro_f1 + self.pcd_w * entry.pcd

    def rank(self, entries: List[TreeEntry]) -> List[TreeEntry]:
        for e in entries:
            e.score = self._score(e)
        return sorted(entries, key=lambda e: e.score, reverse=True)

    @staticmethod
    def build_entries(client_trees: Dict[str, List[Any]],
                      client_metadata: Dict,
                      X_val: Optional[np.ndarray] = None,
                      diversity_service: Optional[IDiversityService] = None) -> List[TreeEntry]:
        """Construye TreeEntry por cada árbol de cada cliente."""
        entries = []
        for cid, trees in client_trees.items():
            meta = client_metadata[cid]
            # Extraer tree_metrics si existen (lista de dicts con 'accuracy' y 'macro_f1')
            tree_metrics = getattr(meta, 'tree_metrics', [])
            
            # Predict each tree if X_val is provided and individual metrics are missing or PCD is needed
            predictions_matrix = None
            if X_val is not None and diversity_service is not None:
                # Optimized: Predict all trees in the client once
                predictions_matrix = np.zeros((X_val.shape[0], len(trees)), dtype=int)
                for i, tree in enumerate(trees):
                    predictions_matrix[:, i] = tree.predict(X_val)

            for local_id, tree in enumerate(trees):
                # Usar métrica individual si está disponible, si no usar la del bosque completo (fallback)
                tree_acc = tree_metrics[local_id]['accuracy'] if local_id < len(tree_metrics) else meta.accuracy
                tree_f1 = tree_metrics[local_id]['macro_f1'] if local_id < len(tree_metrics) else meta.macro_f1
                
                # Para PCD, si tenemos el matrix de predicciones y el servicio, calculamos PCD individual
                tree_pcd = meta.pcd
                if predictions_matrix is not None and diversity_service is not None:
                    # Calculate mean disagreement of this tree with others in the same client forest
                    n_others = predictions_matrix.shape[1] - 1
                    if n_others > 0:
                        # Slice other trees' predictions
                        other_preds = np.delete(predictions_matrix, local_id, axis=1)
                        # Mean disagreement of this tree with all others
                        disagreements = (predictions_matrix[:, [local_id]] != other_preds)
                        tree_pcd = float(np.mean(disagreements))

                entries.append(TreeEntry(
                    tree=tree,
                    client_id=cid,
                    tree_local_id=local_id,
                    accuracy=tree_acc,
                    macro_f1=tree_f1,
                    pcd=tree_pcd,
                ))
        return entries
