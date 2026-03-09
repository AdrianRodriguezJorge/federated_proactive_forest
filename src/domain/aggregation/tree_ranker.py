"""Lógica de ordenamiento de árboles reutilizada por S2–S7."""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List


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
                 f1_weight: float = 0.5, pcd_weight: float = 0.5):
        self.criterion = criterion
        self.f1_w = f1_weight
        self.pcd_w = pcd_weight

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
                      client_metadata: Dict) -> List[TreeEntry]:
        """Construye TreeEntry por cada árbol de cada cliente."""
        entries = []
        for cid, trees in client_trees.items():
            meta = client_metadata[cid]
            for local_id, tree in enumerate(trees):
                entries.append(TreeEntry(
                    tree=tree,
                    client_id=cid,
                    tree_local_id=local_id,
                    accuracy=meta.accuracy,
                    macro_f1=meta.macro_f1,
                    pcd=meta.pcd,
                ))
        return entries
