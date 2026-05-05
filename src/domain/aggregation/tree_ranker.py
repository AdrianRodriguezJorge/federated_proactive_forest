"""Lógica de ordenamiento de árboles reutilizada por S2–S7."""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np
from src.domain.metrics.metrics_service import IDiversityService
from src.domain.aggregation.services.tree_metric_extractor import TreeMetricExtractor


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
        """Construye TreeEntry por cada árbol de cada cliente delegando en TreeMetricExtractor."""
        extractor = TreeMetricExtractor(diversity_service=diversity_service)
        raw_entries = extractor.extract_metrics(client_trees, client_metadata, X_val)
        
        entries = []
        for raw in raw_entries:
            entries.append(TreeEntry(
                tree=raw['tree'],
                client_id=raw['client_id'],
                tree_local_id=raw['tree_local_id'],
                accuracy=raw['accuracy'],
                macro_f1=raw['macro_f1'],
                pcd=raw['pcd'],
            ))
        return entries
