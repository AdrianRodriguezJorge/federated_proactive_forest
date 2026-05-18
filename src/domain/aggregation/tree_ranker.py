"""Tree ranking logic for ensemble tree aggregation (S2-S7).

Defines criteria, metadata wrapper dataclasses (TreeEntry), and ranker logic
relying on local F1, Accuracy, and Percentage Correct Diversity (PCD) scores.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np

from src.domain.aggregation.services.tree_metric_extractor import (
    TreeMetricExtractor,
)
from src.domain.metrics.metrics_service import IDiversityService


class RankingCriterion(str, Enum):
    """Enumeration of possible ranking metrics for trees."""

    ACCURACY = "accuracy"  # S2, S5
    MACRO_F1 = "macro_f1"  # S3, S6
    F1_PCD = "f1_pcd"  # S4, S7


@dataclass
class TreeEntry:
    """Wrapper encapsulating a decision tree along with local score metadata.

    Attributes:
        tree (Any): Decision tree instance.
        client_id (str): ID of client who trained this tree.
        tree_local_id (int): Index inside client's local ensemble.
        accuracy (float): Client's local forest validation accuracy.
        macro_f1 (float): Client's local forest validation Macro-F1 score.
        pcd (float): Client's local forest Percentage Correct Diversity score.
        score (float): Score assigned dynamically during ranking.
    """

    tree: Any
    client_id: str
    tree_local_id: int
    accuracy: float
    macro_f1: float
    pcd: float
    score: float = 0.0


class TreeRanker:
    """Logic for ranking trees based on performance and diversity criteria.

    Used by strategies S2-S7 to determine which trees to include in the global
    forest model.
    """

    def __init__(
        self,
        criterion: RankingCriterion,
        f1_weight: float = 0.5,
        pcd_weight: float = 0.5,
        diversity_service: Optional[IDiversityService] = None,
    ):
        """Initializes the tree ranker.

        Args:
            criterion (RankingCriterion): Metric used for ranking.
            f1_weight (float): Weight for F1-score when using F1+PCD criterion.
            pcd_weight (float): Weight for PCD when using F1+PCD criterion.
            diversity_service (Optional[IDiversityService]): Service for PCD.
        """
        self.criterion = criterion
        self.f1_w = f1_weight
        self.pcd_w = pcd_weight
        self.diversity_service = diversity_service

    def _score(self, entry: TreeEntry) -> float:
        """Calculate score value for a single TreeEntry.

        Args:
            entry (TreeEntry): Target tree entry.

        Returns:
            float: Score value.
        """
        if self.criterion == RankingCriterion.ACCURACY:
            return entry.accuracy
        if self.criterion == RankingCriterion.MACRO_F1:
            return entry.macro_f1
        return self.f1_w * entry.macro_f1 + self.pcd_w * entry.pcd

    def rank(self, entries: List[TreeEntry]) -> List[TreeEntry]:
        """Rank entries in descending order.

        Args:
            entries (List[TreeEntry]): TreeEntry list to rank.

        Returns:
            List[TreeEntry]: Ranked list.
        """
        for e in entries:
            e.score = self._score(e)
        return sorted(entries, key=lambda e: e.score, reverse=True)

    @staticmethod
    def build_entries(
        client_trees: Dict[str, List[Any]],
        client_metadata: Dict[str, Any],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        diversity_service: Optional[IDiversityService] = None,
    ) -> List[TreeEntry]:
        """Build a TreeEntry instance for each client tree.

        Args:
            client_trees (Dict[str, List[Any]]): Client trees dict.
            client_metadata (Dict[str, Any]): Client metadata dict.
            X_val (Optional[np.ndarray]): Validation features.
            y_val (Optional[np.ndarray]): Validation labels.
            diversity_service (Optional[IDiversityService]): Diversity service.

        Returns:
            List[TreeEntry]: Raw constructed TreeEntry list.
        """
        extractor = TreeMetricExtractor(diversity_service=diversity_service)
        raw_entries = extractor.extract_metrics(
            client_trees, client_metadata, X_val, y_val
        )

        entries = []
        for raw in raw_entries:
            entries.append(
                TreeEntry(
                    tree=raw["tree"],
                    client_id=raw["client_id"],
                    tree_local_id=raw["tree_local_id"],
                    accuracy=raw["accuracy"],
                    macro_f1=raw["macro_f1"],
                    pcd=raw["pcd"],
                )
            )
        return entries
