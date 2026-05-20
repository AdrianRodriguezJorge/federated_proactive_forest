"""Base class for per-client strategies (S5-S7) with Progressive Forest.

These strategies rank trees within each client independently and incorporate
them using round-robin scheduling with Progressive Forest early stopping.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.domain.aggregation.services.progressive_selector import (
    ProgressiveSelector,
)
from src.domain.metrics.metrics_service import IDiversityService, IMetricsService
from src.domain.services.label_service import SimpleLabelService
from ..tree_ranker import RankingCriterion, TreeEntry, TreeRanker


class PerClientProgressiveStrategy(ABC):
    """Base class for per-client ranking strategies with Progressive Forest.

    Each client's trees are ranked independently, then incorporated via
    round-robin. EPISODE size is fixed to the number of clients (W), adding
    one tree per client each time.

    Aggregation stops by:
    1. Convergence (2 consecutive episodes with improvement < CONVERGENCE)
    2. T_MAX trees reached
    3. All trees exhausted
    """

    CONVERGENCE = 0.002
    T_MAX = 100

    def __init__(
        self,
        metrics_service: Optional[IMetricsService] = None,
        diversity_service: Optional[IDiversityService] = None,
    ):
        """Initializes the per-client progressive strategy.

        Args:
            metrics_service (Optional[IMetricsService]): Performance metrics.
            diversity_service (Optional[IDiversityService]): Diversity service.
        """
        self.metrics_svc = metrics_service
        self.diversity_svc = diversity_service

    @property
    @abstractmethod
    def ranking_criterion(self) -> RankingCriterion:
        """Return the ranking criterion for this strategy.

        Returns:
            RankingCriterion: Enum choice.
        """
        pass

    @property
    def strategy_id(self) -> str:
        """Get unique strategy ID.

        Returns:
            str: Identifier code.
        """
        return (
            self.__class__.__name__.replace("Strategy", "")
            .replace("PerClient", "S")
        )

    def aggregate(
        self,
        client_trees: Dict[str, List[Any]],
        client_metadata: Dict[str, Any],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        max_trees: Optional[int] = None,
        max_trees_per_client: Optional[int] = None,
        t_max: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[
        List[Any],
        Dict[str, List[int]],
        List[TreeEntry],
        Optional[int],
        List[Dict[str, Any]],
    ]:
        """Aggregates trees from multiple clients using per-client ranking.

        This method follows a three-step process:
        1. Ranks trees for each client independently.
        2. Interleaves the ranked trees using a round-robin approach.
        3. Applies Progressive Forest selection with early stopping.

        Args:
            client_trees (Dict[str, List[Any]]): Client local trees mapping.
            client_metadata (Dict[str, Any]): Client metadata mapping.
            X_val (Optional[np.ndarray]): Validation features.
            y_val (Optional[np.ndarray]): Validation labels.
            max_trees (Optional[int]): Unused fallback parameter.
            max_trees_per_client (Optional[int]): Max trees per client limit.
            t_max (Optional[int]): Max trees total allowed.
            **kwargs (Any): Additional parameters.

        Returns:
            Tuple: A tuple containing:
                - List[Any]: The final list of aggregated trees.
                - Dict[str, List[int]]: Mapping of client IDs to tree indices.
                - List[TreeEntry]: Interleaved candidates.
                - Optional[int]: The episode round where convergence occurred.
                - List[Dict[str, Any]]: Progressive execution logs.

        Raises:
            ValueError: If validation data is not provided.
        """
        if X_val is None or y_val is None:
            raise ValueError(
                f"CRITICAL: Server-side validation dataset "
                f"(X_val, y_val) must be provided "
                f"for ranking strategy {self.strategy_id}."
            )

        client_ids = list(client_trees.keys())
        if not client_ids:
            return [], {}, [], None, []

        if max_trees_per_client is None and max_trees is not None:
            max_trees_per_client = max(1, max_trees // len(client_ids))

        # Step 1: Rank trees within each client independently
        client_ranked_entries: Dict[str, List[TreeEntry]] = {}
        criterion = self._get_ranking_criterion(**kwargs)
        diversity_svc = self.diversity_svc or kwargs.get("diversity_service")
        ranker = TreeRanker(
            criterion=criterion,
            f1_weight=kwargs.get("f1_weight", 0.5),
            pcd_weight=kwargs.get("pcd_weight", 0.5),
            diversity_service=diversity_svc,
        )

        for client_id, trees in client_trees.items():
            meta = client_metadata[client_id]
            entries = TreeRanker.build_entries(
                {client_id: trees},
                {client_id: meta},
                X_val=X_val,
                y_val=y_val,
                diversity_service=diversity_svc,
            )
            ranked = ranker.rank(entries)
            if max_trees_per_client is not None:
                ranked = ranked[:max_trees_per_client]
            client_ranked_entries[client_id] = ranked

        # Step 2: Interleave trees from all clients using round-robin
        round_robin_entries = self._interleave_round_robin(
            client_ranked_entries, client_ids
        )

        # Normalize y_val to numeric indices if class_names available
        class_names = kwargs.get("class_names", [])
        label_svc = SimpleLabelService(class_names) if class_names else None
        if label_svc:
            y_val_norm = label_svc.transform(y_val)
        else:
            y_val_norm = np.asarray(y_val)

        # Ensure y_val_norm is not object dtype
        if y_val_norm.dtype == object:
            try:
                y_val_norm = np.array(y_val_norm.tolist())
            except Exception:
                pass

        # Step 3: Apply Progressive Forest with early stopping
        selector = ProgressiveSelector(
            metrics_service=self.metrics_svc or kwargs.get("metrics_service"),
            diversity_service=diversity_svc,
        )
        trees_per_client = int(kwargs.get("trees_per_client_per_episode", 1))
        global_trees, selected_entries, conv_round, logs = selector.select(
            candidate_entries=round_robin_entries,
            X_val=X_val,
            y_val_norm=y_val_norm,
            episode_size=trees_per_client * len(client_ids),
            t_max=t_max if t_max is not None else self.T_MAX,
            convergence_threshold=kwargs.get(
                "global_convergence_threshold", self.CONVERGENCE
            ),
            min_episodes=kwargs.get("min_episodes", 4),
            label_service=label_svc,
            ranker=ranker,
        )

        selected_ids = {cid: [] for cid in client_trees.keys()}
        for entry in selected_entries:
            selected_ids[entry.client_id].append(entry.tree_local_id)

        return (
            global_trees,
            selected_ids,
            round_robin_entries,
            conv_round,
            logs,
        )

    def _get_ranking_criterion(self, **kwargs: Any) -> RankingCriterion:
        """Helper to get criterion."""
        return self.ranking_criterion

    def _interleave_round_robin(
        self,
        client_ranked_entries: Dict[str, List[TreeEntry]],
        client_ids: List[str],
    ) -> List[TreeEntry]:
        """Interleave trees across clients using round-robin scheduling."""
        round_robin_entries = []
        client_indices = {cid: 0 for cid in client_ids}
        if client_ranked_entries:
            max_trees_any_client = max(
                len(entries) for entries in client_ranked_entries.values()
            )
        else:
            max_trees_any_client = 0

        for _ in range(max_trees_any_client):
            for client_id in client_ids:
                entries = client_ranked_entries[client_id]
                idx = client_indices[client_id]
                if idx < len(entries):
                    round_robin_entries.append(entries[idx])
                    client_indices[client_id] += 1
        return round_robin_entries


class S5PerClientAccuracyStrategy(PerClientProgressiveStrategy):
    """Strategy S5: Per-client ranking based on local accuracy."""

    @property
    def ranking_criterion(self) -> RankingCriterion:
        return RankingCriterion.ACCURACY

    @property
    def strategy_id(self) -> str:
        return "S5"


class S6PerClientF1Strategy(PerClientProgressiveStrategy):
    """Strategy S6: Per-client ranking based on local Macro-F1 score."""

    @property
    def ranking_criterion(self) -> RankingCriterion:
        return RankingCriterion.MACRO_F1

    @property
    def strategy_id(self) -> str:
        return "S6"


class S7PerClientF1PCDStrategy(PerClientProgressiveStrategy):
    """Strategy S7: Per-client ranking combination of F1 and PCD."""

    @property
    def ranking_criterion(self) -> RankingCriterion:
        return RankingCriterion.F1_PCD

    @property
    def strategy_id(self) -> str:
        return "S7"
