"""Abstract base interfaces for Federated Forest aggregation strategies.

Specifies parameters and signatures required by all global federated trees
selection strategies (e.g. S1-S7, S9, Progressive Windows).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.domain.metrics.metrics_service import IDiversityService, IMetricsService


class IAggregationStrategy(ABC):
    """Abstract baseline class for all Federated Forest aggregation strategies."""

    def __init__(
        self,
        metrics_service: Optional[IMetricsService] = None,
        diversity_service: Optional[IDiversityService] = None,
    ):
        """Initializes strategy dependency state.

        Args:
            metrics_service (Optional[IMetricsService]): Performance metrics.
            diversity_service (Optional[IDiversityService]): Diversity metrics.
        """
        self.metrics_svc = metrics_service
        self.diversity_svc = diversity_service

    @abstractmethod
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
        List[Any],
        Optional[int],
        List[Dict[str, Any]],
    ]:
        """Aggregate trees for the global model.

        Args:
            client_trees (Dict[str, List[Any]]): Dict mapping client_id to list.
            client_metadata (Dict[str, Any]): Dict mapping client_id to meta.
            X_val (Optional[np.ndarray]): Validation features.
            y_val (Optional[np.ndarray]): Validation labels.
            max_trees (Optional[int]): Maximum number of trees (S2-S4).
            max_trees_per_client (Optional[int]): Max trees per client (S5-S7).
            t_max (Optional[int]): Maximum trees in global model (T_MAX).
            **kwargs (Any): Strategy-specific parameters.

        Returns:
            Tuple: A tuple containing:
                - List[Any]: Trees in global forest.
                - Dict[str, List[int]]: Global indices of selected trees.
                - List[Any]: TreeEntry list in ranking order.
                - Optional[int]: Selected T_MAX tree count.
                - List[Dict[str, Any]]: Strategy evaluation log metrics.
        """
        pass

    @property
    @abstractmethod
    def strategy_id(self) -> str:
        """Get unique strategy identifier.

        Returns:
            str: Strategy code name.
        """
        pass
