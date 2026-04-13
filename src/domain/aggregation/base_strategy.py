from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from src.domain.metrics.metrics_service import IMetricsService, IDiversityService


class IAggregationStrategy(ABC):
    def __init__(self, 
                 metrics_service: Optional[IMetricsService] = None,
                 diversity_service: Optional[IDiversityService] = None):
        self.metrics_svc = metrics_service
        self.diversity_svc = diversity_service

    @abstractmethod
    def aggregate(
        self,
        client_trees: Dict[str, List[Any]],
        client_metadata: Dict,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        max_trees: Optional[int] = None,
        max_trees_per_client: Optional[int] = None,
        t_max: Optional[int] = None,
        **kwargs
    ) -> Tuple[List[Any], Dict[str, List[int]], List[Any]]:
        """Aggregate trees for the global model.

        Args:
            client_trees: Dict mapping client_id to list of trees
            client_metadata: Dict mapping client_id to metadata
            X_val: Validation features for Progressive Forest convergence (S2-S7)
            y_val: Validation labels for Progressive Forest convergence (S2-S7)
            max_trees: Maximum number of trees for global strategies (S2-S4)
            max_trees_per_client: Maximum trees per client for per-client strategies (S5-S7)
            t_max: Maximum number of trees in global model (T_MAX en tesis, default: 100)
            **kwargs: Strategy-specific parameters (e.g., f1_weight, pcd_weight)

        Returns:
          - List[DecisionTree]: árboles del bosque global
          - Dict[str, List[int]]: {client_id: [global_indices_of_selected_trees]}
          - List[TreeEntry]: todas las entradas (trees + metadata) en el orden usado para el ranking
        """
        ...

    @property
    @abstractmethod
    def strategy_id(self) -> str: ...
