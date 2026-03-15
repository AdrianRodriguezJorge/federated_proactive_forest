"""Strategy Pattern — 7 implementaciones concretas."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class IAggregationStrategy(ABC):
    @abstractmethod
    def aggregate(
        self,
        client_trees: Dict[str, List[Any]],
        client_metadata: Dict,
    ) -> Tuple[List[Any], Dict[str, List[int]], List[Any]]:
        """Aggregate trees for the global model.

        Returns:
          - List[DecisionTree]: árboles del bosque global
          - Dict[str, List[int]]: {client_id: [global_indices_of_selected_trees]}
          - List[TreeEntry]: todas las entradas (trees + metadata) en el orden usado para el ranking
        """
        ...

    @property
    @abstractmethod
    def strategy_id(self) -> str: ...
