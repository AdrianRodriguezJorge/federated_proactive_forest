"""Strategy Pattern — 7 implementaciones concretas."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class IAggregationStrategy(ABC):
    @abstractmethod
    def aggregate(
        self,
        client_trees: Dict[str, List[Any]],
        client_metadata: Dict,
    ) -> Tuple[List[Any], Dict[str, List[int]]]:
        """
        Retorna:
          - List[DecisionTree]: árboles del bosque global
          - Dict[str, List[int]]: {client_id: [local_tree_ids seleccionados]}
        """
        ...

    @property
    @abstractmethod
    def strategy_id(self) -> str: ...
