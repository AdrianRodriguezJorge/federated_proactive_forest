"""Metadatos obligatorios que cada cliente envía al servidor junto con sus árboles."""
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ClientMetadata:
    client_id: str
    n_trees: int
    accuracy: float        # Accuracy del bosque local sobre el conjunto local val
    macro_f1: float        # Macro-F1 del bosque local sobre el conjunto local val
    pcd: float             # Pair Classifier Disagreement (diversidad) sobre el conjunto local val
    # Métricas individuales de cada árbol (Accuracy, F1)
    tree_metrics: List[Dict[str, float]] = field(default_factory=list)
    # IDs de árboles del bosque LOCAL seleccionados durante la agregación
    # (llenado por el servidor, devuelto al cliente para el No-Repeat Merge)
    selected_local_tree_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'client_id': self.client_id,
            'n_trees': self.n_trees,
            'accuracy': self.accuracy,
            'macro_f1': self.macro_f1,
            'pcd': self.pcd,
        }
