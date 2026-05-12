"""Metadatos obligatorios que cada cliente envía al servidor junto con sus árboles."""
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ClientMetadata:
    client_id: str = "unknown"
    n_trees: int = 0
    accuracy: float = 0.0
    macro_f1: float = 0.0
    pcd: float = 0.0
    # Métricas individuales de cada árbol (Accuracy, F1)
    tree_metrics: List[Dict[str, float]] = field(default_factory=list)
    # IDs de árboles del bosque LOCAL seleccionados durante la agregación
    # (llenado por el servidor, devuelto al cliente para el No-Repeat Merge)
    selected_local_tree_ids: List[int] = field(default_factory=list)
    has_converged: bool = False
    stop_counter: int = 0
    prev_episode_acc: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'client_id': self.client_id,
            'n_trees': self.n_trees,
            'accuracy': self.accuracy,
            'macro_f1': self.macro_f1,
            'pcd': self.pcd,
            'has_converged': self.has_converged,
            'stop_counter': self.stop_counter,
            'prev_episode_acc': self.prev_episode_acc
        }
