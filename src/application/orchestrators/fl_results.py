from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
from src.domain.metadata.client_metadata import ClientMetadata

@dataclass
class FLResults:
    strategy_id: str
    global_accuracy: float
    global_macro_f1: float
    n_trees_global: int
    client_ids: List[str]
    client_accuracies: Dict[str, float]
    client_f1_scores: Dict[str, float]
    client_metadata: Dict[str, ClientMetadata] = field(default_factory=dict)
    client_reports: Dict[str, Any] = field(default_factory=dict)
    selected_ids: Dict[str, List[int]] = field(default_factory=dict)
    all_tree_entries: List[Any] = field(default_factory=list)
    global_report: Any = None
    global_predictions: np.ndarray = None
    num_rounds: int = 1
    communication_cost: float = 0.0
    client_hybrid_predictions: Dict[str, np.ndarray] = field(default_factory=dict)
    y_test: np.ndarray = None
    class_names: List[str] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)
    client_hybrid_forest_sizes: Dict[str, int] = field(default_factory=dict)
    convergence_round: Optional[int] = None
    round_logs: List[Dict[str, Any]] = field(default_factory=list)
    hybrid_weights: Dict[str, float] = field(default_factory=lambda: {'local_weight': 0.4, 'global_weight': 0.6})
