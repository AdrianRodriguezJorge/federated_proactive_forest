"""Federated Learning results dataclass definition.

Stores global metrics, client reports, convergence logs, and hybrid predictions
data generated during FL orchestrations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from src.domain.metadata.client_metadata import ClientMetadata


def default_hybrid_weights() -> Dict[str, float]:
    """Helper to provide default hybrid prediction weights.

    Returns:
        Dict[str, float]: Standard local and global ensemble mixing weights.
    """
    return {"local_weight": 0.4, "global_weight": 0.6}


@dataclass
class FLResults:
    """Stores the aggregated results and logs of a federated training run."""

    strategy_id: str
    client_ids: List[str]
    global_accuracy: float = 0.0
    global_macro_f1: float = 0.0
    n_trees_global: int = 0
    client_accuracies: Dict[str, float] = field(default_factory=dict)
    client_f1_scores: Dict[str, float] = field(default_factory=dict)
    client_metadata: Dict[str, ClientMetadata] = field(default_factory=dict)
    client_reports: Dict[str, Any] = field(default_factory=dict)
    selected_ids: Dict[str, List[int]] = field(default_factory=dict)
    all_tree_entries: List[Any] = field(default_factory=list)
    global_report: Any = None
    global_predictions: Optional[np.ndarray] = None
    num_rounds: int = 1
    communication_cost: float = 0.0
    client_hybrid_predictions: Dict[str, np.ndarray] = field(
        default_factory=dict
    )
    y_test: Optional[np.ndarray] = None
    class_names: List[str] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)
    client_hybrid_forest_sizes: Dict[str, int] = field(default_factory=dict)
    convergence_round: Optional[int] = None
    round_logs: List[Dict[str, Any]] = field(default_factory=list)
    hybrid_weights: Dict[str, float] = field(
        default_factory=default_hybrid_weights
    )
