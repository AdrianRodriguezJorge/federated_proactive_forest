"""
S9 Roulette Aggregation Strategy — Global Attribute Roulette.

Aggregates *feature-probability vectors* (roulettes) from all clients
into a single global roulette.  Five methodological variants are
provided, selectable via the ``method`` parameter:

  - ``weighted_average``: Weighted by each client's dataset size.
  - ``simple_mean``:      Unweighted arithmetic mean (democratic).
  - ``median``:           Coordinate-wise median (robust to outliers).
  - ``consensus``:        Softmax-weighted by client local performance (F1).
  - ``proactive_pcd``:    Weighted by client's local Percentage Correct Diversity (PCD).
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.domain.metrics.metrics_service import IMetricsService, IDiversityService


# ---------------------------------------------------------------------------
# Base interface for roulette aggregation (vector-based, NOT tree-based)
# ---------------------------------------------------------------------------

class IRouletteAggregationStrategy:
    """Interface for roulette aggregation strategies.

    Unlike ``IAggregationStrategy`` (which operates on trees), this
    interface operates on numeric probability vectors.
    """

    def aggregate_vectors(
        self,
        client_vectors: Dict[str, np.ndarray],
        client_dataset_sizes: Optional[Dict[str, int]] = None,
        client_f1_scores: Optional[Dict[str, float]] = None,
        client_pcd_scores: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Aggregate client roulette vectors into a global roulette."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------

class S9WeightedAverageStrategy(IRouletteAggregationStrategy):
    """Weighted average by dataset size."""
    strategy_id = "S9_WEIGHTED"

    def aggregate_vectors(self, client_vectors, client_dataset_sizes=None,
                          client_f1_scores=None, client_pcd_scores=None) -> np.ndarray:
        ids = list(client_vectors.keys())
        vectors = np.array([client_vectors[cid] for cid in ids])
        if client_dataset_sizes is None:
            raise ValueError("S9_WEIGHTED requires client_dataset_sizes.")
        weights = np.array([client_dataset_sizes[cid] for cid in ids], dtype=np.float64)
        weights /= weights.sum()
        result = np.average(vectors, axis=0, weights=weights)
        result /= result.sum()
        return result

class S9SimpleMeanStrategy(IRouletteAggregationStrategy):
    """Simple arithmetic mean (democratic)."""
    strategy_id = "S9_MEAN"

    def aggregate_vectors(self, client_vectors, client_dataset_sizes=None,
                          client_f1_scores=None, client_pcd_scores=None) -> np.ndarray:
        vectors = np.array(list(client_vectors.values()))
        result = np.mean(vectors, axis=0)
        result /= result.sum()
        return result

class S9MedianStrategy(IRouletteAggregationStrategy):
    """Coordinate-wise median."""
    strategy_id = "S9_MEDIAN"

    def aggregate_vectors(self, client_vectors, client_dataset_sizes=None,
                          client_f1_scores=None, client_pcd_scores=None) -> np.ndarray:
        vectors = np.array(list(client_vectors.values()))
        result = np.median(vectors, axis=0)
        total = result.sum()
        if total > 0:
            result /= total
        else:
            result = np.ones(result.shape) / result.shape[0]
        return result

class S9ConsensusStrategy(IRouletteAggregationStrategy):
    """Performance-based consensus using Softmax over local Macro-F1."""
    strategy_id = "S9_CONSENSUS"

    def aggregate_vectors(self, client_vectors, client_dataset_sizes=None,
                          client_f1_scores=None, client_pcd_scores=None) -> np.ndarray:
        if client_f1_scores is None:
            raise ValueError("S9_CONSENSUS requires client_f1_scores.")
        ids = list(client_vectors.keys())
        vectors = np.array([client_vectors[cid] for cid in ids])
        metrics = np.array([client_f1_scores[cid] for cid in ids], dtype=np.float64)
        exp_metrics = np.exp(metrics - np.max(metrics))
        consensus_weights = exp_metrics / exp_metrics.sum()
        result = np.average(vectors, axis=0, weights=consensus_weights)
        result /= result.sum()
        return result

class S9ProactivePCDStrategy(IRouletteAggregationStrategy):
    """Proactive consensus using Softmax over local PCD (Cepero Diversity)."""
    strategy_id = "S9_PROACTIVE_PCD"

    def aggregate_vectors(self, client_vectors, client_dataset_sizes=None,
                          client_f1_scores=None, client_pcd_scores=None) -> np.ndarray:
        if client_pcd_scores is None:
            # Fallback to mean if PCD not available
            return S9SimpleMeanStrategy().aggregate_vectors(client_vectors)
        
        ids = list(client_vectors.keys())
        vectors = np.array([client_vectors[cid] for cid in ids])
        pcds = np.array([client_pcd_scores.get(cid, 0.0) for cid in ids], dtype=np.float64)
        
        # Softmax over PCD to give more weight to diverse clients
        exp_pcds = np.exp(pcds - np.max(pcds))
        weights = exp_pcds / exp_pcds.sum()
        
        result = np.average(vectors, axis=0, weights=weights)
        result /= result.sum()
        return result


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

_VARIANT_MAP: Dict[str, type] = {
    "S9_WEIGHTED":  S9WeightedAverageStrategy,
    "S9_MEAN":      S9SimpleMeanStrategy,
    "S9_MEDIAN":    S9MedianStrategy,
    "S9_CONSENSUS": S9ConsensusStrategy,
    "S9_PROACTIVE_PCD": S9ProactivePCDStrategy,
}

def create_roulette_strategy(variant: str) -> IRouletteAggregationStrategy:
    key = variant.upper().replace("-", "_")
    if not key.startswith("S9_"):
        key = f"S9_{key}"
    if key not in _VARIANT_MAP:
        return _VARIANT_MAP["S9_MEAN"]() # Fallback
    return _VARIANT_MAP[key]()

__all__ = [
    "IRouletteAggregationStrategy",
    "S9WeightedAverageStrategy",
    "S9SimpleMeanStrategy",
    "S9MedianStrategy",
    "S9ConsensusStrategy",
    "S9ProactivePCDStrategy",
    "create_roulette_strategy",
]
