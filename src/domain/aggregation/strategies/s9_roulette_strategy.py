"""
S9 Roulette Aggregation Strategy — Global Attribute Roulette.

Aggregates *feature-probability vectors* (roulettes) from all clients
into a single global roulette.  Four methodological variants are
provided, selectable via the ``method`` parameter:

  - ``weighted_average``: Weighted by each client's dataset size.
  - ``simple_mean``:      Unweighted arithmetic mean (democratic).
  - ``median``:           Coordinate-wise median (robust to outliers).
  - ``consensus``:        Softmax-weighted by client local performance (F1).
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
    ) -> np.ndarray:
        """Aggregate client roulette vectors into a global roulette.

        Args:
            client_vectors: ``{client_id: np.ndarray(n_features,)}``.
            client_dataset_sizes: ``{client_id: n_samples}`` — required
                for ``weighted_average``.
            client_f1_scores: ``{client_id: macro_f1}`` — required for
                ``consensus``.

        Returns:
            np.ndarray of shape ``(n_features,)`` summing to 1.0.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------

class S9WeightedAverageStrategy(IRouletteAggregationStrategy):
    """Weighted average by dataset size."""

    strategy_id = "S9_WEIGHTED"

    def aggregate_vectors(self, client_vectors, client_dataset_sizes=None,
                          client_f1_scores=None) -> np.ndarray:
        ids = list(client_vectors.keys())
        vectors = np.array([client_vectors[cid] for cid in ids])

        if client_dataset_sizes is None:
            raise ValueError("S9_WEIGHTED requires client_dataset_sizes.")

        weights = np.array([client_dataset_sizes[cid] for cid in ids], dtype=np.float64)
        weights /= weights.sum()

        result = np.average(vectors, axis=0, weights=weights)
        result /= result.sum()  # safety re-normalisation
        return result


class S9SimpleMeanStrategy(IRouletteAggregationStrategy):
    """Simple arithmetic mean (democratic)."""

    strategy_id = "S9_MEAN"

    def aggregate_vectors(self, client_vectors, client_dataset_sizes=None,
                          client_f1_scores=None) -> np.ndarray:
        vectors = np.array(list(client_vectors.values()))
        result = np.mean(vectors, axis=0)
        result /= result.sum()
        return result


class S9MedianStrategy(IRouletteAggregationStrategy):
    """Coordinate-wise median — robust against outlier clients."""

    strategy_id = "S9_MEDIAN"

    def aggregate_vectors(self, client_vectors, client_dataset_sizes=None,
                          client_f1_scores=None) -> np.ndarray:
        vectors = np.array(list(client_vectors.values()))
        result = np.median(vectors, axis=0)
        # Median of probability vectors does NOT sum to 1.0 → re-normalise
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
                          client_f1_scores=None) -> np.ndarray:
        if client_f1_scores is None:
            raise ValueError("S9_CONSENSUS requires client_f1_scores.")

        ids = list(client_vectors.keys())
        vectors = np.array([client_vectors[cid] for cid in ids])

        metrics = np.array([client_f1_scores[cid] for cid in ids], dtype=np.float64)
        # Softmax to accentuate performance differences
        exp_metrics = np.exp(metrics - np.max(metrics))  # shift for numerical stability
        consensus_weights = exp_metrics / exp_metrics.sum()

        result = np.average(vectors, axis=0, weights=consensus_weights)
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
}


def create_roulette_strategy(variant: str) -> IRouletteAggregationStrategy:
    """Instantiate a roulette aggregation strategy by variant name.

    Args:
        variant: One of ``'S9_WEIGHTED'``, ``'S9_MEAN'``,
                 ``'S9_MEDIAN'``, ``'S9_CONSENSUS'`` (case-insensitive).

    Returns:
        An ``IRouletteAggregationStrategy`` instance.
    """
    key = variant.upper().replace("-", "_")
    # Allow shorthand like "weighted" → "S9_WEIGHTED"
    if not key.startswith("S9_"):
        key = f"S9_{key}"
    if key not in _VARIANT_MAP:
        available = sorted(_VARIANT_MAP.keys())
        raise ValueError(f"Unknown S9 variant '{variant}'. Available: {available}")
    return _VARIANT_MAP[key]()


__all__ = [
    "IRouletteAggregationStrategy",
    "S9WeightedAverageStrategy",
    "S9SimpleMeanStrategy",
    "S9MedianStrategy",
    "S9ConsensusStrategy",
    "create_roulette_strategy",
]
