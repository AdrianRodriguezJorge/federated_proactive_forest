"""S9 Roulette Aggregation Strategy — Global Attribute Roulette.

Aggregates feature-probability vectors (roulettes) from all clients into a
single global roulette using various mathematical variants:
- weighted_average: Weighted by each client's dataset size.
- simple_mean: Unweighted arithmetic mean.
- median: Coordinate-wise median.
- consensus: Proportional to local F1-score performance.
- proactive_pcd: Proportional to local Percentage Correct Diversity.
"""

from typing import Any, Dict, List, Optional
import numpy as np


class IRouletteAggregationStrategy:
    """Interface for roulette aggregation strategies.

    Unlike tree aggregation, operates exclusively on probability vectors.
    """

    def aggregate_vectors(
        self,
        client_vectors: Dict[str, np.ndarray],
        client_dataset_sizes: Optional[Dict[str, int]] = None,
        client_f1_scores: Optional[Dict[str, float]] = None,
        client_pcd_scores: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Aggregate client roulette vectors into a global roulette.

        Args:
            client_vectors (Dict[str, np.ndarray]): Client roulette vectors.
            client_dataset_sizes (Optional[Dict[str, int]]): Client sample
                counts.
            client_f1_scores (Optional[Dict[str, float]]): Client local F1
                scores.
            client_pcd_scores (Optional[Dict[str, float]]): Client local PCD
                scores.

        Returns:
            np.ndarray: Unified global roulette probability vector.
        """
        raise NotImplementedError


class S9WeightedAverageStrategy(IRouletteAggregationStrategy):
    """Weighted average by dataset size."""

    strategy_id = "S9_WEIGHTED"

    def aggregate_vectors(
        self,
        client_vectors: Dict[str, np.ndarray],
        client_dataset_sizes: Optional[Dict[str, int]] = None,
        client_f1_scores: Optional[Dict[str, float]] = None,
        client_pcd_scores: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Aggregate vectors weighted by client dataset size."""
        ids = list(client_vectors.keys())
        vectors = np.array([client_vectors[cid] for cid in ids])
        if client_dataset_sizes is None:
            raise ValueError("S9_WEIGHTED requires client_dataset_sizes.")
        weights = np.array(
            [client_dataset_sizes[cid] for cid in ids], dtype=np.float64
        )
        weights /= weights.sum()
        result = np.average(vectors, axis=0, weights=weights)
        result /= result.sum()
        return result


class S9SimpleMeanStrategy(IRouletteAggregationStrategy):
    """Simple arithmetic mean (democratic)."""

    strategy_id = "S9_MEAN"

    def aggregate_vectors(
        self,
        client_vectors: Dict[str, np.ndarray],
        client_dataset_sizes: Optional[Dict[str, int]] = None,
        client_f1_scores: Optional[Dict[str, float]] = None,
        client_pcd_scores: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Aggregate vectors using arithmetic average."""
        vectors = np.array(list(client_vectors.values()))
        result = np.mean(vectors, axis=0)
        result /= result.sum()
        return result


class S9MedianStrategy(IRouletteAggregationStrategy):
    """Coordinate-wise median."""

    strategy_id = "S9_MEDIAN"

    def aggregate_vectors(
        self,
        client_vectors: Dict[str, np.ndarray],
        client_dataset_sizes: Optional[Dict[str, int]] = None,
        client_f1_scores: Optional[Dict[str, float]] = None,
        client_pcd_scores: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Aggregate vectors using median values."""
        vectors = np.array(list(client_vectors.values()))
        result = np.median(vectors, axis=0)
        total = result.sum()
        if total > 0:
            result /= total
        else:
            result = np.ones(result.shape) / result.shape[0]
        return result


class S9ConsensusStrategy(IRouletteAggregationStrategy):
    """Performance-based consensus using Macro-F1 scores."""

    strategy_id = "S9_CONSENSUS"

    def aggregate_vectors(
        self,
        client_vectors: Dict[str, np.ndarray],
        client_dataset_sizes: Optional[Dict[str, int]] = None,
        client_f1_scores: Optional[Dict[str, float]] = None,
        client_pcd_scores: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Aggregate vectors weighted by client F1 scores."""
        if client_f1_scores is None:
            raise ValueError("S9_CONSENSUS requires client_f1_scores.")
        ids = list(client_vectors.keys())
        vectors = np.array([client_vectors[cid] for cid in ids])
        metrics = np.array(
            [client_f1_scores[cid] for cid in ids], dtype=np.float64
        )
        if metrics.sum() > 0:
            consensus_weights = metrics / metrics.sum()
        else:
            consensus_weights = np.ones(len(metrics)) / len(metrics)
        result = np.average(vectors, axis=0, weights=consensus_weights)
        result /= result.sum()
        return result


class S9ProactivePCDStrategy(IRouletteAggregationStrategy):
    """Proactive consensus using local PCD (Cepero Diversity)."""

    strategy_id = "S9_PROACTIVE_PCD"

    def aggregate_vectors(
        self,
        client_vectors: Dict[str, np.ndarray],
        client_dataset_sizes: Optional[Dict[str, int]] = None,
        client_f1_scores: Optional[Dict[str, float]] = None,
        client_pcd_scores: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Aggregate vectors weighted by client PCD scores."""
        if client_pcd_scores is None:
            # Fallback to mean if PCD not available
            return S9SimpleMeanStrategy().aggregate_vectors(client_vectors)

        ids = list(client_vectors.keys())
        vectors = np.array([client_vectors[cid] for cid in ids])
        pcds = np.array(
            [client_pcd_scores.get(cid, 0.0) for cid in ids], dtype=np.float64
        )

        if pcds.sum() > 0:
            weights = pcds / pcds.sum()
        else:
            weights = np.ones(len(pcds)) / len(pcds)

        result = np.average(vectors, axis=0, weights=weights)
        result /= result.sum()
        return result


_VARIANT_MAP: Dict[str, type] = {
    "S9_WEIGHTED": S9WeightedAverageStrategy,
    "S9_WEIGHTED_AVERAGE": S9WeightedAverageStrategy,
    "S9_MEAN": S9SimpleMeanStrategy,
    "S9_SIMPLE_MEAN": S9SimpleMeanStrategy,
    "S9_MEDIAN": S9MedianStrategy,
    "S9_CONSENSUS": S9ConsensusStrategy,
    "S9_PROACTIVE_PCD": S9ProactivePCDStrategy,
}


def create_roulette_strategy(variant: str) -> IRouletteAggregationStrategy:
    """Factory helper to resolve roulette strategies.

    Args:
        variant (str): Variant code.

    Returns:
        IRouletteAggregationStrategy: Resolved strategy instance.
    """
    key = variant.upper().replace("-", "_")
    if not key.startswith("S9_"):
        key = f"S9_{key}"
    if key not in _VARIANT_MAP:
        return _VARIANT_MAP["S9_MEAN"]()
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
