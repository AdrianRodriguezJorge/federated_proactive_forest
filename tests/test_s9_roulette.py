import pytest
import numpy as np
from src.domain.aggregation.strategies.s9_roulette_strategy import (
    create_roulette_strategy,
    S9WeightedAverageStrategy,
    S9SimpleMeanStrategy,
    S9MedianStrategy,
    S9ConsensusStrategy,
    S9ProactivePCDStrategy
)

def test_s9_weighted_average():
    """Verify weighted average strategy weights vectors based on dataset sizes."""
    strategy = S9WeightedAverageStrategy()
    client_vectors = {
        "client_0": np.array([0.1, 0.9]),
        "client_1": np.array([0.5, 0.5])
    }
    client_dataset_sizes = {
        "client_0": 100,
        "client_1": 300
    }
    
    # Weights: client_0 = 0.25, client_1 = 0.75
    # pos_0: 0.1 * 0.25 + 0.5 * 0.75 = 0.4
    # pos_1: 0.9 * 0.25 + 0.5 * 0.75 = 0.6
    res = strategy.aggregate_vectors(client_vectors, client_dataset_sizes=client_dataset_sizes)
    assert np.allclose(res, [0.4, 0.6])
    
    # Missing dataset sizes raises ValueError
    with pytest.raises(ValueError):
        strategy.aggregate_vectors(client_vectors, client_dataset_sizes=None)


def test_s9_simple_mean():
    """Verify simple mean aggregates democratic unweighted average."""
    strategy = S9SimpleMeanStrategy()
    client_vectors = {
        "client_0": np.array([0.1, 0.9]),
        "client_1": np.array([0.5, 0.5])
    }
    
    # Democratic: [0.3, 0.7]
    res = strategy.aggregate_vectors(client_vectors)
    assert np.allclose(res, [0.3, 0.7])


def test_s9_median():
    """Verify median aggregation takes coordinate-wise median."""
    strategy = S9MedianStrategy()
    client_vectors = {
        "client_0": np.array([0.1, 0.9]),
        "client_1": np.array([0.3, 0.7]),
        "client_2": np.array([0.5, 0.5])
    }
    
    # Coordinate-wise median: [0.3, 0.7]
    res = strategy.aggregate_vectors(client_vectors)
    assert np.allclose(res, [0.3, 0.7])
    
    # Zero sum case:
    client_vectors_zero = {
        "client_0": np.array([0.0, 0.0]),
        "client_1": np.array([0.0, 0.0])
    }
    res_zero = strategy.aggregate_vectors(client_vectors_zero)
    assert np.allclose(res_zero, [0.5, 0.5])


def test_s9_consensus():
    """Verify consensus uses local F1 scores for weighting."""
    strategy = S9ConsensusStrategy()
    client_vectors = {
        "client_0": np.array([0.2, 0.8]),
        "client_1": np.array([0.6, 0.4])
    }
    client_f1 = {
        "client_0": 0.8,
        "client_1": 0.2
    }
    
    # Weights: client_0 = 0.8, client_1 = 0.2
    # pos_0: 0.2 * 0.8 + 0.6 * 0.2 = 0.28
    # pos_1: 0.8 * 0.8 + 0.4 * 0.2 = 0.72
    res = strategy.aggregate_vectors(client_vectors, client_f1_scores=client_f1)
    assert np.allclose(res, [0.28, 0.72])
    
    # Missing F1 scores raises ValueError
    with pytest.raises(ValueError):
        strategy.aggregate_vectors(client_vectors, client_f1_scores=None)
        
    # Zero sum F1 falls back to democratic unweighted mean
    client_f1_zero = {
        "client_0": 0.0,
        "client_1": 0.0
    }
    res_zero = strategy.aggregate_vectors(client_vectors, client_f1_scores=client_f1_zero)
    assert np.allclose(res_zero, [0.4, 0.6])


def test_s9_proactive_pcd():
    """Verify proactive PCD strategy weights client roulettes proportional to PCD."""
    strategy = S9ProactivePCDStrategy()
    client_vectors = {
        "client_0": np.array([0.2, 0.8]),
        "client_1": np.array([0.6, 0.4])
    }
    client_pcd = {
        "client_0": 0.3,
        "client_1": 0.1
    }
    
    # Weights: client_0 = 0.75, client_1 = 0.25
    # pos_0: 0.2 * 0.75 + 0.6 * 0.25 = 0.30
    # pos_1: 0.8 * 0.75 + 0.4 * 0.25 = 0.70
    res = strategy.aggregate_vectors(client_vectors, client_pcd_scores=client_pcd)
    assert np.allclose(res, [0.3, 0.7])
    
    # Missing PCD scores falls back to Simple Mean
    res_fallback = strategy.aggregate_vectors(client_vectors, client_pcd_scores=None)
    assert np.allclose(res_fallback, [0.4, 0.6])
    
    # Zero sum PCD falls back to democratic unweighted mean
    client_pcd_zero = {
        "client_0": 0.0,
        "client_1": 0.0
    }
    res_zero = strategy.aggregate_vectors(client_vectors, client_pcd_scores=client_pcd_zero)
    assert np.allclose(res_zero, [0.4, 0.6])


def test_factory_helper():
    """Verify that factory resolves strings to correct roulette strategies."""
    assert isinstance(create_roulette_strategy("S9_WEIGHTED"), S9WeightedAverageStrategy)
    assert isinstance(create_roulette_strategy("weighted"), S9WeightedAverageStrategy)
    assert isinstance(create_roulette_strategy("S9_MEAN"), S9SimpleMeanStrategy)
    assert isinstance(create_roulette_strategy("mean"), S9SimpleMeanStrategy)
    assert isinstance(create_roulette_strategy("S9_MEDIAN"), S9MedianStrategy)
    assert isinstance(create_roulette_strategy("median"), S9MedianStrategy)
    assert isinstance(create_roulette_strategy("S9_CONSENSUS"), S9ConsensusStrategy)
    assert isinstance(create_roulette_strategy("consensus"), S9ConsensusStrategy)
    assert isinstance(create_roulette_strategy("S9_PROACTIVE_PCD"), S9ProactivePCDStrategy)
    assert isinstance(create_roulette_strategy("proactive_pcd"), S9ProactivePCDStrategy)
    
    # Fallback to Simple Mean for unrecognized inputs
    assert isinstance(create_roulette_strategy("unrecognized"), S9SimpleMeanStrategy)


def test_aggregate_roulettes_unbiased_and_fail_fast():
    """Verify that aggregate_roulettes prioritizes server_eval_f1 and raises error when missing."""
    from src.infrastructure.flex.flex_roulette_pf import aggregate_roulettes
    
    weights = [
        {"client_id": "client_0", "roulette": [0.2, 0.8], "n_samples": 100, "macro_f1": 0.9, "pcd": 0.8},
        {"client_id": "client_1", "roulette": [0.6, 0.4], "n_samples": 100, "macro_f1": 0.1, "pcd": 0.1}
    ]
    
    # 1. S9_CONSENSUS with server evaluations provided
    agg_model = {"weights": weights}
    aggregate_roulettes(
        agg_model,
        None, 
        variant="S9_CONSENSUS", 
        server_eval_f1={"client_0": 0.8, "client_1": 0.2}
    )
    res = agg_model["aggregated_weights"]
    # Weights based on server_eval_f1: 0.8 and 0.2
    # pos_0: 0.2 * 0.8 + 0.6 * 0.2 = 0.28
    # pos_1: 0.8 * 0.8 + 0.4 * 0.2 = 0.72
    assert np.allclose(res["global_roulette"], [0.28, 0.72])
    
    # 2. S9_CONSENSUS raises ValueError when server evaluations are missing (strict fail-fast)
    print("\n[DEBUG TEST] Calling second time (without server_eval_f1)...")
    agg_model_fail = {"weights": weights}
    with pytest.raises(ValueError, match="CRITICAL: Unbiased server-side F1-score evaluation is missing"):
        aggregate_roulettes(agg_model_fail, None, variant="S9_CONSENSUS")
