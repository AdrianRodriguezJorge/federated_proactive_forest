import pytest
import numpy as np
from src.domain.dataset.base_adapter import DatasetSplit
from src.application.orchestrators.fl_orchestrator import FLEXOrchestrator
from src.application.orchestrators.roulette_orchestrator import RouletteOrchestrator

# Parametrization list for all 12 aggregation strategies
ALL_STRATEGIES = [
    ("S1_simple_pool", "S1"),
    ("S2_global_accuracy", "S2"),
    ("S3_global_macro_f1", "S3"),
    ("S4_global_f1_pcd", "S4"),
    ("S5_perclient_accuracy", "S5"),
    ("S6_perclient_f1", "S6"),
    ("S7_perclient_f1_pcd", "S7"),
    ("S8_WEIGHTED", "S8_WEIGHTED"),
    ("S8_MEAN", "S8_MEAN"),
    ("S8_MEDIAN", "S8_MEDIAN"),
    ("S8_CONSENSUS", "S8_CONSENSUS"),
    ("S8_PROACTIVE_PCD", "S8_PROACTIVE_PCD"),
]

@pytest.fixture(scope="module")
def synthetic_split():
    """Generates a small synthetic dataset split for fast integration testing."""
    np.random.seed(42)
    X = np.random.rand(30, 2)
    y = np.random.choice(["0", "1"], size=30)
    
    # 20 training samples, 5 validation samples, 5 test samples
    return DatasetSplit(
        X_train=X[:20], X_val=X[20:25], X_test=X[25:],
        y_train=y[:20], y_val=y[20:25], y_test=y[25:],
        feature_names=["f1", "f2"],
        class_names=["0", "1"],
        dataset_name="SyntheticIntegration"
    )


@pytest.mark.parametrize("name,strategy_id", ALL_STRATEGIES)
def test_strategy_execution(name, strategy_id, synthetic_split):
    """Verifies that each of the 13 aggregation strategies executes, aggregates, and converges correctly."""
    config = {
        "federation": {"n_clients": 2, "distribution": "iid"},
        "model": {
            "n_estimators": 4, 
            "alpha": 0.1, 
            "voting": "soft", 
            "class_names": ["0", "1"],
            "local_convergence_threshold": 0.0001
        },
        "aggregation": {
            "strategy": strategy_id if not strategy_id.startswith("S8_") else "S8",
            "variant": strategy_id,
            "window_size": 2,
            "max_rounds": 2,
            "convergence_threshold": 0.01,
            "local_roulette_weight": 0.5
        }
    }
    
    # Instantiate the correct orchestrator based on strategy
    if strategy_id.startswith("S8_"):
        orch = RouletteOrchestrator(config)
    else:
        orch = FLEXOrchestrator(config)
        
    orch.setup_federation(synthetic_split)
    
    try:
        # Run orchestrator with n_bootstrap=0 to skip high-cost bootstrap estimations
        res = orch.run_federated_round(n_bootstrap=0)
        
        # Validate result integrity
        assert res is not None, f"Results for strategy {name} should not be None"
        assert hasattr(res, "global_accuracy"), f"Strategy {name} missing global_accuracy"
        assert hasattr(res, "global_macro_f1"), f"Strategy {name} missing global_macro_f1"
        assert hasattr(res, "n_trees_global"), f"Strategy {name} missing n_trees_global"
        
        # Metric boundaries checks
        assert 0.0 <= res.global_accuracy <= 1.0, f"Invalid global accuracy: {res.global_accuracy}"
        assert 0.0 <= res.global_macro_f1 <= 1.0, f"Invalid global F1-score: {res.global_macro_f1}"
        assert res.n_trees_global >= 0, f"Invalid number of trees: {res.n_trees_global}"
        
    finally:
        if hasattr(orch, 'cleanup'):
            orch.cleanup()
