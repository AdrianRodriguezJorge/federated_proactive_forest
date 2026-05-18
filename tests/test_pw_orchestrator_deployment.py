import pytest
import numpy as np
from src.domain.dataset.base_adapter import DatasetSplit
from src.application.orchestrators.progressive_windows_orchestrator import ProgressiveWindowsOrchestrator

def test_pw_orchestrator_deployment():
    # 1. Minimum synthetic dataset
    np.random.seed(42)
    X = np.random.rand(40, 2)
    y = np.random.choice(["0", "1"], size=40)
    
    split = DatasetSplit(
        X_train=X[:30], X_val=X[30:35], X_test=X[35:],
        y_train=y[:30], y_val=y[30:35], y_test=y[35:],
        feature_names=["f1", "f2"],
        class_names=["0", "1"],
        dataset_name="TestDS"
    )
    
    config = {
        "federation": {"n_clients": 2, "distribution": "iid"},
        "model": {"n_estimators": 5, "alpha": 0.1, "class_names": ["0", "1"]},
        "aggregation": {
            "strategy": "PW",
            "window_size": 2,
            "max_rounds": 2,
            "convergence_threshold": 0.0001
        }
    }
    
    # Initialize orchestrator
    orch = ProgressiveWindowsOrchestrator(config)
    orch.setup_federation(split)
    
    try:
        results = orch.run_federated_round(n_bootstrap=0)
        
        # Verify server state
        server_model = orch.flex_pool._models["server"].get("model")
        server_trees = orch.flex_pool._models["server"].get("trees", [])
        
        print(f"\n[TEST PW] Server model object: {server_model}")
        print(f"[TEST PW] Number of trees in server model: {len(server_trees)}")
        
        # Assertions
        assert server_model is not None, "Server model should be initialized"
        assert len(server_trees) > 0, f"Server model should have aggregated trees, but found {len(server_trees)}"
        
    finally:
        if hasattr(orch, 'cleanup'):
            orch.cleanup()
