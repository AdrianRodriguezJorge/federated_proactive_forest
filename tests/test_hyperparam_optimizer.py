import pytest
import numpy as np
import optuna
from src.domain.dataset.base_adapter import DatasetSplit
from src.application.hyperparam_optimizer import HyperparamOptimizer, OptimizationConfig

@pytest.fixture
def synthetic_split():
    """Generates a small synthetic dataset split for testing optimizer."""
    np.random.seed(42)
    X = np.random.rand(20, 2)
    y = np.random.choice(["0", "1"], size=20)
    
    return DatasetSplit(
        X_train=X[:12], X_val=X[12:16], X_test=X[16:],
        y_train=y[:12], y_val=y[12:16], y_test=y[16:],
        feature_names=["f1", "f2"],
        class_names=["0", "1"],
        dataset_name="SyntheticOptimizer"
    )

def test_hyperparam_optimizer_invalid_strategy(synthetic_split):
    """Verify invalid strategy throws a ValueError."""
    base_config = {"model": {}, "aggregation": {}}
    search_space = {}
    with pytest.raises(ValueError, match="Invalid strategy"):
        HyperparamOptimizer(synthetic_split, "INVALID_S9", base_config, search_space)

def test_hyperparam_optimizer_get_default_params():
    """Verify get_default_params extracts correct values and standard names."""
    base_config = {
        "model": {
            "n_estimators": 50,
            "alpha_pf": 0.15,
            "local_convergence_threshold": 0.005
        },
        "aggregation": {
            "max_trees": 120,
            "f1_weight": 0.6,
            "global_convergence_threshold": 0.001
        },
        "prediction": {
            "local_weight": 0.7,
            "use_weighted": False
        }
    }
    
    opt = HyperparamOptimizer(None, "S1", base_config, {}, verbose=False)
    defaults = opt.get_default_params()
    
    assert defaults["alpha_pf"] == 0.15
    assert defaults["local_convergence_threshold"] == 0.005
    assert defaults["max_trees"] == 120
    assert defaults["global_convergence_threshold"] == 0.001
    assert defaults["f1_weight"] == 0.6
    assert defaults["local_weight"] == 0.7
    assert defaults["use_weighted"] is False

def test_hyperparam_optimizer_build_config():
    """Verify _build_config transforms parameters correctly."""
    base_config = {
        "model": {"n_estimators": 10},
        "aggregation": {},
        "prediction": {}
    }
    search_space = {}
    opt = HyperparamOptimizer(None, "S1", base_config, search_space)
    
    trial_params = {
        "f1_weight": 0.4,
        "local_weight": 0.3,
        "use_weighted": True,
        "max_trees": 80,
        "alpha_pf": 0.05,
        "global_convergence_threshold": 0.005,
        "local_convergence_threshold": 0.002
    }
    
    config = opt._build_config(trial_params)
    
    # Check aggregation mappings
    assert config["aggregation"]["f1_weight"] == 0.4
    assert config["aggregation"]["pcd_weight"] == pytest.approx(0.6)
    assert config["aggregation"]["max_trees"] == 80
    assert config["aggregation"]["global_convergence_threshold"] == 0.005
    
    # Check model mappings
    assert config["model"]["alpha_pf"] == 0.05
    assert config["model"]["local_convergence_threshold"] == 0.002
    
    # Check prediction mappings
    assert config["prediction"]["local_weight"] == 0.3
    assert config["prediction"]["global_weight"] == pytest.approx(0.7)
    assert config["prediction"]["use_weighted"] is True

def test_hyperparam_optimizer_optimization_run(synthetic_split):
    """Verify standard optimization executes trials and logs results successfully."""
    base_config = {
        "federation": {"n_clients": 2, "distribution": "iid"},
        "model": {
            "n_estimators": 2,
            "alpha_pf": 0.1,
            "class_names": ["0", "1"],
            "local_convergence_threshold": 0.01
        },
        "aggregation": {
            "max_trees": 4,
            "window_size": 2,
            "max_rounds": 2,
            "global_convergence_threshold": 0.01,
            "local_roulette_weight": 0.5
        }
    }
    
    search_space = {
        "local_convergence_threshold": {"type": "categorical", "choices": [0.001, 0.002]},
        "max_trees": {"type": "int", "low": 2, "high": 4, "step": 2}
    }
    
    opt = HyperparamOptimizer(
        synthetic_split,
        strategy="S1",
        base_config=base_config,
        search_space=search_space,
        verbose=False
    )
    
    study = opt.optimize(
        n_trials=2,
        metric="accuracy",
        seed=42
    )
    
    assert isinstance(study, optuna.Study)
    assert len(study.trials) == 2
    assert "local_convergence_threshold" in study.best_params
    assert "max_trees" in study.best_params
