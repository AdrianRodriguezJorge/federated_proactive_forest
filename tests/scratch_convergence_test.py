import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.datasets import load_iris
from src.domain.dataset.base_adapter import DatasetSplit
from src.application.orchestrators.fl_orchestrator import FLEXOrchestrator

def run():
    print("Iniciando prueba de convergencia y limites (t_max=100)...")
    iris = load_iris()
    X = iris.data
    y = iris.target.astype(str)
    
    np.random.seed(42)
    shuffled_idx = np.random.permutation(len(X))
    X, y = X[shuffled_idx], y[shuffled_idx]
    
    X_train, y_train = X[:100], y[:100]
    X_val, y_val = X[100:125], y[100:125]
    X_test, y_test = X[125:], y[125:]
    
    split = DatasetSplit(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        feature_names=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        class_names=["0", "1", "2"],
        dataset_name="Iris"
    )
    
    config = {
        "federation": {"n_clients": 3, "distribution": "iid"},
        "model": {
            "n_estimators": 100, 
            "alpha": 0.1, 
            "voting": "soft", 
            "class_names": ["0", "1", "2"],
            "local_convergence_threshold": 0.005
        },
        "aggregation": {
            "strategy": "S2",
            "variant": "S2",
            "window_size": 2,
            "max_rounds": 1,
            "convergence_threshold": 0.002,
            "t_max": 100,
            "global_episode_size": 5
        }
    }
    
    orch = FLEXOrchestrator(config)
    orch.setup_federation(split)
    res = orch.run_federated_round(n_bootstrap=0)
    
    print("\n[RESULTADOS S2]")
    print(f"Global Trees: {res.n_trees_global}")
    print(f"Convergence Round (Global): {res.convergence_round}")
    print(f"Accuracy: {res.global_accuracy}")
    if hasattr(orch, 'cleanup'): orch.cleanup()

    print("\nIniciando prueba S9 en entorno Non-IID (simulando clases faltantes)...")
    from src.application.orchestrators.roulette_orchestrator import RouletteOrchestrator
    config_s9 = {
        "federation": {"n_clients": 3, "distribution": "dirichlet", "alpha": 0.1},
        "model": {
            "n_estimators": 10, 
            "alpha": 0.1, 
            "voting": "soft", 
            "class_names": ["0", "1", "2"]
        },
        "aggregation": {
            "strategy": "S9",
            "variant": "S9_WEIGHTED",
            "max_rounds": 1
        }
    }
    try:
        orch_s9 = RouletteOrchestrator(config_s9)
        orch_s9.setup_federation(split)
        res_s9 = orch_s9.run_federated_round(n_bootstrap=0)
        print("[S9] Ejecución exitosa.")
        print(f"S9 Global Trees: {res_s9.n_trees_global}")
    except Exception as e:
        print(f"[S9 ERROR]: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if hasattr(orch_s9, 'cleanup'): orch_s9.cleanup()

if __name__ == "__main__":
    run()
