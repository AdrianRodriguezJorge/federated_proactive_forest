import os
import sys
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.getcwd())

from src.application.orchestrators.fl_orchestrator import FLEXOrchestrator
from src.domain.dataset.base_adapter import DatasetSplit

def test_run():
    print("Iniciando prueba de robustez de tipos (Mix of label types)...")
    
    config = {
        "dataset": {"name": "Iris", "split_config": {"n_clients": 3, "distribution": "iid"}},
        "model": {
            "n_estimators": 10,
            "alpha": 0.1,
            "use_progressive_stopping": True,
            "convergence": 0.002,
            "episode_size": 5
        },
        "federation": {"rounds": 1, "clients_per_round": 3},
        "aggregation": {
            "strategy": "s4_global_f1_pcd", # Test strategy normalization
            "convergence": 0.5,             # High threshold for early stopping test
            "f1_weight": 0.5
        }
    }
    
    # Load data with string labels to force "Mix of labels" scenario if not handled
    iris = load_iris()
    X_train, X_test, y_train_int, y_test_int = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    
    # Convert to strings
    class_names = iris.target_names.tolist()
    y_train = np.array([class_names[i] for i in y_train_int])
    y_test = np.array([class_names[i] for i in y_test_int])
    
    ds = DatasetSplit(
        X_train=X_train, X_test=X_test, 
        y_train=y_train, y_test=y_test,
        X_val=X_test, y_val=y_test, # Use test as val for simplicity
        feature_names=iris.feature_names,
        class_names=class_names,
        dataset_name="IrisTest"
    )
    
    orch = FLEXOrchestrator(config)
    print("Configurando federación...")
    orch.setup_federation(ds)
    
    print("Ejecutando ronda federada...")
    results = orch.run_federated_round()
    
    print("\n[OK] Ronda completada sin errores de tipos.")
    print(f"Rondas de convergencia: {results.round_logs[-1]['episode']}")
    print(f"Árboles finales: {len(results.global_trees)}")
    
    if len(results.round_logs) > 0:
        print(f"Mejora última ronda: {results.round_logs[-1]['accuracy']:.4f}")

if __name__ == "__main__":
    try:
        test_run()
        print("\nPrueba de CLI completada con éxito.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
