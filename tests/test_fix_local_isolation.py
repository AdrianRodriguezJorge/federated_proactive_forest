import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Añadir src al path
sys.path.append(os.getcwd())

from src.domain.dataset.base_adapter import DatasetSplit
from src.application.orchestrators.fed_data_distributor import FedDataDistributor
from src.domain.model.proactive_forest import ProactiveForest
from src.domain.metrics.forest_evaluator import ForestEvaluator

def test_local_isolation_flow():
    print("Iniciando test rápido de local_isolation...")
    
    # 1. Crear datos sintéticos mínimos
    X = np.random.rand(50, 4)
    y = np.random.choice(["0", "1"], size=50)
    
    split = DatasetSplit(
        X_train=X[:40], X_val=X[40:45], X_test=X[45:],
        y_train=y[:40], y_val=y[40:45], y_test=y[45:],
        feature_names=["f1", "f2", "f3", "f4"],
        class_names=["0", "1"],
        dataset_name="TestDS"
    )
    
    config = {
        "federation": {"n_clients": 2, "distribution": "iid"},
        "model": {"n_estimators": 5, "alpha": 0.1},
        "aggregation": {"strategy": "local_isolation"}
    }
    
    # 2. Ejecutar lógica de distribución (lo que fallaba)
    print("Distribuyendo datos...")
    distributor = FedDataDistributor(config, use_flex_pool=False)
    _, fed_data = distributor.distribute(split)
    
    print(f"Tipo de fed_data: {type(fed_data)}")
    
    # 3. Probar iteración (el punto del error)
    client_reports = []
    try:
        print("Iterando sobre fed_data.items()...")
        for cid, client_dataset in fed_data.items():
            print(f"  Procesando cliente {cid}...")
            X_c, y_c = client_dataset.to_numpy()
            
            model = ProactiveForest(
                n_estimators=5, alpha=0.1, 
                class_names=split.class_names,
                convergence_threshold=0.002
            )
            model.fit(X_c, y_c)
            preds = model.predict(split.X_test)
            report = ForestEvaluator.evaluate_from_predictions(
                preds, split.y_test, split.class_names, len(model.get_trees())
            )
            client_reports.append(report)
        
        print(f"SUCCESS: Se procesaron {len(client_reports)} clientes correctamente.")
        
    except AttributeError as e:
        print(f"FAILURE: AttributeError detectado: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"FAILURE: Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_local_isolation_flow()
