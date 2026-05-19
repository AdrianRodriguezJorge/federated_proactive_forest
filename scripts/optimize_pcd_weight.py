import sys
import os
import json
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.application.orchestrators.fl_orchestrator import FLEXOrchestrator
from src.infrastructure.datasets.dataset_factory import DatasetFactory

def run_optimization():
    print("Iniciando Optimización Exhaustiva del Peso PCD (Maximizar Accuracy)")
    
    datasets = ["Iris", "Wine", "Breast", "Glass", "Vehicle", "Segment", "Optdigits", "Letter"]
    strategies = ["S4", "S7"]
    pcd_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    results = {}
    
    start_time = time.time()
    
    for ds_name in datasets:
        print(f"\n======================================")
        print(f" DATASET: {ds_name}")
        print(f"======================================")
        results[ds_name] = {}
        
        # Load dataset
        try:
            dataset_split = DatasetFactory.get_dataset(ds_name)
        except Exception as e:
            print(f"Error cargando {ds_name}: {e}")
            continue
            
        for strategy in strategies:
            print(f"\n  --- Estrategia: {strategy} ---")
            results[ds_name][strategy] = []
            
            for pcd in pcd_weights:
                f1_w = round(1.0 - pcd, 2)
                pcd_w = round(pcd, 2)
                
                config = {
                    "federation": {"n_clients": 3, "distribution": "dirichlet", "alpha": 0.5},
                    "model": {
                        "n_estimators": 50, 
                        "alpha": 0.1, 
                        "voting": "soft",
                        "local_convergence_threshold": 0.005
                    },
                    "aggregation": {
                        "strategy": strategy,
                        "variant": strategy,
                        "window_size": 2,
                        "max_rounds": 10,
                        "convergence_threshold": 0.002,
                        "t_max": 150,  # Suficientes árboles para permitir diversidad
                        "global_episode_size": 5,
                        "f1_weight": f1_w,
                        "pcd_weight": pcd_w
                    }
                }
                
                try:
                    orch = FLEXOrchestrator(config)
                    orch.setup_federation(dataset_split)
                    res = orch.run_federated_round(n_bootstrap=0)
                    
                    acc = res.global_accuracy
                    trees = res.n_trees_global
                    
                    results[ds_name][strategy].append({
                        "pcd_weight": pcd_w,
                        "accuracy": acc,
                        "n_trees": trees
                    })
                    
                    print(f"    PCD: {pcd_w:.1f} | F1_W: {f1_w:.1f} --> Acc: {acc:.4f} (Trees: {trees})")
                    
                    if hasattr(orch, 'cleanup'):
                        orch.cleanup()
                        
                except Exception as e:
                    print(f"    Error con PCD {pcd_w}: {e}")
                    
    end_time = time.time()
    print(f"\nOptimización finalizada en {end_time - start_time:.2f} segundos.")
    
    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "pcd_optimization_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Resultados guardados en {out_path}")

if __name__ == "__main__":
    run_optimization()
