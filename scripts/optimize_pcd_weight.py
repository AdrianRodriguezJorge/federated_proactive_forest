import sys
import os
import time
import warnings

# Silenciar el RuntimeWarning molesto de FLEX sobre los arreglos Numpy vs Listas
warnings.filterwarnings("ignore", message="X_array or y_array are not a list nor a numpy array", category=RuntimeWarning)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.application.orchestrators.fl_orchestrator import FLEXOrchestrator
from src.application.orchestrators.progressive_tree_orchestrator import ProgressiveTreeOrchestrator
from src.domain.aggregation.aggregation_factory import AggregationFactory
from src.infrastructure.dataset.dataset_factory import DatasetFactory
from src.infrastructure.persistence.experiment_results import save_experiment_results_json

def run_optimization():
    print("Iniciando Optimización Exhaustiva del Peso PCD (Maximizar Accuracy Híbrida Media)")
    
    # Datasets comparables con hyperparameter_search.py
    datasets = ["Car", "Sonar", "Vowel", "Spambase"]
    strategies = ["S4", "S7", "pw"]
    pcd_weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    records = []
    
    start_time = time.time()
    
    for ds_name in datasets:
        print(f"\n======================================")
        print(f" DATASET: {ds_name}")
        print(f"======================================")
        
        # Load dataset
        try:
            ds_config = {
                "type": ds_name.lower(),
                "test_size": 0.2,
                "scale": True,
                "scaler_type": "standard",
                "seed": 42,
            }
            dataset_split = DatasetFactory.load_from_config(ds_config)
        except Exception as e:
            print(f"Error cargando {ds_name}: {e}")
            continue
            
        for strategy in strategies:
            print(f"\n  --- Estrategia: {strategy} ---")
            
            for pcd in pcd_weights:
                f1_w = round(1.0 - pcd, 2)
                pcd_w = round(pcd, 2)
                norm_strat = AggregationFactory.normalize_strategy_name(strategy)

                global_ep_size = 5
                trees_per_client_ep = 1
                trees_per_rnd_client = 1
                win_size = 5

                if norm_strat == "S4":
                    global_ep_size = 10
                elif norm_strat == "S7":
                    trees_per_client_ep = 3
                elif norm_strat == "PW":
                    win_size = 10
                    trees_per_rnd_client = 3

                config = {
                    "federation": {"n_clients": 3, "distribution": "iid"},
                    "model": {
                        "n_estimators": 50,
                        "alpha": 0.1,
                        "voting": "soft",
                        "local_convergence_threshold": 0.002,
                    },
                    "aggregation": {
                        "strategy": norm_strat,
                        "variant": norm_strat,
                        "window_size": win_size,
                        "max_rounds": 15,
                        "convergence_threshold": 0.002,
                        "t_max": 150,
                        "global_episode_size": global_ep_size,
                        "trees_per_client_per_episode": trees_per_client_ep,
                        "trees_per_round_per_client": trees_per_rnd_client,
                        "min_episodes": 3,
                        "min_rounds": 3,
                        "f1_weight": f1_w,
                        "pcd_weight": pcd_w,
                    },
                }
                
                try:
                    if norm_strat == "PW":
                        orch = ProgressiveTreeOrchestrator(config)
                    else:
                        orch = FLEXOrchestrator(config)
                        
                    orch.setup_federation(dataset_split)
                    res = orch.run_federated_round(n_bootstrap=0)
                    
                    acc = res.hybrid_accuracy_mean
                    trees = res.n_trees_global
                    
                    records.append({
                        "dataset": ds_name,
                        "strategy": strategy,
                        "strategy_normalized": norm_strat,
                        "pcd_weight": pcd_w,
                        "f1_weight": f1_w,
                        "accuracy": acc,
                        "n_trees": trees,
                    })
                    
                    print(f"    PCD: {pcd_w:.1f} | F1_W: {f1_w:.1f} --> Hybrid Acc Mean: {acc:.4f} (Trees: {trees})")
                    
                    if hasattr(orch, "cleanup"):
                        orch.cleanup()
                        
                except Exception as e:
                    print(f"    Error con PCD {pcd_w}: {e}")
                    
    end_time = time.time()
    print(f"\nOptimización finalizada en {end_time - start_time:.2f} segundos.")
    
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "pcd_optimization_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_experiment_results_json(
        out_path,
        {
            "script_name": "optimize_pcd_weight.py",
            "experiment_type": "pcd_optimization",
            "objective": "hybrid_accuracy",
            "dataset_config": {
                "test_size": 0.2,
                "scale": True,
                "scaler_type": "standard",
                "seed": 42,
            },
            "common_config": {
                "federation": {"n_clients": 3, "distribution": "iid"},
                "model": {"n_estimators": 50, "alpha": 0.1, "voting": "soft"},
            },
            "created_by": "optimize_pcd_weight.py",
        },
        records,
    )
    print(f"Resultados guardados en {out_path}")

if __name__ == "__main__":
    run_optimization()

