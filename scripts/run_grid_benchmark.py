import os
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

# Asegurar que el src/ pueda ser importado
sys.path.append(os.getcwd())

from src.application.orchestrators.fl_orchestrator import FLEXOrchestrator
from src.infrastructure.dataset.dataset_factory import DatasetFactory
from src.domain.metrics.forest_evaluator import ForestEvaluator

def run_benchmark(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        grid_config = yaml.safe_load(f)

    datasets = grid_config.get("datasets", [])
    strategies = grid_config.get("strategies", [])
    federation = grid_config.get("federation", {})
    model = grid_config.get("model", {})
    output_file = grid_config.get("output_file", "results/benchmark_results.csv")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    results_list = []

    for ds_info in datasets:
        ds_type = ds_info.get("type")
        print(f"\n=========================================")
        print(f"Dataset: {ds_type}")
        print(f"=========================================")
        
        # Cargar dataset
        try:
            ds = DatasetFactory.load_from_config(ds_info)
        except Exception as e:
            print(f"Error cargando dataset {ds_type}: {e}")
            continue

        for strategy in strategies:
            print(f"\n  -> Estrategia: {strategy}")
            
            # Construir configuración para esta combinación
            config = {
                "dataset": ds_info,
                "federation": federation,
                "model": model.copy(),
                "aggregation": {
                    "strategy": strategy,
                    "convergence": 0.002,
                    "f1_weight": 0.5,
                    "pcd_weight": 0.5,
                    "window_size": 5,    
                    "max_rounds": 20,    
                    "alpha": 1.0         
                },
                "prediction": {
                    "use_weighted": True,
                    "local_weight": 0.4,
                    "global_weight": 0.6
                }
            }

            try:
                orch = FLEXOrchestrator(config)
                orch.setup_federation(ds)
                res = orch.run_federated_round()
                
                precisions = []
                f1_scores = []
                
                for cid in res.client_ids:
                    y_pred = res.client_hybrid_predictions.get(cid)
                    if y_pred is not None:
                        hybrid_report = ForestEvaluator.evaluate_from_predictions(
                            y_pred, res.y_test, res.class_names, 0, 0.0
                        )
                        # Usamos "accuracy" como la precisión clásica (porcentaje total de aciertos)
                        precisions.append(hybrid_report.accuracy)
                        f1_scores.append(hybrid_report.macro_f1)
                
                avg_accuracy = np.mean(precisions) if precisions else 0.0
                avg_f1 = np.mean(f1_scores) if f1_scores else 0.0

                results_list.append({
                    "Dataset": ds_type,
                    "Strategy": strategy,
                    "Avg_Client_Accuracy": avg_accuracy,
                    "Avg_Client_F1": avg_f1,
                    "Global_Accuracy": res.global_accuracy,
                    "Global_Macro_F1": res.global_macro_f1,
                    "Status": "OK"
                })
                
                print(f"     [OK] Accuracy={avg_accuracy:.4f}, F1={avg_f1:.4f}")
                
            except Exception as e:
                print(f"     [ERROR] {e}")
                results_list.append({
                    "Dataset": ds_type,
                    "Strategy": strategy,
                    "Avg_Client_Accuracy": None,
                    "Avg_Client_F1": None,
                    "Global_Accuracy": None,
                    "Global_Macro_F1": None,
                    "Status": f"Error: {e}"
                })
                
            # Ir guardando incrementalmente
            df = pd.DataFrame(results_list)
            df.to_csv(output_file, index=False)

    print(f"\nResultados guardados en {output_file}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_benchmark(sys.argv[1])
    else:
        print("Uso: python scripts/run_benchmark.py <config.yaml>")
