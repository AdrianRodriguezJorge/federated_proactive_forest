"""Unified Hyperparameter Grid Search comparing Paso de 3 vs Paso de 6 across all progressive selection strategies.

Compares:
- Paso 3: Global (S2, S3, S4 with global_episode_size=3), Per-Client (S5, S6, S7 with trees_per_client=1), PW (trees_per_round=1)
- Paso 6: Global (S2, S3, S4 with global_episode_size=6), Per-Client (S5, S6, S7 with trees_per_client=2), PW (trees_per_round=2)

Over the 4 standard datasets: Sonar, Vowel, Spambase, Nursery.
Fixed hyperparameters: patience=4, pcd_weight=0.7 (where applicable), window_size=5 (for PW).
"""

import sys
import os
import time
import warnings
import argparse
import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Silence FLEX warnings about numpy arrays
warnings.filterwarnings(
    "ignore",
    message="X_array or y_array are not a list nor a numpy array",
    category=RuntimeWarning,
)

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.application.orchestrators import FLEXOrchestrator
from src.application.orchestrators.progressive_tree_orchestrator import (
    ProgressiveTreeOrchestrator,
)
from src.domain.dataset.base_adapter import DatasetSplit
from src.infrastructure.dataset.dataset_factory import DatasetFactory
from src.infrastructure.persistence.experiment_results import save_experiment_results_json


def evaluate_config(
    strategy: str,
    ds_name: str,
    pace: int,
    patience: int,
    threshold: float,
    n_estimators_max: int,
    local_convergence: float,
    split: DatasetSplit,
) -> Dict[str, Any]:
    """Evalúa una estrategia específica con paso de 3 o paso de 6."""
    pcd_w = 0.7
    f1_w = 0.3

    # Traducir paso a los hiperparámetros específicos
    is_global = strategy in ["S2", "S3", "S4"]
    is_perclient = strategy in ["S5", "S6", "S7"]

    if is_global:
        global_ep_size = pace
        trees_per_client_ep = 2  # No se usa pero se declara
        trees_per_rnd_client = 2
    elif is_perclient:
        trees_per_client_ep = pace // 3  # 1 o 2
        global_ep_size = trees_per_client_ep * 3
        trees_per_rnd_client = 2
    elif strategy == "PW":
        trees_per_rnd_client = pace // 3  # 1 o 2
        global_ep_size = trees_per_rnd_client * 3
        trees_per_client_ep = 2

    # Normalizar nombre de estrategia para FLEXOrchestrator
    strategy_map = {
        "S2": "s2_global_accuracy",
        "S3": "s3_global_f1",
        "S4": "s4_global_f1_pcd",
        "S5": "s5_perclient_accuracy",
        "S6": "s6_perclient_f1",
        "S7": "s7_perclient_f1_pcd",
        "PW": "pw",
    }
    flex_strategy = strategy_map[strategy]

    aggregation_config = {
        "strategy": flex_strategy,
        "variant": flex_strategy,
        "max_rounds": 20,
        "global_convergence_threshold": threshold,
        "convergence_threshold": threshold,
        "global_episode_size": global_ep_size,
        "trees_per_client_per_episode": trees_per_client_ep,
        "trees_per_round_per_client": trees_per_rnd_client,
        "min_episodes": patience,
        "min_rounds": patience,
        "window_size": 5,
        "f1_weight": f1_w,
        "pcd_weight": pcd_w,
        "t_max": 100,
    }

    config = {
        "federation": {"n_clients": 3, "distribution": "iid"},
        "model": {
            "n_estimators": n_estimators_max,
            "alpha": 0.1,
            "voting": "soft",
            "local_convergence_threshold": local_convergence,
        },
        "aggregation": aggregation_config,
        "prediction": {
            "local_weight": 0.4,
            "use_weighted": True
        }
    }

    if strategy == "PW":
        orch = ProgressiveTreeOrchestrator(config)
    else:
        orch = FLEXOrchestrator(config)
        
    try:
        orch.setup_federation(split)
        res = orch.run_federated_round(n_bootstrap=0)
        
        # Métricas promedio de los bosques híbridos de los clientes
        acc = res.hybrid_accuracy_mean
        f1 = res.hybrid_f1_mean
        pcd = res.hybrid_pcd_mean
        avg_trees = float(np.mean([r.forest_size for r in res.client_reports.values()]))
        
        convergence_round = res.convergence_round
    except Exception as e:
        print(f"Error en {strategy} (Paso={pace}) en dataset {ds_name}: {e}")
        acc, f1, pcd, avg_trees, convergence_round = 0.0, 0.0, 0.0, 0, None
    finally:
        if hasattr(orch, "cleanup"):
            orch.cleanup()
            
    return {
        "strategy": strategy,
        "dataset": ds_name,
        "pace": pace,
        "accuracy": acc,
        "f1_macro": f1,
        "pcd": pcd,
        "forest_size": avg_trees,
        "convergence_round": convergence_round,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Unified Grid Search comparing Paso de 3 vs Paso de 6 across all S strategies + PW."
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs to run (default: -1)",
    )
    parser.add_argument(
        "--n_estimators_max",
        type=int,
        default=100,
        help="Maximum estimators in server model (default: 100)",
    )
    parser.add_argument(
        "--local_convergence",
        type=float,
        default=0.002,
        help="Local convergence threshold for client training (default: 0.002)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data splitting (default: 42)",
    )
    
    args = parser.parse_args()

    datasets = ["Sonar", "Vowel", "Spambase", "Nursery"]
    strategies = ["S2", "S3", "S4", "S5", "S6", "S7", "PW"]
    paces = [3, 6]
    
    # Hiperparámetros de control fijos
    patience = 4
    threshold = 0.002

    print("======================================================================")
    print("COMPARATIVA DE PASO DE CRECIMIENTO: PASO DE 3 VS PASO DE 6 (TODAS LAS ESTRATEGIAS S + PW)")
    print("======================================================================")
    print(f"Parámetros Fijos: Patience = {patience} | Threshold = {threshold} | PCD Weight = 0.7 (donde aplica)")
    print(f"Estrategias: {strategies} | Pasos: {paces}")
    print(f"Datasets: {datasets}")

    dataset_base_config = {
        "test_size": 0.2,
        "scale": True,
        "scaler_type": "standard",
        "seed": args.seed,
    }

    splits = {}
    for d in datasets:
        print(f"Cargando dataset {d}...")
        splits[d] = DatasetFactory.load_from_config(
            {**dataset_base_config, "type": d.lower()}
        )

    # Generación de tareas (combinatoria)
    tasks = []
    for d in datasets:
        for strat in strategies:
            for pace in paces:
                tasks.append((strat, d, pace))

    print(f"\nEjecutando {len(tasks)} combinaciones en paralelo (n_jobs={args.n_jobs})...")
    start_time = time.time()
    
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(evaluate_config)(
            task[0],  # strategy
            task[1],  # ds_name
            task[2],  # pace
            patience,
            threshold,
            args.n_estimators_max,
            args.local_convergence,
            splits[task[1]],
        )
        for task in tasks
    )

    end_time = time.time()
    print(f"\nComparativa finalizada en {end_time - start_time:.2f} segundos.")

    print("\nGuardando resultados en JSON...")
    out_file = "results/unified_hyperparam_search_results.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    save_experiment_results_json(
        out_file,
        {
            "script_name": "unified_hyperparameter_search.py",
            "experiment_type": "pace_comparison_all_s_strategies_3_vs_6",
            "objective": "hybrid_accuracy",
            "dataset_config": dataset_base_config,
            "common_config": {
                "federation": {"n_clients": 3, "distribution": "iid"},
                "model": {
                    "n_estimators": args.n_estimators_max,
                    "alpha": 0.1,
                    "voting": "soft",
                    "local_convergence_threshold": args.local_convergence,
                },
                "patience": patience,
                "threshold": threshold,
            },
            "created_by": "unified_hyperparameter_search.py",
        },
        results,
    )
    print(f"Resultados guardados exitosamente en: {out_file}")

    # Analizar y mostrar la tabla comparativa por estrategia y paso
    df = pd.DataFrame(results)
    
    grouped = df.groupby(["strategy", "pace"]).agg(
        mean_acc=("accuracy", "mean"),
        mean_f1=("f1_macro", "mean"),
        mean_pcd=("pcd", "mean"),
        mean_forest=("forest_size", "mean")
    ).reset_index()

    # Reordenar las estrategias en un orden lógico
    strategy_order = {"S2": 1, "S3": 2, "S4": 3, "S5": 4, "S6": 5, "S7": 6, "PW": 7}
    grouped["order"] = grouped["strategy"].map(strategy_order)
    grouped = grouped.sort_values(by=["order", "pace"])

    print("\n======================================================================")
    print("TABLA COMPARATIVA GLOBAL (PASO DE 3 VS PASO DE 6)")
    print("======================================================================")
    print(f"{'Estrategia':<10} | {'Paso':<5} | {'Acc Promedio':<12} | {'F1 Promedio':<12} | {'PCD Promedio':<12} | {'Bosque Promedio':<15}")
    print("-" * 75)
    for _, row in grouped.iterrows():
        print(
            f"{row['strategy']:<10} | {row['pace']:<5} | "
            f"{row['mean_acc']*100:11.2f}% | {row['mean_f1']*100:11.2f}% | "
            f"{row['mean_pcd']:11.4f} | {row['mean_forest']:14.1f}"
        )


if __name__ == "__main__":
    main()
