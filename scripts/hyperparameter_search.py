"""Hyperparameter Grid Search for Convergence Mechanisms.

This script explores the optimal combination of early stopping parameters
across representative strategies and datasets using Joblib for parallelization.

Parameters explored:
- min_episodes (Patience): [5]
- global_episode_size (Episode Size): [5, 10, 15] (explicit)
- convergence_threshold: [0.001, 0.002, 0.005]
"""
import sys
import os
from typing import Dict, Any
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.application.orchestrators import FLEXOrchestrator
from src.application.orchestrators.progressive_tree_orchestrator import ProgressiveTreeOrchestrator
from src.domain.aggregation.aggregation_factory import AggregationFactory
from src.domain.dataset.base_adapter import DatasetSplit
from src.infrastructure.dataset.dataset_factory import DatasetFactory
from src.infrastructure.persistence.experiment_results import save_experiment_results_json


def load_dataset(ds_name: str) -> DatasetSplit:
    dataset_config = {
        "type": ds_name,
        "scale": True,
        "scaler_type": "standard",
        "seed": 42,
    }
    return DatasetFactory.load_from_config(dataset_config)

def evaluate_hyperparams(
    ds_name: str,
    strategy: str,
    patience: int,
    global_episode_size: int,
    threshold: float,
    split: DatasetSplit,
) -> Dict[str, Any]:
    # Use explicit parameters instead of `size_scale`.
    # For S4 and S7 we expose `global_episode_size` (total trees added per episode).
    # For PW we fix/override window size and trees_per_round_per_client below.
    trees_per_client = max(1, global_episode_size // 3) if global_episode_size is not None else 1
    strategy_norm = AggregationFactory.normalize_strategy_name(strategy)

    aggregation_config = {
        "strategy": strategy_norm,
        "max_rounds": 15,
        "global_convergence_threshold": threshold,
        "convergence_threshold": threshold,
        "trees_per_client_per_episode": trees_per_client,
        "min_episodes": patience,
        "min_rounds": patience,
        # PW explicit defaults: window_size=5, trees_per_round_per_client=2
        "window_size": 5,
        "trees_per_round_per_client": 2 if strategy_norm == "PW" else trees_per_client,
        "f1_weight": 0.3,
        "pcd_weight": 0.7,
    }

    if strategy_norm != "PW":
        aggregation_config["global_episode_size"] = global_episode_size

    config = {
        "federation": {"n_clients": 3, "distribution": "iid"},
        "model": {"n_estimators": 40, "alpha": 0.1, "voting": "soft"},
        "aggregation": aggregation_config,
    }

    if strategy_norm == "PW":
        orch = ProgressiveTreeOrchestrator(config)
    else:
        orch = FLEXOrchestrator(config)
        
    try:
        orch.setup_federation(split)
        res = orch.run_federated_round(n_bootstrap=0)
        acc = res.hybrid_accuracy_mean
        f1 = res.hybrid_f1_mean
        n_trees = len(orch.flex_pool._models["server"].get("trees", []))
    except Exception as e:
        print(
            f"Error en {strategy} ({patience}, {global_episode_size}, {threshold}) sobre {ds_name}: {e}"
        )
        acc, f1, n_trees = 0.0, 0.0, 0
    finally:
        if hasattr(orch, "cleanup"):
            orch.cleanup()
            
    return {
        "strategy": strategy,
        "strategy_normalized": strategy_norm,
        "dataset": ds_name,
        "patience": patience,
        "global_episode_size": global_episode_size,
        "threshold": threshold,
        "accuracy": acc,
        "f1_macro": f1,
        "n_trees": n_trees,
    }


def main():
    datasets = ["Car", "Sonar", "Vowel", "Spambase"]
    strategies = ["s4_global_f1_pcd", "s7_perclient_f1_pcd", "pw"]
    
    # patience (lower bound 5 as requested)
    patiences = [5]
    # explicit global episode sizes to explore for S4/S7
    global_episode_sizes = [5, 10, 15]
    thresholds = [0.001, 0.002, 0.005]

    print("======================================================================")
    print("HYPERPARAMETER SEARCH: CONVERGENCE MECHANISMS")
    print("======================================================================")

    dataset_base_config = {
        "scale": True,
        "scaler_type": "standard",
        "seed": 42,
    }

    splits = {}
    for d in datasets:
        print(f"Cargando dataset {d}...")
        splits[d] = DatasetFactory.load_from_config({**dataset_base_config, "type": d})

    tasks = []
    for d in datasets:
        for strat in strategies:
            for pat in patiences:
                for ges in global_episode_sizes:
                    for thresh in thresholds:
                        tasks.append((d, strat, pat, ges, thresh))

    print(f"\nEjecutando {len(tasks)} combinaciones en paralelo (n_jobs=2)...")
    results = Parallel(n_jobs=2, verbose=10)(
        delayed(evaluate_hyperparams)(
            task[0], task[1], task[2], task[3], task[4], splits[task[0]]
        )
        for task in tasks
    )

    print("\nGuardando resultados en JSON...")
    out_file = "results/hyperparam_search_results.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    save_experiment_results_json(
        out_file,
        {
            "script_name": "hyperparameter_search.py",
            "experiment_type": "hyperparameter_search",
            "objective": "hybrid_accuracy",
            "dataset_config": dataset_base_config,
            "common_config": {
                "federation": {"n_clients": 3, "distribution": "iid"},
                "model": {"n_estimators": 40, "alpha": 0.1, "voting": "soft"},
            },
            "created_by": "hyperparameter_search.py",
        },
        results,
    )
    print(f"Resultados guardados exitosamente en: {out_file}")

    df_grid = pd.DataFrame(results)

    print("\n======================================================================")
    print("TOP 3 CONFIGURACIONES POR ESTRATEGIA (Basado en Accuracy Promedio)")
    print("======================================================================")

    for strat in strategies:
        strat_norm = AggregationFactory.normalize_strategy_name(strat)
        print(f"\nEstrategia: {strat_norm}")
        df_strat = df_grid[df_grid["strategy_normalized"] == strat_norm]
        
        grouped = df_strat.groupby(["patience", "global_episode_size", "threshold"]).agg(
            avg_acc=("accuracy", "mean"),
            avg_f1=("f1_macro", "mean"),
            avg_trees=("n_trees", "mean")
        ).reset_index()
        
        top3 = grouped.sort_values(by="avg_acc", ascending=False).head(3)
        
        for _, row in top3.iterrows():
            print(
                f"  Paciencia: {int(row['patience'])} | Tamaño: {int(row['global_episode_size']):2d} | "
                f"Umbral: {row['threshold']:.3f} --> "
                f"Acc: {row['avg_acc']:.4f} | F1: {row['avg_f1']:.4f} | Árboles: {row['avg_trees']:.1f}"
            )

if __name__ == "__main__":
    main()
