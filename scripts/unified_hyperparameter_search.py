"""Unified Hyperparameter Grid Search for Federated Proactive Forest.

This script executes a comprehensive grid search over both convergence thresholds,
episode sizes, and PCD weights for S4 and S7 strategies (excluding PW).
It maintains a homologous configuration across all tasks and datasets.
"""

import sys
import os
import time
import warnings
import argparse
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
from src.domain.aggregation.aggregation_factory import AggregationFactory
from src.domain.dataset.base_adapter import DatasetSplit
from src.infrastructure.dataset.dataset_factory import DatasetFactory
from src.infrastructure.persistence.experiment_results import save_experiment_results_json


def evaluate_hyperparams(
    ds_name: str,
    strategy: str,
    patience: int,
    global_episode_size: int,
    threshold: float,
    pcd_weight: float,
    n_estimators: int,
    local_convergence: float,
    split: DatasetSplit,
) -> Dict[str, Any]:
    """Evaluate a single hyperparameter combination on a specific dataset split."""
    strategy_norm = AggregationFactory.normalize_strategy_name(strategy)
    
    # Standardize trees_per_client based on global_episode_size and 3 clients
    trees_per_client = max(1, global_episode_size // 3)
    f1_w = round(1.0 - pcd_weight, 2)
    pcd_w = round(pcd_weight, 2)

    aggregation_config = {
        "strategy": strategy_norm,
        "variant": strategy_norm,
        "max_rounds": 15,
        "global_convergence_threshold": threshold,
        "convergence_threshold": threshold,
        "trees_per_client_per_episode": trees_per_client,
        "min_episodes": patience,
        "min_rounds": patience,
        "window_size": 5,
        "trees_per_round_per_client": trees_per_client,
        "f1_weight": f1_w,
        "pcd_weight": pcd_w,
        "t_max": 100,
    }

    if strategy_norm != "PW":
        aggregation_config["global_episode_size"] = global_episode_size

    config = {
        "federation": {"n_clients": 3, "distribution": "iid"},
        "model": {
            "n_estimators": n_estimators,
            "alpha": 0.1,
            "voting": "soft",
            "local_convergence_threshold": local_convergence,
        },
        "aggregation": aggregation_config,
    }

    orch = FLEXOrchestrator(config)
        
    try:
        orch.setup_federation(split)
        res = orch.run_federated_round(n_bootstrap=0)
        acc = res.hybrid_accuracy_mean
        f1 = res.hybrid_f1_mean
        n_trees = len(orch.flex_pool._models["server"].get("trees", []))
    except Exception as e:
        print(
            f"Error on {strategy_norm} (pat={patience}, size={global_episode_size}, thres={threshold}, pcd={pcd_w}) on {ds_name}: {e}"
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
        "pcd_weight": pcd_w,
        "f1_weight": f1_w,
        "accuracy": acc,
        "f1_macro": f1,
        "n_trees": n_trees,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Unified Grid Search for early-stopping and PCD weight hyperparameters."
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs to run (default: -1)",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Number of estimators for local training (default: 100)",
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
    strategies = ["s4_global_f1_pcd", "s7_perclient_f1_pcd"]  # PW is excluded
    
    # Grid Search Space
    patiences = [3, 4]
    global_episode_sizes = [5, 10, 15]
    thresholds = [0.002]
    pcd_weights = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]

    print("======================================================================")
    print("UNIFIED HYPERPARAMETER GRID SEARCH (S4 & S7)")
    print("======================================================================")
    print(f"Estimators: {args.n_estimators} | Local threshold: {args.local_convergence} | Seed: {args.seed}")

    dataset_base_config = {
        "test_size": 0.2,
        "scale": True,
        "scaler_type": "standard",
        "seed": args.seed,
    }

    splits = {}
    for d in datasets:
        print(f"Loading dataset {d}...")
        splits[d] = DatasetFactory.load_from_config(
            {**dataset_base_config, "type": d.lower()}
        )

    tasks = []
    for d in datasets:
        for strat in strategies:
            for pat in patiences:
                for ges in global_episode_sizes:
                    for thresh in thresholds:
                        for pcd in pcd_weights:
                            tasks.append((d, strat, pat, ges, thresh, pcd))

    print(f"\nRunning {len(tasks)} combinations in parallel (n_jobs={args.n_jobs})...")
    start_time = time.time()
    
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(evaluate_hyperparams)(
            task[0],
            task[1],
            task[2],
            task[3],
            task[4],
            task[5],
            args.n_estimators,
            args.local_convergence,
            splits[task[0]],
        )
        for task in tasks
    )

    end_time = time.time()
    print(f"\nGrid search completed in {end_time - start_time:.2f} seconds.")

    print("\nSaving results to JSON...")
    out_file = "results/unified_hyperparam_search_results.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    save_experiment_results_json(
        out_file,
        {
            "script_name": "unified_hyperparameter_search.py",
            "experiment_type": "unified_grid_search",
            "objective": "hybrid_accuracy",
            "dataset_config": dataset_base_config,
            "common_config": {
                "federation": {"n_clients": 3, "distribution": "iid"},
                "model": {
                    "n_estimators": args.n_estimators,
                    "alpha": 0.1,
                    "voting": "soft",
                    "local_convergence_threshold": args.local_convergence,
                },
            },
            "created_by": "unified_hyperparameter_search.py",
        },
        results,
    )
    print(f"Results successfully saved at: {out_file}")

    # Analyze and display results
    df_grid = pd.DataFrame(results)

    print("\n======================================================================")
    print("TOP 3 CONFIGURATIONS PER STRATEGY (Based on Average Accuracy)")
    print("======================================================================")

    for strat in strategies:
        strat_norm = AggregationFactory.normalize_strategy_name(strat)
        print(f"\nStrategy: {strat_norm}")
        df_strat = df_grid[df_grid["strategy_normalized"] == strat_norm]
        
        grouped = df_strat.groupby(
            ["patience", "global_episode_size", "threshold", "pcd_weight"]
        ).agg(
            avg_acc=("accuracy", "mean"),
            avg_f1=("f1_macro", "mean"),
            avg_trees=("n_trees", "mean")
        ).reset_index()
        
        top3 = grouped.sort_values(by="avg_acc", ascending=False).head(3)
        
        for _, row in top3.iterrows():
            print(
                f"  Patience: {int(row['patience'])} | Size: {int(row['global_episode_size']):2d} | "
                f"Threshold: {row['threshold']:.3f} | PCD Weight: {row['pcd_weight']:.1f} --> "
                f"Acc: {row['avg_acc']:.4f} | F1: {row['avg_f1']:.4f} | Trees: {row['avg_trees']:.1f}"
            )


if __name__ == "__main__":
    main()
