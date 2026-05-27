#!/usr/bin/env python
"""Experiment Script to Test the Winning Hyperparameters (Trial 13).

Evaluates all 14 strategies on Sonar and Vowel datasets using a 10-Fold CV protocol
with the optimal hyperparameters found during the unified Bayesian optimization run.
"""

import os
import sys
import csv
import time
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.application.orchestrators import FLEXOrchestrator
from src.application.orchestrators.fed_data_distributor import FedDataDistributor
from src.application.orchestrators.roulette_orchestrator import RouletteOrchestrator
from src.domain.dataset.base_adapter import DatasetSplit
from src.domain.metrics.forest_evaluator import ForestEvaluator
from src.domain.model.proactive_forest import ProactiveForest
from src.infrastructure.dataset.dataset_factory import DATASET_METADATA

# ===========================================================================
# CONFIGURACIÓN DEL EXPERIMENTO (TRIAL 13 WINNING HYPERPARAMETERS)
# ===========================================================================
N_CLIENTS = 3
K_FOLDS = 2
REPETITIONS = 1

# Winning Hyperparameters (Trial 13)
# Hyperparameters of Trial 13
WINNING_PARAMS = {
    "local_convergence_threshold": 0.002,
    "local_weight": 0.5,
    "use_weighted": True,
    "max_trees": 130,
    "global_convergence_threshold": 0.001,
    "f1_weight": 0.5,
    "pcd_weight": 0.5,
    "local_roulette_weight": 0.1,
    "global_episode_size": 3,
    "trees_per_client_per_episode": 1,
    "window_size": 10,
    "max_rounds_roulette": 15,
    "max_rounds_flex": 20,
}

STRATEGIES = [
    "local_isolation",
    "s1_simple_pool",
    "s2_global_accuracy",
    "s3_global_f1",
    "s4_global_f1_pcd",
    "s5_perclient_accuracy",
    "s6_perclient_f1",
    "s7_perclient_f1_pcd",
    "s8_weighted_average",
    "s8_simple_mean",
    "s8_median",
    "s8_consensus",
    "s8_proactive_pcd",
]

DATASETS = ["Nursery"]
RESULTS_FILE = "results/results_trial13.csv"

def run_single_strategy(
    strategy: str,
    ds_name: str,
    precomputed_splits: List[DatasetSplit],
) -> Dict[str, Any]:
    """Helper function to run all folds of a strategy for a dataset."""
    fold_results = []
    
    # Extract values from Winning Params
    l_conv = WINNING_PARAMS["local_convergence_threshold"]
    l_weight = WINNING_PARAMS["local_weight"]
    use_weighted = WINNING_PARAMS["use_weighted"]
    max_trees = WINNING_PARAMS["max_trees"]
    g_conv = WINNING_PARAMS["global_convergence_threshold"]
    f1_w = WINNING_PARAMS["f1_weight"]
    pcd_w = WINNING_PARAMS["pcd_weight"]
    l_roulette_w = WINNING_PARAMS["local_roulette_weight"]
    global_ep_size = WINNING_PARAMS["global_episode_size"]
    trees_per_client_ep = WINNING_PARAMS["trees_per_client_per_episode"]
    window_size = WINNING_PARAMS["window_size"]
    
    for fold_idx, split in enumerate(precomputed_splits):
        # Base config with winning params
        config = {
            "federation": {"n_clients": N_CLIENTS, "distribution": "iid"},
            "model": {
                "n_estimators": 150,
                "alpha_pf": 0.1,
                "voting": "soft",
                "local_convergence_threshold": l_conv,
            },
            "aggregation": {
                "strategy": strategy,
                "max_trees": max_trees,
                "t_max": max_trees,
                "global_convergence_threshold": g_conv,
                "global_episode_size": global_ep_size,
                "trees_per_client_per_episode": trees_per_client_ep,
                "trees_per_round_per_client": trees_per_client_ep,
                "min_episodes": 4,
                "min_rounds": 4,
                "window_size": window_size,
                "f1_weight": f1_w,
                "pcd_weight": pcd_w,
            },
            "prediction": {
                "local_weight": l_weight,
                "global_weight": 1.0 - l_weight,
                "use_weighted": use_weighted,
            }
        }
        
        if strategy == "local_isolation":
            distributor = FedDataDistributor(config, use_flex_pool=False)
            _, fed_data = distributor.distribute(split)
            client_reports = []
            for _, client_dataset in fed_data.items():
                X_c, y_c = client_dataset.to_numpy()
                model = ProactiveForest(
                    n_estimators=100,
                    alpha_pf=0.1,
                    class_names=split.class_names,
                    local_convergence_threshold=l_conv,
                )
                model.fit(X_c, y_c)
                preds = model.predict(split.X_test)
                client_reports.append(
                    ForestEvaluator.evaluate_from_predictions(
                        preds,
                        split.y_test,
                        split.class_names,
                        len(model.get_trees()),
                    )
                )
            fold_results.append({
                "f1": np.mean([r.macro_f1 for r in client_reports]),
                "acc": np.mean([r.accuracy for r in client_reports]),
                "recall": np.mean([r.macro_recall for r in client_reports]),
                "prec": np.mean([r.macro_precision for r in client_reports]),
                "pcd": 0.0,
            })
        else:
            if strategy.startswith("s8_"):
                variant = strategy.replace("s8_", "S8_").upper()
                if variant == "S8_SIMPLE_MEAN":
                    variant = "S8_MEAN"
                elif variant == "S8_WEIGHTED_AVERAGE":
                    variant = "S8_WEIGHTED"
                
                config["aggregation"]["strategy"] = "S8"
                config["aggregation"]["variant"] = variant
                config["aggregation"]["local_roulette_weight"] = l_roulette_w
                config["aggregation"]["max_rounds"] = WINNING_PARAMS["max_rounds_roulette"]
                orch = RouletteOrchestrator(config)
            else:
                config["aggregation"]["max_rounds"] = WINNING_PARAMS["max_rounds_flex"]
                orch = FLEXOrchestrator.from_config(config)
                
            try:
                orch.setup_federation(split)
                res = orch.run_federated_round(n_bootstrap=0)
                fold_results.append({
                    "f1": res.hybrid_f1_mean,
                    "acc": res.hybrid_accuracy_mean,
                    "recall": res.hybrid_recall_mean,
                    "prec": res.hybrid_precision_mean,
                    "pcd": res.hybrid_pcd_mean,
                })
            except Exception as e:
                print(f"Error in fold {fold_idx} for strategy {strategy}: {e}")
                fold_results.append({"f1": 0.0, "acc": 0.0, "recall": 0.0, "prec": 0.0, "pcd": 0.0})
            finally:
                if hasattr(orch, "cleanup"):
                    orch.cleanup()
                    
    return {
        "strategy": strategy,
        "dataset": ds_name,
        "f1_mean": np.mean([r["f1"] for r in fold_results]),
        "f1_std": np.std([r["f1"] for r in fold_results]),
        "acc_mean": np.mean([r["acc"] for r in fold_results]),
        "acc_std": np.std([r["acc"] for r in fold_results]),
        "recall_mean": np.mean([r["recall"] for r in fold_results]),
        "recall_std": np.std([r["recall"] for r in fold_results]),
        "prec_mean": np.mean([r["prec"] for r in fold_results]),
        "prec_std": np.std([r["prec"] for r in fold_results]),
        "pcd_mean": np.mean([r["pcd"] for r in fold_results]),
    }

def main():
    print("==================================================================")
    print("EXPERIMENTO CON HIPERPARÁMETROS DEL TRIAL 13")
    print("==================================================================")
    print(f"Datasets: {DATASETS} | Folds: {K_FOLDS} | Clientes: {N_CLIENTS}")
    print("Hiperparámetros usados:")
    for k, v in WINNING_PARAMS.items():
        print(f"  - {k}: {v}")
    print("==================================================================\n")
    
    os.makedirs("results", exist_ok=True)
    
    # Initialize output file
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset", "strategy", "acc_mean", "acc_std", 
            "f1_mean", "f1_std", "pcd_mean", "timestamp"
        ])
        
    for ds_name in DATASETS:
        print(f"\n>>> Procesando dataset: {ds_name}...")
        preset = DATASET_METADATA[ds_name.lower()]
        try:
            df = pd.read_csv(preset["file_path"], sep=preset["sep"])
        except Exception as e:
            print(f"Error cargando {ds_name}: {e}")
            continue
            
        cols_to_drop = preset.get("columns_to_drop", [])
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
        
        target = preset["target_column"]
        X_raw = df.drop(columns=[target])
        y_raw = df[target].astype(str).values
        
        # Subset Nursery to 2000 samples to run fast while maintaining class ratios
        if ds_name == "Nursery" and len(X_raw) > 2000:
            X_raw, _, y_raw, _ = train_test_split(
                X_raw, y_raw,
                train_size=2000,
                random_state=42,
                stratify=y_raw
            )
            
        # Categorical detection
        cat_cols = list(preset.get("categorical_features", []))
        for col in X_raw.columns:
            if (X_raw[col].dtype == "object" or pd.api.types.is_string_dtype(X_raw[col])) and col not in cat_cols:
                cat_cols.append(col)
                
        # Label Encoder
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(y_raw)
        
        # Pre-compute Folds
        print(f"  - Pre-calculando {K_FOLDS} folds...")
        skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
        precomputed_splits = []
        
        for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(X_raw, y_raw)):
            if fold_idx >= 1:
                break
            X_train_val_raw = X_raw.iloc[train_val_idx]
            X_test_raw = X_raw.iloc[test_idx]
            y_train_val_raw = y_raw[train_val_idx]
            y_test_raw = y_raw[test_idx]
            
            _, counts_y = np.unique(y_train_val_raw, return_counts=True)
            can_stratify = np.min(counts_y) >= 2
            
            X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
                X_train_val_raw,
                y_train_val_raw,
                test_size=0.15,  # 70/15/15 ratio
                random_state=42 + fold_idx,
                stratify=y_train_val_raw if can_stratify else None,
            )
            
            X_train = X_train_raw.values.copy()
            X_val = X_val_raw.values.copy()
            X_test = X_test_raw.values.copy()
            
            cat_cols_idx = [X_raw.columns.get_loc(c) for c in cat_cols] if cat_cols else []
            num_cols_idx = [X_raw.columns.get_loc(c) for c in X_raw.columns if c not in cat_cols]
            
            if cat_cols_idx:
                enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                X_train[:, cat_cols_idx] = enc.fit_transform(X_train[:, cat_cols_idx].astype(str))
                X_val[:, cat_cols_idx] = enc.transform(X_val[:, cat_cols_idx].astype(str))
                X_test[:, cat_cols_idx] = enc.transform(X_test[:, cat_cols_idx].astype(str))
                
            if num_cols_idx:
                scaler = StandardScaler()
                X_train[:, num_cols_idx] = scaler.fit_transform(X_train[:, num_cols_idx])
                X_val[:, num_cols_idx] = scaler.transform(X_val[:, num_cols_idx])
                X_test[:, num_cols_idx] = scaler.transform(X_test[:, num_cols_idx])
                
            split = DatasetSplit(
                X_train=X_train.astype(np.float64),
                X_val=X_val.astype(np.float64),
                X_test=X_test.astype(np.float64),
                y_train=le.transform(y_train_raw),
                y_val=le.transform(y_val_raw),
                y_test=le.transform(y_test_raw),
                feature_names=list(X_raw.columns),
                class_names=[str(c) for c in le.classes_],
                dataset_name=ds_name,
            )
            precomputed_splits.append(split)
            
        print(f"  - Ejecutando las {len(STRATEGIES)} estrategias en 2 hilos paralelos...")
        start_ds = time.time()
        
        # Parallel execution across strategies
        results = Parallel(n_jobs=2)(
            delayed(run_single_strategy)(strat, ds_name, precomputed_splits)
            for strat in STRATEGIES
        )
        
        end_ds = time.time()
        print(f"  - Completado en {(end_ds - start_ds)/60:.2f} minutos.")
        
        # Save and print results for this dataset
        print("\n  Resultados:")
        print(f"  {'Estrategia':<25} | {'Accuracy':<15} | {'Macro-F1':<15} | {'Diversidad (PCD)':<15}")
        print("  " + "-" * 78)
        
        with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for r in results:
                writer.writerow([
                    r["dataset"], r["strategy"],
                    f"{r['acc_mean']:.4f}", f"{r['acc_std']:.4f}",
                    f"{r['f1_mean']:.4f}", f"{r['f1_std']:.4f}",
                    f"{r['pcd_mean']:.4f}", datetime.now().isoformat()
                ])
                print(f"  {r['strategy']:<25} | {r['acc_mean']:.4f}±{r['acc_std']:.3f} | {r['f1_mean']:.4f}±{r['f1_std']:.3f} | {r['pcd_mean']:.4f}")
                
    print(f"\nExperimento finalizado. Resultados consolidados en: {RESULTS_FILE}")

if __name__ == "__main__":
    main()
