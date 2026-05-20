"""Hyperparameter Grid Search for Convergence Mechanisms.

This script explores the optimal combination of early stopping parameters
across representative strategies and datasets using Joblib for parallelization.

Parameters explored:
- min_episodes (Patience): [3, 4, 5]
- size_scale (Episode Size): [5, 10, 15]
- convergence_threshold: [0.001, 0.002, 0.005]
"""
import sys, os, csv
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from joblib import Parallel, delayed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.application.orchestrators import FLEXOrchestrator
from src.application.orchestrators.progressive_tree_orchestrator import ProgressiveTreeOrchestrator
from src.domain.dataset.base_adapter import DatasetSplit
from src.infrastructure.dataset.dataset_factory import DATASET_METADATA

def load_dataset(ds_name: str) -> DatasetSplit:
    preset = DATASET_METADATA[ds_name.lower()]
    df = pd.read_csv(preset["file_path"], sep=preset["sep"])
    cols_to_drop = preset.get("columns_to_drop", [])
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
    
    target = preset["target_column"]
    X_raw = df.drop(columns=[target])
    y_raw = df[target].astype(str).values
    
    cat_cols = list(preset.get("categorical_features", []))
    for col in X_raw.columns:
        is_obj = X_raw[col].dtype == "object"
        is_str = is_string_dtype(X_raw[col])
        if (is_obj or is_str) and col not in cat_cols:
            cat_cols.append(col)
            
    le = LabelEncoder()
    y_raw_encoded = le.fit_transform(y_raw)
    
    # 60/20/20 train/val/test split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_raw, y_raw_encoded, test_size=0.20, random_state=42, stratify=y_raw_encoded
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    
    X_tr = X_train.values.copy()
    X_va = X_val.values.copy()
    X_te = X_test.values.copy()
    
    cat_cols_idx = [X_raw.columns.get_loc(c) for c in cat_cols] if cat_cols else []
    num_cols_idx = [X_raw.columns.get_loc(c) for c in X_raw.columns if c not in cat_cols]
    
    if cat_cols_idx:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_tr[:, cat_cols_idx] = enc.fit_transform(X_tr[:, cat_cols_idx].astype(str))
        X_va[:, cat_cols_idx] = enc.transform(X_va[:, cat_cols_idx].astype(str))
        X_te[:, cat_cols_idx] = enc.transform(X_te[:, cat_cols_idx].astype(str))
        
    if num_cols_idx:
        scaler = StandardScaler()
        X_tr[:, num_cols_idx] = scaler.fit_transform(X_tr[:, num_cols_idx])
        X_va[:, num_cols_idx] = scaler.transform(X_va[:, num_cols_idx])
        X_te[:, num_cols_idx] = scaler.transform(X_te[:, num_cols_idx])
        
    return DatasetSplit(
        X_train=X_tr.astype(np.float64),
        X_val=X_va.astype(np.float64),
        X_test=X_te.astype(np.float64),
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=list(X_raw.columns),
        class_names=[str(c) for c in le.classes_],
        dataset_name=ds_name
    )

def evaluate_hyperparams(
    ds_name: str, 
    strategy: str, 
    patience: int, 
    size_scale: int, 
    threshold: float, 
    split: DatasetSplit
) -> Dict[str, Any]:
    # Translate size_scale into the correct parameters per strategy
    # If size_scale is 15, we want:
    # S4 -> global_episode_size = 15
    # S7 -> trees_per_client_per_episode = 5 (since 15/3 clients = 5)
    # PW -> window_size = 15, trees_per_round_per_client = 5
    
    trees_per_client = max(1, size_scale // 3)
    
    config = {
        "federation": {"n_clients": 3, "distribution": "iid"},
        "model": {"n_estimators": 40, "alpha": 0.1, "voting": "soft"},
        "aggregation": {
            "strategy": strategy,
            "max_rounds": 15,
            "global_convergence_threshold": threshold,
            "convergence_threshold": threshold,
            "global_episode_size": size_scale,
            "trees_per_client_per_episode": trees_per_client,
            "trees_per_round_per_client": trees_per_client,
            "min_episodes": patience,
            "min_rounds": patience,
            "window_size": size_scale if strategy == "pw" else 5
        },
    }

    if strategy == "pw":
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
        print(f"Error en {strategy} ({patience}, {size_scale}, {threshold}) sobre {ds_name}: {e}")
        acc, f1, n_trees = 0.0, 0.0, 0
    finally:
        if hasattr(orch, "cleanup"):
            orch.cleanup()
            
    return {
        "strategy": strategy,
        "dataset": ds_name,
        "patience": patience,
        "size_scale": size_scale,
        "threshold": threshold,
        "accuracy": acc,
        "f1_macro": f1,
        "n_trees": n_trees
    }

def main():
    datasets = ["Car", "Sonar", "Vowel", "Spambase"]
    strategies = ["s4_global_f1_pcd", "s7_perclient_f1_pcd", "pw"]
    
    patiences = [3, 4, 5]
    size_scales = [5, 10, 15]
    thresholds = [0.001, 0.002, 0.005]

    print("======================================================================")
    print("HYPERPARAMETER SEARCH: CONVERGENCE MECHANISMS")
    print("======================================================================")

    splits = {}
    for d in datasets:
        print(f"Cargando dataset {d}...")
        splits[d] = load_dataset(d)

    tasks = []
    for d in datasets:
        for strat in strategies:
            for pat in patiences:
                for scale in size_scales:
                    for thresh in thresholds:
                        tasks.append((d, strat, pat, scale, thresh))

    print(f"\nEjecutando {len(tasks)} combinaciones en paralelo (n_jobs=2)...")
    results = Parallel(n_jobs=2, verbose=10)(
        delayed(evaluate_hyperparams)(task[0], task[1], task[2], task[3], task[4], splits[task[0]])
        for task in tasks
    )

    df_grid = pd.DataFrame(results)
    out_file = "results/hyperparam_search_results.csv"
    os.makedirs("results", exist_ok=True)
    df_grid.to_csv(out_file, index=False)
    print(f"\nResultados guardados exitosamente en: {out_file}")

    print("\n======================================================================")
    print("TOP 3 CONFIGURACIONES POR ESTRATEGIA (Basado en Accuracy Promedio)")
    print("======================================================================")

    for strat in strategies:
        print(f"\nEstrategia: {strat.upper()}")
        df_strat = df_grid[df_grid["strategy"] == strat]
        
        # Agrupar por hiperparámetros y promediar métricas entre los datasets
        grouped = df_strat.groupby(["patience", "size_scale", "threshold"]).agg(
            avg_acc=("accuracy", "mean"),
            avg_f1=("f1_macro", "mean"),
            avg_trees=("n_trees", "mean")
        ).reset_index()
        
        # Ordenar de mayor a menor accuracy
        top3 = grouped.sort_values(by="avg_acc", ascending=False).head(3)
        
        for idx, row in top3.iterrows():
            print(f"  Paciencia: {int(row['patience'])} | Tamaño: {int(row['size_scale']):2d} | "
                  f"Umbral: {row['threshold']:.3f} --> "
                  f"Acc: {row['avg_acc']:.4f} | F1: {row['avg_f1']:.4f} | Árboles: {row['avg_trees']:.1f}")

if __name__ == "__main__":
    main()
