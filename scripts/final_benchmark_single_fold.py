"""Benchmark de un solo fold sin local_isolation.

Ejecuta todas las estrategias federadas del experimento definitivo,
pero sin la estrategia 'local_isolation' y sin validación cruzada.
Solo calcula F1, accuracy y PCD.
"""

import csv
from datetime import datetime
import os
import sys
from typing import Any, Dict, List, Set, Tuple
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

# Add src to path
sys.path.append(os.getcwd())

from src.application.orchestrators import FLEXOrchestrator
from src.application.orchestrators.progressive_tree_orchestrator import (
    ProgressiveTreeOrchestrator,
)
from src.application.orchestrators.roulette_orchestrator import (
    RouletteOrchestrator,
)
from src.domain.dataset.base_adapter import DatasetSplit
from src.infrastructure.dataset.dataset_factory import DATASET_METADATA

# ==========================================================================
# CONFIGURACIÓN DEL EXPERIMENTO (UN SOLO FOLD)
# ==========================================================================
REPETITIONS = 1
TEST_SIZE = 0.1
N_CLIENTS = 3

STRATEGIES = [
    "s1_simple_pool",
    "s2_global_accuracy",
    "s3_global_f1",
    "s4_global_f1_pcd",
    "s5_perclient_accuracy",
    "s6_perclient_f1",
    "s7_perclient_f1_pcd",
    "pw",
    "s9_weighted_average",
    "s9_simple_mean",
    "s9_median",
    "s9_consensus",
    "s9_proactive_pcd",
]

DATASETS = [
    "Iris",
    "Car",
    "Nursery",
    "Vowel",
    "Letter",
    "Optdigits",
    "Sonar",
    "Spambase",
]
RESULTS_FILE = "results/results_final_benchmark_single_fold.csv"


def get_completed_work() -> Set[Tuple[int, str, str]]:
    completed = set()
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    completed.add(
                        (
                            int(row["rep"]),
                            row["strategy"],
                            row["dataset"],
                        )
                    )
        except Exception:
            pass
    return completed


def run_single_strategy(
    strategy: str,
    ds_name: str,
    rep: int,
    precomputed_splits: List[DatasetSplit],
) -> Dict[str, Any]:
    fold_results = []

    from src.domain.aggregation.aggregation_factory import AggregationFactory

    norm_strat = AggregationFactory.normalize_strategy_name(strategy)

    global_ep_size = 5
    trees_per_client_ep = 1
    trees_per_rnd_client = 1
    win_size = 5
    f1_w = 0.5
    pcd_w = 0.5

    if norm_strat == "S4":
        global_ep_size = 5
        f1_w = 0.3
        pcd_w = 0.7
    elif norm_strat == "S7":
        trees_per_client_ep = 2
        global_ep_size = trees_per_client_ep * N_CLIENTS
        f1_w = 0.3
        pcd_w = 0.7
    elif norm_strat == "PW":
        win_size = 5
        trees_per_rnd_client = 3
        f1_w = 0.3
        pcd_w = 0.7

    for split in precomputed_splits:
        config = {
            "federation": {"n_clients": N_CLIENTS, "distribution": "iid"},
            "model": {"n_estimators": 100, "alpha": 0.1, "voting": "soft"},
            "aggregation": {
                "strategy": strategy,
                "max_rounds": 15,
                "global_convergence_threshold": 0.002,
                "convergence_threshold": 0.002,
                "global_episode_size": global_ep_size,
                "trees_per_client_per_episode": trees_per_client_ep,
                "trees_per_round_per_client": trees_per_rnd_client,
                "min_episodes": 4,
                "min_rounds": 4,
                "window_size": win_size,
                "f1_weight": f1_w,
                "pcd_weight": pcd_w,
            },
        }

        if strategy.startswith("s9_"):
            variant = strategy.replace("s9_", "S9_").upper()
            config["aggregation"]["variant"] = variant
            config["aggregation"]["local_roulette_weight"] = 0.1
            orch = RouletteOrchestrator(config)
        elif strategy == "pw":
            orch = ProgressiveTreeOrchestrator(config)
        else:
            orch = FLEXOrchestrator(config)

        try:
            orch.setup_federation(split)
            res = orch.run_federated_round(n_bootstrap=0)
            fold_results.append(
                {
                    "f1": res.hybrid_f1_mean,
                    "acc": res.hybrid_accuracy_mean,
                    "pcd": res.hybrid_pcd_mean,
                    "avg_trees": float(
                        np.mean(
                            [r.forest_size for r in res.client_reports.values()]
                        )
                    ),
                }
            )
        finally:
            if hasattr(orch, "cleanup"):
                orch.cleanup()

    return {
        "rep": rep,
        "strategy": strategy,
        "dataset": ds_name,
        "f1_mean": np.mean([r["f1"] for r in fold_results]),
        "acc_mean": np.mean([r["acc"] for r in fold_results]),
        "pcd_mean": np.mean([r["pcd"] for r in fold_results]),
        "avg_trees_mean": np.mean([r["avg_trees"] for r in fold_results]),
        "timestamp": datetime.now().isoformat(),
    }


def run_final_benchmark() -> None:
    print("=== BENCHMARK DE UN SOLO FOLD: FEDERATED PROACTIVE FOREST ===")
    print(
        f"Repeticiones: {REPETITIONS} | Single Fold | Clientes: {N_CLIENTS}"
    )

    completed = get_completed_work()

    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "rep",
                    "strategy",
                    "dataset",
                    "f1_mean",
                    "acc_mean",
                    "pcd_mean",
                    "avg_trees_mean",
                    "timestamp",
                ]
            )

    for rep in range(1, REPETITIONS + 1):
        for ds_name in DATASETS:
            if all((rep, strat, ds_name) in completed for strat in STRATEGIES):
                continue

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

            cat_cols = list(preset.get("categorical_features", []))
            for col in X_raw.columns:
                is_obj = X_raw[col].dtype == "object"
                is_str = is_string_dtype(X_raw[col])
                if (is_obj or is_str) and col not in cat_cols:
                    cat_cols.append(col)

            le = LabelEncoder()
            le.fit(y_raw)

            if len(np.unique(y_raw)) > 1:
                stratify_values = y_raw
            else:
                stratify_values = None

            X_train_val_raw, X_test_raw, y_train_val_raw, y_test_raw = train_test_split(
                X_raw,
                y_raw,
                test_size=TEST_SIZE,
                random_state=42 * rep,
                stratify=stratify_values,
            )

            can_stratify = len(np.unique(y_train_val_raw)) > 1
            X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
                X_train_val_raw,
                y_train_val_raw,
                test_size=0.1111,
                random_state=42 * rep + 1,
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

            y_train = le.transform(y_train_raw)
            y_val = le.transform(y_val_raw)
            y_test = le.transform(y_test_raw)

            split = DatasetSplit(
                X_train=X_train.astype(np.float64),
                X_val=X_val.astype(np.float64),
                X_test=X_test.astype(np.float64),
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                feature_names=list(X_raw.columns),
                class_names=[str(c) for c in le.classes_],
                dataset_name=ds_name,
            )
            precomputed_splits = [split]

            pending_strategies = [
                s for s in STRATEGIES if (rep, s, ds_name) not in completed
            ]
            if not pending_strategies:
                continue

            print(
                f"  [PARALLEL] Ejecutando {len(pending_strategies)} estrategias "
                f"con n_jobs=-1..."
            )

            results_gen = Parallel(n_jobs=-1, return_as="generator")(
                delayed(run_single_strategy)(
                    strat, ds_name, rep, precomputed_splits
                )
                for strat in pending_strategies
            )

            for r in results_gen:
                with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            r["rep"],
                            r["strategy"],
                            r["dataset"],
                            r["f1_mean"],
                            r["acc_mean"],
                            r["pcd_mean"],
                            r["avg_trees_mean"],
                            r["timestamp"],
                        ]
                    )
                print(
                    f"  [CHECKPOINT] {r['strategy']} guardada | "
                    f"F1={r['f1_mean']:.4f} | Acc={r['acc_mean']:.4f} | "
                    f"PCD={r['pcd_mean']:.4f} | "
                    f"Árboles={r['avg_trees_mean']:.2f}"
                )

    print(f"\n=== BENCHMARK FINALIZADO. Resultados en {RESULTS_FILE} ===")


if __name__ == "__main__":
    run_final_benchmark()
