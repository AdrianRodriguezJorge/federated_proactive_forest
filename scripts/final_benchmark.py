import os
import sys
import pandas as pd
import numpy as np
import csv
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder

# Añadir src al path
sys.path.append(os.getcwd())

from src.domain.dataset.base_adapter import DatasetSplit
from src.application.orchestrators.fl_orchestrator import FLEXOrchestrator
from src.application.orchestrators.roulette_orchestrator import RouletteOrchestrator
from src.application.orchestrators.progressive_tree_orchestrator import ProgressiveTreeOrchestrator
from src.domain.model.proactive_forest import ProactiveForest
from src.domain.metrics.forest_evaluator import ForestEvaluator
from src.application.orchestrators.fed_data_distributor import FedDataDistributor
from src.interfaces.streamlit.components.constants import DATASET_PRESETS

# ===========================================================================
# CONFIGURACIÓN DEL EXPERIMENTO DEFINITIVO (NAYMA RIGOR)
# ===========================================================================
REPETITIONS = 1  # Cambiar a 5 para igualar exactamente a Nayma
K_FOLDS = 10     # Protocolo 10-Fold CV
N_CLIENTS = 3    # Configuración de clientes federados

STRATEGIES = [
    "local_isolation",
    "s1_simple_pool", "s2_global_accuracy", "s3_global_f1", "s4_global_f1_pcd", 
    "s5_perclient_accuracy", "s6_perclient_f1", "s7_perclient_f1_pcd", "pw", 
    "s9_weighted_average", "s9_simple_mean", "s9_median", "s9_consensus", "s9_proactive_pcd"
]

DATASETS = ["Iris", "Car", "Nursery", "Vowel", "Letter", "Optdigits", "Sonar", "Spambase"]
RESULTS_FILE = "results/results_final_benchmark.csv"

def get_completed_work():
    """Identifica qué combinaciones de (repetición, estrategia, dataset) ya terminaron."""
    completed = set()
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r", encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    completed.add((int(row["rep"]), row["strategy"], row["dataset"]))
        except Exception: pass
    return completed

def run_final_benchmark():
    print(f"=== BENCHMARK DEFINITIVO: FEDERATED PROACTIVE FOREST ===")
    print(f"Repeticiones: {REPETITIONS} | Folds: {K_FOLDS} | Clientes: {N_CLIENTS}")
    
    completed = get_completed_work()
    
    # Inicializar archivo con todas las métricas de Nayma
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "rep", "strategy", "dataset", 
                "f1_mean", "f1_std", 
                "acc_mean", "acc_std", 
                "recall_mean", "recall_std",
                "prec_mean", "prec_std",
                "pcd_mean", "timestamp"
            ])

    for rep in range(1, REPETITIONS + 1):
        for ds_name in DATASETS:
            # --- Carga Eficiente de Datos ---
            if all((rep, strat, ds_name) in completed for strat in STRATEGIES):
                continue
                
            preset = DATASET_PRESETS[ds_name]
            try:
                df = pd.read_csv(preset["file_path"], sep=preset["sep"])
            except Exception as e:
                print(f"Error cargando {ds_name}: {e}")
                continue

            target = preset["target_column"]
            X_raw = df.drop(columns=[target])
            y_raw = df[target].astype(str).values
            
            cat_cols = preset.get("categorical_columns", [])
            for col in X_raw.columns:
                if X_raw[col].dtype == 'object' and col not in cat_cols:
                    cat_cols.append(col)

            # --- PRE-COMPUTE FOLDS FOR THIS REP AND DATASET ---
            # Esto garantiza zero-data-leakage y elimina las 130 copias redundantes
            print(f"\n>>> [REP {rep}/{REPETITIONS}] {ds_name} | Pre-calculando {K_FOLDS} Folds (Aislamiento en Memoria)...")
            skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42 * rep)
            precomputed_splits = []
            
            for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(X_raw, y_raw)):
                X_train_val_raw = X_raw.iloc[train_val_idx]
                X_test_raw = X_raw.iloc[test_idx]
                y_train_val_raw = y_raw[train_val_idx]
                y_test_raw = y_raw[test_idx]
                
                unique_y, counts_y = np.unique(y_train_val_raw, return_counts=True)
                can_stratify = np.min(counts_y) >= 2

                X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
                    X_train_val_raw, y_train_val_raw, 
                    test_size=0.1111, random_state=42 + fold_idx, 
                    stratify=y_train_val_raw if can_stratify else None
                )

                # Inicializar y ajustar LabelEncoder solo con el entrenamiento del fold
                le = LabelEncoder()
                le.fit(y_train_raw)

                # Extraer como numpy arrays desconectados del dataframe original
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
                    # Modificación in-place sobre el slice de numpy para evitar reservar memoria extra
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
                    y_train=y_train, y_val=y_val, y_test=y_test,
                    feature_names=list(X_raw.columns),
                    class_names=[str(c) for c in le.classes_],
                    dataset_name=ds_name
                )
                precomputed_splits.append(split)

            from joblib import Parallel, delayed

            def run_single_strategy(strategy, ds_name, rep, precomputed_splits):
                """Helper function to run all folds of a strategy for a given dataset and rep."""
                # No print here to avoid mixing lines in parallel, we print after return
                fold_results = []
                
                for fold_idx, split in enumerate(precomputed_splits):
                    config = {
                        "federation": {"n_clients": N_CLIENTS, "distribution": "iid"},
                        "model": {"n_estimators": 100, "alpha": 0.1, "voting": "soft"},
                        "aggregation": {"strategy": strategy, "max_rounds": 20}
                    }
                    
                    if strategy == "local_isolation":
                        distributor = FedDataDistributor(config, use_flex_pool=False)
                        _, fed_data = distributor.distribute(split)
                        client_reports = []
                        for _, client_dataset in fed_data.items():
                            X_c, y_c = client_dataset.to_numpy()
                            model = ProactiveForest(
                                n_estimators=100, alpha=0.1, 
                                class_names=split.class_names,
                                convergence_threshold=0.002
                            )
                            model.fit(X_c, y_c)
                            preds = model.predict(split.X_test)
                            client_reports.append(ForestEvaluator.evaluate_from_predictions(
                                preds, split.y_test, split.class_names, len(model.get_trees())
                            ))
                    else:
                        if strategy.startswith("s9_"):
                            config["aggregation"]["variant"] = strategy.replace("s9_", "S9_").upper()
                            orch = RouletteOrchestrator(config)
                        elif strategy == "pw":
                            orch = ProgressiveTreeOrchestrator(config)
                        else:
                            orch = FLEXOrchestrator(config)
                        
                        orch.setup_federation(split)
                        res = orch.run_federated_round(n_bootstrap=0)
                        client_reports = list(res.client_reports.values())
                        if hasattr(orch, 'cleanup'):
                            orch.cleanup()
                    
                    fold_results.append({
                        "f1": np.mean([r.macro_f1 for r in client_reports]),
                        "acc": np.mean([r.accuracy for r in client_reports]),
                        "recall": np.mean([r.macro_recall for r in client_reports]),
                        "prec": np.mean([r.macro_precision for r in client_reports]),
                        "pcd": np.mean([r.pcd for r in client_reports])
                    })
                
                return {
                    "rep": rep,
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
                    "timestamp": datetime.now().isoformat()
                }

            # Identificar que estrategias faltan para este dataset
            pending_strategies = [s for s in STRATEGIES if (rep, s, ds_name) not in completed]
            
            if not pending_strategies:
                continue

            n_workers = -1  # Usa todos los núcleos disponibles (Ideal para Colab/PC)
            print(f"  [PARALLEL] Ejecutando {len(pending_strategies)} estrategias en {os.cpu_count() if n_workers == -1 else n_workers} hilos (Checkpoint Activo)...")
            
            # Usar return_as='generator' para escribir en el CSV segun van terminando
            results_generator = Parallel(n_jobs=n_workers, return_as="generator")(
                delayed(run_single_strategy)(strat, ds_name, rep, precomputed_splits) 
                for strat in pending_strategies
            )

            for r in results_generator:
                with open(RESULTS_FILE, "a", newline="", encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        r["rep"], r["strategy"], r["dataset"],
                        r["f1_mean"], r["f1_std"],
                        r["acc_mean"], r["acc_std"],
                        r["recall_mean"], r["recall_std"],
                        r["prec_mean"], r["prec_std"],
                        r["pcd_mean"], r["timestamp"]
                    ])
                print(f"  [CHECKPOINT] {r['strategy']} guardada | F1={r['f1_mean']:.4f} | Acc={r['acc_mean']:.4f}")

    print(f"\n=== BENCHMARK FINALIZADO. Resultados en {RESULTS_FILE} ===")

if __name__ == "__main__":
    run_final_benchmark()
