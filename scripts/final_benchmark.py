import os
import sys
import pandas as pd
import numpy as np
import csv
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder

# Añadir src al path
sys.path.append(os.getcwd())

from src.domain.dataset.base_adapter import DatasetSplit
from src.application.orchestrators.fl_orchestrator import FLEXOrchestrator
from src.application.orchestrators.roulette_orchestrator import RouletteOrchestrator
from src.application.orchestrators.progressive_tree_orchestrator import ProgressiveTreeOrchestrator
from src.interfaces.streamlit.components.constants import DATASET_PRESETS

# ===========================================================================
# CONFIGURACIÓN DEL EXPERIMENTO DEFINITIVO (NAYMA RIGOR)
# ===========================================================================
REPETITIONS = 1  # Cambiar a 5 para igualar exactamente a Nayma
K_FOLDS = 10     # Protocolo 10-Fold CV
N_CLIENTS = 3    # Configuración de clientes federados

STRATEGIES = [
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
            for strategy in STRATEGIES:
                if (rep, strategy, ds_name) in completed:
                    continue
                
                print(f"\n>>> [REP {rep}/{REPETITIONS}] {ds_name} | Estrategia: {strategy}")
                
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

                # Inicializar y ajustar LabelEncoder con todos los datos
                le = LabelEncoder()
                le.fit(y_raw)

                # SKF con semilla dependiente de la repetición para variabilidad
                skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42 * rep)
                fold_results = []
                
                for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(X_raw, y_raw)):
                    print(f"  - Fold {fold_idx+1}/{K_FOLDS}... ", end="", flush=True)
                    
                    X_train_val_raw, X_test_raw = X_raw.iloc[train_val_idx], X_raw.iloc[test_idx]
                    y_train_val_raw, y_test_raw = y_raw[train_val_idx], y_raw[test_idx]
                    
                    # Verificación de seguridad para estratificación (evitar error en clases raras)
                    unique_y, counts_y = np.unique(y_train_val_raw, return_counts=True)
                    can_stratify = np.min(counts_y) >= 2

                    from sklearn.model_selection import train_test_split
                    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
                        X_train_val_raw, y_train_val_raw, 
                        test_size=0.1111, random_state=42 + fold_idx, 
                        stratify=y_train_val_raw if can_stratify else None
                    )
                    
                    # Preprocesamiento
                    X_train, X_val, X_test = X_train_raw.copy(), X_val_raw.copy(), X_test_raw.copy()
                    if cat_cols:
                        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                        X_train[cat_cols] = enc.fit_transform(X_train[cat_cols].astype(str))
                        X_val[cat_cols] = enc.transform(X_val[cat_cols].astype(str))
                        X_test[cat_cols] = enc.transform(X_test[cat_cols].astype(str))
                    
                    scaler = StandardScaler()
                    num_cols = [c for c in X_train.columns if c not in cat_cols]
                    if num_cols:
                        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
                        X_val[num_cols] = scaler.transform(X_val[num_cols])
                        X_test[num_cols] = scaler.transform(X_test[num_cols])
                    
                    # Label Encoding (Clases) ya está pre-ajustado globalmente, solo transformamos
                    y_train = le.transform(y_train_raw)
                    y_val = le.transform(y_val_raw)
                    y_test = le.transform(y_test_raw)
                    
                    split = DatasetSplit(
                        X_train=X_train.values.astype(np.float64), 
                        X_val=X_val.values.astype(np.float64), 
                        X_test=X_test.values.astype(np.float64),
                        y_train=y_train, y_val=y_val, y_test=y_test,
                        feature_names=list(X_raw.columns),
                        class_names=[str(c) for c in le.classes_],
                        dataset_name=ds_name
                    )
                    
                    config = {
                        "federation": {"n_clients": N_CLIENTS, "distribution": "iid"},
                        "model": {"n_estimators": 50, "alpha": 0.1, "voting": "soft"},
                        "aggregation": {"strategy": strategy, "max_rounds": 10}
                    }
                    
                    if strategy.startswith("s9_"):
                        config["aggregation"]["variant"] = strategy.replace("s9_", "S9_").upper()
                        orch = RouletteOrchestrator(config)
                    elif strategy == "pw":
                        orch = ProgressiveTreeOrchestrator(config)
                    else:
                        orch = FLEXOrchestrator(config)
                    
                    orch.setup_federation(split)
                    res = orch.run_federated_round()
                    
                    # Extraer el promedio de los reportes de los clientes (Modelos Híbridos)
                    # Esto es lo correcto metodológicamente, no el modelo global del servidor.
                    client_reports = list(res.client_reports.values())
                    
                    fold_results.append({
                        "f1": np.mean([r.macro_f1 for r in client_reports]),
                        "acc": np.mean([r.accuracy for r in client_reports]),
                        "recall": np.mean([r.macro_recall for r in client_reports]),
                        "prec": np.mean([r.macro_precision for r in client_reports]),
                        "pcd": np.mean([r.pcd for r in client_reports])
                    })
                    orch.cleanup() if hasattr(orch, 'cleanup') else None
                    print(f"ok (F1: {fold_results[-1]['f1']:.4f}, PCD: {fold_results[-1]['pcd']:.4f})")

                # Consolidar Repetición
                with open(RESULTS_FILE, "a", newline="", encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        rep, strategy, ds_name,
                        np.mean([r["f1"] for r in fold_results]), np.std([r["f1"] for r in fold_results]),
                        np.mean([r["acc"] for r in fold_results]), np.std([r["acc"] for r in fold_results]),
                        np.mean([r["recall"] for r in fold_results]), np.std([r["recall"] for r in fold_results]),
                        np.mean([r["prec"] for r in fold_results]), np.std([r["prec"] for r in fold_results]),
                        np.mean([r["pcd"] for r in fold_results]), datetime.now().isoformat()
                    ])
                print(f"  [DONE] F1={np.mean([r['f1'] for r in fold_results]):.4f} | Acc={np.mean([r['acc'] for r in fold_results]):.4f}")

    print(f"\n=== BENCHMARK FINALIZADO. Resultados en {RESULTS_FILE} ===")

if __name__ == "__main__":
    run_final_benchmark()
