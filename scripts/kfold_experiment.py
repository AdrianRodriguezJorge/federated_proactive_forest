import os
import sys
import pandas as pd
import numpy as np
import csv
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder

# Añadir src al path para que las importaciones funcionen correctamente
sys.path.append(os.getcwd())

from src.domain.dataset.base_adapter import DatasetSplit
from src.application.orchestrators.fl_orchestrator import FLEXOrchestrator
from src.application.orchestrators.roulette_orchestrator import RouletteOrchestrator
from src.interfaces.streamlit.components.constants import DATASET_PRESETS

# Configuración del experimento (Protocolo de Rigor Científico Completo)
STRATEGIES = [
    "s1_simple_pool", "s2_global_accuracy", "s3_global_f1", "s4_global_f1_pcd", 
    "s5_perclient_accuracy", "s6_perclient_f1", "s7_perclient_f1_pcd", "pw", 
    "s9_weighted_average", "s9_simple_mean", "s9_median", "s9_consensus", "s9_proactive_pcd"
]
DATASETS = ["Iris", "Nursery"]
K_FOLDS = 10
RESULTS_FILE = "results_kfold.csv"

RESULTS_FILE = "results_kfold.csv"

def get_completed_experiments():
    completed = set()
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r", encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    completed.add((row["strategy"], row["dataset"]))
        except Exception:
            pass
    return completed

def run_kfold_experiment():
    print(f"=== INICIANDO PROTOCOLO 10-FOLD CV (80/10/10) ===")
    print(f"Estrategias: {len(STRATEGIES)} | Datasets: {len(DATASETS)} | Folds: {K_FOLDS}")
    
    completed = get_completed_experiments()
    
    # Crear cabecera si el archivo no existe
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["strategy", "dataset", "f1_mean", "f1_std", "acc_mean", "acc_std", "pcd_mean", "timestamp"])

    for ds_name in DATASETS:
        preset = DATASET_PRESETS[ds_name]
        print(f"\n>>> Procesando Dataset: {ds_name}")
        
        # Cargar datos base desde el CSV
        try:
            df = pd.read_csv(preset["file_path"], sep=preset["sep"])
        except Exception as e:
            print(f"Error cargando {ds_name}: {e}")
            continue

        target = preset["target_column"]
        X_raw = df.drop(columns=[target])
        y_raw = df[target].astype(str).values
        
        # Identificar columnas categóricas para el OrdinalEncoder
        cat_cols = preset.get("categorical_columns", [])
        for col in X_raw.columns:
            if X_raw[col].dtype == 'object' and col not in cat_cols:
                cat_cols.append(col)

        # Inicializar Cross-Validation Estratificado
        skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
        
        for strategy in STRATEGIES:
            if (strategy, ds_name) in completed:
                print(f"  > Saltando {strategy} para {ds_name} (ya completado).")
                continue
                
            print(f"  > Evaluando Estrategia: {strategy}")
            fold_results = []
            
            # Bucle de Folds (10 iteraciones por cada estrategia/dataset)
            for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(X_raw, y_raw)):
                # 1. Separar Test (10% - 1 pliegue)
                X_train_val_raw, X_test_raw = X_raw.iloc[train_val_idx], X_raw.iloc[test_idx]
                y_train_val_raw, y_test_raw = y_raw[train_val_idx], y_raw[test_idx]
                
                # 2. Separar Validación (10% del total -> ~11.11% de los datos restantes)
                # Verificación de seguridad para estratificación (evitar error en clases raras)
                unique_y, counts_y = np.unique(y_train_val_raw, return_counts=True)
                can_stratify = np.min(counts_y) >= 2

                from sklearn.model_selection import train_test_split
                X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
                    X_train_val_raw, y_train_val_raw, 
                    test_size=0.1111, 
                    random_state=42 + fold_idx,
                    stratify=y_train_val_raw if can_stratify else None
                )
                
                # 3. Preprocesamiento Atómico (Fit SOLO en Train para evitar Leakage)
                X_train, X_val, X_test = X_train_raw.copy(), X_val_raw.copy(), X_test_raw.copy()
                
                # Codificación Categórica
                if cat_cols:
                    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                    X_train[cat_cols] = enc.fit_transform(X_train[cat_cols].astype(str))
                    X_val[cat_cols] = enc.transform(X_val[cat_cols].astype(str))
                    X_test[cat_cols] = enc.transform(X_test[cat_cols].astype(str))
                
                # Escalado Numérico
                scaler = StandardScaler()
                num_cols = [c for c in X_train.columns if c not in cat_cols]
                if num_cols:
                    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
                    X_val[num_cols] = scaler.transform(X_val[num_cols])
                    X_test[num_cols] = scaler.transform(X_test[num_cols])
                
                # Label Encoding (Clases)
                le = LabelEncoder()
                y_train = le.fit_transform(y_train_raw)
                y_val = le.transform(y_val_raw)
                y_test = le.transform(y_test_raw)
                
                # 4. Crear DatasetSplit del Dominio
                split = DatasetSplit(
                    X_train=X_train.values.astype(np.float64), 
                    X_val=X_val.values.astype(np.float64), 
                    X_test=X_test.values.astype(np.float64),
                    y_train=y_train, y_val=y_val, y_test=y_test,
                    feature_names=list(X_raw.columns),
                    class_names=[str(c) for c in le.classes_],
                    dataset_name=ds_name
                )
                
                # 5. Configuración de la Federación
                config = {
                    "federation": {"n_clients": 3, "distribution": "iid"},
                    "model": {"n_estimators": 50, "alpha": 0.1, "voting": "soft"},
                    "aggregation": {"strategy": strategy, "max_rounds": 10}
                }
                
                # 6. Selección de Orquestador (S9 usa uno especial)
                if strategy.startswith("s9_"):
                    variant = strategy.replace("s9_", "S9_").upper()
                    config["aggregation"]["variant"] = variant
                    orch = RouletteOrchestrator(config)
                else:
                    orch = FLEXOrchestrator(config)
                
                # Ejecutar Ronda Federada
                orch.setup_federation(split)
                results = orch.run_federated_round()
                
                fold_results.append({
                    "f1": results.global_macro_f1,
                    "acc": results.global_accuracy,
                    "pcd": results.global_report.pcd if results.global_report else 0.0
                })
                orch.cleanup() if hasattr(orch, 'cleanup') else None
            
            # Consolidar métricas del Fold: Media y Desviación Estándar
            f1_vals = [r["f1"] for r in fold_results]
            acc_vals = [r["acc"] for r in fold_results]
            pcd_vals = [r["pcd"] for r in fold_results]
            
            with open(RESULTS_FILE, "a", newline="", encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    strategy, ds_name,
                    np.mean(f1_vals), np.std(f1_vals),
                    np.mean(acc_vals), np.std(acc_vals),
                    np.mean(pcd_vals), datetime.now().isoformat()
                ])
            print(f"  [RESULTADO] {strategy}: F1 = {np.mean(f1_vals):.4f} (+/- {np.std(f1_vals):.4f})")

    print(f"\n=== EXPERIMENTO FINALIZADO. Resultados guardados en {RESULTS_FILE} ===")

if __name__ == "__main__":
    run_kfold_experiment()
