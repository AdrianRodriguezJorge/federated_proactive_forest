import os
import sys
import pandas as pd
import numpy as np
import csv
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Añadir src al path
sys.path.append(os.getcwd())

from src.domain.dataset.base_adapter import DatasetSplit
from src.application.orchestrators.roulette_orchestrator import RouletteOrchestrator
from src.interfaces.streamlit.components.constants import DATASET_PRESETS

# ===========================================================================
# CONFIGURACIÓN DEL EXPERIMENTO (CONTROLADO)
# ===========================================================================
K_FOLDS = 10
N_CLIENTS = 3    # Ajustado a 3 clientes según petición
DATASET_NAME = "Optdigits" 
RESULTS_FILE = "results/results_local_vs_s9.csv"

def get_completed_folds():
    """Identifica qué combinaciones de (fold, estrategia) ya terminaron."""
    completed = set()
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r", encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    completed.add((int(row["fold"]), row["strategy"]))
        except Exception: pass
    return completed

def run_local_vs_s9_robust():
    print(f"=== EXPERIMENTO RIGUROSO: LOCAL VS S9 FEDERATED ===")
    print(f"Dataset: {DATASET_NAME} | Clientes: {N_CLIENTS} | Folds: {K_FOLDS}")
    
    # 1. Cargar y preparar datos (Lógica idéntica al Benchmark Principal)
    preset = DATASET_PRESETS[DATASET_NAME]
    try:
        df = pd.read_csv(preset["file_path"], sep=preset["sep"])
    except Exception as e:
        print(f"Error cargando datos: {e}")
        return

    target = preset["target_column"]
    X_raw = df.drop(columns=[target])
    y_raw = df[target].astype(str).values
    
    # Inicializar archivo de resultados si no existe
    if not os.path.exists(RESULTS_FILE):
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        with open(RESULTS_FILE, "w", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["fold", "strategy", "dataset", "f1", "acc", "timestamp"])

    completed = get_completed_folds()
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    # 2. Bucle de Folds
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_raw, y_raw)):
        X_train_raw = X_raw.iloc[train_idx]
        X_test_raw = X_raw.iloc[test_idx]
        y_train_raw = y_raw[train_idx]
        y_test_raw = y_raw[test_idx]
        
        # Simular split para el orquestador
        ds_split = DatasetSplit(
            X_train_raw.values, X_test_raw.values, 
            y_train_raw, y_test_raw,
            feature_names=X_raw.columns.tolist(),
            class_names=np.unique(y_raw).tolist(),
            dataset_name=DATASET_NAME
        )
        
        # Definición de estrategias
        # beta=1.0 -> Local (ignore server)
        # beta=0.0 -> Federated (standard S9)
        configs = [
            ("local_proactive", 1.0, "S9_SIMPLE_MEAN"),
            ("s9_federated_consensus", 0.0, "S9_CONSENSUS")
        ]
        
        for strat_name, beta_val, variant in configs:
            if (fold_idx, strat_name) in completed:
                print(f"--- Skip: Fold {fold_idx} | {strat_name} (Ya completado) ---")
                continue
                
            print(f"\n>>> [FOLD {fold_idx+1}/{K_FOLDS}] Estrategia: {strat_name}...")
            
            config = {
                'n_clients': N_CLIENTS,
                'model': {
                    'n_estimators': 100, 
                    'alpha': 0.1,
                    'local_convergence_threshold': 0.0 # Desactivar para asegurar 100 árboles
                },
                'aggregation': {
                    'variant': variant, 
                    'beta': beta_val, 
                    'max_rounds': 20, 
                    'window_size': 5
                }
            }
            
            try:
                orch = RouletteOrchestrator(config)
                orch.setup_federation(ds_split)
                res = orch.run_federated_round()
                
                # Métrica: Si es local, promediamos el rendimiento de los 3 clientes individuales
                # Si es federado, usamos la métrica global del consenso
                if beta_val == 1.0:
                    f1 = np.mean(list(res.client_f1_scores.values()))
                    acc = np.mean(list(res.client_accuracies.values()))
                else:
                    f1 = res.global_macro_f1
                    acc = res.global_accuracy
                
                # Guardar checkpoint
                with open(RESULTS_FILE, "a", newline="", encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([fold_idx, strat_name, DATASET_NAME, f1, acc, datetime.now().isoformat()])
                
                print(f"    Resultado Guardado: F1={f1:.4f}")
                orch.cleanup()
                
            except Exception as e:
                print(f"    Error en ejecución: {e}")

    print(f"\n=== EXPERIMENTO FINALIZADO: Resultados en {RESULTS_FILE} ===")

if __name__ == "__main__":
    run_local_vs_s9_robust()
