import os
import csv
import sys
import time
import numpy as np
from pathlib import Path

# Agregar el directorio raíz al path para importar src
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.application.orchestrators.fl_orchestrator import FLEXOrchestrator
from src.application.orchestrators.roulette_orchestrator import RouletteOrchestrator
from src.infrastructure.dataset.dataset_factory import DatasetFactory

# --- CONFIGURACIÓN MAESTRA ---
N_CLIENTS = 3
DISTRIBUTION = "iid"
SEED = 42
HYPERPARAMS = {
    "model": {
        "n_estimators": 100,
        "alpha": 0.1,
        "max_depth": None,  # Basado en la tesis de Nayma
        "split_criterion": "entropy",
        "t_max": 100
    },
    "prediction": {
        "local_weight": 0.3,
        "global_weight": 0.7,
        "use_weighted": True
    },
    "federation": {
        "n_clients": N_CLIENTS,
        "distribution": DISTRIBUTION,
        "seed": SEED
    }
}

STRATEGIES = [
    "S1", "S2", "S3", "S4", "S5", "S6", "S7", "PW",
    "S9_WEIGHTED", "S9_MEAN", "S9_MEDIAN", "S9_CONSENSUS", "S9_PROACTIVE_PCD"
]

DATASETS = ["iris", "car", "nursery", "vowel", "letter", "optdigits", "sonar", "spambase"]

RESULTS_FILE = "results_master.csv"

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

def main():
    print(f"=== INICIANDO EXPERIMENTO MAESTRO (Nayma Comparison) ===")
    completed = get_completed_experiments()
    
    # Escribir cabecera si el archivo es nuevo
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["strategy", "dataset", "avg_f1", "avg_accuracy", "avg_pcd", "n_clients", "timestamp"])

    for strategy in STRATEGIES:
        for ds_name in DATASETS:
            if (strategy, ds_name) in completed:
                print(f"[SKIP] {strategy} en {ds_name} ya completado.")
                continue
            
            print(f"\n>>> Ejecutando {strategy} en {ds_name}...")
            
            try:
                # 1. Configuración dinámica
                cfg = {
                    "model": HYPERPARAMS["model"].copy(),
                    "prediction": HYPERPARAMS["prediction"].copy(),
                    "federation": HYPERPARAMS["federation"].copy(),
                    "aggregation": {"strategy": strategy},
                    "dataset": {"type": ds_name},
                    "seed": SEED,
                    "verbose": False
                }
                
                # S4/S7/PW requieren pesos internos de agregación (F1 vs PCD)
                if strategy in ["S4", "S7", "PW"]:
                    cfg["aggregation"]["f1_weight"] = 0.5
                    cfg["aggregation"]["pcd_weight"] = 0.5

                # 2. Cargar Dataset
                ds = DatasetFactory.load_from_config(cfg["dataset"])
                
                # 3. Selección de Orquestador (S9 requiere RouletteOrchestrator)
                if strategy.startswith("S9"):
                    cfg["aggregation"]["variant"] = strategy
                    cfg["aggregation"]["beta"] = 0.0  # Adopción global por defecto
                    cfg["aggregation"]["max_rounds"] = 20
                    cfg["aggregation"]["window_size"] = 5
                    orch = RouletteOrchestrator(cfg)
                else:
                    orch = FLEXOrchestrator(cfg)
                
                orch.setup_federation(ds)
                results = orch.run_federated_round()
                
                # 4. Extraer métricas atómicas (Promedio de clientes)
                c_ids = [str(cid) for cid in results.client_ids]
                c_f1s = [results.client_f1_scores[cid] for cid in c_ids]
                c_accs = [results.client_accuracies[cid] for cid in c_ids]
                
                # Extraer PCD manejando tanto objetos como diccionarios
                c_pcds = []
                for cid in c_ids:
                    meta = results.client_metadata.get(cid)
                    if meta is None:
                        c_pcds.append(0.0)
                    elif isinstance(meta, dict):
                        c_pcds.append(meta.get("pcd", 0.0))
                    else:
                        c_pcds.append(getattr(meta, "pcd", 0.0))
                
                avg_f1 = np.mean(c_f1s)
                avg_acc = np.mean(c_accs)
                avg_pcd = np.mean(c_pcds) if c_pcds else 0.0
                
                # 5. Guardar inmediatamente
                with open(RESULTS_FILE, "a", newline="", encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([strategy, ds_name, f"{avg_f1:.4f}", f"{avg_acc:.4f}", f"{avg_pcd:.4f}", N_CLIENTS, time.strftime("%Y-%m-%d %H:%M:%S")])
                
                print(f"  [DONE] F1={avg_f1:.4f} | Acc={avg_acc:.4f} | PCD={avg_pcd:.4f}")
                
            except Exception as e:
                print(f"  [ERROR] Falló {strategy} en {ds_name}: {e}")
                continue

    print(f"\n=== EXPERIMENTO FINALIZADO. Resultados en {RESULTS_FILE} ===")

if __name__ == "__main__":
    main()
