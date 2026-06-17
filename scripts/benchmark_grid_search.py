import json
import logging
import os
import sys
import time
sys.path.append('.')

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pandas.api.types import is_string_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

from src.application.orchestrators import FLEXOrchestrator
from src.application.orchestrators.roulette_orchestrator import RouletteOrchestrator
from src.domain.dataset.base_adapter import DatasetSplit
from src.infrastructure.dataset.dataset_factory import DATASET_METADATA

# Configuración básica de logging para consola y archivo
os.makedirs("results/logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/logs/benchmark_grid_search_run.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("BenchmarkGridSearch")

# ==========================================================================
# CONFIGURACIÓN DEL EXPERIMENTO
# ==========================================================================
N_CLIENTS = 3
TEST_SIZE = 0.1
VAL_SIZE_RELATIVE = 0.1111  # 10% del total

DATASETS = ["Iris", "Car", "Nursery", "Vowel", "Optdigits", "Sonar", "Spambase"]
STRATEGIES = ["s1_simple_pool", "s4_global_f1_pcd", "s7_perclient_f1_pcd", "s8_simple_mean"]

GRID = {
    "s1_simple_pool": {
        "local_weight": [0.4],
        "max_trees": [110, 130],
        "global_convergence_threshold": [0.002],
        "local_roulette_weight": [0.0],
    },
    "s4_global_f1_pcd": {
        "local_weight": [0.4, 0.5],
        "max_trees": [110, 130],
        "global_convergence_threshold": [0.001, 0.002],
        "local_roulette_weight": [0.0],
    },
    "s7_perclient_f1_pcd": {
        "local_weight": [0.4, 0.5],
        "max_trees": [110, 130],
        "global_convergence_threshold": [0.001, 0.002],
        "local_roulette_weight": [0.0],
    },
    "s8_simple_mean": {
        "local_weight": [0.4, 0.5],
        "max_trees": [150],  # Fijo por ventana (10) * max_rounds (15)
        "global_convergence_threshold": [0.002],
        "local_roulette_weight": [0.6, 0.7],
    }
}

# Habilitar modo prueba si se pasa el flag --test
if len(sys.argv) > 1 and sys.argv[1] == "--test":
    logger.info("MODO PRUEBA ACTIVADO: Limitando la ejecución a 1 dataset, 1 estrategia y 1 combinación.")
    DATASETS = ["Iris"]
    STRATEGIES = ["s1_simple_pool"]
    GRID = {
        "s1_simple_pool": {
            "local_weight": [0.4],
            "max_trees": [110],
            "global_convergence_threshold": [0.002],
            "local_roulette_weight": [0.0],
        }
    }

RESULTS_JSON = "results/grid_search_results.json"


def get_config_key(local_weight: float, max_trees: int, global_conv: float, local_roulette_weight: float) -> str:
    """Genera una clave única legible para identificar una combinación de parámetros."""
    return f"lw{local_weight:.1f}_mt{max_trees}_gc{global_conv}_f10.5_lrw{local_roulette_weight:.1f}"


def get_strategy_config(
    strategy: str,
    local_weight: float,
    max_trees: int,
    global_conv: float,
    local_roulette_weight: float,
) -> Dict[str, Any]:
    """Genera la configuración estructurada del orquestador para una combinación dada."""
    from src.domain.aggregation.aggregation_factory import AggregationFactory
    norm_strat = AggregationFactory.normalize_strategy_name(strategy)

    # Valores fijos optimizados basados en HPO
    global_ep_size = 10
    trees_per_client_ep = 3
    trees_per_rnd_client = 3
    win_size = 10
    max_rounds = 15
    f1_w = 0.5
    pcd_w = 0.5
    min_ep = 4
    min_rounds = 4

    config = {
        "federation": {
            "n_clients": N_CLIENTS,
            "distribution": "iid"
        },
        "model": {
            "n_estimators": max_trees,
            "alpha": 0.1,
            "voting": "soft",
            "local_convergence_threshold": 0.002  # Fijo por HPO
        },
        "aggregation": {
            "strategy": norm_strat,
            "max_rounds": max_rounds,
            "global_convergence_threshold": global_conv,
            "global_episode_size": global_ep_size,
            "trees_per_client_per_episode": trees_per_client_ep,
            "trees_per_round_per_client": trees_per_rnd_client,
            "min_episodes": min_ep,
            "min_rounds": min_rounds,
            "window_size": win_size,
            "f1_weight": f1_w,
            "pcd_weight": pcd_w,
        },
        "prediction": {
            "local_weight": local_weight,
            "use_weighted": True  # Fijo por HPO
        }
    }

    if strategy.startswith("s8_"):
        variant = strategy.replace("s8_", "S8_").upper()
        config["aggregation"]["variant"] = variant
        config["aggregation"]["local_roulette_weight"] = local_roulette_weight
        config["model"]["n_estimators"] = max_rounds * win_size

    return config


def load_json_results() -> Dict[str, Any]:
    """Carga los resultados previos desde el archivo JSON si existe."""
    if os.path.exists(RESULTS_JSON):
        try:
            with open(RESULTS_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"No se pudo leer {RESULTS_JSON}. Se creará uno nuevo. Error: {e}")
    
    return {
        "timestamp_start": datetime.now().isoformat(),
        "experiment_metadata": {
            "n_clients": N_CLIENTS,
            "test_size_percent": 10,
            "val_size_percent": 10,
            "train_size_percent": 80,
            "cross_validation": False,
            "pruned_grid": True
        },
        "results": {}
    }


def save_json_results(data: Dict[str, Any]) -> None:
    """Guarda los resultados de forma segura en formato JSON."""
    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def compile_tasks(json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compila la lista de tareas pendientes cruzando Datasets × Estrategias × Parámetros."""
    tasks = []
    for ds_name in DATASETS:
        for strategy in STRATEGIES:
            grid_opts = GRID[strategy]
            for lw in grid_opts["local_weight"]:
                for mt in grid_opts["max_trees"]:
                    for gc in grid_opts["global_convergence_threshold"]:
                        for lrw in grid_opts["local_roulette_weight"]:
                            config_key = get_config_key(lw, mt, gc, lrw)
                            
                            # Validar si ya existe en checkpoint
                            ds_res = json_data["results"].get(ds_name, {})
                            strat_res = ds_res.get(strategy, {})
                            combo_res = strat_res.get(config_key, {})
                            
                            if combo_res.get("success", False):
                                continue
                                
                            tasks.append({
                                "dataset": ds_name,
                                "strategy": strategy,
                                "local_weight": lw,
                                "max_trees": mt,
                                "global_conv": gc,
                                "local_roulette_weight": lrw,
                                "config_key": config_key
                            })
    return tasks


def run_single_task(task: Dict[str, Any], splits: Dict[str, DatasetSplit]) -> Dict[str, Any]:
    """Ejecuta un solo ensayo de combinación de parámetros y devuelve los resultados."""
    start_time = time.time()
    ds_name = task["dataset"]
    strategy = task["strategy"]
    lw = task["local_weight"]
    mt = task["max_trees"]
    gc = task["global_conv"]
    lrw = task["local_roulette_weight"]
    config_key = task["config_key"]
    split = splits[ds_name]
    
    config = get_strategy_config(strategy, lw, mt, gc, lrw)
    msg_prefix = f"[{ds_name} | {strategy} | {config_key}]"
    logger.info(f"{msg_prefix} Iniciando ejecución...")

    if strategy.startswith("s8_"):
        orch = RouletteOrchestrator(config)
    else:
        orch = FLEXOrchestrator(config)

    try:
        orch.setup_federation(split)
        res = orch.run_federated_round(n_bootstrap=0)
        
        # Calcular métricas promedio de los bosques híbridos de los clientes
        f1_mean = res.hybrid_f1_mean
        acc_mean = res.hybrid_accuracy_mean
        pcd_mean = res.hybrid_pcd_mean
        avg_trees = float(np.mean([r.forest_size for r in res.client_reports.values()]))
        
        duration = time.time() - start_time
        logger.info(
            f"{msg_prefix} Finalizado en {duration:.2f}s | "
            f"F1={f1_mean:.4f} | Acc={acc_mean:.4f} | "
            f"PCD={pcd_mean:.4f} | Árboles Promedio={avg_trees:.2f}"
        )

        return {
            "success": True,
            "task": task,
            "metrics": {
                "f1": f1_mean,
                "accuracy": acc_mean,
                "pcd": pcd_mean,
                "forest_size": avg_trees
            },
            "convergence_round": res.convergence_round,
            "duration_seconds": duration,
            "error": None
        }
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"{msg_prefix} Error durante la ejecución: {e}", exc_info=True)
        return {
            "success": False,
            "task": task,
            "metrics": None,
            "convergence_round": None,
            "duration_seconds": duration,
            "error": str(e)
        }
    finally:
        if hasattr(orch, "cleanup"):
            orch.cleanup()


def load_and_preprocess_datasets() -> Dict[str, DatasetSplit]:
    """Carga y preprocesa de forma secuencial todos los datasets a evaluar."""
    splits = {}
    for ds_name in DATASETS:
        logger.info(f"Cargando y preprocesando dataset: {ds_name}...")
        preset = DATASET_METADATA[ds_name.lower()]
        
        try:
            df = pd.read_csv(preset["file_path"], sep=preset["sep"])
        except Exception as e:
            logger.error(f"Error cargando el dataset {ds_name}: {e}")
            continue

        # Limpiar columnas de metadatos no deseadas
        cols_to_drop = preset.get("columns_to_drop", [])
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

        target = preset["target_column"]
        X_raw = df.drop(columns=[target])
        y_raw = np.array(df[target].astype(str).values, dtype=object)

        # Detectar columnas categóricas
        cat_cols = list(preset.get("categorical_features", []))
        for col in X_raw.columns:
            is_obj = X_raw[col].dtype == "object"
            is_str = is_string_dtype(X_raw[col])
            if (is_obj or is_str) and col not in cat_cols:
                cat_cols.append(col)

        # Preparar encoder de labels
        le = LabelEncoder()
        le.fit(y_raw)
        
        stratify_values = y_raw if len(np.unique(y_raw)) > 1 else None

        # División 80% Train, 10% Val, 10% Test
        X_raw_val = np.array(X_raw.values, dtype=object)
        X_train_val_raw, X_test_raw, y_train_val_raw, y_test_raw = train_test_split(
            X_raw_val,
            y_raw,
            test_size=TEST_SIZE,
            random_state=42,
            stratify=stratify_values,
        )

        can_stratify = len(np.unique(y_train_val_raw)) > 1
        X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
            X_train_val_raw,
            y_train_val_raw,
            test_size=VAL_SIZE_RELATIVE,
            random_state=43,
            stratify=y_train_val_raw if can_stratify else None,
        )

        X_train = X_train_raw.copy()
        X_val = X_val_raw.copy()
        X_test = X_test_raw.copy()

        # Codificar variables categóricas
        cat_cols_idx = [X_raw.columns.get_loc(c) for c in cat_cols] if cat_cols else []
        num_cols_idx = [X_raw.columns.get_loc(c) for c in X_raw.columns if c not in cat_cols]

        if cat_cols_idx:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_train[:, cat_cols_idx] = enc.fit_transform(X_train[:, cat_cols_idx].astype(str))
            X_val[:, cat_cols_idx] = enc.transform(X_val[:, cat_cols_idx].astype(str))
            X_test[:, cat_cols_idx] = enc.transform(X_test[:, cat_cols_idx].astype(str))

        # Estandarizar variables numéricas
        if num_cols_idx:
            scaler = StandardScaler()
            X_train[:, num_cols_idx] = scaler.fit_transform(X_train[:, num_cols_idx])
            X_val[:, num_cols_idx] = scaler.transform(X_val[:, num_cols_idx])
            X_test[:, num_cols_idx] = scaler.transform(X_test[:, num_cols_idx])

        y_train = le.transform(y_train_raw)
        y_val = le.transform(y_val_raw)
        y_test = le.transform(y_test_raw)

        # Construir DatasetSplit
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
        splits[ds_name] = split
        logger.info(f"Dataset {ds_name} preprocesado: Train={X_train.shape[0]} | Val={X_val.shape[0]} | Test={X_test.shape[0]}")
    return splits


def run_grid_search() -> None:
    logger.info("=== BENCHMARK DE GRID SEARCH DE HIPERPARÁMETROS: INICIANDO EXPERIMENTO ===")
    logger.info(f"Datasets a evaluar: {DATASETS}")
    logger.info(f"Estrategias: {STRATEGIES}")
    
    # Cargar base de datos de resultados existentes (soporte de checkpoints)
    json_data = load_json_results()
    
    # Compilar lista de tareas pendientes
    pending_tasks = compile_tasks(json_data)
    logger.info(f"Tareas totales pendientes: {len(pending_tasks)}")
    
    if not pending_tasks:
        logger.info("Todas las combinaciones del grid search ya han sido completadas con éxito.")
        return

    # Cargar y preprocesar todos los datasets necesarios una sola vez en el hilo principal
    splits = load_and_preprocess_datasets()

    n_workers = 2
    logger.info(f"Ejecutando {len(pending_tasks)} tareas pendientes en paralelo con {n_workers} hilos...")

    # Generador paralelo de tareas
    results_generator = Parallel(n_jobs=n_workers, return_as="generator")(
        delayed(run_single_task)(task, splits)
        for task in pending_tasks
    )

    # Actualizar checkpoints progresivamente a medida que cada tarea se completa
    for r in results_generator:
        task = r["task"]
        ds_name = task["dataset"]
        strategy = task["strategy"]
        config_key = task["config_key"]
        
        # Estructurar almacenamiento en el diccionario JSON
        ds_res = json_data["results"].setdefault(ds_name, {})
        strat_res = ds_res.setdefault(strategy, {})
        
        strat_res[config_key] = {
            "success": r["success"],
            "metrics": r["metrics"],
            "convergence_round": r["convergence_round"],
            "duration_seconds": round(r["duration_seconds"], 2),
            "error": r["error"],
            "timestamp": datetime.now().isoformat()
        }
        
        # Persistir a disco de forma inmediata
        save_json_results(json_data)
        
        if r["success"]:
            logger.info(
                f"[CHECKPOINT GUARDADO] {ds_name} | {strategy} | {config_key} | "
                f"Acc={r['metrics']['accuracy']:.4f} | F1={r['metrics']['f1']:.4f}"
            )
        else:
            logger.error(f"[CHECKPOINT ERROR] {ds_name} | {strategy} | {config_key} falló: {r['error']}")

    logger.info("=== EXPERIMENTO DE GRID SEARCH FINALIZADO CON ÉXITO ===")
    logger.info(f"Resultados persistidos en: {RESULTS_JSON}")


if __name__ == "__main__":
    run_grid_search()
