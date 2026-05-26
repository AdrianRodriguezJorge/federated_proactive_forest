"""Benchmark de un solo fold con hiperparámetros comparables y guardado en JSON.

Ejecuta las 13 estrategias federadas en 7 datasets (excluyendo Letter)
con una partición de 80% train, 10% server validation y 10% test.
Utiliza 2 hilos de ejecución paralela y muestra un log detallado.
Los resultados se guardan en results/benchmark_single_fold_results.json.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
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
from src.application.orchestrators.roulette_orchestrator import (
    RouletteOrchestrator,
)
from src.domain.dataset.base_adapter import DatasetSplit
from src.infrastructure.dataset.dataset_factory import DATASET_METADATA

# Configuración básica de logging para consola y archivo
os.makedirs("results/logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/logs/benchmark_single_fold_run.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("BenchmarkSingleFold")

# ==========================================================================
# CONFIGURACIÓN DEL EXPERIMENTO
# ==========================================================================
N_CLIENTS = 3
TEST_SIZE = 0.1
VAL_SIZE_RELATIVE = 0.1111  # 10% del total (0.1111 * 90% restante)

STRATEGIES = [
    "s1_simple_pool",
    "s2_global_accuracy",
    "s3_global_f1",
    "s4_global_f1_pcd",
    "s5_perclient_accuracy",
    "s6_perclient_f1",
    "s7_perclient_f1_pcd",
    "pw",
    "s8_weighted_average",
    "s8_simple_mean",
    "s8_median",
    "s8_consensus",
    "s8_proactive_pcd",
]

DATASETS = [
    "Iris",
    "Car",
    "Nursery",
    "Vowel",
    "Optdigits",
    "Sonar",
    "Spambase",
]

RESULTS_JSON = "results/benchmark_single_fold_results.json"


def get_strategy_config(strategy: str) -> Dict[str, Any]:
    """Genera la configuración de hiperparámetros para una estrategia específica.

    Garantiza que sean directamente comparables entre sí.
    """
    from src.domain.aggregation.aggregation_factory import AggregationFactory
    norm_strat = AggregationFactory.normalize_strategy_name(strategy)

    # Valores base por defecto
    global_ep_size = 3
    trees_per_client_ep = 1
    trees_per_rnd_client = 1
    win_size = 5
    f1_w = 0.3
    pcd_w = 0.7
    min_ep = 4
    min_rnd = 4
    conv_thresh = 0.002

    config = {
        "federation": {
            "n_clients": N_CLIENTS,
            "distribution": "iid"
        },
        "model": {
            "n_estimators": 100,
            "alpha": 0.1,
            "voting": "soft",
            "local_convergence_threshold": conv_thresh
        },
        "aggregation": {
            "strategy": strategy,
            "max_rounds": 20,
            "global_convergence_threshold": conv_thresh,
            "global_episode_size": global_ep_size,
            "trees_per_client_per_episode": trees_per_client_ep,
            "trees_per_round_per_client": trees_per_rnd_client,
            "min_episodes": min_ep,
            "min_rounds": min_rnd,
            "window_size": win_size,
            "f1_weight": f1_w,
            "pcd_weight": pcd_w,
        },
        "prediction": {
            "local_weight": 0.4,
            "use_weighted": True
        }
    }

    if strategy.startswith("s8_"):
        variant = strategy.replace("s8_", "S8_").upper()
        config["aggregation"]["variant"] = variant
        config["aggregation"]["local_roulette_weight"] = 0.1

    return config


def load_json_results() -> Dict[str, Any]:
    """Carga los resultados previos desde el archivo JSON si existe."""
    if os.path.exists(RESULTS_JSON):
        try:
            with open(RESULTS_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"No se pudo leer {RESULTS_JSON}. Se creará uno nuevo. Error: {e}")
    
    # Estructura inicial del JSON
    hyperparams_summary = {strat: get_strategy_config(strat) for strat in STRATEGIES}
    return {
        "timestamp": datetime.now().isoformat(),
        "experiment_metadata": {
            "n_clients": N_CLIENTS,
            "test_size_percent": 10,
            "val_size_percent": 10,
            "train_size_percent": 80,
            "cross_validation": False
        },
        "hyperparameters": hyperparams_summary,
        "results": {}
    }


def save_json_results(data: Dict[str, Any]) -> None:
    """Guarda los resultados estructurados en formato JSON."""
    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_single_strategy(
    strategy: str,
    ds_name: str,
    split: DatasetSplit,
) -> Dict[str, Any]:
    """Ejecuta una estrategia individual en un dataset y devuelve las métricas."""
    start_time = time.time()
    config = get_strategy_config(strategy)
    
    msg_prefix = f"[{ds_name} | {strategy}]"
    logger.info(f"{msg_prefix} Iniciando ejecución...")
    logger.info(f"{msg_prefix} Configuración utilizada: {json.dumps(config['aggregation'])}")

    if strategy.startswith("s8_"):
        orch = RouletteOrchestrator(config)
    else:
        orch = FLEXOrchestrator(config)

    try:
        logger.info(f"{msg_prefix} Configurando federación...")
        orch.setup_federation(split)
        
        logger.info(f"{msg_prefix} Ejecutando rondas federadas...")
        res = orch.run_federated_round(n_bootstrap=0)
        
        # Calcular métricas promedio de los bosques híbridos de los clientes (excluyendo servidor)
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
            "strategy": strategy,
            "dataset": ds_name,
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
            "strategy": strategy,
            "dataset": ds_name,
            "metrics": None,
            "convergence_round": None,
            "duration_seconds": duration,
            "error": str(e)
        }
    finally:
        if hasattr(orch, "cleanup"):
            orch.cleanup()


def run_benchmark() -> None:
    logger.info("=== BENCHMARK DE UN SOLO FOLD: INICIANDO EXPERIMENTO ===")
    logger.info(f"Datasets: {DATASETS}")
    logger.info(f"Estrategias (13): {STRATEGIES}")
    
    # Cargar base de datos de resultados existentes (para soporte de reanudación)
    json_data = load_json_results()
    
    for ds_name in DATASETS:
        logger.info(f"\n>>> Procesando dataset: {ds_name} <<<")
        
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
        y_raw = df[target].astype(str).values

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

        # División 80% Entrenamiento, 10% Server Validación, 10% Test
        # 1. Separar 10% de Test del 90% restante (Train + Val)
        X_train_val_raw, X_test_raw, y_train_val_raw, y_test_raw = train_test_split(
            X_raw,
            y_raw,
            test_size=TEST_SIZE,
            random_state=42,
            stratify=stratify_values,
        )

        # 2. Del 90% restante, separar el 11.11% para validación en el servidor
        # (0.1111 * 0.9 = 0.1 de los datos originales)
        can_stratify = len(np.unique(y_train_val_raw)) > 1
        X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
            X_train_val_raw,
            y_train_val_raw,
            test_size=VAL_SIZE_RELATIVE,
            random_state=43,
            stratify=y_train_val_raw if can_stratify else None,
        )

        X_train = X_train_raw.values.copy()
        X_val = X_val_raw.values.copy()
        X_test = X_test_raw.values.copy()

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

        logger.info(
            f"Dataset {ds_name} estructurado correctamente. "
            f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}"
        )

        # Filtrar las estrategias que ya se ejecutaron para este dataset
        ds_results = json_data["results"].setdefault(ds_name, {})
        pending_strategies = [
            s for s in STRATEGIES if s not in ds_results or not ds_results[s].get("success", False)
        ]

        if not pending_strategies:
            logger.info(f"Todas las estrategias para {ds_name} ya han sido completadas.")
            continue

        # Ejecutar en paralelo (2 hilos como solicita el usuario)
        n_workers = 2
        logger.info(f"Ejecutando {len(pending_strategies)} estrategias pendientes en paralelo (n_jobs={n_workers})...")

        results_gen = Parallel(n_jobs=n_workers, return_as="generator")(
            delayed(run_single_strategy)(strat, ds_name, split)
            for strat in pending_strategies
        )

        # Ir guardando los checkpoints a medida que cada tarea finaliza
        for r in results_gen:
            strat = r["strategy"]
            ds_results[strat] = {
                "success": r["success"],
                "metrics": r["metrics"],
                "convergence_round": r["convergence_round"],
                "duration_seconds": round(r["duration_seconds"], 2),
                "error": r["error"],
                "timestamp": datetime.now().isoformat()
            }
            # Guardar el JSON actualizado inmediatamente
            save_json_results(json_data)
            
            if r["success"]:
                logger.info(
                    f"[CHECKPOINT] {ds_name} - {strat} guardada con éxito | "
                    f"F1={r['metrics']['f1']:.4f} | Acc={r['metrics']['accuracy']:.4f}"
                )
            else:
                logger.error(f"[CHECKPOINT ERROR] {ds_name} - {strat} falló: {r['error']}")

    logger.info(f"\n=== EXPERIMENTO FINALIZADO CON ÉXITO ===")
    logger.info(f"Resultados totales guardados en: {RESULTS_JSON}")


if __name__ == "__main__":
    run_benchmark()
