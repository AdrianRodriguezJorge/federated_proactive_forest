"""Benchmark Definitivo: Federated Proactive Forest.

Evalúa 13 estrategias federadas (incluyendo baseline local_isolation)
en 10 datasets bajo un protocolo riguroso de 10-Fold Stratified CV.

Características del script:
  - Hiperparámetros centralizados y fácilmente editables en HYPERPARAMS.
  - Sistema de checkpoint granular a nivel de fold individual (JSON).
  - Corrección de PCD real para local_isolation.
  - Compatibilidad con PyArrow (conversión explícita a numpy).
  - Modo de prueba (--test) para verificación rápida.

Uso:
  python scripts/final_benchmark.py          # Ejecución completa
  python scripts/final_benchmark.py --test   # Prueba mínima (1 dataset, 1 fold)
"""

import json
import logging
import os
import sys
import time

# Agregar raíz del proyecto al path para importar src.*
sys.path.append(os.getcwd())

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pandas.api.types import is_string_dtype
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

from src.application.orchestrators import FLEXOrchestrator
from src.application.orchestrators.fed_data_distributor import FedDataDistributor
from src.application.orchestrators.roulette_orchestrator import RouletteOrchestrator
from src.domain.dataset.base_adapter import DatasetSplit
from src.domain.metrics.forest_evaluator import ForestEvaluator
from src.domain.model.proactive_forest import ProactiveForest
from src.infrastructure.dataset.dataset_factory import DATASET_METADATA

# ===========================================================================
# LOGGING
# ===========================================================================
os.makedirs("results/logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            "results/logs/final_benchmark_run.log", encoding="utf-8"
        ),
    ],
)
logger = logging.getLogger("FinalBenchmark")

# ===========================================================================
# HIPERPARÁMETROS OPTIMIZADOS (HPO UNIFICADA + GRID SEARCH)
# Editar estos valores directamente para ajustes manuales.
# ===========================================================================
HYPERPARAMS = {
    # --- Modelo ---
    "n_estimators": 110,            # Tamaño máximo del pool global (S1-S7)
    "alpha": 0.1,                   # Parámetro de diversidad del ProactiveForest
    "voting": "soft",               # Tipo de votación del ensamble

    # --- Convergencia ---
    "local_convergence_threshold": 0.002,   # Umbral parada temprana local
    "global_convergence_threshold": 0.002,  # Umbral convergencia global

    # --- Pacing Progresivo (pace_combination = 10_3) ---
    "global_episode_size": 10,          # Árboles por episodio global (S2-S4)
    "trees_per_client_per_episode": 3,  # Árboles por cliente por episodio (S5-S7)
    "trees_per_round_per_client": 3,    # Árboles por ronda por cliente
    "min_episodes": 4,                  # Episodios mínimos antes de convergencia

    # --- Roulette S8 (roulette_pace = 10_15) ---
    "window_size": 10,              # Ventana de árboles locales antes de intercambio
    "max_rounds": 15,               # Máximo de rondas federadas
    "min_rounds": 4,                # Rondas mínimas antes de convergencia
    "local_roulette_weight": 0.6,   # Peso del vector local de importancia

    # --- Selección de Métricas (para S4, S7) ---
    "f1_weight": 0.5,              # Peso F1 en selección mixta F1+PCD
    "pcd_weight": 0.5,             # Peso PCD (complementario a f1_weight)

    # --- Predicción Híbrida ---
    "local_weight": 0.4,           # Peso del modelo local en predicción
    "use_weighted": True,          # Activar votación híbrida ponderada
}

# ===========================================================================
# CONFIGURACIÓN DEL EXPERIMENTO
# ===========================================================================
K_FOLDS = 10
N_CLIENTS = 3
N_WORKERS = 2  # Hilos de ejecución paralela

DATASETS = [
    "Iris", "Car", "Nursery", "Vowel", "Optdigits",
    "Sonar", "Spambase", "Glass", "Molecular", "Pendigits",
]

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

RESULTS_JSON = "results/final_benchmark_results.json"

# --- Modo de prueba ---
if len(sys.argv) > 1 and sys.argv[1] == "--test":
    logger.info("MODO PRUEBA: 1 dataset (Iris), 2 estrategias, 2 folds.")
    DATASETS = ["Iris"]
    STRATEGIES = ["local_isolation", "s1_simple_pool"]
    K_FOLDS = 2


# ===========================================================================
# FUNCIONES DE CONFIGURACIÓN
# ===========================================================================

def build_orchestrator_config(strategy: str) -> Dict[str, Any]:
    """Construye el diccionario de configuración para un orquestador.

    Los valores se leen de HYPERPARAMS para garantizar que todas las
    estrategias y local_isolation usen los mismos hiperparámetros base.

    Args:
        strategy: Nombre de la estrategia (e.g. 's4_global_f1_pcd').

    Returns:
        Diccionario de configuración listo para el orquestador.
    """
    h = HYPERPARAMS

    config = {
        "federation": {
            "n_clients": N_CLIENTS,
            "distribution": "iid",
        },
        "model": {
            "n_estimators": h["n_estimators"],
            "alpha": h["alpha"],
            "voting": h["voting"],
            "local_convergence_threshold": h["local_convergence_threshold"],
        },
        "aggregation": {
            "strategy": strategy,
            "max_rounds": h["max_rounds"],
            "global_convergence_threshold": h["global_convergence_threshold"],
            "global_episode_size": h["global_episode_size"],
            "trees_per_client_per_episode": h["trees_per_client_per_episode"],
            "trees_per_round_per_client": h["trees_per_round_per_client"],
            "min_episodes": h["min_episodes"],
            "min_rounds": h["min_rounds"],
            "window_size": h["window_size"],
            "f1_weight": h["f1_weight"],
            "pcd_weight": h["pcd_weight"],
        },
        "prediction": {
            "local_weight": h["local_weight"],
            "use_weighted": h["use_weighted"],
        },
    }

    # Configuración especial para variantes S8 (Roulette)
    if strategy.startswith("s8_"):
        variant = strategy.replace("s8_", "S8_").upper()
        config["aggregation"]["variant"] = variant
        config["aggregation"]["local_roulette_weight"] = h["local_roulette_weight"]
        # En S8, n_estimators = window_size × max_rounds
        config["model"]["n_estimators"] = h["window_size"] * h["max_rounds"]

    return config


# ===========================================================================
# FUNCIONES DE CHECKPOINT (JSON)
# ===========================================================================

def load_results() -> Dict[str, Any]:
    """Carga los resultados previos desde el archivo JSON.

    Returns:
        Diccionario con los resultados y metadatos del experimento.
    """
    if os.path.exists(RESULTS_JSON):
        try:
            with open(RESULTS_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"No se pudo leer {RESULTS_JSON}: {e}")

    return {
        "timestamp_start": datetime.now().isoformat(),
        "experiment_metadata": {
            "k_folds": K_FOLDS,
            "n_clients": N_CLIENTS,
            "n_workers": N_WORKERS,
            "datasets": DATASETS,
            "strategies": STRATEGIES,
        },
        "hyperparams": HYPERPARAMS,
        "results": {},
    }


def save_results(data: Dict[str, Any]) -> None:
    """Persiste los resultados a disco en formato JSON.

    Args:
        data: Diccionario completo de resultados del experimento.
    """
    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def is_fold_completed(json_data: Dict, ds: str, strat: str, fold: int) -> bool:
    """Verifica si un fold específico ya fue completado exitosamente.

    Args:
        json_data: Diccionario de resultados cargado.
        ds: Nombre del dataset.
        strat: Nombre de la estrategia.
        fold: Índice del fold (0-based).

    Returns:
        True si el fold tiene success=True en el checkpoint.
    """
    fold_key = f"fold_{fold}"
    return (
        json_data.get("results", {})
        .get(ds, {})
        .get(strat, {})
        .get(fold_key, {})
        .get("success", False)
    )


# ===========================================================================
# PREPROCESAMIENTO DE DATASETS
# ===========================================================================

def precompute_cv_splits(
    ds_name: str, k_folds: int
) -> Tuple[List[DatasetSplit], List[str]]:
    """Carga un dataset y pre-calcula los K splits de validación cruzada.

    Aplica OrdinalEncoder para categóricas y StandardScaler para numéricas,
    ajustados SOLO sobre el conjunto de entrenamiento de cada fold para
    evitar data leakage.

    Args:
        ds_name: Nombre del dataset (clave en DATASET_METADATA).
        k_folds: Número de folds para StratifiedKFold.

    Returns:
        Tupla con la lista de DatasetSplits y los nombres de las clases.

    Raises:
        FileNotFoundError: Si el archivo CSV del dataset no existe.
    """
    preset = DATASET_METADATA[ds_name.lower()]

    df = pd.read_csv(preset["file_path"], sep=preset["sep"])

    # Eliminar columnas de metadatos no deseadas (e.g. 'instance' en molecular)
    cols_to_drop = preset.get("columns_to_drop", [])
    df = df.drop(
        columns=[c for c in cols_to_drop if c in df.columns], errors="ignore"
    )

    target = preset["target_column"]
    X_raw = df.drop(columns=[target])
    y_raw = np.array(df[target].astype(str).values, dtype=object)

    # Detectar columnas categóricas (explícitas + autodetectadas)
    cat_cols = list(preset.get("categorical_features", []))
    for col in X_raw.columns:
        if (X_raw[col].dtype == "object" or is_string_dtype(X_raw[col])) \
                and col not in cat_cols:
            cat_cols.append(col)

    cat_cols_idx = [X_raw.columns.get_loc(c) for c in cat_cols] if cat_cols else []
    num_cols_idx = [
        X_raw.columns.get_loc(c) for c in X_raw.columns if c not in cat_cols
    ]

    # Encoder de labels (ajustado sobre TODO y_raw para consistencia)
    le = LabelEncoder()
    le.fit(y_raw)

    # StratifiedKFold con semilla fija para reproducibilidad
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    splits = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(X_raw, y_raw)):
        # Separar train+val vs test
        X_tv_raw = np.array(X_raw.iloc[train_val_idx].values, dtype=object)
        X_test_raw = np.array(X_raw.iloc[test_idx].values, dtype=object)
        y_tv_raw = y_raw[train_val_idx]
        y_test_raw = y_raw[test_idx]

        # Sub-split: 10% del train+val como server validation
        _, counts_y = np.unique(y_tv_raw, return_counts=True)
        can_stratify = np.min(counts_y) >= 2

        X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
            X_tv_raw,
            y_tv_raw,
            test_size=0.1111,  # ~10% del total original
            random_state=42 + fold_idx,
            stratify=y_tv_raw if can_stratify else None,
        )

        # Copias para transformación in-place
        X_train = X_train_raw.copy()
        X_val = X_val_raw.copy()
        X_test = X_test_raw.copy()

        # Codificar variables categóricas (ajustado solo en train)
        if cat_cols_idx:
            enc = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            X_train[:, cat_cols_idx] = enc.fit_transform(
                X_train[:, cat_cols_idx].astype(str)
            )
            X_val[:, cat_cols_idx] = enc.transform(
                X_val[:, cat_cols_idx].astype(str)
            )
            X_test[:, cat_cols_idx] = enc.transform(
                X_test[:, cat_cols_idx].astype(str)
            )

        # Estandarizar variables numéricas (ajustado solo en train)
        if num_cols_idx:
            scaler = StandardScaler()
            X_train[:, num_cols_idx] = scaler.fit_transform(
                X_train[:, num_cols_idx]
            )
            X_val[:, num_cols_idx] = scaler.transform(
                X_val[:, num_cols_idx]
            )
            X_test[:, num_cols_idx] = scaler.transform(
                X_test[:, num_cols_idx]
            )

        # Codificar labels a enteros
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
        splits.append(split)

    class_names = [str(c) for c in le.classes_]
    return splits, class_names


# ===========================================================================
# EJECUCIÓN DE UN FOLD INDIVIDUAL
# ===========================================================================

def run_local_isolation_fold(
    split: DatasetSplit,
    fold_idx: int,
) -> Dict[str, Any]:
    """Ejecuta la estrategia baseline local_isolation para un fold.

    Cada cliente entrena un ProactiveForest en aislamiento total
    (sin federación), usando solo su partición IID de los datos.
    Las métricas se evalúan sobre X_test y se promedian.

    Args:
        split: DatasetSplit pre-procesado del fold.
        fold_idx: Índice del fold actual.

    Returns:
        Diccionario con métricas del fold.
    """
    h = HYPERPARAMS
    config = build_orchestrator_config("local_isolation")

    # Distribuir datos entre clientes (misma distribución IID que federadas)
    distributor = FedDataDistributor(config, use_flex_pool=False)
    updated_split, fed_data = distributor.distribute(split)

    client_reports = []
    for client_id, client_dataset in fed_data.items():
        X_c, y_c = client_dataset.to_numpy()

        # Entrenar modelo local con los mismos hiperparámetros que las federadas
        model = ProactiveForest(
            n_estimators=h["n_estimators"],
            alpha_pf=h["alpha"],
            class_names=updated_split.class_names,
            local_convergence_threshold=h["local_convergence_threshold"],
        )
        model.fit(X_c, y_c)

        # Predecir sobre el test set global del fold
        preds = model.predict(updated_split.X_test)

        # Calcular PCD real del bosque local (NO hardcoded a 0.0)
        try:
            pcd = float(model.diversity_measure(
                updated_split.X_test, updated_split.y_test, diversity="pcd"
            ))
        except (ValueError, TypeError, AttributeError):
            pcd = 0.0

        # Evaluar con el mismo evaluador que las estrategias federadas
        report = ForestEvaluator.evaluate_from_predictions(
            preds,
            updated_split.y_test,
            updated_split.class_names,
            len(model.get_trees()),
            pcd=pcd,
        )
        client_reports.append(report)

    # Promediar métricas de los 3 clientes (mismo criterio que federadas)
    return {
        "f1": float(np.mean([r.macro_f1 for r in client_reports])),
        "accuracy": float(np.mean([r.accuracy for r in client_reports])),
        "recall": float(np.mean([r.macro_recall for r in client_reports])),
        "precision": float(np.mean([r.macro_precision for r in client_reports])),
        "pcd": float(np.mean([r.pcd for r in client_reports])),
        "forest_size": float(np.mean([r.forest_size for r in client_reports])),
        "convergence_round": None,  # No aplica a local_isolation
    }


def run_federated_fold(
    strategy: str,
    split: DatasetSplit,
    fold_idx: int,
) -> Dict[str, Any]:
    """Ejecuta una estrategia federada (S1-S8) para un fold.

    Crea el orquestador apropiado (FLEX o Roulette), ejecuta la ronda
    federada completa, y extrae las métricas del bosque híbrido.

    Args:
        strategy: Nombre de la estrategia federada.
        split: DatasetSplit pre-procesado del fold.
        fold_idx: Índice del fold actual.

    Returns:
        Diccionario con métricas del fold.
    """
    config = build_orchestrator_config(strategy)

    # Seleccionar orquestador según tipo de estrategia
    if strategy.startswith("s8_"):
        orch = RouletteOrchestrator(config)
    else:
        orch = FLEXOrchestrator(config)

    try:
        orch.setup_federation(split)
        res = orch.run_federated_round(n_bootstrap=0)

        # Extraer métricas híbridas (promedio de los 3 clientes)
        avg_trees = float(np.mean([
            r.forest_size for r in res.client_reports.values()
        ]))

        return {
            "f1": res.hybrid_f1_mean,
            "accuracy": res.hybrid_accuracy_mean,
            "recall": res.hybrid_recall_mean,
            "precision": res.hybrid_precision_mean,
            "pcd": res.hybrid_pcd_mean,
            "forest_size": avg_trees,
            "convergence_round": res.convergence_round,
        }
    finally:
        if hasattr(orch, "cleanup"):
            orch.cleanup()


def run_single_fold(
    strategy: str,
    ds_name: str,
    fold_idx: int,
    split: DatasetSplit,
) -> Dict[str, Any]:
    """Ejecuta un fold para una estrategia y dataset dados.

    Función wrapper que delega en run_local_isolation_fold o
    run_federated_fold según corresponda, capturando errores.

    Args:
        strategy: Nombre de la estrategia.
        ds_name: Nombre del dataset.
        fold_idx: Índice del fold (0-based).
        split: DatasetSplit pre-procesado del fold.

    Returns:
        Diccionario con los resultados del fold y metadatos.
    """
    start_time = time.time()
    prefix = f"[{ds_name} | {strategy} | fold_{fold_idx}]"
    logger.info(f"{prefix} Iniciando...")

    try:
        if strategy == "local_isolation":
            metrics = run_local_isolation_fold(split, fold_idx)
        else:
            metrics = run_federated_fold(strategy, split, fold_idx)

        duration = time.time() - start_time
        logger.info(
            f"{prefix} OK en {duration:.1f}s | "
            f"F1={metrics['f1']:.4f} Acc={metrics['accuracy']:.4f} "
            f"PCD={metrics['pcd']:.4f}"
        )

        return {
            "success": True,
            "dataset": ds_name,
            "strategy": strategy,
            "fold_idx": fold_idx,
            "metrics": metrics,
            "duration_seconds": round(duration, 2),
            "error": None,
        }

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"{prefix} ERROR en {duration:.1f}s: {e}", exc_info=True)
        return {
            "success": False,
            "dataset": ds_name,
            "strategy": strategy,
            "fold_idx": fold_idx,
            "metrics": None,
            "duration_seconds": round(duration, 2),
            "error": str(e),
        }


# ===========================================================================
# BUCLE PRINCIPAL DEL BENCHMARK
# ===========================================================================

def run_final_benchmark() -> None:
    """Ejecuta el benchmark completo con checkpoint granular por fold.

    Flujo de ejecución:
      1. Carga resultados previos (soporte de checkpoint).
      2. Para cada dataset: pre-calcula los K splits de CV.
      3. Para cada estrategia: identifica folds pendientes.
      4. Ejecuta los folds pendientes en paralelo (n_jobs=N_WORKERS).
      5. Persiste cada resultado inmediatamente tras completarse.
    """
    logger.info("=" * 70)
    logger.info("BENCHMARK DEFINITIVO: FEDERATED PROACTIVE FOREST")
    logger.info(f"Datasets: {len(DATASETS)} | Estrategias: {len(STRATEGIES)} "
                f"| Folds: {K_FOLDS} | Total: {len(DATASETS)*len(STRATEGIES)*K_FOLDS}")
    logger.info("=" * 70)

    json_data = load_results()

    for ds_name in DATASETS:
        logger.info(f"\n>>> Dataset: {ds_name} — Precalculando {K_FOLDS} folds...")

        try:
            splits, class_names = precompute_cv_splits(ds_name, K_FOLDS)
        except Exception as e:
            logger.error(f"Error cargando dataset {ds_name}: {e}")
            continue

        logger.info(
            f"  {ds_name}: {splits[0].X_train.shape[0]} train | "
            f"{splits[0].X_val.shape[0]} val | "
            f"{splits[0].X_test.shape[0]} test | "
            f"{len(class_names)} clases"
        )

        for strategy in STRATEGIES:
            # Compilar lista de folds pendientes para esta combinación
            pending_folds = [
                fold_idx
                for fold_idx in range(K_FOLDS)
                if not is_fold_completed(json_data, ds_name, strategy, fold_idx)
            ]

            if not pending_folds:
                logger.info(f"  [{strategy}] Todos los folds completados. Saltando.")
                continue

            logger.info(
                f"  [{strategy}] {len(pending_folds)}/{K_FOLDS} folds pendientes. "
                f"Ejecutando en {N_WORKERS} hilos..."
            )

            # Ejecución paralela de folds pendientes
            results_gen = Parallel(
                n_jobs=N_WORKERS, return_as="generator"
            )(
                delayed(run_single_fold)(
                    strategy, ds_name, fold_idx, splits[fold_idx]
                )
                for fold_idx in pending_folds
            )

            # Guardar checkpoint inmediatamente tras cada fold completado
            for r in results_gen:
                fold_key = f"fold_{r['fold_idx']}"

                # Crear estructura anidada en el JSON
                ds_results = json_data["results"].setdefault(ds_name, {})
                strat_results = ds_results.setdefault(strategy, {})

                strat_results[fold_key] = {
                    "success": r["success"],
                    "metrics": r["metrics"],
                    "duration_seconds": r["duration_seconds"],
                    "error": r["error"],
                    "timestamp": datetime.now().isoformat(),
                }

                # Persistir a disco de forma inmediata
                save_results(json_data)

                if r["success"]:
                    m = r["metrics"]
                    logger.info(
                        f"  [CHECKPOINT] {ds_name} | {strategy} | {fold_key} | "
                        f"F1={m['f1']:.4f} Acc={m['accuracy']:.4f} PCD={m['pcd']:.4f}"
                    )
                else:
                    logger.error(
                        f"  [CHECKPOINT ERROR] {ds_name} | {strategy} | "
                        f"{fold_key}: {r['error']}"
                    )

    logger.info("=" * 70)
    logger.info(f"BENCHMARK FINALIZADO. Resultados en: {RESULTS_JSON}")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_final_benchmark()
