#!/usr/bin/env python
"""Unified Hyperparameter Optimization Script for Federated Proactive Forest.

Optimizes a single joint hyperparameter vector across representative strategies
(S1, S4, S7, S8) and datasets (Sonar, Vowel, Spambase, Nursery) to find a robust
unified configuration profile.
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from typing import Any, Dict, List
import yaml
import numpy as np
import pandas as pd
import optuna
from joblib import Parallel, delayed

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.application.orchestrators import FLEXOrchestrator
from src.application.orchestrators.roulette_orchestrator import RouletteOrchestrator
from src.infrastructure.dataset.dataset_factory import DatasetFactory

# Dataset bounds for Min-Max normalization to prevent easy/hard dataset scaling biases
DATASET_BOUNDS = {
    "sonar": {"min": 0.60, "max": 0.88},
    "vowel": {"min": 0.65, "max": 0.95},
    "spambase": {"min": 0.80, "max": 0.96},
    "nursery": {"min": 0.85, "max": 0.99}
}

def load_search_space() -> Dict[str, Any]:
    """Load search space definition from YAML config under the UNIFIED key."""
    search_spaces_path = Path(ROOT) / "configs" / "optimization" / "search_spaces.yaml"
    if not search_spaces_path.exists():
        raise FileNotFoundError(f"Search spaces config not found at: {search_spaces_path}")
    with open(search_spaces_path, "r", encoding="utf-8") as f:
        all_spaces = yaml.safe_load(f)
    if "UNIFIED" not in all_spaces:
        raise ValueError("No UNIFIED search space defined in search_spaces.yaml")
    return all_spaces["UNIFIED"]

def sample_params(trial: optuna.Trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
    """Sample hyperparameters dynamically from the loaded search space configuration."""
    params = {}
    for param_name, param_config in search_space.items():
        param_type = param_config.get("type", "float")
        if param_type == "int":
            params[param_name] = trial.suggest_int(
                param_name,
                low=param_config["low"],
                high=param_config["high"],
                step=param_config.get("step", 1),
            )
        elif param_type == "float":
            if param_config.get("log", False):
                params[param_name] = trial.suggest_float(
                    param_name,
                    low=param_config["low"],
                    high=param_config["high"],
                    log=True,
                )
            else:
                params[param_name] = trial.suggest_float(
                    param_name,
                    low=param_config["low"],
                    high=param_config["high"],
                    step=param_config.get("step", None),
                )
        elif param_type == "categorical":
            params[param_name] = trial.suggest_categorical(
                param_name, param_config["choices"]
            )
    return params

def evaluate_combination(
    strategy: str,
    ds_name: str,
    params: Dict[str, Any],
    split: Any,
    n_clients: int,
    seed: int
) -> float:
    """Evaluate a single strategy on a single dataset with the given parameters.

    Returns:
        float: Hybrid accuracy mean across clients.
    """
    # Decode compound categorical parameters
    pace = params.get("pace_combination", "3_1")
    if pace == "3_1":
        global_episode_size = 3
        trees_per_client_per_episode = 1
    elif pace == "6_2":
        global_episode_size = 6
        trees_per_client_per_episode = 2
    elif pace == "9_3":
        global_episode_size = 9
        trees_per_client_per_episode = 3
    elif pace == "5_1":
        global_episode_size = 5
        trees_per_client_per_episode = 1
    elif pace == "10_3":
        global_episode_size = 10
        trees_per_client_per_episode = 3
    else:
        global_episode_size = 3
        trees_per_client_per_episode = 1

    r_pace = params.get("roulette_pace", "5_30")
    if r_pace == "5_30":
        window_size = 5
        max_rounds = 30
    elif r_pace == "10_15":
        window_size = 10
        max_rounds = 15
    elif r_pace == "15_10":
        window_size = 15
        max_rounds = 10
    else:
        window_size = 5
        max_rounds = 30

    # Common config construction
    config = {
        "federation": {
            "n_clients": n_clients,
            "distribution": "iid",
            "seed": seed,
        },
        "model": {
            "n_estimators": 150,  # robust upper limit
            "alpha_pf": 0.1,  # fixed to maintain baseline fairness
            "split_criterion": "entropy",
            "use_progressive_stopping": True,
            "local_convergence_threshold": params["local_convergence_threshold"],
            "local_episode_size": 5,
            "verbose": False,
        },
        "aggregation": {
            "max_trees": params["max_trees"],
            "t_max": params["max_trees"],  # for backwards compatibility
            "global_convergence_threshold": params["global_convergence_threshold"],
            "global_episode_size": 3,
            "min_episodes": 4,
            "min_rounds": 4,
        },
        "prediction": {
            "local_weight": params["local_weight"],
            "global_weight": 1.0 - params["local_weight"],
            "use_weighted": params["use_weighted"],
        },

        "verbose": False,
        "seed": seed,
    }

    # Strategy-specific config tuning
    if strategy == "S1":
        config["aggregation"]["strategy"] = "s1_simple_pool"
    elif strategy == "S4":
        config["aggregation"]["strategy"] = "s4_global_f1_pcd"
        config["aggregation"]["global_episode_size"] = global_episode_size
        config["aggregation"]["f1_weight"] = params["f1_weight"]
        config["aggregation"]["pcd_weight"] = 1.0 - params["f1_weight"]
    elif strategy == "S7":
        config["aggregation"]["strategy"] = "s7_perclient_f1_pcd"
        config["aggregation"]["trees_per_client_per_episode"] = trees_per_client_per_episode
        config["aggregation"]["f1_weight"] = params["f1_weight"]
        config["aggregation"]["pcd_weight"] = 1.0 - params["f1_weight"]
    elif strategy == "S8":
        config["aggregation"].update({
            "strategy": "S8",
            "variant": "S8_MEAN",
            "local_roulette_weight": params["local_roulette_weight"],
            "window_size": window_size,
            "max_rounds": max_rounds,
            "global_convergence_threshold": params["global_convergence_threshold"],
        })

    # Run orchestrator
    orchestrator = None
    try:
        if strategy == "S8":
            orchestrator = RouletteOrchestrator(config)
        else:
            orchestrator = FLEXOrchestrator.from_config(config)

        orchestrator.setup_federation(split, seed=seed)
        results = orchestrator.run_federated_round()
        accuracy = results.hybrid_accuracy_mean
    except Exception as e:
        print(f"Error evaluating strategy {strategy} on dataset {ds_name}: {e}")
        accuracy = 0.0
    finally:
        if orchestrator and hasattr(orchestrator, "cleanup"):
            orchestrator.cleanup()

    return accuracy

def make_objective(splits: Dict[str, Any], search_space: Dict[str, Any], n_clients: int, seed: int):
    """Generate the objective function closure for Optuna."""
    trial_counter = 0

    def objective(trial: optuna.Trial) -> float:
        nonlocal trial_counter
        trial_counter += 1
        params = sample_params(trial, search_space)

        # Prepare parallel evaluations (S1, S4, S7, S8) x (Sonar, Vowel, Spambase, Nursery)
        strategies = ["S1", "S4", "S7", "S8"]
        datasets = list(splits.keys())

        tasks = []
        for strat in strategies:
            for ds_name in datasets:
                tasks.append((strat, ds_name))

        # Run all evaluations in parallel with 2 jobs
        accuracies = Parallel(n_jobs=2, verbose=5)(
            delayed(evaluate_combination)(
                strat, ds_name, params, splits[ds_name], n_clients, seed
            )
            for strat, ds_name in tasks
        )

        # Calculate normalized average accuracy
        normalized_scores = []
        for i, (strat, ds_name) in enumerate(tasks):
            acc = accuracies[i]
            bounds = DATASET_BOUNDS[ds_name]
            norm_acc = (acc - bounds["min"]) / (bounds["max"] - bounds["min"])
            norm_acc = max(0.0, min(1.0, norm_acc)) # Clip between 0 and 1
            normalized_scores.append(norm_acc)

        mean_score = float(np.mean(normalized_scores))
        print(f"\n[Trial {trial_counter} Completed] Mean Normalized Accuracy: {mean_score:.4f}")
        print(f"Params: {params}\n")
        return mean_score

    return objective

def main():
    parser = argparse.ArgumentParser(
        description="Unified Hyperparameter Optimization across multiple strategies and datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n_trials", type=int, default=50, help="Number of trials (default: 50)")
    parser.add_argument("--n_clients", type=int, default=3, help="Number of clients (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default="results/unified_optimization", help="Output directory")
    args = parser.parse_args()

    # Load dataset splits
    print("📊 Cargando datasets representativos...")
    dataset_base_config = {
        "test_size": 0.15,
        "scale": True,
        "scaler_type": "standard",
        "seed": args.seed,
    }
    splits = {}
    for d in DATASET_BOUNDS.keys():
        print(f"   - {d.upper()}...")
        splits[d] = DatasetFactory.load_from_config(
            {**dataset_base_config, "type": d.lower()}
        )

    # Load search space
    print("🔧 Cargando espacio de búsqueda unificado...")
    search_space = load_search_space()

    # Setup output directory
    os.makedirs(args.output, exist_ok=True)

    # Create Optuna study
    study_db = Path(args.output) / "unified_study.db"
    storage_url = f"sqlite:///{study_db.resolve().as_posix()}"
    
    # Use TPESampler with multivariate option enabled
    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
    study = optuna.create_study(
        study_name="unified_hpo",
        direction="maximize",
        sampler=sampler,
        storage=storage_url,
        load_if_exists=True
    )

    # Objective function
    objective_fn = make_objective(splits, search_space, args.n_clients, args.seed)

    print(f"\n🚀 Iniciando optimización unificada ({args.n_trials} trials, n_jobs=2)...")
    start_time = time.time()
    study.optimize(objective_fn, n_trials=args.n_trials)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"\n✅ Optimización completada en {duration/60:.2f} minutos.")

    # Save results using the unified OptimizationResultsLogger
    from src.infrastructure.persistence.results_logger import OptimizationResultsLogger
    
    logger = OptimizationResultsLogger(args.output)
    
    metadata = {
        "strategy": "UNIFIED_BAYESIAN",
        "dataset": "sonar_vowel_spambase_nursery",
        "metric": "mean_normalized_accuracy",
        "n_trials": args.n_trials,
        "n_clients": args.n_clients,
        "seed": args.seed,
        "duration_seconds": duration,
    }
    
    logger.save_study(study, metadata=metadata)
    logger.generate_plots(study)
    
    try:
        logger.generate_summary_report(study, metadata=metadata)
    except Exception as e:
        print(f"⚠️ No se pudo generar el reporte de texto: {e}")

    # Save JSON summary
    best_params = study.best_params
    best_value = study.best_value
    print(f"🏆 Mejor puntuación media normalizada: {best_value:.4f}")
    print(f"🔧 Mejores parámetros encontrados: {best_params}")

    results_file = Path(args.output) / "unified_best_params.json"
    summary = {
        "best_value": best_value,
        "best_params": best_params,
        "n_trials": args.n_trials,
        "n_clients": args.n_clients,
        "seed": args.seed,
        "duration_seconds": duration,
        "dataset_bounds": DATASET_BOUNDS,
    }
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"💾 Resultados guardados en: {results_file}")

if __name__ == "__main__":
    main()
