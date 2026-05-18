#!/usr/bin/env python
"""Hyperparameter Optimization Script for Federated Proactive Forest.

Uses Optuna for Bayesian optimization of aggregation strategy hyperparameters.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict
import numpy as np
import yaml

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.application.hyperparam_optimizer import HyperparamOptimizer
from src.infrastructure.dataset.dataset_factory import DatasetFactory
from src.infrastructure.persistence.results_logger import (
    OptimizationResultsLogger,
)


def load_base_config(
    strategy: str, n_clients: int = 5, seed: int = 42
) -> Dict[str, Any]:
    """Build base configuration for a given strategy.

    Args:
        strategy: Strategy name (S1-S7, PW, S9)
        n_clients: Number of federated clients
        seed: Random seed

    Returns:
        Base configuration dict
    """
    config = {
        "federation": {
            "n_clients": n_clients,
            "distribution": "iid",
            "seed": seed,
        },
        "model": {
            "n_estimators": 100,  # Fixed for now
            "alpha": 0.1,
            "split_criterion": "entropy",
            "use_progressive_stopping": True,
            "local_convergence_threshold": 0.002,
            "episode_size": 5,
            "verbose": False,
        },
        "aggregation": {
            "strategy": strategy,
            "t_max": 100,
            "f1_weight": 0.5,
            "pcd_weight": 0.5,
            "global_convergence_threshold": 0.002,
            "episode_size": 5,
        },
        "prediction": {
            "local_weight": 0.4,
            "global_weight": 0.6,
        },
        "verbose": False,
        "seed": seed,
    }

    # Strategy-specific adjustments
    if strategy == "PW":
        config["aggregation"].update(
            {
                "window_size": 5,
                "max_rounds": 20,
                "alpha": 0.5,
                "global_convergence_threshold": 0.002,
            }
        )
        config["prediction"] = {
            "local_weight": 0.5,
            "global_weight": 0.5,
        }
    elif strategy == "S9":
        config["aggregation"].update(
            {
                "variant": "S9_MEAN",
                "beta": 0.0,
                "window_size": 5,
                "max_rounds": 20,
                "global_convergence_threshold": 0.002,
            }
        )

    return config


def load_search_space(strategy: str) -> Dict[str, Any]:
    """Load search space definition from YAML config.

    Args:
        strategy: Strategy name (S1-S7, PW)

    Returns:
        Search space dict
    """
    project_root = Path(__file__).parent.parent
    search_spaces_path = (
        project_root / "configs" / "optimization" / "search_spaces.yaml"
    )

    if not search_spaces_path.exists():
        raise FileNotFoundError(
            f"Search spaces config not found at: {search_spaces_path}"
        )

    with open(search_spaces_path, "r", encoding="utf-8") as f:
        all_spaces = yaml.safe_load(f)

    if strategy not in all_spaces:
        avail = list(all_spaces.keys())
        raise ValueError(
            f"No search space defined for strategy '{strategy}'. "
            f"Available: {avail}"
        )

    return all_spaces[strategy]


def main() -> None:
    """Entry point for hyperparameter optimization execution."""
    parser = argparse.ArgumentParser(
        description=(
            "Hyperparameter optimization for Federated Proactive Forest"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=["S1", "S2", "S3", "S4", "S5", "S6", "S7", "PW", "S9"],
        help="Aggregation strategy to optimize",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["car", "iris", "letter"],
        help="Dataset to use for optimization",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of optimization trials (default: 50)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="macro_f1",
        choices=["macro_f1", "accuracy", "hybrid_macro_f1"],
        help="Metric to optimize (default: macro_f1)",
    )
    parser.add_argument(
        "--n_clients",
        type=int,
        default=5,
        help="Number of federated clients (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging during optimization",
    )

    args = parser.parse_args()

    # Set output directory
    if args.output is None:
        project_root = Path(__file__).parent.parent
        target_dir = f"opt_{args.strategy.lower()}_{args.dataset}"
        args.output = str(project_root / "results" / target_dir)

    # Set seed
    np.random.seed(args.seed)

    # Load dataset using factory
    print(f"\n{'=' * 60}")
    print(f"📊 CARGANDO DATASET: {args.dataset.upper()}")
    print(f"{'=' * 60}")

    ds_config = {
        "type": args.dataset.capitalize(),
        "test_size": 0.2,
        "scale": True,
        "scaler_type": "standard",
        "seed": args.seed,
        "file_path": f"data/{args.dataset}.csv",
    }

    dataset_split = DatasetFactory.load_from_config(
        ds_config, project_root=Path(ROOT)
    )

    print(f"   Dataset: {dataset_split.dataset_name}")
    print(f"   Train samples: {dataset_split.X_train.shape[0]}")
    print(f"   Test samples: {dataset_split.X_test.shape[0]}")
    print(f"   Features: {dataset_split.X_train.shape[1]}")
    print(
        f"   Classes: {len(dataset_split.class_names)} - "
        f"{dataset_split.class_names}"
    )
    print(f"{'=' * 60}\n")

    # Load base config and search space
    base_config = load_base_config(
        args.strategy, n_clients=args.n_clients, seed=args.seed
    )
    search_space = load_search_space(args.strategy)

    print("🔧 CONFIGURACIÓN DE OPTIMIZACIÓN")
    print(f"{'=' * 60}")
    print(f"   Estrategia: {args.strategy}")
    print(f"   Métrica objetivo: {args.metric}")
    print(f"   Trials: {args.n_trials}")
    print("   Hiperparámetros a optimizar:")
    for param_name, param_config in search_space.items():
        p_low = param_config.get("low")
        p_high = param_config.get("high")
        p_step = param_config.get("step", 1)
        if param_config["type"] == "int":
            print(
                f"      - {param_name}: int[{p_low}-{p_high}, "
                f"step={p_step}]"
            )
        elif param_config["type"] == "float":
            f_step = param_config.get("step", "N/A")
            print(
                f"      - {param_name}: float[{p_low}-{p_high}, "
                f"step={f_step}]"
            )
        elif param_config["type"] == "categorical":
            p_choices = param_config["choices"]
            print(f"      - {param_name}: categorical{p_choices}")
    print(f"{'=' * 60}\n")

    # Create optimizer
    optimizer = HyperparamOptimizer(
        dataset_split=dataset_split,
        strategy=args.strategy,
        base_config=base_config,
        search_space=search_space,
        verbose=args.verbose,
    )

    # Run optimization
    study = optimizer.optimize(
        n_trials=args.n_trials, metric=args.metric, seed=args.seed
    )

    # Save results
    print("\n💾 GUARDANDO RESULTADOS")
    print(f"{'=' * 60}")

    logger = OptimizationResultsLogger(args.output)

    metadata = {
        "strategy": args.strategy,
        "dataset": args.dataset,
        "metric": args.metric,
        "n_trials": args.n_trials,
        "n_clients": args.n_clients,
        "seed": args.seed,
        "n_classes": len(dataset_split.class_names),
        "n_features": dataset_split.X_train.shape[1],
        "n_train_samples": dataset_split.X_train.shape[0],
        "n_test_samples": dataset_split.X_test.shape[0],
    }

    saved_files = logger.save_study(study, metadata=metadata)
    print("\n✅ Archivos guardados:")
    for file_type, file_path in saved_files.items():
        print(f"   - {file_type}: {file_path}")

    # Generate plots
    print("\n📊 GENERANDO VISUALIZACIONES")
    print(f"{'=' * 60}")
    plot_files = logger.generate_plots(study)
    for plot_type, plot_path in plot_files.items():
        print(f"   - {plot_type}: {plot_path}")

    # Generate summary report
    print("\n📝 GENERANDO REPORTE RESUMEN")
    print(f"{'=' * 60}")
    logger.generate_summary_report(study, metadata=metadata)

    print(f"\n{'=' * 60}")
    print("✅ OPTIMIZACIÓN COMPLETADA EXITOSAMENTE")
    print(f"{'=' * 60}")
    print(f"Resultados guardados en: {args.output}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
