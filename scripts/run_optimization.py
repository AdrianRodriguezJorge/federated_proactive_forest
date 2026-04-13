#!/usr/bin/env python
"""
Hyperparameter Optimization Script for Federated Proactive Forest.

Uses Optuna for Bayesian optimization of aggregation strategy hyperparameters.

Usage examples:
    # Optimize S6 with Car dataset (10 trials for testing)
    python scripts/run_optimization.py \
        --strategy S6 \
        --dataset car \
        --n_trials 10 \
        --metric macro_f1

    # Optimize S7 with Letter dataset (50 trials)
    python scripts/run_optimization.py \
        --strategy S7 \
        --dataset letter \
        --n_trials 50 \
        --metric macro_f1

    # Optimize Progressive Windows
    python scripts/run_optimization.py \
        --strategy PW \
        --dataset letter \
        --n_trials 100 \
        --metric macro_f1
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np
import yaml
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.infrastructure.dataset.dataset_factory import DatasetFactory
from src.domain.dataset.base_adapter import DatasetSplit


# ============================================================================
# Dataset Loading Functions
# ============================================================================

# Redundant dataset loading functions removed.
# All datasets are now handled by DatasetFactory.load_from_config()


# ============================================================================
# Configuration Loading
# ============================================================================

def load_base_config(strategy: str, n_clients: int = 5, seed: int = 42) -> dict:
    """
    Build base configuration for a given strategy.

    Args:
        strategy: Strategy name (S1-S7, PW)
        n_clients: Number of federated clients
        seed: Random seed

    Returns:
        Base configuration dict
    """
    config = {
        'federation': {
            'n_clients': n_clients,
            'distribution': 'iid',
            'seed': seed,
        },
        'model': {
            'n_estimators': 100,  # Fixed for now
            'alpha': 0.1,
            'split_criterion': 'entropy',
            'use_progressive_stopping': True,
            'convergence': 0.002,
            'episode_size': 5,
            'verbose': False,
        },
        'aggregation': {
            'strategy': strategy,
            't_max': 100,
            'f1_weight': 0.5,
            'pcd_weight': 0.5,
            'convergence': 0.002,
            'episode_size': 5,
        },
        'prediction': {
            'local_weight': 0.4,
            'global_weight': 0.6,
        },
        'verbose': False,
        'seed': seed,
    }

    # Strategy-specific adjustments
    if strategy == 'PW':
        config['aggregation'].update({
            'window_size': 5,
            'max_rounds': 20,
            'alpha': 0.5,
            'convergence_threshold': 0.002,
        })
        config['prediction'] = {
            'local_weight': 0.5,
            'global_weight': 0.5,
        }

    return config


def load_search_space(strategy: str) -> dict:
    """
    Load search space definition from YAML config.

    Args:
        strategy: Strategy name (S1-S7, PW)

    Returns:
        Search space dict
    """
    project_root = Path(__file__).parent.parent
    search_spaces_path = project_root / 'configs' / 'experiments' / 'optimization' / 'search_spaces.yaml'

    if not search_spaces_path.exists():
        raise FileNotFoundError(f"Search spaces config not found at: {search_spaces_path}")

    with open(search_spaces_path, 'r', encoding='utf-8') as f:
        all_spaces = yaml.safe_load(f)

    if strategy not in all_spaces:
        raise ValueError(f"No search space defined for strategy '{strategy}'. Available: {list(all_spaces.keys())}")

    return all_spaces[strategy]


# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for Federated Proactive Forest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with Car dataset
  python scripts/run_optimization.py --strategy S6 --dataset car --n_trials 10

  # Full optimization with Letter dataset
  python scripts/run_optimization.py --strategy S7 --dataset letter --n_trials 50

  # Progressive Windows optimization
  python scripts/run_optimization.py --strategy PW --dataset letter --n_trials 100
        """
    )

    parser.add_argument(
        '--strategy',
        type=str,
        required=True,
        choices=['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'PW'],
        help='Aggregation strategy to optimize'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['car', 'iris', 'letter', 'students'],
        help='Dataset to use for optimization'
    )
    parser.add_argument(
        '--n_trials',
        type=int,
        default=50,
        help='Number of optimization trials (default: 50)'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='macro_f1',
        choices=['macro_f1', 'accuracy', 'hybrid_macro_f1'],
        help='Metric to optimize (default: macro_f1)'
    )
    parser.add_argument(
        '--n_clients',
        type=int,
        default=5,
        help='Number of federated clients (default: 5)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for results (default: results/opt_{strategy}_{dataset})'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging during optimization'
    )

    args = parser.parse_args()

    # Set output directory
    if args.output is None:
        project_root = Path(__file__).parent.parent
        args.output = str(project_root / 'results' / f'opt_{args.strategy.lower()}_{args.dataset}')

    # Set seed
    np.random.seed(args.seed)

    # Load dataset using factory
    print(f"\n{'='*60}")
    print(f"📊 CARGANDO DATASET: {args.dataset.upper()}")
    print(f"{'='*60}")

    ds_config = {
        "type": args.dataset.capitalize() if args.dataset != "students" else "Students Dropout",
        "test_size": 0.2,
        "scale": True,
        "scaler_type": "standard",
        "seed": args.seed
    }
    
    # Custom mapping for folder structure consistency
    if args.dataset == "students":
        ds_config["file_path"] = "data/students_dropout.csv"
    else:
        ds_config["file_path"] = f"data/{args.dataset}.csv"

    dataset_split = DatasetFactory.load_from_config(ds_config, project_root=Path(ROOT))

    print(f"   Dataset: {dataset_split.dataset_name}")
    print(f"   Train samples: {dataset_split.X_train.shape[0]}")
    print(f"   Test samples: {dataset_split.X_test.shape[0]}")
    print(f"   Features: {dataset_split.X_train.shape[1]}")
    print(f"   Classes: {len(dataset_split.class_names)} - {dataset_split.class_names}")
    print(f"{'='*60}\n")

    # Load base config and search space
    base_config = load_base_config(args.strategy, n_clients=args.n_clients, seed=args.seed)
    search_space = load_search_space(args.strategy)

    print(f"🔧 CONFIGURACIÓN DE OPTIMIZACIÓN")
    print(f"{'='*60}")
    print(f"   Estrategia: {args.strategy}")
    print(f"   Métrica objetivo: {args.metric}")
    print(f"   Trials: {args.n_trials}")
    print(f"   Hiperparámetros a optimizar:")
    for param_name, param_config in search_space.items():
        if param_config['type'] == 'int':
            print(f"      - {param_name}: int[{param_config['low']}-{param_config['high']}, step={param_config.get('step', 1)}]")
        elif param_config['type'] == 'float':
            print(f"      - {param_name}: float[{param_config['low']}-{param_config['high']}, step={param_config.get('step', 'N/A')}]")
        elif param_config['type'] == 'categorical':
            print(f"      - {param_name}: categorical{param_config['choices']}")
    print(f"{'='*60}\n")

    # Create optimizer
    optimizer = HyperparamOptimizer(
        dataset_split=dataset_split,
        strategy=args.strategy,
        base_config=base_config,
        search_space=search_space,
        verbose=args.verbose
    )

    # Run optimization
    study = optimizer.optimize(
        n_trials=args.n_trials,
        metric=args.metric,
        seed=args.seed
    )

    # Save results
    print(f"\n💾 GUARDANDO RESULTADOS")
    print(f"{'='*60}")

    logger = OptimizationResultsLogger(args.output)

    metadata = {
        'strategy': args.strategy,
        'dataset': args.dataset,
        'metric': args.metric,
        'n_trials': args.n_trials,
        'n_clients': args.n_clients,
        'seed': args.seed,
        'n_classes': len(dataset_split.class_names),
        'n_features': dataset_split.X_train.shape[1],
        'n_train_samples': dataset_split.X_train.shape[0],
        'n_test_samples': dataset_split.X_test.shape[0],
    }

    saved_files = logger.save_study(study, metadata=metadata)
    print(f"\n✅ Archivos guardados:")
    for file_type, file_path in saved_files.items():
        print(f"   - {file_type}: {file_path}")

    # Generate plots
    print(f"\n📊 GENERANDO VISUALIZACIONES")
    print(f"{'='*60}")
    plot_files = logger.generate_plots(study)
    for plot_type, plot_path in plot_files.items():
        print(f"   - {plot_type}: {plot_path}")

    # Generate summary report
    print(f"\n📝 GENERANDO REPORTE RESUMEN")
    print(f"{'='*60}")
    logger.generate_summary_report(study, metadata=metadata)

    print(f"\n{'='*60}")
    print(f"✅ OPTIMIZACIÓN COMPLETADA EXITOSAMENTE")
    print(f"{'='*60}")
    print(f"Resultados guardados en: {args.output}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
