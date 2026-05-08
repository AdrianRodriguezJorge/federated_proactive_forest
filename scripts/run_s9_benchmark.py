"""
S9 Global Attribute Roulette - Benchmark Script.

Runs all 4 S9 variants (Weighted, Mean, Median, Consensus) on a given
dataset and produces a consolidated results table comparing:
  - Global Accuracy
  - Global Macro-F1
  - Total Communication Bytes (upload + download across all rounds)
  - Number of Rounds to Convergence
  - Total Trees Built

Usage:
    python scripts/run_s9_benchmark.py
    python scripts/run_s9_benchmark.py --dataset Iris --clients 5 --alpha 0.5
    python scripts/run_s9_benchmark.py --dataset Letter --clients 10 --alpha 0.3
"""

import argparse
import sys
import os
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.infrastructure.dataset.dataset_factory import DatasetFactory
from src.application.orchestrators.roulette_orchestrator import RouletteOrchestrator


VARIANTS = ["S9_WEIGHTED", "S9_MEAN", "S9_MEDIAN", "S9_CONSENSUS"]

BETA_VALUES = [0.0]  # Default: full global adoption


def build_config(dataset_type: str, n_clients: int, dirichlet_alpha: float,
                 variant: str, beta: float, window_size: int = 5,
                 max_rounds: int = 20, n_estimators: int = 100) -> dict:
    """Build the configuration dict for a single experiment run."""
    return {
        "dataset": {
            "type": dataset_type,
            "scale": True,
            "scaler_type": "standard",
            "test_size": 0.2,
        },
        "federation": {
            "n_clients": n_clients,
            "distribution": "noniid_dirichlet",
            "dirichlet_alpha": dirichlet_alpha,
            "seed": 42,
        },
        "model": {
            "n_estimators": n_estimators,
            "alpha": 0.1,
            "split_criterion": "entropy",
            "verbose": False,
        },
        "aggregation": {
            "strategy": "S9",
            "variant": variant,
            "beta": beta,
            "window_size": window_size,
            "max_rounds": max_rounds,
            "convergence_threshold": 0.002,
            "t_max": n_estimators,
        },
        "prediction": {
            "local_weight": 0.0,
            "global_weight": 1.0,
        },
        "verbose": False,
        "seed": 42,
    }


def run_experiment(config: dict, dataset_type: str) -> dict:
    """Run a single S9 experiment and return results dict."""
    # Load dataset
    adapter = DatasetFactory.get_adapter_from_config(config["dataset"])
    dataset_split = adapter.load()

    # Create and run orchestrator
    orchestrator = RouletteOrchestrator(config)
    orchestrator.setup_federation(dataset_split, seed=config.get("seed", 42))

    start_time = time.time()
    results = orchestrator.run_federated_round()
    elapsed = time.time() - start_time

    return {
        "Dataset": dataset_type,
        "Variant": results.roulette_variant,
        "Beta": results.beta,
        "Global Accuracy": round(results.global_accuracy, 4),
        "Global Macro-F1": round(results.global_macro_f1, 4),
        "Total Comm. Bytes": results.total_communication_bytes,
        "Comm. KB": round(results.total_communication_bytes / 1024, 2),
        "Rounds": results.num_rounds,
        "Convergence Round": results.convergence_round or "-",
        "Total Trees": results.n_trees_global,
        "Time (s)": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="S9 Roulette Benchmark")
    parser.add_argument("--dataset", type=str, default="Iris",
                        help="Dataset type (Iris, Letter, etc.)")
    parser.add_argument("--clients", type=int, default=5,
                        help="Number of federated clients")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet alpha for non-IID partitioning")
    parser.add_argument("--window-size", type=int, default=5,
                        help="Trees per window/round")
    parser.add_argument("--max-rounds", type=int, default=20,
                        help="Maximum federated rounds")
    parser.add_argument("--n-estimators", type=int, default=100,
                        help="Max trees per client (T_max)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: results/s9_benchmark.csv)")
    args = parser.parse_args()

    output_path = args.output or os.path.join("results", "s9_benchmark.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("=" * 70)
    print("  S9 GLOBAL ATTRIBUTE ROULETTE - BENCHMARK")
    print("  Dataset: %s | Clients: %s | alpha=%s" % (args.dataset, args.clients, args.alpha))
    print("=" * 70)

    all_results = []

    for variant in VARIANTS:
        for beta in BETA_VALUES:
            print("\n>> Running %s (beta=%s)..." % (variant, beta))
            config = build_config(
                dataset_type=args.dataset,
                n_clients=args.clients,
                dirichlet_alpha=args.alpha,
                variant=variant,
                beta=beta,
                window_size=args.window_size,
                max_rounds=args.max_rounds,
                n_estimators=args.n_estimators,
            )
            try:
                result = run_experiment(config, args.dataset)
                all_results.append(result)
                print("  [OK] Acc=%.4f  F1=%.4f  Bytes=%sKB  Rounds=%s  Time=%ss" % (
                      result['Global Accuracy'], result['Global Macro-F1'],
                      result['Comm. KB'], result['Rounds'], result['Time (s)']))
            except Exception as e:
                print("  [FAIL] %s" % e)
                import traceback
                traceback.print_exc()
                all_results.append({
                    "Dataset": args.dataset,
                    "Variant": variant,
                    "Beta": beta,
                    "Global Accuracy": 0.0,
                    "Global Macro-F1": 0.0,
                    "Total Comm. Bytes": 0,
                    "Comm. KB": 0.0,
                    "Rounds": 0,
                    "Convergence Round": "-",
                    "Total Trees": 0,
                    "Time (s)": 0.0,
                })

    # Consolidate results
    df = pd.DataFrame(all_results)
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(df.to_string(index=False))
    print("\nResults saved to: %s" % output_path)

    # Highlight best variant
    if len(df[df["Global Accuracy"] > 0]) > 0:
        best = df.loc[df["Global Accuracy"].idxmax()]
        print("\n** Best variant: %s (Acc=%.4f, F1=%.4f)" % (
              best['Variant'], best['Global Accuracy'], best['Global Macro-F1']))


if __name__ == "__main__":
    main()
