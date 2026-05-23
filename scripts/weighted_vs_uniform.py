"""Experiment: Optimization of local_weight vs Uniform Tree Voting.

Compares hybrid prediction metrics when weighting by origin (local/global) with
local_weight ranging from 0.1 to 0.9, vs uniform tree voting (use_weighted=False)
across Iris, Car, and Spambase datasets.
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.application.orchestrators import FLEXOrchestrator
from src.infrastructure.dataset.dataset_factory import DatasetFactory


def run_experiment(
    dataset_cfg: Dict[str, Any],
    use_weighted: bool,
    local_weight: float = 0.4,
    strategy: str = "s6_perclient_f1",
    seed: int = 42,
) -> Dict[str, Any]:
    """Runs a single federated experiment and returns comparative metrics.

    Args:
        dataset_cfg (Dict[str, Any]): Config map for the dataset adapter.
        use_weighted (bool): Whether to weigh global vs local trees.
        local_weight (float): Vote weight to allocate to local trees.
        strategy (str): Aggregation strategy identifier.
        seed (int): Random number generator seed.

    Returns:
        Dict[str, Any]: Dict containing global and per-client metrics.
    """
    from src.domain.aggregation.aggregation_factory import AggregationFactory
    norm_strat = AggregationFactory.normalize_strategy_name(strategy)

    global_ep_size = 3
    trees_per_client_ep = 1
    trees_per_rnd_client = 1
    win_size = 5
    f1_w = 0.3
    pcd_w = 0.7

    config = {
        "dataset": dataset_cfg,
        "federation": {
            "n_clients": 3,
            "distribution": "iid",
            "dirichlet_alpha": 0.5,
        },
        "model": {
            "n_estimators": 50,
            "alpha": 0.1,
            "split_criterion": "entropy",
            "feature_selection": "prob",
            "use_progressive_stopping": True,
            "local_convergence_threshold": 0.002,
            "episode_size": 5,
        },
        "aggregation": {
            "strategy": strategy,
            "f1_weight": f1_w,
            "pcd_weight": pcd_w,
            "global_convergence_threshold": 0.002,
            "global_episode_size": global_ep_size,
            "trees_per_client_per_episode": trees_per_client_ep,
            "trees_per_round_per_client": trees_per_rnd_client,
            "window_size": win_size,
            "max_rounds": 20,
            "min_episodes": 4,
            "min_rounds": 4,
        },
        "prediction": {
            "local_weight": local_weight,
            "use_weighted": use_weighted,
        },
        "verbose": False,
        "seed": seed,
    }

    ds = DatasetFactory.load_from_config(
        config["dataset"], project_root=PROJECT_ROOT
    )
    orchestrator = FLEXOrchestrator(config)
    orchestrator.setup_federation(ds, seed=seed)

    t0 = time.time()
    results = orchestrator.run_federated_round()
    elapsed = time.time() - t0

    # Compute per-client hybrid metrics
    hybrid_accs = []
    hybrid_f1s = []
    for cid in results.client_ids:
        preds = results.client_hybrid_predictions.get(cid)
        if preds is not None:
            y = results.y_test
            hybrid_accs.append(float(accuracy_score(y, preds)))
            hybrid_f1s.append(
                float(
                    f1_score(
                        y, preds, average="macro", zero_division=0
                    )
                )
            )

    # Per-client local tree counts
    local_counts = []
    for cid in results.client_ids:
        meta = results.client_metadata.get(cid)
        if meta:
            local_counts.append(meta.n_trees)

    return {
        "global_acc": results.global_accuracy,
        "global_f1": results.global_macro_f1,
        "n_global_trees": results.n_trees_global,
        "avg_hybrid_acc": np.mean(hybrid_accs) if hybrid_accs else 0.0,
        "avg_hybrid_f1": np.mean(hybrid_f1s) if hybrid_f1s else 0.0,
        "local_tree_counts": local_counts,
        "convergence_round": results.convergence_round,
        "elapsed_s": elapsed,
    }


# ── Datasets ──────────────────────────────────────────────────────────────────
DATASETS = {
    "Iris": {
        "type": "Iris",
        "file_path": "data/iris.csv",
        "target_column": "class",
        "test_size": 0.2,
        "scale": True,
        "sep": ",",
    },
    "Car": {
        "type": "Car",
        "file_path": "data/car.csv",
        "target_column": "class",
        "test_size": 0.2,
        "scale": True,
        "sep": ",",
    },
    "Spambase": {
        "type": "Spambase",
        "file_path": "data/spambase.csv",
        "target_column": "class",
        "test_size": 0.2,
        "scale": True,
        "sep": ",",
    },
}


def main() -> None:
    """Executes local_weight optimization sweep."""
    all_results = {}

    for ds_name, ds_cfg in DATASETS.items():
        print(f"\n{'=' * 70}")
        print(f"  OPTIMIZING LOCAL WEIGHT ON DATASET: {ds_name}")
        print(f"{'=' * 70}")

        ds_results = []

        # 1. Run Uniform Voting (use_weighted = False)
        print("  >> Evaluating Uniform (1/N) voting...")
        try:
            r = run_experiment(ds_cfg, use_weighted=False)
            ds_results.append({
                "label": "Uniforme (1/N)",
                "use_weighted": False,
                "local_weight": 0.0,
                **r
            })
            print(f"     Uniform Hybrid Acc: {r['avg_hybrid_acc']:.4f} | Hybrid F1: {r['avg_hybrid_f1']:.4f}")
        except Exception as exc:
            print(f"     ERROR: {exc}")

        # 2. Sweep local_weight from 0.1 to 0.9
        for lw in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            print(f"  >> Evaluating Weighted (local_weight={lw:.1f})...")
            try:
                r = run_experiment(ds_cfg, use_weighted=True, local_weight=lw)
                ds_results.append({
                    "label": f"Ponderado (lw={lw:.1f})",
                    "use_weighted": True,
                    "local_weight": lw,
                    **r
                })
                print(f"     Weighted Hybrid Acc: {r['avg_hybrid_acc']:.4f} | Hybrid F1: {r['avg_hybrid_f1']:.4f}")
            except Exception as exc:
                print(f"     ERROR: {exc}")

        all_results[ds_name] = ds_results

    # ── Summary tables by Dataset ─────────────────────────────────────────────
    print(f"\n\n{'=' * 90}")
    print("  OPTIMIZATION EXPERIMENT DETAILED SUMMARY")
    print(f"{'=' * 90}")

    for ds_name, ds_results in all_results.items():
        print(f"\nDataset: {ds_name}")
        print("-" * 90)
        print(
            f"{'Prediction Mode':<22} {'Global Acc':>10} {'Global F1':>10} "
            f"{'Hybrid Acc':>10} {'Hybrid F1':>10} {'#Glob Trees':>11} {'Local Trees':<15}"
        )
        print("-" * 90)

        best_config = None
        best_f1 = -1.0

        for r in ds_results:
            print(
                f"{r['label']:<22} "
                f"{r['global_acc']:>10.4f} {r['global_f1']:>10.4f} "
                f"{r['avg_hybrid_acc']:>10.4f} {r['avg_hybrid_f1']:>10.4f} "
                f"{r['n_global_trees']:>11} {str(r['local_tree_counts']):<15}"
            )
            # Find the configuration with the highest hybrid F1 score
            if r['avg_hybrid_f1'] > best_f1:
                best_f1 = r['avg_hybrid_f1']
                best_config = r

        print("-" * 90)
        if best_config:
            print(
                f"[BEST] Best configuration for {ds_name}: {best_config['label']} "
                f"with Hybrid F1 = {best_config['avg_hybrid_f1']:.4f} "
                f"(Acc = {best_config['avg_hybrid_acc']:.4f})"
            )
        print("-" * 90)

    # ── Final Global Conclusion ───────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("  FINAL OPTIMIZATION CONCLUSION & RECOMMENDATION")
    print(f"{'=' * 90}")
    
    overall_f1_gains = []
    
    for ds_name, ds_results in all_results.items():
        uniform_cfg = next((r for r in ds_results if not r["use_weighted"]), None)
        best_weighted_cfg = None
        best_wf1 = -1.0
        for r in ds_results:
            if r["use_weighted"] and r["avg_hybrid_f1"] > best_wf1:
                best_wf1 = r["avg_hybrid_f1"]
                best_weighted_cfg = r
                
        if uniform_cfg and best_weighted_cfg:
            gain = best_weighted_cfg["avg_hybrid_f1"] - uniform_cfg["avg_hybrid_f1"]
            overall_f1_gains.append(gain)
            print(
                f"  * {ds_name:<10}: Best Weighted F1 = {best_weighted_cfg['avg_hybrid_f1']:.4f} "
                f"(lw={best_weighted_cfg['local_weight']:.1f}) vs Uniform F1 = {uniform_cfg['avg_hybrid_f1']:.4f}. "
                f"Delta F1: {gain:+.4f}"
            )
            
    avg_gain = np.mean(overall_f1_gains) if overall_f1_gains else 0.0
    print(f"\n  Average F1 gain of the optimal Weighted config over Uniform: {avg_gain:+.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
