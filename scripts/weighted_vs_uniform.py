"""Experiment: use_weighted=True vs False on 2 datasets.

Compares hybrid prediction metrics when weighting by origin (local/global)
vs uniform tree voting.
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.application.orchestrators import FLEXOrchestrator
from src.infrastructure.dataset.dataset_factory import DatasetFactory


def run_experiment(
    dataset_cfg: Dict[str, Any],
    use_weighted: bool,
    strategy: str = "s6_perclient_f1",
    seed: int = 42,
) -> Dict[str, Any]:
    """Runs a single federated experiment and returns comparative metrics.

    Args:
        dataset_cfg (Dict[str, Any]): Config map for the dataset adapter.
        use_weighted (bool): Whether to weigh global vs local trees.
        strategy (str): Aggregation strategy identifier.
        seed (int): Random number generator seed.

    Returns:
        Dict[str, Any]: Dict containing global and per-client metrics.
    """
    from src.domain.aggregation.aggregation_factory import AggregationFactory
    norm_strat = AggregationFactory.normalize_strategy_name(strategy)

    global_ep_size = 5
    trees_per_client_ep = 1
    trees_per_rnd_client = 1
    win_size = 5

    if norm_strat == "S4":
        global_ep_size = 10
    elif norm_strat == "S7":
        trees_per_client_ep = 3
    elif norm_strat == "PW":
        win_size = 10
        trees_per_rnd_client = 3

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
            "f1_weight": 0.5,
            "pcd_weight": 0.5,
            "global_convergence_threshold": 0.002,
            "convergence_threshold": 0.002,
            "global_episode_size": global_ep_size,
            "trees_per_client_per_episode": trees_per_client_ep,
            "trees_per_round_per_client": trees_per_rnd_client,
            "window_size": win_size,
            "max_rounds": 20,
            "min_episodes": 3,
            "min_rounds": 3,
        },
        "prediction": {
            "local_weight": 0.4,
            "global_weight": 0.6,
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

    # Per-client local tree counts and global tree count
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
        "hybrid_accs": hybrid_accs,
        "hybrid_f1s": hybrid_f1s,
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
    """Executes the comparative weighted vs uniform experiment."""
    results_table = []

    for ds_name, ds_cfg in DATASETS.items():
        print(f"\n{'=' * 60}")
        print(f"  Dataset: {ds_name}")
        print(f"{'=' * 60}")

        for weighted in [True, False]:
            label = "Ponderado (lambda)" if weighted else "Uniforme (1/N)"
            print(f"\n  >> {label} (use_weighted={weighted})...")
            try:
                r = run_experiment(ds_cfg, use_weighted=weighted)
                results_table.append(
                    {
                        "dataset": ds_name,
                        "mode": label,
                        "use_weighted": weighted,
                        **r,
                    }
                )
                print(
                    f"     Global:  Acc={r['global_acc']:.4f}  "
                    f"F1={r['global_f1']:.4f}  Trees={r['n_global_trees']}"
                )
                print(
                    f"     Hybrid:  Acc={r['avg_hybrid_acc']:.4f}  "
                    f"F1={r['avg_hybrid_f1']:.4f}"
                )
                print(f"     Local trees per client: {r['local_tree_counts']}")
                print(f"     Convergence round: {r['convergence_round']}")
                print(f"     Time: {r['elapsed_s']:.1f}s")
            except Exception as exc:
                import traceback

                print(f"     ERROR: {exc}")
                traceback.print_exc()

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 80}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 80}")
    print(
        f"{'Dataset':<12} {'Mode':<22} {'Global Acc':>10} "
        f"{'Global F1':>10} {'Hybrid Acc':>10} {'Hybrid F1':>10} "
        f"{'#Trees':>7}"
    )
    print("-" * 83)

    for r in results_table:
        print(
            f"{r['dataset']:<12} {r['mode']:<22} "
            f"{r['global_acc']:>10.4f} {r['global_f1']:>10.4f} "
            f"{r['avg_hybrid_acc']:>10.4f} {r['avg_hybrid_f1']:>10.4f} "
            f"{r['n_global_trees']:>7}"
        )

    # ── Differences ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("  DIFFERENCES (Ponderado - Uniforme)")
    print(f"{'=' * 80}")

    for ds_name in DATASETS:
        weighted_r = next(
            (
                res
                for res in results_table
                if res["dataset"] == ds_name and res["use_weighted"]
            ),
            None,
        )
        uniform_r = next(
            (
                res
                for res in results_table
                if res["dataset"] == ds_name and not res["use_weighted"]
            ),
            None,
        )
        if weighted_r and uniform_r:
            d_acc = (
                weighted_r["avg_hybrid_acc"] - uniform_r["avg_hybrid_acc"]
            )
            d_f1 = weighted_r["avg_hybrid_f1"] - uniform_r["avg_hybrid_f1"]
            print(
                f"  {ds_name:<12}  Delta Hybrid Acc: {d_acc:+.4f}   "
                f"Delta Hybrid F1: {d_f1:+.4f}"
            )
            print(
                f"               Local trees: "
                f"{weighted_r['local_tree_counts']}  Global trees: "
                f"{weighted_r['n_global_trees']}"
            )
            ratio = (
                max(weighted_r["local_tree_counts"])
                / weighted_r["n_global_trees"]
                if weighted_r["n_global_trees"] > 0
                else 0
            )
            print(f"               Ratio max_local/global: {ratio:.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
