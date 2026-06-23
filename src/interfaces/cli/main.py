"""CLI tool for executing Federated Proactive Forest experiments from YAML.

Usage:
    python -m src.interfaces.cli.main --config configs/experiments/exp_s1.yaml
"""

import argparse
import os
import sys
from typing import Any, Dict

import numpy as np
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, ROOT)

from src.application.orchestrators import FLEXOrchestrator
from src.application.orchestrators.roulette_orchestrator import RouletteOrchestrator
from src.domain.aggregation.aggregation_factory import AggregationFactory
from src.domain.dataset.base_adapter import DatasetSplit
from src.infrastructure.dataset.dataset_factory import DatasetFactory


def load_dataset(cfg: Dict[str, Any]) -> DatasetSplit:
    """Loads target dataset split based on configuration dict.

    Args:
        cfg (Dict[str, Any]): Entire experiments configuration.

    Returns:
        DatasetSplit: Initialized train/test dataset split.
    """
    return DatasetFactory.load_from_config(cfg["dataset"])


def get_default_config() -> Dict[str, Any]:
    """Provides standard default configuration dict.

    Returns:
        Dict[str, Any]: Default configuration dict.
    """
    return {
        "dataset": {
            "type": "Iris",
            "file_path": "",
            "target_column": "class",
            "test_size": 0.15,
            "scale": True,
            "scaler_type": "standard",
        },
        "federation": {
            "n_clients": 3,
            "distribution": "iid",
            "dirichlet_alpha": 0.5,
        },
        "model": {
            "n_estimators": 100,
            "alpha_pf": 0.1,
            "split_criterion": "entropy",
            "feature_selection": "prob",
            "use_progressive_stopping": True,
            "local_convergence_threshold": 0.002,
            "local_episode_size": 5,
        },
        "aggregation": {
            "strategy": "s6_perclient_f1",
            "f1_weight": 0.5,
            "pcd_weight": 0.5,
            "global_convergence_threshold": 0.002,
            "global_episode_size": 5,
            "window_size": 5,
            "max_rounds": 20,
            "alpha_pf": 0.5,
        },
        "prediction": {
            "local_weight": 0.4,
            "global_weight": 0.6,
            "use_weighted": True,
        },
        "verbose": False,
        "seed": 42,
    }


def main() -> None:
    """Executes the federated training round from CLI configurations."""
    parser = argparse.ArgumentParser(
        description="Run federated proactively forest experiments."
    )
    parser.add_argument(
        "--config", required=False, help="Path to experiment YAML config file."
    )
    parser.add_argument(
        "--dataset", required=False, help="Target dataset name (e.g. Iris, Letter)."
    )
    parser.add_argument(
        "--clients", required=False, type=int, help="Number of federated clients."
    )
    parser.add_argument(
        "--strategy", required=False, help="Aggregation strategy (e.g. S1, s7_perclient_f1_pcd, S8)."
    )
    parser.add_argument(
        "--trees", required=False, type=int, help="Maximum number of trees per client."
    )
    parser.add_argument(
        "--seed", required=False, type=int, help="Random seed for reproducibility."
    )
    args = parser.parse_args()

    # Load base configuration
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = get_default_config()

    # Apply overrides
    if args.dataset:
        cfg["dataset"]["type"] = args.dataset
    if args.clients:
        cfg["federation"]["n_clients"] = args.clients
    if args.strategy:
        cfg["aggregation"]["strategy"] = args.strategy
    if args.trees:
        cfg["model"]["n_estimators"] = args.trees
        cfg["aggregation"]["max_trees"] = args.trees
    if args.seed:
        cfg["seed"] = args.seed
        if "federation" in cfg:
            cfg["federation"]["seed"] = args.seed

    seed = cfg.get("seed", 42)
    np.random.seed(seed)

    print("[CLI] Cargando dataset...")
    ds = load_dataset(cfg)
    cfg["_dataset_split"] = ds
    print(
        f"[CLI] Dataset '{ds.dataset_name}': "
        f"{ds.X_train.shape[0]} train / {ds.X_test.shape[0]} test, "
        f"{len(ds.class_names)} clases"
    )

    def cb(msg: str, pct: int, detail: str = "") -> None:
        print(f"  [{pct:3d}%] {msg} {detail}")

    strategy_key = cfg.get("aggregation", {}).get("strategy", "S1")
    normalized_name = AggregationFactory.normalize_strategy_name(str(strategy_key))
    is_s8 = (normalized_name == "S8")

    if is_s8:
        print(f"[CLI] Usando RouletteOrchestrator para estrategia {normalized_name}...")
        orch = RouletteOrchestrator(cfg, step_callback=cb)
    else:
        print(f"[CLI] Usando FLEXOrchestrator para estrategia {normalized_name}...")
        orch = FLEXOrchestrator.from_config(cfg, step_callback=cb)

    orch.setup_federation(cfg.get("_dataset_split"))
    results = orch.run_federated_round()

    print("\n" + "=" * 60)
    print(f"Estrategia:      {results.strategy_id}")
    print(f"Accuracy global: {results.global_accuracy:.4f}")
    print(f"Macro-F1 global: {results.global_macro_f1:.4f}")
    if is_s8:
        print(f"Rondas totales:  {results.num_rounds}")
        print(f"Coste Comm.:     {results.communication_cost / 1024:.2f} KB")
    else:
        print(f"Árboles global:  {results.n_trees_global}")

    print("\nPor cliente:")
    for cid in results.client_ids:
        acc = results.client_accuracies.get(cid, 0.0)
        f1 = results.client_f1_scores.get(cid, 0.0)
        meta = results.client_metadata.get(cid)
        sel = len(results.selected_ids.get(str(cid), [])) if results.selected_ids else 0
        report = results.client_reports.get(cid)
        pcd = report.pcd if report else 0.0

        if meta:
            # Safe property getters that work on both ClientMetadata instances and dicts
            get_val = lambda key, default: meta.get(key, default) if isinstance(meta, dict) else getattr(meta, key, default)

            n_trees = get_val("n_trees", 0)
            has_converged = get_val("has_converged", False)
            stop_counter = get_val("stop_counter", 0)

            if is_s8:
                print(
                    f"  {cid}: acc={acc:.4f} | f1={f1:.4f} | "
                    f"convergido={has_converged} | rondas_loc={stop_counter}"
                )
            else:
                print(
                    f"  {cid}: acc={acc:.4f} | f1={f1:.4f} | "
                    f"pcd={pcd:.4f} | árboles_loc={n_trees} | "
                    f"sel={sel}"
                )
        else:
            print(f"  {cid}: acc={acc:.4f} | f1={f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
