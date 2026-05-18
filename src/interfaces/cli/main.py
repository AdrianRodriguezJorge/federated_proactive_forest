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


def main() -> None:
    """Executes the federated training round from CLI configurations."""
    parser = argparse.ArgumentParser(
        description="Run federated proactively forest experiments from YAML."
    )
    parser.add_argument(
        "--config", required=True, help="Path to experiment YAML config file."
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

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

    orch = FLEXOrchestrator.from_config(cfg, step_callback=cb)
    orch.setup_federation(cfg.get("_dataset_split"))
    results = orch.run_federated_round()

    print("\n" + "=" * 60)
    print(f"Estrategia:      {results.strategy_id}")
    print(f"Accuracy global: {results.global_accuracy:.4f}")
    print(f"Macro-F1 global: {results.global_macro_f1:.4f}")
    print(f"Árboles global:  {results.n_trees_global}")
    print("\nPor cliente:")
    for cid in results.client_ids:
        acc = results.client_accuracies.get(cid, 0.0)
        f1 = results.client_f1_scores.get(cid, 0.0)
        meta = results.client_metadata.get(cid)
        sel = len(results.selected_ids.get(str(cid), []))
        if meta:
            print(
                f"  {cid}: acc={acc:.4f} | f1={f1:.4f} | "
                f"pcd={meta.pcd:.4f} | árboles_loc={meta.n_trees} | "
                f"sel={sel}"
            )
        else:
            print(f"  {cid}: acc={acc:.4f} | f1={f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
