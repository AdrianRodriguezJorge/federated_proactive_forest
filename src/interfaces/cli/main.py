"""
CLI para ejecutar experimentos desde YAML.
Uso: python -m src.interfaces.cli.main --config configs/experiments/exp001_s1_simple.yaml
"""
import argparse
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, ROOT)

import yaml
import numpy as np


from src.infrastructure.dataset.dataset_factory import DatasetFactory

def load_dataset(cfg: dict):
    return DatasetFactory.load_from_config(cfg["dataset"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Ruta al YAML de experimento")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    np.random.seed(seed)

    print("[CLI] Cargando dataset...")
    ds = load_dataset(cfg)
    cfg["_dataset_split"] = ds
    print(f"[CLI] Dataset '{ds.dataset_name}': "
          f"{ds.X_train.shape[0]} train / {ds.X_test.shape[0]} test, "
          f"{len(ds.class_names)} clases")

    from src.application.orchestrators import FLEXOrchestrator

    def cb(msg, pct, detail=""):
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
        sel = len(results.selected_ids.get(cid, []))
        if meta:
            print(f"  {cid}: acc={acc:.4f} | f1={f1:.4f} | "
                  f"pcd={meta.pcd:.4f} | árboles_loc={meta.n_trees} | sel={sel}")
        else:
            print(f"  {cid}: acc={acc:.4f} | f1={f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
