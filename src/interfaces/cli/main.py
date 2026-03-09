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


def load_dataset(cfg: dict):
    d = cfg["dataset"]
    t = d.get("type", "NSL-KDD")
    if t == "NSL-KDD":
        from src.infrastructure.dataset.nslkdd_adapter import NslKddAdapter
        return NslKddAdapter(train_path=d["train_path"], test_path=d["test_path"],
                             scale=d.get("scale", True),
                             scaler_type=d.get("scaler_type", "standard")).load()
    else:
        from src.infrastructure.dataset.csv_adapter import GenericCsvAdapter
        return GenericCsvAdapter(
            name=d.get("name", "custom"),
            train_path=d["train_path"],
            test_path=d.get("test_path") or None,
            target_column=d["target_column"],
            categorical_features=d.get("categorical_features", []),
            scale=d.get("scale", True),
            scaler_type=d.get("scaler_type", "standard"),
        ).load()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Ruta al YAML de experimento")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    np.random.seed(seed)

    print(f"[CLI] Cargando dataset...")
    ds = load_dataset(cfg)
    cfg["_dataset_split"] = ds
    print(f"[CLI] Dataset '{ds.dataset_name}': "
          f"{ds.X_train.shape[0]} train / {ds.X_test.shape[0]} test, "
          f"{len(ds.class_names)} clases")

    from src.application.fl_orchestrator import FLOrchestrator

    def cb(msg, pct, detail=""):
        print(f"  [{pct:3d}%] {msg} {detail}")

    orch    = FLOrchestrator.from_config(cfg, step_callback=cb)
    results = orch.run()

    print("\n" + "=" * 60)
    print(f"Estrategia:      {results.strategy_id}")
    print(f"Accuracy global: {results.global_accuracy:.4f}")
    print(f"Macro-F1 global: {results.global_macro_f1:.4f}")
    print(f"Árboles global:  {results.n_trees_global}")
    print("\nPor cliente:")
    for cid in results.client_ids:
        rep  = results.client_reports[cid]
        meta = results.client_metadata[cid]
        sel  = len(results.selected_ids.get(cid, []))
        print(f"  {cid}: acc={rep.accuracy:.4f} | f1={rep.macro_f1:.4f} | "
              f"pcd={meta.pcd:.4f} | árboles_loc={meta.n_trees} | sel={sel}")
    print("=" * 60)


if __name__ == "__main__":
    main()
