"""Script to execute a federated round using the last saved UI configuration."""

import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.getcwd())

from src.application.orchestrators import FLEXOrchestrator
from src.application.orchestrators.roulette_orchestrator import (
    RouletteOrchestrator,
)
from src.infrastructure.dataset.dataset_factory import DatasetFactory


def run_last_config() -> None:
    """Loads configs/last_config.json and runs a federated round."""
    config_path = Path("configs/last_config.json")
    if not config_path.exists():
        print("Error: No se encontró configs/last_config.json")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    ds_type = config["dataset"]["type"]
    n_cl = config["federation"]["n_clients"]
    strat = config["aggregation"]["strategy"]
    glob_conv = config["aggregation"].get("global_convergence_threshold")
    loc_conv = config["model"].get("local_convergence_threshold")

    print(f"Cargada configuración: {ds_type} con {n_cl} clientes.")
    print(f"Estrategia: {strat} | Umbral Global: {glob_conv}")
    print(f"Umbral Local (CPF): {loc_conv}")

    # Load data using the same factory as UI
    ds = DatasetFactory.load_from_config(config["dataset"])

    is_s9 = strat == "s9_roulette"
    if is_s9:
        orch = RouletteOrchestrator(config)
        print("\nConfigurando federación (S9)...")
        orch.setup_federation(ds, seed=config.get("seed", 42))
    else:
        orch = FLEXOrchestrator(config)
        print("\nConfigurando federación...")
        orch.setup_federation(ds)

    print("Ejecutando ronda federada...")
    results = orch.run_federated_round()

    print("\n" + "=" * 40)
    print("RESULTADOS FINALES (CLI)")
    print("=" * 40)
    status_text = (
        "Convergió" if results.convergence_round else "Límite alcanzado"
    )
    print(f"Estatus: {status_text}")
    if results.convergence_round:
        print(f"Ronda de convergencia: {results.convergence_round}")

    if not is_s9:
        print(f"\nÁrboles en el Bosque Global Final: {results.n_trees_global}")

    print("\nDetalle por Cliente:")
    for cid in results.client_ids:
        meta = results.client_metadata.get(cid)
        sel_count = len(results.selected_ids.get(str(cid), []))
        hybrid_size = results.client_hybrid_forest_sizes.get(cid, 0)

        if meta:
            n_trees_val = int(getattr(meta, "n_trees", 0))
            print(f"  - Cliente {cid}:")
            print(f"    * Árboles entrenados localmente: {n_trees_val}")
            if not is_s9:
                print(f"    * Árboles seleccionados para el global: {sel_count}")
                print(
                    f"    * Tamaño del Bosque Híbrido (Local + Global "
                    f"Externo): {hybrid_size}"
                )

                # Explain the hybrid size
                if hybrid_size > n_trees_val:
                    external = hybrid_size - n_trees_val
                    print(
                        f"      (Incluye {external} árboles externos del "
                        f"modelo global)"
                    )
                else:
                    print(
                        "      (No se añadieron árboles externos; el modelo "
                        "global está compuesto por sus propios árboles)"
                    )
            else:
                rep = results.client_reports.get(cid)
                if rep:
                    print(f"    * Accuracy: {rep.accuracy:.4f}")
                    print(f"    * Macro-F1: {rep.macro_f1:.4f}")

    print("=" * 40)


if __name__ == "__main__":
    try:
        run_last_config()
    except Exception as exc:
        import traceback

        traceback.print_exc()
        sys.exit(1)
