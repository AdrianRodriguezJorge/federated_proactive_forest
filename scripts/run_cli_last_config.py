import json
import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.getcwd())

from src.application.orchestrators.fl_orchestrator import FLEXOrchestrator
from src.infrastructure.dataset.dataset_factory import DatasetFactory

def run_last_config():
    config_path = Path("configs/last_config.json")
    if not config_path.exists():
        print("Error: No se encontró configs/last_config.json")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"Cargada configuración: {config['dataset']['type']} con {config['federation']['n_clients']} clientes.")
    print(f"Estrategia: {config['aggregation']['strategy']} | Umbral Global: {config['aggregation'].get('convergence')}")
    print(f"Umbral Local (CPF): {config['model'].get('convergence')}")

    # Load data using the same factory as UI
    ds = DatasetFactory.load_from_config(config["dataset"])

    orch = FLEXOrchestrator(config)
    print("\nConfigurando federación...")
    orch.setup_federation(ds)
    
    print("Ejecutando ronda federada...")
    results = orch.run_federated_round()
    
    print("\n" + "="*40)
    print("RESULTADOS FINALES (CLI)")
    print("="*40)
    print(f"Estatus: {'Convergió' if results.convergence_round else 'Límite alcanzado'}")
    if results.convergence_round:
        print(f"Ronda de convergencia: {results.convergence_round}")
    
    print(f"\nÁrboles en el Bosque Global Final: {results.n_trees_global}")
    print("\nDetalle por Cliente (Post-Actualización Híbrida):")
    for cid in results.client_ids:
        meta = results.client_metadata.get(cid)
        sel_count = len(results.selected_ids.get(str(cid), []))
        hybrid_size = results.client_hybrid_forest_sizes.get(cid, 0)
        
        if meta:
            print(f"  - Cliente {cid}:")
            print(f"    * Árboles entrenados localmente: {int(meta.n_trees)}")
            print(f"    * Árboles seleccionados para el global: {sel_count}")
            print(f"    * Tamaño del Bosque Híbrido (Local + Global Externo): {hybrid_size}")
            
            # Explain the hybrid size
            if hybrid_size > int(meta.n_trees):
                external = hybrid_size - int(meta.n_trees)
                print(f"      (Incluye {external} árboles externos del modelo global)")
            else:
                print(f"      (No se añadieron árboles externos; el modelo global está compuesto por sus propios árboles)")
    
    print("="*40)

if __name__ == "__main__":
    try:
        run_last_config()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
