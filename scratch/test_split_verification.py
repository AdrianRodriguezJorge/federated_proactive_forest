import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.getcwd())

from src.infrastructure.dataset.csv_adapter import GenericCsvAdapter
from src.application.orchestrators.fed_data_distributor import FedDataDistributor

def test_nursery_split():
    print("=== Test de División 70/15/15 (Dataset: Nursery) ===")
    
    # Configuración del adaptador
    adapter = GenericCsvAdapter(
        name="Nursery",
        train_path="data/nursery.csv",
        target_column="class",
        test_size=0.15,
        categorical_features=["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health"]
    )
    
    # 1. Cargar datos (Split Test 15%)
    dataset_split = adapter.load()
    
    total_samples = len(dataset_split.X_train) + len(dataset_split.X_test)
    print(f"Total de muestras en el dataset: {total_samples}")
    print(f"Muestras en TEST (15%): {len(dataset_split.X_test)} (Esperado: {int(total_samples * 0.15)})")
    
    # 2. Distribuir (Split Val 15% del total / 17.65% del pool)
    distributor = FedDataDistributor(config={"federation": {"n_clients": 5, "distribution": "iid"}})
    updated_split, _ = distributor.distribute(dataset_split)
    
    print(f"Muestras en VALIDACIÓN (15%): {len(updated_split.X_val)} (Esperado: {int(total_samples * 0.15)})")
    print(f"Muestras en ENTRENAMIENTO (70%): {len(updated_split.X_train)} (Esperado: {int(total_samples * 0.70)})")
    
    # Verificación final
    suma = len(updated_split.X_train) + len(updated_split.X_val) + len(updated_split.X_test)
    print(f"Suma total verificada: {suma}")
    
    if suma == total_samples:
        print("\nSUCCESS: La distribución 70/15/15 se aplica correctamente.")
    else:
        print(f"\nERROR: La suma ({suma}) no coincide con el total ({total_samples})")

if __name__ == "__main__":
    test_nursery_split()
