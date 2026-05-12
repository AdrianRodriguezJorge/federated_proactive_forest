import os
import sys

# Añadir src al path
sys.path.append(os.getcwd())

import scripts.final_benchmark as fb

def run_integration_test():
    print("=== INICIANDO TEST DE INTEGRACIÓN RÁPIDO ===")
    
    # Sobrescribir constantes para que el test sea instantáneo
    fb.REPETITIONS = 1
    fb.K_FOLDS = 2
    fb.N_CLIENTS = 2
    fb.STRATEGIES = ["s1_simple_pool", "s6_perclient_f1", "pw"]
    fb.DATASETS = ["Iris"]
    fb.RESULTS_FILE = "results/results_integration_test.csv"
    
    # Limpiar archivo previo si existe
    if os.path.exists(fb.RESULTS_FILE):
        os.remove(fb.RESULTS_FILE)
        
    try:
        fb.run_final_benchmark()
        
        # Validar si se creó el archivo y hay resultados
        if not os.path.exists(fb.RESULTS_FILE):
            print("ERROR: No se generó el archivo de resultados.")
            return False
            
        import pandas as pd
        df = pd.read_csv(fb.RESULTS_FILE)
        print(f"\nResultados generados exitosamente: {len(df)} filas.")
        print("El sistema no se rompió y completó el ciclo HFL correctamente con todas las optimizaciones.")
        return True
    except Exception as e:
        print(f"\nERROR CRÍTICO DURANTE LA EJECUCIÓN: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    run_integration_test()
