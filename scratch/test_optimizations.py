import numpy as np
import sys
import os

# Añadir src al path
sys.path.append(os.getcwd())

from src.domain.model.cpf_implementation.estimator import DecisionTreeClassifier

def test_correctness():
    print("Iniciando prueba de integridad de las optimizaciones...")
    
    # Crear un dataset sintetico
    X = np.random.rand(200, 10)
    # Regla simple: si la primera característica es > 0.5, clase 1, si no 0.
    y = (X[:, 0] > 0.5).astype(int)
    
    try:
        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(X, y)
        print("OK: Fit completado con exito (Cambio 2 funcionando)")
        
        preds = clf.predict(X)
        print(f"OK: Predict completado con exito (Cambio 1 funcionando)")
        
        acc = np.mean(preds == y)
        print(f"Accuracy en datos simples: {acc:.2f}")
        
        if acc > 0.8:
            print("--- PRUEBA SUPERADA ---")
        else:
            print("--- AVISO: El modelo no aprendio la regla simple, revisar logica ---")
            
    except Exception as e:
        print(f"ERROR durante la prueba: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_correctness()
