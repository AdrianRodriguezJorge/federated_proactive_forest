import pytest
import numpy as np
from src.domain.model.proactive_forest import ProactiveForest

def test_proactive_forest_fit_predict():
    """Prueba básica de entrenamiento y predicción."""
    # Dataset sintético simple
    X = np.random.rand(50, 2)
    y = np.random.randint(0, 2, 50)
    
    model = ProactiveForest(n_estimators=5, alpha=0.1)
    model.fit(X, y)
    
    assert len(model.estimators_) > 0
    
    preds = model.predict(X)
    assert preds.shape == (50,)
    # Las predicciones vienen como strings por el LabelService interno
    assert np.all(np.isin(preds.astype(str), ["0", "1"]))

def test_proactive_forest_progressive_stopping():
    """Verifica que el criterio de parada progresivo funcione."""
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    # Con un umbral de convergencia muy alto, debería parar pronto
    # El parámetro es convergence_threshold
    model = ProactiveForest(
        n_estimators=100, 
        convergence_threshold=0.5 # Muy alto, parará casi de inmediato
    )
    model.fit(X, y)
    
    # Debería parar antes de los 100 árboles
    assert len(model.estimators_) < 100
