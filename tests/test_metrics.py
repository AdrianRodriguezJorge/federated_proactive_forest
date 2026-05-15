import pytest
import numpy as np
from src.infrastructure.metrics.diversity_service import PredictionBasedDiversityService

def test_pcd_calculation_zero_diversity():
    """Si todos los modelos predicen lo mismo, la diversidad PCD debe ser 0."""
    y_true = np.array([0, 1, 0, 1])
    # Tres modelos que predicen exactamente lo mismo (y todo correcto)
    # Shape debe ser (n_classifiers, n_samples) para transponerlo si es necesario, 
    # pero calculate_pcd espera (n_samples, n_classifiers)
    predictions = np.array([
        [0, 0, 0], # Muestra 0: todos dicen 0 (Acierto total)
        [1, 1, 1], # Muestra 1: todos dicen 1 (Acierto total)
        [0, 0, 0], # Muestra 2: todos dicen 0 (Acierto total)
        [1, 1, 1]  # Muestra 3: todos dicen 1 (Acierto total)
    ])
    service = PredictionBasedDiversityService()
    pcd = service.calculate_pcd(predictions, y_true)
    assert pcd == 0.0

def test_pcd_calculation_high_diversity():
    """Si los modelos fallan en diferentes instancias, el PCD debe ser > 0."""
    y_true = np.array([0, 0, 0, 0])
    # 10 modelos para que el rango 10% - 90% sea amplio (1 a 9 aciertos)
    # Creamos una matriz donde en cada muestra hay diversidad de aciertos
    predictions = np.zeros((4, 10))
    predictions[0, 0] = 1 # 1 fallo, 9 aciertos -> diverso
    predictions[1, 0:5] = 1 # 5 fallos, 5 aciertos -> muy diverso
    
    service = PredictionBasedDiversityService()
    pcd = service.calculate_pcd(predictions, y_true)
    assert pcd > 0.0
