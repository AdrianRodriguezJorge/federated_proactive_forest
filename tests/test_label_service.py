import pytest
import numpy as np
from src.domain.services.label_service import SimpleLabelService

def test_label_service_consistency():
    """Verifica que el mapeo de etiquetas sea consistente al entrenar con todas las clases."""
    service = SimpleLabelService()
    
    # En FL real, solemos conocer todas las clases posibles al inicio
    all_classes = np.array(["0", "1", "2"])
    service.fit(all_classes)
    
    assert list(service.classes) == ["0", "1", "2"]
    
    # Transformación de datos parciales
    y_partial = np.array(["0", "2"])
    y_trans = service.transform(y_partial)
    assert np.array_equal(y_trans, [0, 2])

def test_label_service_inverse():
    """Verifica que podemos volver a las etiquetas originales."""
    service = SimpleLabelService()
    y = np.array(["cat", "dog", "bird"])
    service.fit(y)
    
    encoded = service.transform(np.array(["dog", "cat"]))
    decoded = service.inverse_transform(encoded)
    
    assert np.array_equal(decoded, ["dog", "cat"])
