import pytest
import numpy as np
from src.domain.aggregation.strategies.s1_simple_pool import S1SimplePoolStrategy
from src.domain.model.proactive_forest import ProactiveForest
from src.domain.metadata.client_metadata import ClientMetadata

def test_s1_aggregation_logic():
    """Verifica que la agregación S1 combine correctamente los árboles de los clientes."""
    strategy = S1SimplePoolStrategy()
    
    # Dataset sintético para el fit
    X = np.random.rand(20, 2)
    y = np.random.randint(0, 2, 20)

    # Simulamos 2 clientes con sus bosques entrenados
    f1 = ProactiveForest(n_estimators=2)
    f1.fit(X, y)
    f2 = ProactiveForest(n_estimators=2)
    f2.fit(X, y)
    
    # S1 espera un Dict[client_id, List[Trees]]
    client_trees = {
        "client_0": f1.estimators_,
        "client_1": f2.estimators_
    }
    
    # S1 espera metadatos de clientes como objetos ClientMetadata
    client_metadata = {
        "client_0": ClientMetadata(client_id="client_0", accuracy=0.8),
        "client_1": ClientMetadata(client_id="client_1", accuracy=0.7)
    }
    
    global_trees, selected_indices, all_entries, _, _ = strategy.aggregate(
        client_trees, 
        client_metadata
    )
    
    # S1 simplemente suma todos los árboles
    expected_total = len(f1.estimators_) + len(f2.estimators_)
    assert len(global_trees) == expected_total
    assert "client_0" in selected_indices
    assert "client_1" in selected_indices
