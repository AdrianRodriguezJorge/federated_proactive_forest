"""FLEX evaluation primitives for Proactive Forest."""
from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def evaluate_global_pf_model(server_flex_model: Dict[str, Any], 
                              X_test: np.ndarray,
                              y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate global model on server side using test data.
    
    FLEX Primitive for server evaluation.
    
    Args:
        server_flex_model: Server model containing trained global_model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Dict with accuracy and f1_score
    """
    model = server_flex_model.get('model')
    if model is None:
        return {'accuracy': 0.0, 'macro_f1': 0.0}

    try:
        predictions = model.predict(X_test)
        accuracy = float(accuracy_score(y_test, predictions))
        macro_f1 = float(f1_score(y_test, predictions, average='macro', zero_division=0))

        server_flex_model['global_accuracy'] = accuracy
        server_flex_model['global_f1'] = macro_f1
        
        return {'accuracy': accuracy, 'macro_f1': macro_f1}
    except Exception as e:
        print(f"[ERROR] evaluate_global_pf_model: {e}")
        return {'accuracy': 0.0, 'macro_f1': 0.0}


def evaluate_global_pf_model_at_clients(client_flex_model: Dict[str, Any]) -> Dict[str, float]:
    """
    Evaluate global model at client side using local test data.
    
    FLEX Primitive for client-side global model evaluation.
    
    Returns:
        Dict with accuracy and f1_score
    """
    global_model = client_flex_model.get('global_model')
    if global_model is None:
        return {'accuracy': 0.0, 'macro_f1': 0.0}

    X_test = client_flex_model.get('X_test')
    y_test = client_flex_model.get('y_test')

    if X_test is None or y_test is None:
        return {'accuracy': 0.0, 'macro_f1': 0.0}

    try:
        predictions = global_model.predict(X_test)
        accuracy = float(accuracy_score(y_test, predictions))
        macro_f1 = float(f1_score(y_test, predictions, average='macro', zero_division=0))

        client_flex_model['global_accuracy'] = accuracy
        client_flex_model['global_f1'] = macro_f1
        
        return {'accuracy': accuracy, 'macro_f1': macro_f1}
    except Exception as e:
        print(f"[ERROR] evaluate_global_pf_model_at_clients: {e}")
        return {'accuracy': 0.0, 'macro_f1': 0.0}


def evaluate_local_pf_model_at_clients(client_flex_model: Dict[str, Any]) -> Dict[str, float]:
    """
    Evaluate local (client's own) model using local test data.
    
    FLEX Primitive for client-side local model evaluation.
    
    Returns:
        Dict with accuracy and f1_score
    """
    local_model = client_flex_model.get('model')
    if local_model is None:
        return {'accuracy': 0.0, 'macro_f1': 0.0}

    X_test = client_flex_model.get('X_test')
    y_test = client_flex_model.get('y_test')

    if X_test is None or y_test is None:
        return {'accuracy': 0.0, 'macro_f1': 0.0}

    try:
        predictions = local_model.predict(X_test)
        accuracy = float(accuracy_score(y_test, predictions))
        macro_f1 = float(f1_score(y_test, predictions, average='macro', zero_division=0))

        client_flex_model['local_accuracy'] = accuracy
        client_flex_model['local_f1'] = macro_f1
        
        return {'accuracy': accuracy, 'macro_f1': macro_f1}
    except Exception as e:
        print(f"[ERROR] evaluate_local_pf_model_at_clients: {e}")
        return {'accuracy': 0.0, 'macro_f1': 0.0}


__all__ = [
    'evaluate_global_pf_model',
    'evaluate_global_pf_model_at_clients', 
    'evaluate_local_pf_model_at_clients'
]