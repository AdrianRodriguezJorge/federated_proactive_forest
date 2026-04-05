"""FLEX evaluation primitives for Proactive Forest."""
from typing import Dict, Any, Optional
import warnings
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
        
        # Ensure y_test and predictions are in the same format
        y_test_arr = np.asarray(y_test)
        predictions_arr = np.asarray(predictions)
        
        # If types don't match, convert both to integers using class_names as mapping
        if len(y_test_arr) > 0 and len(predictions_arr) > 0:
            y_is_string = isinstance(y_test_arr.flat[0], (str, np.str_))
            pred_is_string = isinstance(predictions_arr.flat[0], (str, np.str_))

            if y_is_string != pred_is_string:
                # Type mismatch - need to convert to same format
                # Get class_names from model if available
                class_names = getattr(model, 'class_names', None)
                if class_names is not None:
                    # Create mapping from class name to index
                    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

                    # Convert y_test to indices if it's strings
                    if y_is_string:
                        y_test_arr = np.array([class_to_idx.get(str(y), int(y)) for y in y_test_arr])

                    # Convert predictions to indices if they're strings
                    if pred_is_string:
                        predictions_arr = np.array([class_to_idx.get(str(p), -1) for p in predictions_arr])
                        # Check if any conversion failed
                        if -1 in predictions_arr:
                            failed_preds = [p for p in predictions_arr if p == -1]
                            raise ValueError(f"Failed to convert predictions to indices: {failed_preds[:5]}. Available classes: {class_names}")
                else:
                    raise ValueError(f"Type mismatch but no class_names available. y_test type: {y_test_arr.dtype}, predictions type: {predictions_arr.dtype}")
        
        accuracy = float(accuracy_score(y_test_arr, predictions_arr))
        macro_f1 = float(f1_score(y_test_arr, predictions_arr, average='macro', zero_division=0))

        server_flex_model['global_accuracy'] = accuracy
        server_flex_model['global_f1'] = macro_f1

        return {'accuracy': accuracy, 'macro_f1': macro_f1}
    except Exception as e:
        warnings.warn(f"evaluate_global_pf_model failed: {e}")
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
        
        # Ensure y_test and predictions are in the same format
        y_test_arr = np.asarray(y_test)
        predictions_arr = np.asarray(predictions)
        
        # If types don't match, convert both to integers using class_names as mapping
        if len(y_test_arr) > 0 and len(predictions_arr) > 0:
            y_is_string = isinstance(y_test_arr.flat[0], (str, np.str_))
            pred_is_string = isinstance(predictions_arr.flat[0], (str, np.str_))

            if y_is_string != pred_is_string:
                # Type mismatch - need to convert to same format
                # Get class_names from model if available
                class_names = getattr(global_model, 'class_names', None)
                if class_names is not None:
                    # Create mapping from class name to index
                    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

                    # Convert y_test to indices if it's strings
                    if y_is_string:
                        y_test_arr = np.array([class_to_idx.get(str(y), int(y)) for y in y_test_arr])

                    # Convert predictions to indices if they're strings
                    if pred_is_string:
                        predictions_arr = np.array([class_to_idx.get(str(p), -1) for p in predictions_arr])
                        # Check if any conversion failed
                        if -1 in predictions_arr:
                            failed_preds = [p for p in predictions_arr if p == -1]
                            raise ValueError(f"Failed to convert predictions to indices: {failed_preds[:5]}. Available classes: {class_names}")
                else:
                    raise ValueError(f"Type mismatch but no class_names available. y_test type: {y_test_arr.dtype}, predictions type: {predictions_arr.dtype}")
        
        accuracy = float(accuracy_score(y_test_arr, predictions_arr))
        macro_f1 = float(f1_score(y_test_arr, predictions_arr, average='macro', zero_division=0))

        client_flex_model['global_accuracy'] = accuracy
        client_flex_model['global_f1'] = macro_f1

        return {'accuracy': accuracy, 'macro_f1': macro_f1}
    except Exception as e:
        warnings.warn(f"evaluate_global_pf_model_at_clients failed: {e}")
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

        # Ensure y_test and predictions are in the same format
        # The issue: y_test might be strings while predictions are integers (or vice versa)
        y_test_arr = np.asarray(y_test)
        predictions_arr = np.asarray(predictions)

        # If types don't match, convert both to integers using class_names as mapping
        if len(y_test_arr) > 0 and len(predictions_arr) > 0:
            y_is_string = isinstance(y_test_arr.flat[0], (str, np.str_))
            pred_is_string = isinstance(predictions_arr.flat[0], (str, np.str_))

            if y_is_string != pred_is_string:
                # Type mismatch - need to convert to same format
                # Get class_names from model if available
                class_names = getattr(local_model, 'class_names', None)
                if class_names is not None:
                    # Create mapping from class name to index
                    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

                    # Convert y_test to indices if it's strings
                    if y_is_string:
                        y_test_arr = np.array([class_to_idx.get(str(y), int(y)) for y in y_test_arr])

                    # Convert predictions to indices if they're strings
                    if pred_is_string:
                        predictions_arr = np.array([class_to_idx.get(str(p), -1) for p in predictions_arr])
                        # Check if any conversion failed
                        if -1 in predictions_arr:
                            failed_preds = [p for p in predictions_arr if p == -1]
                            raise ValueError(f"Failed to convert predictions to indices: {failed_preds[:5]}. Available classes: {class_names}")
                else:
                    raise ValueError(f"Type mismatch but no class_names available. y_test type: {y_test_arr.dtype}, predictions type: {predictions_arr.dtype}")

        accuracy = float(accuracy_score(y_test_arr, predictions_arr))
        macro_f1 = float(f1_score(y_test_arr, predictions_arr, average='macro', zero_division=0))

        client_flex_model['local_accuracy'] = accuracy
        client_flex_model['local_f1'] = macro_f1

        return {'accuracy': accuracy, 'macro_f1': macro_f1}
    except Exception as e:
        warnings.warn(f"evaluate_local_pf_model_at_clients failed: {e}")
        return {'accuracy': 0.0, 'macro_f1': 0.0}


__all__ = [
    'evaluate_global_pf_model',
    'evaluate_global_pf_model_at_clients', 
    'evaluate_local_pf_model_at_clients'
]