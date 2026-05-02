from typing import Dict, Any, Optional, List
import warnings
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from flex.model import FlexModel
from flex.pool.decorators import evaluate_server_model


def _align_labels(y_true: np.ndarray, y_pred: np.ndarray, class_names: Optional[List[str]]) -> tuple:
    """Helper to ensure y_true and y_pred are comparable (both indices or both strings)."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    
    if class_names is None or len(class_names) == 0:
        return y_true_arr, y_pred_arr

    # Case 1: Convert everything to indices (integers)
    class_to_idx = {str(name): i for i, name in enumerate(class_names)}
    
    # Process y_true
    try:
        if isinstance(y_true_arr.flat[0], (str, np.str_)):
            y_true_arr = np.array([class_to_idx.get(str(y), 0) for y in y_true_arr])
    except Exception:
        pass
    
    # Process y_pred
    try:
        if isinstance(y_pred_arr.flat[0], (str, np.str_)):
            y_pred_arr = np.array([class_to_idx.get(str(p), 0) for p in y_pred_arr])
    except Exception:
        pass
        
    return y_true_arr, y_pred_arr


@evaluate_server_model
def evaluate_global_pf_model(server_flex_model: FlexModel, test_data: Any = None, *args, **kwargs) -> Dict[str, Any]:
    """
    Evaluate global model on server side.
    FLEX Primitive for @evaluate_server_model.
    """
    model = server_flex_model.get('model')
    if model is None:
        return {'accuracy': 0.0, 'macro_f1': 0.0, 'pcd': 0.0, 'per_class_metrics': {}, 'confusion_matrix': []}

    try:
        # Extract X and y from various possible formats
        X_test, y_test = None, None
        
        if hasattr(test_data, 'to_numpy'):
            X_test, y_test = test_data.to_numpy()
        elif test_data is not None and isinstance(test_data, np.ndarray):
            X_test = test_data
            y_test = kwargs.get('y_test')
        
        # Fallback to kwargs
        if X_test is None: X_test = kwargs.get('X_test')
        if y_test is None: y_test = kwargs.get('y_test')
            
        if X_test is None or y_test is None:
            return {
                'accuracy': 0.0, 'macro_f1': 0.0, 'pcd': 0.0, 
                'per_class_metrics': {}, 'confusion_matrix': []
            }

        # 1. Predictions
        predictions = model.predict(X_test)
        
        # Robust label alignment
        class_names = getattr(model, 'class_names', None)
        if not class_names:
            # Try to get from server model state
            config = server_flex_model.get('config', {})
            class_names = config.get('class_names') or config.get('model', {}).get('class_names')
            
        y_test_arr, predictions_arr = _align_labels(y_test, predictions, class_names)

        # Basic metrics
        accuracy = float(accuracy_score(y_test_arr, predictions_arr))
        macro_f1 = float(f1_score(y_test_arr, predictions_arr, average='macro', zero_division=0))

        # Detailed metrics
        labels = list(range(len(class_names))) if class_names else None
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test_arr, predictions_arr, average=None, labels=labels, zero_division=0
        )
        conf_mat = confusion_matrix(y_test_arr, predictions_arr, labels=labels)

        per_class_metrics = {}
        if class_names:
            for i, name in enumerate(class_names):
                if i < len(prec):
                    per_class_metrics[name] = {
                        'precision': float(prec[i]),
                        'recall': float(rec[i]),
                        'f1': float(f1[i])
                    }

        # 2. Diversity Metric (PCD)
        pcd = 0.0
        if hasattr(model, 'diversity_measure'):
            try:
                pcd = float(model.diversity_measure(X_test, y_test, diversity='pcd'))
            except Exception as div_exc:
                warnings.warn(f"PCD calculation failed: {div_exc}")
        
        # Store in server model state for persistence
        server_flex_model['global_accuracy'] = accuracy
        server_flex_model['global_f1'] = macro_f1
        server_flex_model['global_pcd'] = pcd

        return {
            'accuracy': accuracy, 
            'macro_f1': macro_f1, 
            'pcd': pcd,
            'forest_size': len(server_flex_model.get('trees', [])),
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': conf_mat.tolist()
        }
    except Exception as e:
        warnings.warn(f"evaluate_global_pf_model failed: {e}")
        import traceback
        traceback.print_exc()
        return {'accuracy': 0.0, 'macro_f1': 0.0, 'pcd': 0.0, 'per_class_metrics': {}, 'confusion_matrix': []}


def evaluate_global_pf_model_at_clients(client_flex_model: FlexModel, client_data: Any, *args, **kwargs) -> Dict[str, float]:
    global_model = client_flex_model.get('global_model')
    if global_model is None or client_data is None:
        return {'accuracy': 0.0, 'macro_f1': 0.0}

    try:
        X_test, y_test = client_data.to_numpy()
        predictions = global_model.predict(X_test)
        class_names = getattr(global_model, 'class_names', None)
        y_test_arr, predictions_arr = _align_labels(y_test, predictions, class_names)
        
        accuracy = float(accuracy_score(y_test_arr, predictions_arr))
        macro_f1 = float(f1_score(y_test_arr, predictions_arr, average='macro', zero_division=0))
        
        client_flex_model['global_accuracy'] = accuracy
        client_flex_model['global_f1'] = macro_f1
        return {'accuracy': accuracy, 'macro_f1': macro_f1}
    except Exception as e:
        return {'accuracy': 0.0, 'macro_f1': 0.0}


def evaluate_local_pf_model_at_clients(client_flex_model: FlexModel, client_data: Any, *args, **kwargs) -> Dict[str, float]:
    local_model = client_flex_model.get('model')
    if local_model is None or client_data is None:
        return {'accuracy': 0.0, 'macro_f1': 0.0}

    try:
        X_test, y_test = client_data.to_numpy()
        predictions = local_model.predict(X_test)
        class_names = getattr(local_model, 'class_names', None)
        y_test_arr, predictions_arr = _align_labels(y_test, predictions, class_names)

        accuracy = float(accuracy_score(y_test_arr, predictions_arr))
        macro_f1 = float(f1_score(y_test_arr, predictions_arr, average='macro', zero_division=0))

        client_flex_model['local_accuracy'] = accuracy
        client_flex_model['local_f1'] = macro_f1
        return {'accuracy': accuracy, 'macro_f1': macro_f1}
    except Exception as e:
        return {'accuracy': 0.0, 'macro_f1': 0.0}


__all__ = [
    'evaluate_global_pf_model',
    'evaluate_global_pf_model_at_clients', 
    'evaluate_local_pf_model_at_clients'
]