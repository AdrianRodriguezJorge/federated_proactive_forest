from typing import Dict, Any, List
import numpy as np
from flex.model import FlexModel
from flex.pool.decorators import (
    collect_clients_weights, 
    aggregate_weights, 
    set_aggregated_weights, 
    deploy_server_model
)
import logging

def train_window_pf_pw(client_flex_model: FlexModel, client_data: Any, active_ids: List[str] = None) -> FlexModel:
    """
    Trains a window of W trees on a client for Progressive Windows.
    Returns the newly built trees and local convergence metadata.
    """
    logger = logging.getLogger("FLEX_Client_PW")
    actor_id = str(getattr(client_flex_model, 'actor_id', 'unknown'))
    if active_ids is not None:
        active_ids_str = [str(aid) for aid in active_ids]
        if actor_id not in active_ids_str:
            return client_flex_model
            
    from sklearn.model_selection import train_test_split
    from src.domain.model.proactive_forest import ProactiveForest
    from src.domain.model.progressive_forest import ComparativeProgressiveForest
    from src.domain.metrics.forest_evaluator import ForestEvaluator
    from src.infrastructure.metrics.sklearn_metrics_service import SklearnMetricsService
    
    X, y = client_data.to_numpy()
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
    except Exception:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    config = client_flex_model.get('config', {})
    window_size = config.get('aggregation', {}).get('window_size', 5)
    
    pf = client_flex_model.get('model')
    if pf is None:
        class_names = config.get('model', {}).get('class_names', [])
        pf = ProactiveForest(
            n_estimators=config.get('model', {}).get('n_estimators', 50),
            alpha=config.get('model', {}).get('alpha', 0.1),
            class_names=class_names,
            convergence_threshold=config.get('model', {}).get('local_convergence_threshold', 0.002)
        )
        pf._is_fitted = True
        pf._cpf = ComparativeProgressiveForest(pf._classifier, convergence_threshold=pf.convergence_threshold)
        pf._classifier._n_instances, pf._classifier._n_features = X_train.shape
        all_labels = np.unique(np.concatenate([y_train, y_val]))
        pf._classifier.classes_ = all_labels
        pf._classifier._encoder_dict = {val: idx for idx, val in enumerate(all_labels)}
        pf._classifier._decoder_dict = {idx: val for idx, val in enumerate(all_labels)}
        pf._classifier._n_classes = len(all_labels)

    classifier = pf._classifier
    
    prev_meta = client_flex_model.get('metadata', {})
    if not isinstance(prev_meta, dict): prev_meta = {}
    
    stop_counter = prev_meta.get('stop_counter', 0)
    prev_episode_acc = prev_meta.get('prev_episode_acc')
    
    # Cantidad de árboles antes de entrenar la ventana
    n_trees_before = len(pf.get_trees())
    
    # Entrenar ventana
    classifier.buildEpisode(X_train, y_train, X_val, y_val, window_size)
    
    n_trees_after = len(pf.get_trees())
    
    # Extraer SOLO los árboles nuevos de esta ventana
    new_trees = pf.get_trees()[n_trees_before:n_trees_after]
    
    # Evaluar convergencia
    min_acc = min(classifier._m_progressive_accuracy) if classifier._m_progressive_accuracy else 0
    max_acc = max(classifier._m_progressive_accuracy) if classifier._m_progressive_accuracy else 0
    episode_acc = max_acc - min_acc
    
    has_converged = False
    acc_diff = 0.002
    if prev_episode_acc is not None:
        acc_diff = episode_acc - prev_episode_acc
        
    if acc_diff < pf.convergence_threshold or episode_acc < pf.convergence_threshold:
        stop_counter += 1
        if stop_counter >= 2:
            has_converged = True
    else:
        stop_counter = 0

    metrics_svc = SklearnMetricsService()
    report = ForestEvaluator.evaluate(pf, X_val, y_val, class_names=pf.class_names, metrics_svc=metrics_svc)
    
    # También calculamos las métricas específicas de CADA árbol nuevo
    # para que el servidor pueda evaluarlos
    tree_metrics = []
    for t in new_trees:
        preds = t.predict(X_val)
        preds_int = np.array([pf._classifier._encoder_dict.get(p, -1) for p in preds])
        y_val_int = np.array([pf._classifier._encoder_dict.get(y_v, -1) for y_v in y_val])
        f1 = float(metrics_svc.f1_score(y_val_int, preds_int, average='macro'))
        tree_metrics.append({'macro_f1': f1})

    client_flex_model.update({
        'model': pf,
        'new_trees': new_trees,
        'tree_metrics': tree_metrics,
        'metadata': {
            'accuracy': report.accuracy,
            'macro_f1': report.macro_f1,
            'n_trees': n_trees_after,
            'pcd': report.pcd,
            'has_converged': has_converged,
            'stop_counter': stop_counter,
            'prev_episode_acc': episode_acc
        }
    })
    
    return client_flex_model

@collect_clients_weights
def collect_new_trees_pw(client_flex_model: FlexModel, *args, **kwargs) -> Dict[str, Any]:
    """
    Collects ONLY the new window of trees from the client to the server.
    """
    cid = str(getattr(client_flex_model, 'actor_id', 'unknown'))
    new_trees = client_flex_model.get('new_trees', [])
    tree_metrics = client_flex_model.get('tree_metrics', [])
    metadata = client_flex_model.get('metadata', {})
    
    return {
        'client_id': cid,
        'trees': new_trees,
        'metrics': tree_metrics,
        'metadata': metadata
    }

@aggregate_weights
def aggregate_trees_pw(weights: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
    """
    Collects all client windows into a single dict for the server.
    """
    client_windows = {}
    for w in weights:
        cid = w.get('client_id', 'unknown')
        trees = w.get('trees', [])
        client_windows[cid] = {
            'trees': trees,
            'metrics': w.get('metrics', []),
            'metadata': w.get('metadata', {})
        }
    return {'client_windows': client_windows}

@set_aggregated_weights
def set_client_windows_pw(server_flex_model: FlexModel, aggregated_data: Dict[str, Any], **kwargs):
    """
    Stores the collected windows in the server model.
    """
    server_flex_model.update({'client_windows': aggregated_data.get('client_windows', {})})
    return server_flex_model

@deploy_server_model
def deploy_global_forest_pw(server_flex_model: FlexModel, *args, **kwargs) -> Dict[str, Any]:
    """
    Deploys the current global forest from the server to the client.
    """
    global_trees = server_flex_model.get('global_trees', [])
    return {
        'global_trees': global_trees
    }
