from typing import Dict, Any, List
import numpy as np
from flex.model import FlexModel
from flex.pool.decorators import collect_clients_weights

def train_window_pf_s9(client_flex_model: FlexModel, client_data: Any, active_ids: List[str] = None) -> FlexModel:
    """
    Trains a window of W trees on a client, preserving the Proactive Forest state.
    Skips training if the client is not in the active_ids list.
    """
    actor_id = str(getattr(client_flex_model, 'actor_id', 'unknown'))
    if active_ids is not None:
        active_ids_str = [str(aid) for aid in active_ids]
        if actor_id not in active_ids_str:
            return client_flex_model
    from sklearn.model_selection import train_test_split
    from src.domain.model.proactive_forest import ProactiveForest
    from src.domain.metrics.forest_evaluator import ForestEvaluator
    from src.infrastructure.metrics.sklearn_metrics_service import SklearnMetricsService
    
    # 1. Prepare data
    X, y = client_data.to_numpy()
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
    except Exception:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    config = client_flex_model.get('config', {})
    window_size = config.get('aggregation', {}).get('window_size', 5)
    
    # 2. Retrieve or initialize model
    pf = client_flex_model.get('model')
    if pf is None:
        class_names = config.get('model', {}).get('class_names', [])
        pf = ProactiveForest(
            n_estimators=config.get('n_estimators', 100),
            alpha=config.get('alpha', 0.1),
            class_names=class_names,
            convergence_threshold=config.get('aggregation', {}).get('convergence', 0.002)
        )
        # Initialize internal structures
        pf._is_fitted = True # We'll manage fit iteratively
        from src.domain.model.progressive_forest import ComparativeProgressiveForest
        pf._cpf = ComparativeProgressiveForest(
            pf._classifier, 
            convergence_threshold=pf.convergence_threshold
        )
        # Setup encoder once
        pf._classifier._n_instances, pf._classifier._n_features = X_train.shape
        all_labels = np.unique(np.concatenate([y_train, y_val]))
        pf._classifier.classes_ = all_labels
        pf._classifier._encoder_dict = {val: idx for idx, val in enumerate(all_labels)}
        pf._classifier._decoder_dict = {idx: val for idx, val in enumerate(all_labels)}
        pf._classifier._n_classes = len(all_labels)

    # 3. Train the window (Episode)
    # We use a custom build logic to ensure continuity
    classifier = pf._classifier
    
    # Get state from previous round
    prev_meta = client_flex_model.get('metadata', {})
    if not isinstance(prev_meta, dict): prev_meta = {}
    
    stop_counter = prev_meta.get('stop_counter', 0)
    prev_episode_acc = prev_meta.get('prev_episode_acc')
    
    # Build trees
    classifier.buildEpisode(X_train, y_train, X_val, y_val, window_size)
    
    # Calculate convergence using CPF logic
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

    # 4. Update metadata and model
    metrics_svc = SklearnMetricsService()
    report = ForestEvaluator.evaluate(pf, X_val, y_val, class_names=pf.class_names, metrics_svc=metrics_svc)
    
    client_flex_model.update({
        'model': pf,
        'metadata': {
            'accuracy': report.accuracy,
            'macro_f1': report.macro_f1,
            'n_trees': len(pf.get_trees()),
            'pcd': report.pcd,
            'has_converged': has_converged,
            'stop_counter': stop_counter,
            'prev_episode_acc': episode_acc
        },
        'X_train': X_train # Needed for roulette collection
    })
    
    return client_flex_model

def check_convergence_s9(client_flex_model: FlexModel, *args, **kwargs) -> bool:
    """Returns True if the client model has converged."""
    meta = client_flex_model.get('metadata', {})
    if isinstance(meta, dict):
        return meta.get('has_converged', False)
    return False
