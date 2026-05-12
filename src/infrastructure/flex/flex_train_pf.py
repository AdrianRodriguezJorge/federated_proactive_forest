from typing import Dict, Any, List
from flex.model import FlexModel
from flex.pool.decorators import init_server_model, collect_clients_weights
import numpy as np


@init_server_model
def init_server_model_pf(config: Dict[str, Any] = None):
    """
    Initialize server model for Proactive Forest.
    FLEX primitive function.
    
    Returns a dict that will be used to update the FlexModel.
    """
    return {
        'model': None,
        'trees': [],
        'config': config or {},
    }


def train_pf(client_flex_model: FlexModel, client_data: Any) -> FlexModel:
    """
    Train Proactive Forest on client data with local validation.
    """
    import logging
    logger = logging.getLogger("FLEX_Client")
    actor_id = getattr(client_flex_model, 'actor_id', 'unknown')
    logger.debug(f"Client {actor_id} starting training...")
    from sklearn.model_selection import train_test_split
    from src.application.commands.train_command import TrainCommand
    from src.domain.model.proactive_forest import ProactiveForest
    from src.domain.metadata.client_metadata import ClientMetadata
    from src.domain.metrics.forest_evaluator import ForestEvaluator
    from src.infrastructure.metrics.sklearn_metrics_service import SklearnMetricsService

    # 1. Prepare data
    X, y = client_data.to_numpy()
    
    # Local validation split (20%)
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
    except Exception:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    config = client_flex_model.get('config', {})
    
    # class_names may be nested under 'model' in the config dict
    class_names = config.get('class_names')
    if not class_names:
        class_names = config.get('model', {}).get('class_names', [])

    # 2. Train local model
    pf = ProactiveForest(
        n_estimators=config.get('n_estimators', 100),
        alpha=config.get('alpha', 0.1),
        verbose=config.get('verbose', False),
        class_names=class_names,
        convergence_threshold=config.get('model', {}).get('local_convergence_threshold', 0.002)
    )
    
    pf.fit(X_train, y_train)
    
    # 3. Local Evaluation for Metadata
    metrics_svc = SklearnMetricsService()
    report = ForestEvaluator.evaluate(
        pf, X_val, y_val, 
        class_names=class_names,
        metrics_svc=metrics_svc
    )
    
    # Individual tree metrics for ranking strategies (S4, S7, PW)
    from src.domain.services.label_service import SimpleLabelService
    label_svc = SimpleLabelService(class_names)
    
    y_val_norm = label_svc.transform(y_val)
    
    tree_metrics = []
    trees = pf.get_trees()
    if trees:
        # Predicción vectorizada en bloque para todos los árboles de una sola vez
        all_preds = np.array([tree.predict(X_val) for tree in trees])
        all_preds_norm = np.array([label_svc.transform(preds) for preds in all_preds])
        
        for preds_norm in all_preds_norm:
            tree_metrics.append({
                'accuracy': float(np.mean(y_val_norm == preds_norm)),
                'macro_f1': metrics_svc.f1_score(y_val_norm, preds_norm, average='macro')
            })

    # 4. Create and store metadata
    # Use actor_id to ensure consistency
    actor_id = getattr(client_flex_model, 'actor_id', 'unknown')
    meta = ClientMetadata(
        client_id=actor_id,
        n_trees=len(pf.get_trees()),
        accuracy=report.accuracy,
        macro_f1=report.macro_f1,
        pcd=report.pcd,
        tree_metrics=tree_metrics
    )
    
    client_flex_model.update({
        'model': pf,
        'trees': pf.get_trees(),
        'metadata': meta,
        'X_train': X_train,
        'y_train': y_train
    })
    
    return client_flex_model


@collect_clients_weights
def collect_clients_trees_pf(client_flex_model: FlexModel, *args, **kwargs: Any) -> List[Any]:
    """
    Collect trees from a single client.
    FLEX primitive function decorated with @collect_clients_weights.
    
    The decorator handles the accumulation of these return values into 
    server_flex_model['weights'].
    
    Args:
        client_flex_model: Client model state (FlexModel)
        
    Returns:
        List of trees from this client
    """
    # Just return what we want to aggregate
    # We can also return a dict if we want to include metadata
    return {
        'client_id': getattr(client_flex_model, 'actor_id', 'unknown'),
        'trees': client_flex_model.get('trees', []),
        'metadata': client_flex_model.get('metadata', {})
    }


__all__ = ['init_server_model_pf', 'train_pf', 'collect_clients_trees_pf']
