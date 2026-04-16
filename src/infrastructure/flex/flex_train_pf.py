"""FLEX primitive for training Proactive Forest on clients."""
from typing import Dict, Any


def init_server_model_pf():
    """
    Initialize server model for Proactive Forest.
    FLEX primitive function (called once at start).
    
    Returns a dict with server-side model state.
    """
    return {
        'model': None,  # Will be set after aggregation
        'trees': [],
        'config': {},
    }


def train_pf(client_flex_model: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train Proactive Forest on client data.
    FLEX primitive function (called on each client).
    
    Assumes data is already in client_flex_model as 'X_train', 'y_train'.
    
    Args:
        client_flex_model: Client model state dict
        
    Returns:
        Updated client_flex_model with trained 'model' and 'trees'
    """
    from src.application.commands.train_command import TrainCommand
    from src.domain.model.proactive_forest import ProactiveForest

    train_cmd = TrainCommand(lambda: ProactiveForest(
        n_estimators=client_flex_model.get('config', {}).get('n_estimators', 100),
        alpha=client_flex_model.get('config', {}).get('alpha', 0.1),
        verbose=client_flex_model.get('config', {}).get('verbose', False),
        class_names=client_flex_model.get('config', {}).get('class_names')
    ))

    # Assume data is already in client_flex_model
    updated_data = train_cmd.execute_on_client(client_flex_model)
    client_flex_model.update(updated_data)
    
    return client_flex_model


def collect_clients_trees_pf(server_flex_model: Dict[str, Any],
                             clients_flex_models: Dict[str, Dict[str, Any]]) -> None:
    """
    Collect trees from all clients into server model.
    FLEX primitive function (called on server/aggregator).
    
    Processes all client models and consolidates their trees for aggregation.
    
    Args:
        server_flex_model: Server-side model to store collected data
        clients_flex_models: Dict of all client models
    """
    all_client_trees = []
    client_metadata = {}

    for client_id, client_model in clients_flex_models.items():
        trees = client_model.get('trees', [])
        all_client_trees.append(trees)
        
        # Extract metadata from client
        metadata = client_model.get('metadata', {})
        client_metadata[client_id] = metadata

    server_flex_model['all_client_trees'] = all_client_trees
    server_flex_model['client_metadata'] = client_metadata


__all__ = ['init_server_model_pf', 'train_pf', 'collect_clients_trees_pf']
