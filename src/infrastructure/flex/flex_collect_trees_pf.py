"""FLEX primitives for collecting, aggregating and setting trees."""
from typing import Dict, Any, List
from copy import deepcopy

try:
    from flex.pool import collect_clients_weights
    HAS_FLEX_DECORATORS = True
except ImportError:
    HAS_FLEX_DECORATORS = False
    def collect_clients_weights(func):
        """Dummy decorator if FLEX not available."""
        return func


def collect_clients_trees_pf(server_flex_model: Dict[str, Any],
                            clients_flex_models: Dict[str, Dict[str, Any]]) -> None:
    """
    Collect trees from all clients into server model.
    Called after local training on clients.
    
    Args:
        server_flex_model: Server model
        clients_flex_models: Dict of all client models
    """
    from .flex_train_pf import collect_clients_trees_pf as collect_impl
    collect_impl(server_flex_model, clients_flex_models)


def aggregate_trees_from_pf(server_flex_model: Dict[str, Any],
                             X_val=None,
                             y_val=None,
                             t_max=None,
                             **kwargs) -> List[Any]:
    """
    Aggregate trees collected from clients.
    Called after collect_clients_trees_pf.
    
    Args:
        server_flex_model: Server model with all_client_trees
        X_val: Validation features
        y_val: Validation labels
        t_max: Max trees limit
        **kwargs: Strategy parameters
    
    Returns:
        List of aggregated trees
    """
    from .flex_aggregate_pf import aggregate_trees_from_pf as aggregate_impl
    return aggregate_impl(server_flex_model, X_val=X_val, y_val=y_val, t_max=t_max, **kwargs)


def set_aggregated_trees_pf(server_flex_model: Dict[str, Any]) -> Dict[str, Any]:
    """
    Set aggregated trees as the global server model.
    Called after aggregation.
    
    Args:
        server_flex_model: Server model with global_trees
    
    Returns:
        Updated server_flex_model with 'model' set to ProactiveForest
    """
    from src.domain.model.proactive_forest import ProactiveForest
    
    global_trees = server_flex_model.get('global_trees', [])
    class_names = server_flex_model.get('config', {}).get('class_names', None)
    
    server_flex_model['model'] = ProactiveForest.from_trees(global_trees, class_names=class_names)
    server_flex_model['trees'] = global_trees
    
    return server_flex_model


# FLEX decorators (if FLEX is available)
if HAS_FLEX_DECORATORS:
    collect_trees_pf = collect_clients_weights(collect_clients_trees_pf)
    aggregate_pf = collect_clients_weights(aggregate_trees_from_pf)
    set_aggregated_pf = collect_clients_weights(set_aggregated_trees_pf)
else:
    # Fallback: direct function references
    collect_trees_pf = collect_clients_trees_pf
    aggregate_pf = aggregate_trees_from_pf
    set_aggregated_pf = set_aggregated_trees_pf


__all__ = ['collect_clients_trees_pf', 'aggregate_trees_from_pf', 'set_aggregated_trees_pf',
           'collect_trees_pf', 'aggregate_pf', 'set_aggregated_pf']
