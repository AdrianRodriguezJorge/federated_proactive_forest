from typing import Dict, List, Any
from flex.model import FlexModel
from flex.pool.decorators import aggregate_weights, set_aggregated_weights
from src.domain.aggregation.aggregation_factory import AggregationFactory
from src.application.commands.aggregate_command import AggregateCommand
from src.domain.model.proactive_forest import ProactiveForest


@aggregate_weights
def aggregate_trees_from_pf(weights: List[Dict[str, Any]], **kwargs) -> List[Any]:
    """
    Aggregate trees using the configured strategy.
    
    FLEX Primitive for @aggregate_weights decorator.
    Receives the list of collected weights (from collect_clients_trees_pf).
    
    Args:
        weights: List of dicts, each containing 'trees' and 'metadata' from a client.
        **kwargs: Strategy-specific parameters (X_val, y_val, config, etc.)
    
    Returns:
        List of aggregated trees
    """
    from src.domain.metadata.client_metadata import ClientMetadata

    # Extract trees and metadata in the format expected by our internal logic
    client_models = {
        f'client_{i}': {'trees': w['trees']}
        for i, w in enumerate(weights)
    }
    client_metadata = {}
    for i, w in enumerate(weights):
        cid = f'client_{i}'
        meta = w['metadata']
        if isinstance(meta, dict):
            client_metadata[cid] = ClientMetadata(**meta)
        else:
            client_metadata[cid] = meta

    server_config = kwargs.get('server_config', {})
    strategy_name = server_config.get('strategy', 'S1')
    agg_config = server_config.get('aggregation', {})
    
    metrics_svc = kwargs.get('metrics_service')
    diversity_svc = kwargs.get('diversity_service')
    
    strategy = AggregationFactory.create_strategy(
        strategy_name, 
        metrics_service=metrics_svc,
        diversity_service=diversity_svc
    )

    X_val = kwargs.get('X_val')
    y_val = kwargs.get('y_val')
    t_max = kwargs.get('t_max')

    # Build aggregate_kwargs based on strategy type
    aggregate_kwargs = {}
    if strategy_name == 'PW':
        aggregate_kwargs['window_size'] = agg_config.get('window_size', 5)
        aggregate_kwargs['max_rounds'] = agg_config.get('max_rounds', 20)
        aggregate_kwargs['f1_weight'] = agg_config.get('f1_weight', 0.5)
        aggregate_kwargs['convergence_threshold'] = agg_config.get('convergence_threshold', 0.002)
        aggregate_kwargs['local_weight'] = server_config.get('prediction', {}).get('local_weight', 0.5)
    elif strategy_name in ['S2', 'S3', 'S4', 'S5', 'S6', 'S7']:
        if 'S4' in strategy_name or 'S7' in strategy_name:
            aggregate_kwargs['f1_weight'] = agg_config.get('f1_weight', 0.5)
            aggregate_kwargs['pcd_weight'] = agg_config.get('pcd_weight', 1.0 - aggregate_kwargs['f1_weight'])

    aggregate_cmd = AggregateCommand(strategy)
    global_data = aggregate_cmd.execute(
        client_models,
        client_metadata,
        X_val=X_val,
        y_val=y_val,
        t_max=t_max,
        metrics_service=metrics_svc,
        diversity_service=diversity_svc,
        **aggregate_kwargs
    )

    return global_data.get('global_trees', [])


@set_aggregated_weights
def set_aggregated_trees_pf(server_flex_model: FlexModel, aggregated_weights: List[Any], **kwargs):
    """
    Set aggregated trees into the global model.
    FLEX Primitive for @set_aggregated_weights.
    """
    config = server_flex_model.get('config', {})
    class_names = config.get('class_names') or config.get('model', {}).get('class_names')
    global_forest = ProactiveForest.from_trees(aggregated_weights, class_names=class_names)
    
    server_flex_model['model'] = global_forest
    server_flex_model['trees'] = aggregated_weights
    
    return server_flex_model


__all__ = ['aggregate_trees_from_pf', 'set_aggregated_trees_pf']
