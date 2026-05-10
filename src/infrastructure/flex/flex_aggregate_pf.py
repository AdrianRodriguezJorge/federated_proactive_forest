from typing import Dict, List, Any
from flex.model import FlexModel
from flex.pool.decorators import aggregate_weights, set_aggregated_weights
from src.domain.aggregation.aggregation_factory import AggregationFactory
from src.application.commands.aggregate_command import AggregateCommand
from src.domain.model.proactive_forest import ProactiveForest


@aggregate_weights
def aggregate_trees_pf(weights: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
    """
    Aggregate trees using the configured strategy.
    
    Returns a dict containing trees and metadata (convergence, logs).
    """
    from src.domain.metadata.client_metadata import ClientMetadata

    # Extract trees and metadata in the format expected by our internal logic
    client_models = {}
    client_metadata = {}
    
    for w in weights:
        cid = str(w.get('client_id', 'unknown'))
        client_models[cid] = {'trees': w['trees']}
        
        meta = w['metadata']
        if isinstance(meta, dict):
            client_metadata[cid] = ClientMetadata(**meta)
        else:
            client_metadata[cid] = meta

    server_config = kwargs.get('server_config', {})
    raw_strategy = server_config.get('strategy') or server_config.get('aggregation', {}).get('strategy', 'S1')
    strategy_name = AggregationFactory.normalize_strategy_name(raw_strategy)

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

    # Build aggregate_kwargs
    aggregate_kwargs = {}
    aggregate_kwargs.update({
        'window_size': agg_config.get('window_size', 5),
        'max_rounds': agg_config.get('max_rounds', 20),
        'f1_weight': agg_config.get('f1_weight', 0.5),
        'pcd_weight': agg_config.get('pcd_weight', 1.0 - agg_config.get('f1_weight', 0.5)),
        'global_convergence_threshold': agg_config.get('global_convergence_threshold', 0.002),
        'global_episode_size': agg_config.get('global_episode_size', 5),
        'local_weight': server_config.get('prediction', {}).get('local_weight', 0.5)
    })

    # Always provide class_names for label normalization in progressive strategies (S2-S7, PW)
    aggregate_kwargs['class_names'] = server_config.get('model', {}).get('class_names', [])

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

    return {
        'trees': global_data.get('global_trees', []),
        'selected_ids': global_data.get('selected_indices', {}),
        'all_tree_entries': global_data.get('all_tree_entries', []),
        'convergence_round': global_data.get('convergence_round'),
        'round_logs': global_data.get('round_logs', [])
    }


@set_aggregated_weights
def set_aggregated_trees_pf(server_flex_model: FlexModel, aggregated_data: Dict[str, Any], **kwargs: Any):
    """
    Set aggregated trees and metadata into the global model.
    """
    trees = aggregated_data.get('trees', [])
    config = server_flex_model.get('config', {})
    class_names = config.get('class_names') or config.get('model', {}).get('class_names')
    
    global_forest = ProactiveForest.from_trees(trees, class_names=class_names)
    
    server_flex_model.update({
        'model': global_forest,
        'trees': trees,
        'selected_ids': aggregated_data.get('selected_ids', {}),
        'all_tree_entries': aggregated_data.get('all_tree_entries', []),
        'convergence_round': aggregated_data.get('convergence_round'),
        'round_logs': aggregated_data.get('round_logs', [])
    })
    
    return server_flex_model


__all__ = ['aggregate_trees_pf', 'set_aggregated_trees_pf']
