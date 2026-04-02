"""FLEX aggregation primitives for Proactive Forest."""
from typing import Dict, List, Any
from src.domain.aggregation.tree_ranker import TreeRanker
from src.domain.aggregation.aggregation_factory import AggregationFactory
from src.application.commands.aggregate_command import AggregateCommand


def aggregate_trees_from_pf(server_flex_model: Dict[str, Any],
                             X_val=None,
                             y_val=None,
                             t_max=None,
                             **kwargs) -> List[Any]:
    """
    Aggregate trees using the configured strategy.
    Receives collected trees and returns aggregated global trees.

    FLEX Primitive for @aggregate_weights decorator.

    Args:
        server_flex_model: Server model containing all_client_trees and client_metadata
        X_val: Validation features for Progressive Forest (S2-S7)
        y_val: Validation labels for Progressive Forest (S2-S7)
        t_max: Maximum trees in global model
        **kwargs: Strategy-specific parameters (f1_weight, pcd_weight, etc.)

    Returns:
        List of aggregated trees
    """
    all_client_trees = server_flex_model.get('all_client_trees', [])
    client_metadata = server_flex_model.get('client_metadata', {})

    # Convert to expected format (model_data dict expected by AggregateCommand)
    client_models = {
        f'client_{i}': {'trees': trees}
        for i, trees in enumerate(all_client_trees)
    }

    strategy_name = server_flex_model.get('config', {}).get('strategy', 'S1')
    config = server_flex_model.get('config', {})
    agg_config = config.get('aggregation', {})
    strategy = AggregationFactory.create_strategy(strategy_name)

    # Build aggregate_kwargs based on strategy type, merging with server_flex_model kwargs
    aggregate_kwargs = {}
    
    # RR_DS specific parameters
    if strategy_name == 'RR_DS':
        aggregate_kwargs['window_size'] = agg_config.get('window_size', kwargs.get('window_size', 5))
        aggregate_kwargs['max_rounds'] = agg_config.get('max_rounds', kwargs.get('max_rounds', 20))
        aggregate_kwargs['alpha'] = agg_config.get('alpha', kwargs.get('alpha', 0.5))
        aggregate_kwargs['convergence_threshold'] = agg_config.get('convergence_threshold', kwargs.get('convergence_threshold', 0.002))
        aggregate_kwargs['verbose'] = config.get('verbose', True)  # Pass verbose flag
        aggregate_kwargs['X_val'] = X_val
        aggregate_kwargs['y_val'] = y_val
        aggregate_kwargs['t_max'] = t_max
    # S2-S4 specific parameters
    elif strategy_name in ['S2', 'S3', 'S4']:
        aggregate_kwargs['X_val'] = X_val
        aggregate_kwargs['y_val'] = y_val
        aggregate_kwargs['max_trees'] = config.get('n_estimators', 100)
        aggregate_kwargs['t_max'] = t_max
        if strategy_name == 'S4':
            aggregate_kwargs['f1_weight'] = agg_config.get('f1_weight', kwargs.get('f1_weight', 0.5))
            aggregate_kwargs['pcd_weight'] = agg_config.get('pcd_weight', kwargs.get('pcd_weight', 0.5))
    # S5-S7 specific parameters
    elif strategy_name in ['S5', 'S6', 'S7']:
        aggregate_kwargs['X_val'] = X_val
        aggregate_kwargs['y_val'] = y_val
        aggregate_kwargs['max_trees_per_client'] = config.get('n_estimators', 100)
        aggregate_kwargs['t_max'] = t_max
        if strategy_name == 'S7':
            aggregate_kwargs['f1_weight'] = agg_config.get('f1_weight', kwargs.get('f1_weight', 0.5))
            aggregate_kwargs['pcd_weight'] = agg_config.get('pcd_weight', kwargs.get('pcd_weight', 0.5))
    else:
        # S1 or other strategies
        aggregate_kwargs['X_val'] = X_val
        aggregate_kwargs['y_val'] = y_val
        aggregate_kwargs['t_max'] = t_max

    aggregate_cmd = AggregateCommand(strategy)
    global_data = aggregate_cmd.execute(
        client_models,
        client_metadata,
        X_val=X_val,
        y_val=y_val,
        t_max=t_max,
        **{k: v for k, v in aggregate_kwargs.items() if k not in ['X_val', 'y_val', 't_max']}
    )

    # Store in server model for set_aggregated_trees_pf
    server_flex_model['global_trees'] = global_data.get('global_trees', [])
    server_flex_model['selected_indices'] = global_data.get('selected_indices', {})
    server_flex_model['all_tree_entries'] = global_data.get('all_tree_entries', [])

    # Store strategy instance for accessing convergence info (RR-DS)
    server_flex_model['strategy_instance'] = strategy

    return server_flex_model['global_trees']


__all__ = ['aggregate_trees_from_pf']
