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
    strategy = AggregationFactory.create_strategy(strategy_name)

    aggregate_cmd = AggregateCommand(strategy)
    global_data = aggregate_cmd.execute(
        client_models,
        client_metadata,
        X_val=X_val,
        y_val=y_val,
        t_max=t_max,
        **kwargs
    )

    # Store in server model for set_aggregated_trees_pf
    server_flex_model['global_trees'] = global_data.get('global_trees', [])
    server_flex_model['selected_indices'] = global_data.get('selected_indices', {})
    server_flex_model['all_tree_entries'] = global_data.get('all_tree_entries', [])

    return server_flex_model['global_trees']


__all__ = ['aggregate_trees_from_pf']