"""FLEX Server-side aggregation primitive procedures for Proactive Forest.

Supports parsing client updates, executing multi-strategy federated forest
aggregation commands, and storing global ensembles in server models.
"""

from typing import Any, Dict, List, Optional

from flex.model import FlexModel
from flex.pool.decorators import aggregate_weights, set_aggregated_weights
from src.application.commands.aggregate_command import AggregateCommand
from src.domain.aggregation.aggregation_factory import AggregationFactory
from src.domain.model.proactive_forest import ProactiveForest


@aggregate_weights
def aggregate_trees_pf(
    weights: List[Dict[str, Any]], **kwargs: Any
) -> Dict[str, Any]:
    """Aggregate client trees using the configured strategy.

    FLEX primitive decorated with aggregate_weights.

    Args:
        weights (List[Dict[str, Any]]): Collected client weights mapping.
        **kwargs (Any): Strategy configuration parameters and validation data.

    Returns:
        Dict[str, Any]: Dictionary containing aggregated global trees, selected
            client local IDs, and performance logs.
    """
    from src.domain.metadata.client_metadata import ClientMetadata

    client_models = {}
    client_metadata = {}

    for w in weights:
        cid = str(w.get("client_id", "unknown"))
        client_models[cid] = {"trees": w["trees"]}

        meta = w["metadata"]
        if isinstance(meta, dict):
            client_metadata[cid] = ClientMetadata.from_dict(meta)
        else:
            client_metadata[cid] = meta

    server_config = kwargs.get("server_config", {})
    raw_strategy = server_config.get("strategy") or server_config.get(
        "aggregation", {}
    ).get("strategy", "S1")
    strategy_name = AggregationFactory.normalize_strategy_name(raw_strategy)

    agg_config = server_config.get("aggregation", {})

    metrics_svc = kwargs.get("metrics_service")
    diversity_svc = kwargs.get("diversity_service")

    strategy = AggregationFactory.create_strategy(
        strategy_name,
        metrics_service=metrics_svc,
        diversity_service=diversity_svc,
    )

    X_val = kwargs.get("X_val")
    y_val = kwargs.get("y_val")
    max_trees = kwargs.get("max_trees")
    current_global_trees = kwargs.get("current_global_trees")
    current_round = kwargs.get("current_round")

    aggregate_kwargs = {}
    aggregate_kwargs.update(
        {
            "window_size": agg_config.get("window_size", 5),
            "max_rounds": agg_config.get("max_rounds", 20),
            "f1_weight": agg_config.get("f1_weight", 0.5),
            "pcd_weight": agg_config.get("pcd_weight", 1.0 - agg_config.get("f1_weight", 0.5)),
            "global_convergence_threshold": agg_config.get(
                "global_convergence_threshold", 0.002
            ),
            "global_episode_size": agg_config.get("global_episode_size", 5),
            "local_weight": server_config.get("prediction", {}).get(
                "local_weight", 0.5
            ),
            "min_episodes": agg_config.get("min_episodes", 5),
            "trees_per_client_per_episode": agg_config.get("trees_per_client_per_episode", 1),
            "current_global_trees": current_global_trees,
            "current_round": current_round,
        }
    )

    # Class names are vital for label mapping in progressive strategies
    aggregate_kwargs["class_names"] = (
        server_config.get("model", {}).get("class_names", [])
    )

    aggregate_cmd = AggregateCommand(strategy)
    global_data = aggregate_cmd.execute(
        client_models,
        client_metadata,
        X_val=X_val,
        y_val=y_val,
        max_trees=max_trees,
        metrics_service=metrics_svc,
        diversity_service=diversity_svc,
        **aggregate_kwargs,
    )

    return {
        "trees": global_data.get("global_trees", []),
        "selected_ids": global_data.get("selected_indices", {}),
        "all_tree_entries": global_data.get("all_tree_entries", []),
        "convergence_round": global_data.get("convergence_round"),
        "round_logs": global_data.get("round_logs", []),
    }


@set_aggregated_weights
def set_aggregated_trees_pf(
    server_flex_model: FlexModel,
    aggregated_data: Dict[str, Any],
    **kwargs: Any,
) -> FlexModel:
    """Set aggregated trees and metadata into the global model.

    FLEX primitive decorated with set_aggregated_weights.

    Args:
        server_flex_model (FlexModel): Server model state wrapper.
        aggregated_data (Dict[str, Any]): Dictionary of aggregated results.
        **kwargs (Any): Keyword args.

    Returns:
        FlexModel: Updated server model state.
    """
    trees = aggregated_data.get("trees", [])
    config = server_flex_model.get("config", {})
    class_names = config.get("class_names") or config.get("model", {}).get(
        "class_names"
    )

    global_forest = ProactiveForest.from_trees(trees, class_names=class_names)

    server_flex_model.update(
        {
            "model": global_forest,
            "trees": trees,
            "selected_ids": aggregated_data.get("selected_ids", {}),
            "all_tree_entries": aggregated_data.get("all_tree_entries", []),
            "convergence_round": aggregated_data.get("convergence_round"),
            "round_logs": aggregated_data.get("round_logs", []),
        }
    )

    return server_flex_model
