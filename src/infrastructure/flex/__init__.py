"""FLEX Infrastructure Package for Proactive Forest Federation.

This package provides FLEX primitives for federated learning with Proactive Forest.
It follows the architecture pattern from flex-trees companion library.

Core Modules:
- flex_train_pf: Initialize and train Proactive Forest locally, collect weights
- flex_deploy_model_pf: Deploy server config and global model to clients
- flex_aggregate_pf: Aggregate collected trees from all clients
- flex_evaluate_pf: Evaluate local and global models
- flex_pool_factory: Factory for creating FLEX pools
"""

from .flex_train_pf import (
    init_server_model_pf,
    train_pf,
    collect_clients_trees_pf,
)

from .flex_deploy_model_pf import (
    deploy_server_config_pf,
    deploy_server_model_pf,
)

from .flex_aggregate_pf import (
    aggregate_trees_from_pf,
    set_aggregated_trees_pf,
)

from .flex_evaluate_pf import (
    evaluate_global_pf_model,
    evaluate_global_pf_model_at_clients,
    evaluate_local_pf_model_at_clients,
)

from .flex_pool_factory import (
    FlexPoolFactory,
)

__all__ = [
    # Training & Initialization
    'init_server_model_pf',
    'train_pf',
    
    # Collection & Aggregation
    'collect_clients_trees_pf',
    'aggregate_trees_from_pf',
    'set_aggregated_trees_pf',
    
    # Deployment
    'deploy_server_config_pf',
    'deploy_server_model_pf',
    
    # Evaluation
    'evaluate_global_pf_model',
    'evaluate_global_pf_model_at_clients',
    'evaluate_local_pf_model_at_clients',
    
    # Factory
    'FlexPoolFactory',
]