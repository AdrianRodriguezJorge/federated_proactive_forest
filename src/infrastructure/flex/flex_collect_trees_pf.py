"""FLEX primitives for collecting and aggregating trees."""
from flex.pool import collect_clients_weights

# Import the functions from flex_train_pf
from .flex_train_pf import (
    collect_clients_trees_pf,
    aggregate_trees_from_pf,
    set_aggregated_trees_pf
)

# FLEX decorators for tree operations
collect_trees_pf = collect_clients_weights(collect_clients_trees_pf)
aggregate_pf = collect_clients_weights(aggregate_trees_from_pf)
set_aggregated_pf = collect_clients_weights(set_aggregated_trees_pf)