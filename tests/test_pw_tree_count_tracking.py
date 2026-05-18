import pytest
import numpy as np
from src.domain.dataset.base_adapter import DatasetSplit
from src.application.orchestrators.progressive_windows_orchestrator import ProgressiveWindowsOrchestrator
from src.infrastructure.flex import train_pf, collect_clients_trees_pf, deploy_server_config_pf

def test_pw_tree_count_tracking():
    # 1. Setup small synthetic dataset
    np.random.seed(42)
    X = np.random.rand(50, 2)
    y = np.random.choice(["0", "1"], size=50)

    split = DatasetSplit(
        X_train=X[:30], X_val=X[30:40], X_test=X[40:],
        y_train=y[:30], y_val=y[30:40], y_test=y[40:],
        feature_names=["f1", "f2"],
        class_names=["0", "1"],
        dataset_name="TrackDS"
    )

    # 2. Config with 2 clients, window_size=3, max_rounds=3
    # This means each client trains 3 trees per round.
    # Total local trees after round 1: 3, round 2: 6, round 3: 9.
    # Shared trees per client per round: exactly 3.
    # Aggregated trees on server:
    #   Round 1: 3 trees (1 per client, total 2? No, n_clients=2, so 2 trees total if 1 selected per client).
    #   Let's check: one tree is selected per client per round, so with 2 clients, 2 trees are added per round.
    #   Server total: Round 1: 2 trees, Round 2: 4 trees, Round 3: 6 trees.
    config = {
        "federation": {"n_clients": 2, "distribution": "iid"},
        "model": {"n_estimators": 3, "alpha": 0.1, "class_names": ["0", "1"]},
        "aggregation": {
            "strategy": "PW",
            "window_size": 3,
            "max_rounds": 3,
            "convergence_threshold": -1.0  # Force it to run all 3 rounds without early stopping
        }
    }

    orch = ProgressiveWindowsOrchestrator(config)
    orch.setup_federation(split)

    # We will step through the orchestrator manually round by round to track exact counts!
    try:
        # Check initial states
        for client_idx in range(2):
            client_model = orch.flex_pool._models[client_idx]
            assert client_model.get('model') is None
            assert client_model.get('trees') is None

        server_model = orch.flex_pool._models["server"]
        assert len(server_model.get("trees", [])) == 0

        # --- ROUND 1 ---
        print("\n=== STEPPING ROUND 1 ===")
        orch.config['model']['n_estimators'] = orch.window_size
        orch.flex_pool.servers.map(deploy_server_config_pf, orch.flex_pool.clients)
        orch.flex_pool.clients.map(train_pf)

        # Inspect local client model size and shared tree lot after round 1 training
        for client_idx in range(2):
            client_model = orch.flex_pool._models[client_idx]
            local_pf = client_model.get('model')
            shared_trees = client_model.get('trees', [])
            
            print(f"[Round 1] Client {client_idx} local model tree count: {len(local_pf.get_trees())}")
            print(f"[Round 1] Client {client_idx} shared tree count: {len(shared_trees)}")
            
            # Assertions
            assert len(local_pf.get_trees()) == 3, "Client local model should contain 3 trees after Round 1"
            assert len(shared_trees) == 3, "Client should share exactly 3 trees in Round 1"

        # Collect and Aggregate
        orch.flex_pool.aggregators.map(collect_clients_trees_pf, orch.flex_pool.clients)
        
        current_global_trees = orch.flex_pool._models["server"].get("trees", [])
        assert len(current_global_trees) == 0, "Server global trees should be empty before round 1 aggregation"

        from src.infrastructure.flex.flex_aggregate_pf import aggregate_trees_pf, set_aggregated_trees_pf
        agg_kwargs = {
            'server_config': orch.config,
            'X_val': split.X_val,
            'y_val': split.y_val,
            'metrics_service': orch.metrics_svc,
            'diversity_service': orch.diversity_svc,
            'current_global_trees': current_global_trees,
            'current_round': 0,
            't_max': 3
        }
        orch.flex_pool.aggregators.map(aggregate_trees_pf, **agg_kwargs)
        orch.flex_pool.aggregators.map(set_aggregated_trees_pf, orch.flex_pool.servers)

        # Inspect server aggregated trees after Round 1
        server_trees_r1 = orch.flex_pool._models["server"].get("trees", [])
        print(f"[Round 1] Server aggregated global trees: {len(server_trees_r1)}")
        assert len(server_trees_r1) == 2, "Server should aggregate 2 trees (1 per client) in Round 1"


        # --- ROUND 2 ---
        print("\n=== STEPPING ROUND 2 ===")
        orch.config['model']['n_estimators'] = orch.window_size
        orch.flex_pool.servers.map(deploy_server_config_pf, orch.flex_pool.clients)
        orch.flex_pool.clients.map(train_pf)

        # Inspect local client model size and shared tree lot after round 2 training
        for client_idx in range(2):
            client_model = orch.flex_pool._models[client_idx]
            local_pf = client_model.get('model')
            shared_trees = client_model.get('trees', [])
            
            print(f"[Round 2] Client {client_idx} local model tree count: {len(local_pf.get_trees())}")
            print(f"[Round 2] Client {client_idx} shared tree count: {len(shared_trees)}")
            
            # Assertions
            assert len(local_pf.get_trees()) == 6, "Client local model should accumulate to 6 trees after Round 2"
            assert len(shared_trees) == 3, "Client should share only the last 3 trees in Round 2"

        # Collect and Aggregate
        orch.flex_pool.aggregators.map(collect_clients_trees_pf, orch.flex_pool.clients)
        
        current_global_trees = orch.flex_pool._models["server"].get("trees", [])
        assert len(current_global_trees) == 2, "Server should have the 2 trees from round 1"

        agg_kwargs['current_global_trees'] = current_global_trees
        agg_kwargs['current_round'] = 1
        agg_kwargs['t_max'] = 6
        orch.flex_pool.aggregators.map(aggregate_trees_pf, **agg_kwargs)
        orch.flex_pool.aggregators.map(set_aggregated_trees_pf, orch.flex_pool.servers)

        # Inspect server aggregated trees after Round 2
        server_trees_r2 = orch.flex_pool._models["server"].get("trees", [])
        print(f"[Round 2] Server aggregated global trees: {len(server_trees_r2)}")
        assert len(server_trees_r2) == 4, "Server should aggregate 4 trees (2 from round 1 + 2 from round 2) in Round 2"


        # --- ROUND 3 ---
        print("\n=== STEPPING ROUND 3 ===")
        orch.config['model']['n_estimators'] = orch.window_size
        orch.flex_pool.servers.map(deploy_server_config_pf, orch.flex_pool.clients)
        orch.flex_pool.clients.map(train_pf)

        # Inspect local client model size and shared tree lot after round 3 training
        for client_idx in range(2):
            client_model = orch.flex_pool._models[client_idx]
            local_pf = client_model.get('model')
            shared_trees = client_model.get('trees', [])
            
            print(f"[Round 3] Client {client_idx} local model tree count: {len(local_pf.get_trees())}")
            print(f"[Round 3] Client {client_idx} shared tree count: {len(shared_trees)}")
            
            # Assertions
            assert len(local_pf.get_trees()) == 9, "Client local model should accumulate to 9 trees after Round 3"
            assert len(shared_trees) == 3, "Client should share only the last 3 trees in Round 3"

        # Collect and Aggregate
        orch.flex_pool.aggregators.map(collect_clients_trees_pf, orch.flex_pool.clients)
        
        current_global_trees = orch.flex_pool._models["server"].get("trees", [])
        assert len(current_global_trees) == 4, "Server should have the 4 trees from rounds 1 & 2"

        agg_kwargs['current_global_trees'] = current_global_trees
        agg_kwargs['current_round'] = 2
        agg_kwargs['t_max'] = 9
        orch.flex_pool.aggregators.map(aggregate_trees_pf, **agg_kwargs)
        orch.flex_pool.aggregators.map(set_aggregated_trees_pf, orch.flex_pool.servers)

        # Inspect server aggregated trees after Round 3
        server_trees_r3 = orch.flex_pool._models["server"].get("trees", [])
        print(f"[Round 3] Server aggregated global trees: {len(server_trees_r3)}")
        assert len(server_trees_r3) == 6, "Server should aggregate 6 trees (4 from previous + 2 new ones) in Round 3"

        print("\n>>> ALL DETAILED QUANTITY VERIFICATIONS ARE 100% CORRECT! <<<")

    finally:
        if hasattr(orch, 'cleanup'):
            orch.cleanup()
