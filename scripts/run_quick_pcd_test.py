import sys
import os
import warnings

# Silenciar advertencia específica (por si acaso)
warnings.filterwarnings("ignore", message="X_array or y_array are not a list nor a numpy array", category=RuntimeWarning)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.application.orchestrators.fl_orchestrator import FLEXOrchestrator
from src.application.orchestrators.progressive_tree_orchestrator import ProgressiveTreeOrchestrator
from src.infrastructure.dataset.dataset_factory import DatasetFactory


def run_quick():
    print("Starting quick PCD sensitivity test (Iris, Car)")
    datasets = ["Iris", "Car"]
    strategies = ["S4", "pw"]
    pcd_weights = [0.0, 0.5, 1.0]

    for ds_name in datasets:
        print(f"\n=== DATASET: {ds_name} ===")
        try:
            ds_config = {"type": ds_name.lower(), "test_size": 0.2, "scale": True, "scaler_type": "standard", "seed": 42}
            dataset_split = DatasetFactory.load_from_config(ds_config)
        except Exception as e:
            print(f"Error loading {ds_name}: {e}")
            continue

        for strategy in strategies:
            print(f"\n-- Strategy: {strategy} --")
            for pcd in pcd_weights:
                f1_w = round(1.0 - pcd, 2)
                pcd_w = round(pcd, 2)

                from src.domain.aggregation.aggregation_factory import AggregationFactory
                norm_strat = AggregationFactory.normalize_strategy_name(strategy)

                global_ep_size = 5
                trees_per_client_ep = 1
                trees_per_rnd_client = 1
                win_size = 5

                if norm_strat == "S4":
                    global_ep_size = 6
                elif norm_strat == "PW":
                    win_size = 6
                    trees_per_rnd_client = 2

                config = {
                    "federation": {"n_clients": 3, "distribution": "dirichlet", "alpha": 0.5},
                    "model": {"n_estimators": 30, "alpha": 0.1, "voting": "soft", "local_convergence_threshold": 0.002},
                    "aggregation": {"strategy": strategy, "variant": strategy, "window_size": win_size, "max_rounds": 6, "convergence_threshold": 0.002, "t_max": 80, "global_episode_size": global_ep_size, "trees_per_client_per_episode": trees_per_client_ep, "trees_per_round_per_client": trees_per_rnd_client, "min_episodes": 2, "min_rounds": 2, "f1_weight": f1_w, "pcd_weight": pcd_w}
                }

                try:
                    if strategy == "pw":
                        orch = ProgressiveTreeOrchestrator(config)
                    else:
                        orch = FLEXOrchestrator(config)

                    orch.setup_federation(dataset_split)
                    res = orch.run_federated_round(n_bootstrap=0)

                    acc = getattr(res, 'hybrid_accuracy_mean', None)
                    trees = getattr(res, 'n_trees_global', None)
                    print(f" PCW {pcd_w:.1f} | F1_W {f1_w:.1f} -> Acc: {acc} (Trees: {trees})")

                    if hasattr(orch, 'cleanup'):
                        orch.cleanup()
                except Exception as e:
                    print(f" Error for pcd {pcd_w}: {e}")


if __name__ == '__main__':
    run_quick()
