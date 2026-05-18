"""Progressive Windows Orchestrator Refactored with native FLEX integration.

Uses FlexPool for orchestrating incremental window training and leverages
official PF FLEX primitives.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from flex.data import Dataset, FedDataDistribution
from flex.model import FlexModel
from src.application.orchestrators.fl_orchestrator import (
    FLEXOrchestrator,
    FLResults,
)
from src.application.orchestrators.result_consolidator import (
    ResultConsolidator,
)
from src.infrastructure.flex.flex_aggregate_pf import (
    aggregate_trees_pf,
    set_aggregated_trees_pf,
)
from src.infrastructure.flex.flex_deploy_model_pf import (
    deploy_server_config_pf,
    deploy_server_model_pf,
)
from src.infrastructure.flex.flex_evaluate_pf import (
    evaluate_global_pf_model,
    evaluate_global_pf_model_at_clients,
)
from src.infrastructure.flex.flex_pool_factory import FlexPoolFactory
from src.infrastructure.flex.flex_train_pf import (
    collect_clients_trees_pf,
    init_server_model_pf,
    train_pf,
)


@dataclass
class ProgressiveWindowsResults(FLResults):
    """Extended results for Progressive Windows with round logs."""

    round_logs: List[Dict[str, Any]] = field(default_factory=list)


class ProgressiveWindowsOrchestrator(FLEXOrchestrator):
    """Refactored Orchestrator for True Incremental PW using FlexPool."""

    def __init__(
        self,
        config: Union[dict, Any],
        step_callback: Optional[Callable[..., Any]] = None,
    ):
        """Initializes ProgressiveWindowsOrchestrator.

        Args:
            config (Union[dict, Any]): Configuration dictionary or object.
            step_callback (Optional[Callable]): Progress reporting callback.
        """
        super().__init__(config, step_callback, use_flex_pool=True)

        agg_cfg = self.config.get("aggregation", {})
        self.window_size = agg_cfg.get("window_size", 5)
        self.max_rounds = agg_cfg.get("max_rounds", 20)
        self.convergence_threshold = agg_cfg.get(
            "convergence_threshold", 0.002
        )

    def run_federated_round(
        self, n_bootstrap: int = 0
    ) -> ProgressiveWindowsResults:
        """Executes a multi-round incremental PW experiment using FlexPool.

        Args:
            n_bootstrap (int): Bootstrap repetitions for statistical CI.

        Returns:
            ProgressiveWindowsResults: Comprehensive results of the run.

        Raises:
            RuntimeError: If FlexPool is not initialized.
        """
        if self.flex_pool is None:
            raise RuntimeError("FlexPool not initialized.")

        self.step_callback(
            "Iniciando Progressive Windows Incremental (FLEX Pool)...", 5
        )

        results = ProgressiveWindowsResults(
            strategy_id="PW",
            client_ids=self.flex_pool.clients.actor_ids,
            class_names=self.dataset_split.class_names,
            y_test=self.label_svc.transform(self.dataset_split.y_test),
        )

        previous_accuracy = None
        stop_counter = 0

        # Incremental loop over windows
        for round_num in range(self.max_rounds):
            round_idx = round_num + 1
            progress_pct = 10 + (round_num * 80 // self.max_rounds)
            self.step_callback(
                f"PW Ronda {round_idx}/{self.max_rounds}...", progress_pct
            )

            # 1. Deploy current server config
            self.config["model"]["n_estimators"] = self.window_size
            self.flex_pool.servers.map(
                deploy_server_config_pf, self.flex_pool.clients
            )

            # 2. Train local window
            self.flex_pool.clients.map(train_pf)

            # 3. Collect window trees
            self.flex_pool.aggregators.map(
                collect_clients_trees_pf, self.flex_pool.clients
            )

            # 4. Aggregate incrementally
            X_val = self.dataset_split.X_val
            y_val = self.dataset_split.y_val

            # Get the current global trees from the server model
            current_global_trees = self.flex_pool._models["server"].get(
                "trees", []
            )

            agg_kwargs = {
                "server_config": self.config,
                "X_val": X_val,
                "y_val": y_val,
                "metrics_service": self.metrics_svc,
                "diversity_service": self.diversity_svc,
                "strategy_instance": None,
                "current_global_trees": current_global_trees,
                "current_round": round_num,
            }
            agg_kwargs["t_max"] = self.window_size * round_idx

            self.flex_pool.aggregators.map(aggregate_trees_pf, **agg_kwargs)
            self.flex_pool.aggregators.map(
                set_aggregated_trees_pf, self.flex_pool.servers
            )

            # 5. Evaluate Convergence
            server_val_dataset = Dataset.from_array(X_val, y_val)
            server_eval = self.flex_pool.servers.map(
                evaluate_global_pf_model, test_data=server_val_dataset
            )

            current_accuracy = self.flex_pool._models["server"].get(
                "global_accuracy", 0.0
            )

            round_log = {
                "round": round_idx,
                "accuracy": current_accuracy,
                "n_trees": len(
                    self.flex_pool._models["server"].get("trees", [])
                ),
            }
            results.round_logs.append(round_log)

            if previous_accuracy is not None:
                acc_gain = current_accuracy - previous_accuracy
                if acc_gain <= self.convergence_threshold:
                    stop_counter += 1
                    if stop_counter >= 2:
                        results.convergence_round = round_idx
                        break
                else:
                    stop_counter = 0

            previous_accuracy = current_accuracy
            results.num_rounds = round_idx

        # Finalize
        self.step_callback("PW Convergencia alcanzada. Finalizando...", 90)

        # 6. Final Deployment and Client Evaluation
        self.flex_pool.servers.map(
            deploy_server_model_pf, self.flex_pool.clients
        )
        self.flex_pool.clients.map(evaluate_global_pf_model_at_clients)

        # 7. Build final results
        consolidator = ResultConsolidator(self.label_svc, self.config)
        final_results_basic = consolidator.consolidate(
            strategy_name="PW",
            flex_pool=self.flex_pool,
            dataset_split=self.dataset_split,
            server_eval=server_eval,
            n_bootstrap=n_bootstrap,
        )

        # Merge properties
        results.global_accuracy = final_results_basic.global_accuracy
        results.global_macro_f1 = final_results_basic.global_macro_f1
        results.n_trees_global = final_results_basic.n_trees_global
        results.client_accuracies = final_results_basic.client_accuracies
        results.client_f1_scores = final_results_basic.client_f1_scores
        results.client_metadata = final_results_basic.client_metadata
        results.client_hybrid_predictions = (
            final_results_basic.client_hybrid_predictions
        )
        results.client_hybrid_forest_sizes = (
            final_results_basic.client_hybrid_forest_sizes
        )
        results.global_report = final_results_basic.global_report

        return results
