"""Progressive Tree Orchestrator.

Episodic Orchestrator for Progressive Windows (PW) and similar tree-exchanging
strategies. Iterates round by round, exchanging W trees and selecting the
best to add to the global forest.
"""

import logging
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from flex.data.dataset import Dataset
from flex.pool import FlexPool
from src.application.orchestrators.fl_results import FLResults
from src.application.orchestrators.result_consolidator import (
    ResultConsolidator,
)
from src.domain.dataset.base_adapter import DatasetSplit
from src.domain.metrics.forest_evaluator import ForestEvaluator
from src.domain.model.proactive_forest import ProactiveForest
from src.domain.services.label_service import SimpleLabelService
from src.infrastructure.flex.flex_deploy_model_pf import (
    deploy_server_config_pf,
)
from src.infrastructure.flex.flex_pw_progressive import (
    aggregate_trees_pw,
    collect_new_trees_pw,
    deploy_global_forest_pw,
    set_client_windows_pw,
    train_window_pf_pw,
)
from src.infrastructure.metrics.diversity_service import (
    PredictionBasedDiversityService,
)
from src.infrastructure.metrics.sklearn_metrics_service import (
    SklearnMetricsService,
)


class ProgressiveTreeOrchestrator:
    """Orchestrator for Progressive Windows (PW) tree exchanging.

    Builds an ensemble by iteratively selecting the single best tree
    from client-built batches based on validation F1 and PCD.
    """

    def __init__(
        self, config: dict, step_callback: Optional[Callable[..., Any]] = None
    ):
        """Initializes ProgressiveTreeOrchestrator.

        Args:
            config (dict): Server configuration mapping.
            step_callback (Optional[Callable]): Progress reporting callback.
        """
        self.logger = logging.getLogger("FLEX_PW_Episodic")
        self.config = config if isinstance(config, dict) else config.dict()
        self.step_callback = step_callback or (lambda *a, **kw: None)

        self.dataset_split: Optional[DatasetSplit] = None
        self.federated_data = None
        self.flex_pool = None

        self.metrics_svc = SklearnMetricsService()
        self.diversity_svc = PredictionBasedDiversityService()
        self.label_svc = SimpleLabelService()

        agg_cfg = self.config.get("aggregation", {})
        self.window_size = int(agg_cfg.get("window_size", 5))
        self.max_rounds = int(agg_cfg.get("max_rounds", 20))
        self.f1_weight = float(agg_cfg.get("f1_weight", 0.5))
        self.convergence_threshold = float(
            agg_cfg.get("convergence_threshold", 0.002)
        )
        self.min_rounds = int(agg_cfg.get("min_rounds", 4))
        self.trees_per_round_per_client = int(
            agg_cfg.get("trees_per_round_per_client", 1)
        )

        self.n_estimators = self.max_rounds * self.window_size
        if "model" not in self.config:
            self.config["model"] = {}
        self.config["model"]["n_estimators"] = self.n_estimators

        self.global_trees = []

    def _get_strategy_name(self) -> str:
        """Returns the strategy identifier."""
        return "PW"

    def setup_federation(
        self, dataset_split: DatasetSplit, seed: int = 42
    ) -> None:
        """Setup federated data distribution.

        Args:
            dataset_split (DatasetSplit): Target dataset adapter split.
            seed (int): Stable random seed.
        """
        self.dataset_split = dataset_split
        self.label_svc.fit(dataset_split.y_train)

        class_names = self.label_svc.classes
        if "model" not in self.config:
            self.config["model"] = {}
        self.config["model"]["class_names"] = class_names

        from src.application.orchestrators.fed_data_distributor import (
            FedDataDistributor,
        )

        distributor = FedDataDistributor(self.config, True)
        self.dataset_split, self.federated_data = distributor.distribute(
            dataset_split, seed=seed
        )

        from src.infrastructure.flex.flex_pool_factory import FlexPoolFactory
        from src.infrastructure.flex.flex_train_pf import init_server_model_pf

        self.flex_pool = FlexPoolFactory.create_client_server_pool(
            federated_data=self.federated_data,
            init_model_func=init_server_model_pf,
            config=self.config,
        )

    def run_federated_round(self, n_bootstrap: int = 0) -> FLResults:
        """Executes episodic training rounds of the Progressive Windows strategy.

        Args:
            n_bootstrap (int): Bootstrap repetitions for statistical CI.

        Returns:
            FLResults: Aggregated FL results and evaluations.

        Raises:
            ValueError: If federation not setup.
            RuntimeError: If a client fails to return new trees during a round.
        """
        if not self.federated_data or self.dataset_split is None:
            raise ValueError("Federation not setup.")

        self.logger.info("Deploying configuration to clients...")
        self.flex_pool.servers.map(
            deploy_server_config_pf, self.flex_pool.clients
        )

        active_client_ids = list(self.flex_pool.clients.actor_ids)
        server_id = "server"

        X_val_server = self.dataset_split.X_val
        y_val_server = self.dataset_split.y_val
        y_val_encoded = self.label_svc.transform(y_val_server)

        global_correct_counts = np.zeros(X_val_server.shape[0], dtype=int)

        self.flex_pool._models[server_id]["global_trees"] = []

        global_stop_counter = 0
        global_prev_acc = None

        for round_idx in range(self.max_rounds):
            self.logger.info(
                f"--- PW Episodic Round {round_idx+1}/{self.max_rounds} ---"
            )
            if not active_client_ids:
                self.logger.info("Todos los clientes convergieron localmente.")
                break

            # 1. Train Window
            self.flex_pool.clients.map(
                train_window_pf_pw, active_ids=active_client_ids
            )

            # 2. Check local convergence
            newly_converged = []
            for cid in active_client_ids:
                client_model = None
                if cid in self.flex_pool._models:
                    client_model = self.flex_pool._models[cid]
                elif str(cid) in self.flex_pool._models:
                    client_model = self.flex_pool._models[str(cid)]
                elif (
                    isinstance(cid, str)
                    and cid.isdigit()
                    and int(cid) in self.flex_pool._models
                ):
                    client_model = self.flex_pool._models[int(cid)]

                if client_model is not None:
                    meta = client_model.get("metadata", {})
                    if isinstance(meta, dict):
                        has_conv = meta.get("has_converged", False)
                    else:
                        has_conv = getattr(meta, "has_converged", False)
                    if has_conv:
                        newly_converged.append(cid)

            # 3. Collect and Aggregate the new window of trees
            self.flex_pool.aggregators.map(
                collect_new_trees_pw, self.flex_pool.clients
            )
            self.flex_pool.aggregators.map(aggregate_trees_pw)
            self.flex_pool.aggregators.map(
                set_client_windows_pw, self.flex_pool.servers
            )

            # 4. Server evaluates the windows and picks 1 tree per client
            client_windows = self.flex_pool._models[server_id].get(
                "client_windows", {}
            )

            for cid in active_client_ids:
                cid_str = str(cid)
                window_data = client_windows.get(cid_str, {})
                w_trees = window_data.get("trees", [])
                w_metrics = window_data.get("metrics", [])

                if not w_trees:
                    raise RuntimeError(
                        f"CRITICAL: Active client {cid_str} returned no "
                        f"trees in round {round_idx+1}. This usually "
                        f"indicates a failure in local training or "
                        f"data corruption."
                    )

                for _ in range(min(self.trees_per_round_per_client, len(w_trees))):
                    best_tree = None
                    best_score = -1e9
                    best_idx = -1
    
                    for idx, tree in enumerate(w_trees):
                        cand_preds_raw = tree.predict(X_val_server)
                        if self.label_svc:
                            cand_preds_norm = self.label_svc.transform(
                                cand_preds_raw
                            )
                        else:
                            cand_preds_norm = cand_preds_raw

                        # Calcular métrica (F1) en el conjunto de validación global del servidor
                        tree_f1 = float(
                            self.metrics_svc.f1_score(
                                y_val_encoded, cand_preds_norm, average="macro"
                            )
                        )
    
                        if len(self.global_trees) > 0:
                            diversity = self.diversity_svc.calculate_marginal_pcd(
                                candidate_predictions=cand_preds_norm,
                                current_hits_per_sample=global_correct_counts,
                                n_existing_trees=len(self.global_trees),
                                y_true=y_val_encoded,
                            )
                        else:
                            diversity = 1.0
    
                        effective_f1_weight = (
                            self.f1_weight if len(self.global_trees) > 0 else 1.0
                        )
                        pcd_weight = 1.0 - effective_f1_weight
                        score = (effective_f1_weight * tree_f1) + (
                            pcd_weight * diversity
                        )
    
                        if score > best_score:
                            best_score = score
                            best_tree = tree
                            best_idx = idx
    
                    if best_tree is not None:
                        self.global_trees.append(best_tree)
                        self.logger.info(
                            f"  - Seleccionado árbol de cliente {cid_str} "
                            f"(score={best_score:.4f})"
                        )
                        preds_raw = best_tree.predict(X_val_server)
                        preds_int = self.label_svc.transform(preds_raw)
                        global_correct_counts += (preds_int == y_val_encoded).astype(
                            int
                        )
                        # Remove selected tree and its metrics to pick the next best
                        w_trees.pop(best_idx)
                        if best_idx < len(w_metrics):
                            w_metrics.pop(best_idx)

            # Update global model
            self.flex_pool._models[server_id]["global_trees"] = (
                self.global_trees
            )
            self.flex_pool._models[server_id]["trees"] = self.global_trees

            # 5. Deploy updated Global Forest to clients
            self.flex_pool.servers.map(
                deploy_global_forest_pw, self.flex_pool.clients
            )

            # Global Convergence Check
            if len(self.global_trees) > 0:
                global_forest_eval = ProactiveForest.from_trees(
                    self.global_trees, class_names=self.label_svc.classes
                )
                report = ForestEvaluator.evaluate(
                    global_forest_eval,
                    X_val_server,
                    y_val_server,
                    class_names=self.label_svc.classes,
                    metrics_svc=self.metrics_svc,
                )

                if global_prev_acc is not None:
                    acc_diff = report.accuracy - global_prev_acc
                else:
                    acc_diff = report.accuracy

                if (
                    global_prev_acc is not None
                    and acc_diff <= self.convergence_threshold
                    and (round_idx + 1) >= self.min_rounds
                ):
                    global_stop_counter += 1
                    if global_stop_counter >= 2:
                        self.logger.info(
                            f"Convergencia GLOBAL alcanzada en ronda "
                            f"{round_idx+1}. Deteniendo federación."
                        )
                        break
                else:
                    global_stop_counter = 0

                global_prev_acc = report.accuracy

            for cid in newly_converged:
                active_client_ids.remove(cid)

        # Final evaluation
        self.step_callback("Evaluación y consolidación de resultados...", 90)

        # Save to server_model for ResultConsolidator
        global_forest = ProactiveForest.from_trees(
            self.global_trees, class_names=self.label_svc.classes
        )
        self.flex_pool._models[server_id]["model"] = global_forest

        server_val_dataset = Dataset.from_array(X_val_server, y_val_server)
        from src.infrastructure.flex.flex_evaluate_pf import (
            evaluate_global_pf_model,
            evaluate_global_pf_model_at_clients,
        )

        server_eval = self.flex_pool.servers.map(
            evaluate_global_pf_model, test_data=server_val_dataset
        )
        self.flex_pool.clients.map(evaluate_global_pf_model_at_clients)

        consolidator = ResultConsolidator(self.label_svc, self.config)
        return consolidator.consolidate(
            strategy_name="PW",
            flex_pool=self.flex_pool,
            dataset_split=self.dataset_split,
            server_eval=server_eval,
            n_bootstrap=n_bootstrap,
        )

    def cleanup(self) -> None:
        """Release resources and terminate FlexPool actors."""
        if self.flex_pool:
            try:
                if hasattr(self.flex_pool, "terminate"):
                    self.flex_pool.terminate()
                elif hasattr(self.flex_pool, "close"):
                    self.flex_pool.close()
            except Exception as e:
                self.logger.error(f"Error terminating FlexPool: {e}")
            finally:
                self.flex_pool = None


__all__ = ["ProgressiveTreeOrchestrator"]
