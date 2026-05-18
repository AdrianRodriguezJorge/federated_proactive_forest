"""Roulette Orchestrator — S9 Global Attribute Roulette strategy.

Implements a multi-round federated loop where, instead of exchanging trees,
clients and server exchange feature-probability vectors (roulettes).
"""

from __future__ import annotations
from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from scipy import stats

try:
    from flex.data import Dataset, FedDataDistribution
    from flex.model import FlexModel
except ImportError:
    FlexModel = Any
    Dataset = Any

from src.application.orchestrators.fed_data_distributor import (
    FedDataDistributor,
)
from src.application.orchestrators.fl_results import FLResults
from src.application.orchestrators.result_consolidator import (
    ResultConsolidator,
)
from src.domain.dataset.base_adapter import DatasetSplit
from src.domain.metrics.forest_evaluator import ForestEvaluator
from src.domain.model.proactive_forest import ProactiveForest
from src.domain.services.label_service import SimpleLabelService
from src.domain.update.roulette_updater import RouletteUpdater
from src.infrastructure.flex.flex_deploy_model_pf import (
    deploy_server_config_pf,
)
from src.infrastructure.flex.flex_pool_factory import FlexPoolFactory
from src.infrastructure.flex.flex_roulette_pf import (
    aggregate_roulettes,
    collect_client_roulette,
    deploy_global_roulette,
    set_global_roulette,
)
from src.infrastructure.flex.flex_s9_progressive import (
    check_convergence_s9,
    train_window_pf_s9,
)
from src.infrastructure.flex.flex_train_pf import (
    init_server_model_pf,
    train_pf,
)
from src.infrastructure.logging.logger_setup import setup_project_logger
from src.infrastructure.metrics.diversity_service import (
    PredictionBasedDiversityService,
)
from src.infrastructure.metrics.sklearn_metrics_service import (
    SklearnMetricsService,
)


@dataclass
class RouletteResults(FLResults):
    """Extended FLResults for the S9 strategy with roulette-specific data."""

    roulette_variant: str = "S9_MEAN"
    beta: float = 0.0
    n_features: int = 0
    total_communication_bytes: int = 0
    upload_bytes_per_round: List[int] = field(default_factory=list)
    download_bytes_per_round: List[int] = field(default_factory=list)
    roulette_history: List[Dict[str, Any]] = field(default_factory=list)


class RouletteOrchestrator:
    """Federated Learning orchestrator for S9 Global Attribute Roulette."""

    def __init__(
        self,
        config: Union[dict, Any],
        step_callback: Optional[Callable[..., Any]] = None,
    ):
        """Initializes RouletteOrchestrator.

        Args:
            config (Union[dict, Any]): Configuration dictionary or object.
            step_callback (Optional[Callable]): Progress reporting callback.
        """
        self.config = config if isinstance(config, dict) else config.dict()
        self.step_callback = step_callback or (lambda *a, **kw: None)

        self.dataset_split: Optional[DatasetSplit] = None
        self.federated_data = None
        self.flex_pool = None

        self.metrics_svc = SklearnMetricsService()
        self.diversity_svc = PredictionBasedDiversityService()
        self.label_svc = SimpleLabelService()

        # S9-specific config
        agg_cfg = self.config.get("aggregation", {})
        self.variant = agg_cfg.get("variant", "S9_MEAN")
        self.beta = float(agg_cfg.get("beta", 0.0))
        self.window_size = int(agg_cfg.get("window_size", 5))
        self.max_rounds = int(agg_cfg.get("max_rounds", 20))
        self.convergence_threshold = float(
            self.config.get("model", {}).get(
                "local_convergence_threshold", 0.002
            )
        )

        calculated_n_estimators = self.max_rounds * self.window_size
        if "model" not in self.config:
            self.config["model"] = {}
        self.config["model"]["n_estimators"] = calculated_n_estimators
        self.t_max = calculated_n_estimators

        self._setup_logging()

    def _setup_logging(self) -> None:
        """Sets up project logger using centralized logging system."""
        log_name = f"Roulette_{self.variant.lower()}"
        self.logger = setup_project_logger(log_name)

    def setup_federation(
        self, dataset_split: DatasetSplit, seed: int = 42
    ) -> None:
        """Distribute data and initialize FlexPool.

        Args:
            dataset_split (DatasetSplit): Target dataset adapter split.
            seed (int): Stable random seed.
        """
        distributor = FedDataDistributor(self.config, use_flex_pool=True)
        self.dataset_split, self.federated_data = distributor.distribute(
            dataset_split, seed
        )

        self.label_svc.fit(
            self.dataset_split.class_names
            or self.dataset_split.get_all_labels()
        )

        model_config = self.config.setdefault("model", {})
        if self.dataset_split.class_names:
            model_config["class_names"] = self.dataset_split.class_names

        try:
            self.flex_pool = FlexPoolFactory.create_client_server_pool(
                federated_data=self.federated_data,
                init_model_func=init_server_model_pf,
                config=self.config,
            )
            self.step_callback("FlexPool inicializado (Roulette S9)", 10)
        except Exception as e:
            self.flex_pool = None
            self.logger.error(f"Error al inicializar FlexPool: {e}")
            raise

    def run_federated_round(self, n_bootstrap: int = 0) -> RouletteResults:
        """Execute the full multi-round S9 federated experiment.

        Args:
            n_bootstrap (int): Bootstrap repetitions for statistical CI.

        Returns:
            RouletteResults: Consolidated FL results from this strategy.

        Raises:
            RuntimeError: If federation not setup or client model state is missing.
            ValueError: If validation dataset is missing under consensus strategies.
        """
        if self.flex_pool is None or self.dataset_split is None:
            raise RuntimeError(
                "Federation not set up. Call setup_federation() first."
            )

        self.step_callback("Iniciando S9 Roulette Federada...", 5)
        self.logger.info(
            f"Starting S9 with variant={self.variant}, beta={self.beta}"
        )

        updater = RouletteUpdater(beta=self.beta)

        total_upload = 0
        total_download = 0
        upload_per_round: List[int] = []
        download_per_round: List[int] = []
        roulette_history: List[Dict[str, Any]] = []

        previous_accuracy = None
        stop_counter = 0
        final_round = 0

        # Track which clients are still training
        active_client_ids = list(self.flex_pool.clients.actor_ids)
        convergence_round_map = {}

        for round_num in range(self.max_rounds):
            round_idx = round_num + 1
            progress = 10 + int(round_num * 75 / self.max_rounds)
            self.step_callback(
                f"S9 Ronda {round_idx}/{self.max_rounds} "
                f"(Activos: {len(active_client_ids)})...",
                progress,
            )
            self.logger.info(
                f"--- Round {round_idx} | Active: {len(active_client_ids)} ---"
            )

            if not active_client_ids:
                self.logger.info("Todos los clientes han convergido localmente.")
                final_round = round_idx - 1
                break

            self.flex_pool.servers.map(
                deploy_server_config_pf, self.flex_pool.clients
            )

            # Local training
            self.flex_pool.clients.map(
                train_window_pf_s9, active_ids=active_client_ids
            )

            # Check convergence for just-trained clients
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

                if meta.get("has_converged", False):
                    newly_converged.append(cid)
                    convergence_round_map[cid] = round_idx
                    self.logger.info(
                        f"Cliente {cid} detuvo su entrenamiento en ronda "
                        f"{round_idx} (Convergencia local alcanzada)."
                    )

            # Collect roulette vectors from ALL clients
            self.flex_pool.aggregators.map(
                collect_client_roulette, self.flex_pool.clients
            )

            # Evaluate client models directly on server's validation dataset
            server_eval_f1 = {}
            server_eval_pcd = {}

            X_val = self.dataset_split.X_val
            y_val = self.dataset_split.y_val

            is_consensus_var = self.variant in (
                "S9_CONSENSUS",
                "S9_PROACTIVE_PCD",
            )
            if is_consensus_var and (X_val is None or y_val is None):
                raise ValueError(
                    f"CRITICAL: Server-side validation dataset "
                    f"(X_val, y_val) must be provided for strategy "
                    f"{self.variant}."
                )

            if X_val is not None and y_val is not None:
                y_val_numeric = self.label_svc.transform(y_val)
                for cid in active_client_ids:
                    s_cid = str(cid)
                    client_model = self.flex_pool._models.get(s_cid)
                    if client_model is None and s_cid.isdigit():
                        client_model = self.flex_pool._models.get(int(s_cid))
                    if client_model is None:
                        client_model = self.flex_pool._models.get(cid)

                    if client_model is not None:
                        pf = client_model.get("model")
                        if pf is not None:
                            preds = pf.predict(X_val)
                            preds_numeric = self.label_svc.transform(preds)
                            f1 = float(
                                self.metrics_svc.f1_score(
                                    y_val_numeric,
                                    preds_numeric,
                                    average="macro",
                                )
                            )
                            server_eval_f1[s_cid] = f1

                            try:
                                pcd = float(
                                    pf.diversity_measure(
                                        X_val, y_val_numeric, diversity="pcd"
                                    )
                                )
                            except Exception:
                                pcd = 0.0
                            server_eval_pcd[s_cid] = pcd

            # Aggregate
            self.flex_pool.aggregators.map(
                aggregate_roulettes,
                variant=self.variant,
                server_eval_f1=server_eval_f1,
                server_eval_pcd=server_eval_pcd,
            )
            self.flex_pool.aggregators.map(
                set_global_roulette, self.flex_pool.servers
            )

            # Track communication cost
            server_model = self.flex_pool._models["server"]
            round_up = server_model.get("roulette_upload_bytes", 0)
            round_down = server_model.get("roulette_download_bytes", 0)
            total_upload += round_up
            total_download += round_down
            upload_per_round.append(round_up)
            download_per_round.append(round_down)

            # Deploy global roulette back to clients
            self.flex_pool.servers.map(
                deploy_global_roulette, self.flex_pool.clients
            )

            # Clients fuse local <-> global roulettes
            global_roulette_list = server_model.get("global_roulette", [])
            if global_roulette_list:
                global_vec = np.array(global_roulette_list, dtype=np.float64)
                for cid in active_client_ids:
                    s_cid = str(cid)
                    client_model = self.flex_pool._models.get(s_cid)
                    if client_model is None and s_cid.isdigit():
                        client_model = self.flex_pool._models.get(int(s_cid))
                    if client_model is None:
                        client_model = self.flex_pool._models.get(cid)

                    if client_model is None:
                        raise RuntimeError(
                            f"CRITICAL: Could not find model state for "
                            f"active client {cid} in FlexPool."
                        )

                    pf_model = client_model.get("model")
                    if pf_model is None:
                        raise RuntimeError(
                            f"CRITICAL: Client {cid} state is missing "
                            f"the 'model' object."
                        )

                    local_vec = pf_model.get_feature_probabilities()
                    if local_vec.shape != global_vec.shape:
                        raise ValueError(
                            f"CRITICAL: Dimension mismatch in client {cid}: "
                            f"local {local_vec.shape} vs global "
                            f"{global_vec.shape}"
                        )

                    fused = updater.fuse(local_vec, global_vec)
                    pf_model.set_feature_probabilities(fused)

            # Update active list for next round
            for cid in newly_converged:
                active_client_ids.remove(cid)

            # Record roulette snapshot
            roulette_history.append(
                {
                    "round": round_idx,
                    "global_roulette": global_roulette_list,
                }
            )

            final_round = round_idx

        # ── Final evaluation & Results Consolidation ─────────────────────────
        self.step_callback("Consolidando resultados S9...", 90)

        X_test = self.dataset_split.X_test
        y_test = self.dataset_split.y_test
        y_test_numeric = self.label_svc.transform(y_test)
        class_names = self.label_svc.classes
        client_ids = list(self.flex_pool.clients.actor_ids)

        client_accuracies = {}
        client_f1_scores = {}
        client_metadata_dict = {}
        client_reports = {}
        client_hybrid_predictions = {}
        client_hybrid_forest_sizes = {}
        all_local_preds = []
        total_trees = 0

        for cid in client_ids:
            s_cid = str(cid)
            state = self.flex_pool._models.get(s_cid)
            if state is None and s_cid.isdigit():
                state = self.flex_pool._models.get(int(s_cid))
            if state is None:
                state = self.flex_pool._models.get(cid)

            if state is None:
                continue

            pf = state.get("model")
            meta = state.get("metadata", {})

            if not isinstance(meta, dict):
                meta = {"n_trees": getattr(meta, "n_trees", 0)}

            client_metadata_dict[s_cid] = meta
            forest_size = meta.get("n_trees", 0)
            total_trees += forest_size
            client_hybrid_forest_sizes[s_cid] = forest_size

            if pf is None:
                raise RuntimeError(
                    f"CRITICAL: Model for client {cid} is None during "
                    f"final evaluation."
                )

            preds = pf.predict(X_test)
            preds_numeric = self.label_svc.transform(preds)

            acc = float(
                self.metrics_svc.accuracy_score(y_test_numeric, preds_numeric)
            )
            f1 = float(
                self.metrics_svc.f1_score(
                    y_test_numeric, preds_numeric, average="macro"
                )
            )

            client_accuracies[s_cid] = acc
            client_f1_scores[s_cid] = f1
            client_hybrid_predictions[s_cid] = preds_numeric
            all_local_preds.append(preds_numeric)

            client_reports[s_cid] = ForestEvaluator.evaluate_from_predictions(
                preds_numeric,
                y_test_numeric,
                class_names,
                forest_size,
                pcd=meta.get("pcd", 0.0),
                n_bootstrap=n_bootstrap,
            )

        if all_local_preds:
            stacked = np.stack(all_local_preds, axis=0)
            global_preds, _ = stats.mode(stacked, axis=0, keepdims=False)
            global_preds = global_preds.flatten().astype(int)
        else:
            global_preds = np.zeros(len(y_test_numeric), dtype=int)

        global_acc = float(
            self.metrics_svc.accuracy_score(y_test_numeric, global_preds)
        )
        global_f1 = float(
            self.metrics_svc.f1_score(
                y_test_numeric, global_preds, average="macro"
            )
        )

        global_report = ForestEvaluator.evaluate_from_predictions(
            global_preds,
            y_test_numeric,
            class_names,
            total_trees,
            pcd=0.0,
            n_bootstrap=n_bootstrap,
        )

        self.step_callback("S9 Roulette completada", 100)

        return RouletteResults(
            strategy_id=f"S9_{self.variant.replace('S9_', '')}",
            global_accuracy=global_acc,
            global_macro_f1=global_f1,
            n_trees_global=total_trees,
            client_ids=list(client_reports.keys()),
            client_accuracies=client_accuracies,
            client_f1_scores=client_f1_scores,
            client_metadata=client_metadata_dict,
            client_reports=client_reports,
            y_test=y_test_numeric,
            class_names=class_names,
            feature_names=self.dataset_split.feature_names,
            num_rounds=final_round,
            convergence_round=final_round if not active_client_ids else None,
            communication_cost=float(total_upload + total_download),
            global_report=global_report,
            global_predictions=global_preds,
            client_hybrid_predictions=client_hybrid_predictions,
            client_hybrid_forest_sizes=client_hybrid_forest_sizes,
            roulette_variant=self.variant,
            beta=self.beta,
            n_features=len(global_roulette_list) if global_roulette_list else 0,
            total_communication_bytes=total_upload + total_download,
            upload_bytes_per_round=upload_per_round,
            download_bytes_per_round=download_per_round,
            roulette_history=roulette_history,
        )

    def cleanup(self) -> None:
        """Release resources and terminate FlexPool actors."""
        if self.flex_pool:
            try:
                if hasattr(self.flex_pool, "terminate"):
                    self.flex_pool.terminate()
                elif hasattr(self.flex_pool, "close"):
                    self.flex_pool.close()
                self.logger.debug(
                    "FlexPool (Roulette) terminated successfully."
                )
            except Exception as e:
                self.logger.error(f"Error terminating FlexPool: {e}")
            finally:
                self.flex_pool = None


__all__ = ["RouletteOrchestrator", "RouletteResults"]
