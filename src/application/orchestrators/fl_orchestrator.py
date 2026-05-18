"""Enhanced FLEXOrchestrator for Proactive Forest.

This module provides complete federated learning orchestration using FLEX
Framework, supporting both IID and Non-IID (Dirichlet) data distributions.
"""

from __future__ import annotations
import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    from flex.data import Dataset, FedDataDistribution, FedDatasetConfig
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
from src.domain.metrics.forest_evaluator import ForestEvaluator, ForestReport
from src.domain.model.proactive_forest import ProactiveForest
from src.domain.services.label_service import SimpleLabelService
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
    evaluate_local_pf_model_at_clients,
)
from src.infrastructure.flex.flex_train_pf import (
    collect_clients_trees_pf,
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


class FLEXOrchestrator:
    """Orchestrator using native FLEX Framework constructs.

    Handles the full lifecycle of a federated round, including:
    - Data distribution (IID/Non-IID).
    - Client initialization and local training.
    - Weight collection and global aggregation.
    - Model deployment and evaluation.
    """

    def __init__(
        self,
        config: Union[dict, Any],
        step_callback: Optional[Callable[..., Any]] = None,
        use_flex_pool: bool = True,
    ):
        """Initializes the orchestrator.

        Args:
            config (Union[dict, Any]): Configuration dictionary or object.
            step_callback (Optional[Callable]): Progress reporting callback.
            use_flex_pool (bool): Whether to use FlexPool for communication.
        """
        self.config = config if isinstance(config, dict) else config.dict()
        self.config_dict = self.config
        self.step_callback = step_callback or (lambda *a, **kw: None)

        self.dataset_split: Optional[DatasetSplit] = None
        self.federated_data = None
        self.use_flex_pool = use_flex_pool
        self.flex_pool = None

        self.metrics_svc = SklearnMetricsService()
        self.diversity_svc = PredictionBasedDiversityService()
        self.label_svc = SimpleLabelService()

        self._setup_logging()

    @classmethod
    def from_config(
        cls,
        config: Union[dict, Any],
        step_callback: Optional[Callable[..., Any]] = None,
        use_flex_pool: bool = True,
    ) -> "FLEXOrchestrator":
        """Create an orchestrator instance from a configuration dict.

        Mirrors the previous API used in notebooks.

        Args:
            config (Union[dict, Any]): Configuration dictionary or object.
            step_callback (Optional[Callable]): Progress reporting callback.
            use_flex_pool (bool): Whether to use FlexPool.

        Returns:
            FLEXOrchestrator: Initialized orchestrator instance.
        """
        return cls(
            config, step_callback=step_callback, use_flex_pool=use_flex_pool
        )

    def _setup_logging(self) -> None:
        """Sets up project logger using centralized logging system."""
        strategy = str(
            self.config.get("aggregation", {}).get("strategy", "S1")
        ).lower()
        log_name = f"FLEX_{strategy}"
        self.logger = setup_project_logger(log_name)

    def _get_config_value(self, *keys: str, default: Any = None) -> Any:
        """Helper to extract nested configuration values safely."""
        val = self.config
        for key in keys:
            if isinstance(val, dict):
                val = val.get(key)
            else:
                return default
        return val if val is not None else default

    def _get_strategy_name(self) -> str:
        """Gets strategy name from configuration."""
        from src.domain.aggregation.aggregation_factory import (
            AggregationFactory,
        )

        raw_strategy = self.config.get("strategy") or self.config.get(
            "aggregation", {}
        ).get("strategy", "S1")
        return AggregationFactory.normalize_strategy_name(str(raw_strategy))

    def _init_flex_pool(self) -> None:
        """Internal helper to initialize FlexPool using the factory."""
        try:
            from src.infrastructure.flex.flex_pool_factory import (
                FlexPoolFactory,
            )

            self.flex_pool = FlexPoolFactory.create_client_server_pool(
                federated_data=self.federated_data,
                init_model_func=init_server_model_pf,
                config=self.config,
            )
            self.step_callback("FlexPool inicializado con éxito", 15)
        except Exception as e:
            self.flex_pool = None
            self.logger.error(f"Error al inicializar FlexPool: {e}")
            warnings.warn(f"FLEX integration downgraded: {e}")

    def setup_federation(
        self, dataset_split: DatasetSplit, seed: int = 42
    ) -> None:
        """Setup federated data distribution using FedDataDistributor.

        Args:
            dataset_split (DatasetSplit): Target dataset adapter split.
            seed (int): Stable random seed.
        """
        distributor = FedDataDistributor(self.config, self.use_flex_pool)
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

        if self.use_flex_pool:
            self._init_flex_pool()

        self.step_callback("Federación FLEX configurada", 10)

    def run_federated_round(self, n_bootstrap: int = 0) -> FLResults:
        """Execute federated round using native FLEX orchestration.

        Args:
            n_bootstrap (int): Bootstrap repetitions for CI computation.

        Returns:
            FLResults: Consolidated results from this round.

        Raises:
            ValueError: If federation has not been set up.
            RuntimeError: If FlexPool is not initialized.
        """
        if not self.federated_data or self.dataset_split is None:
            raise ValueError("Federation not setup.")

        if self.flex_pool is None:
            raise RuntimeError(
                "FlexPool not initialized. Native integration required."
            )

        self.step_callback("Iniciando ronda federada nativa FLEX...", 5)

        self.logger.debug("Deploying configuration to clients...")
        self.flex_pool.servers.map(
            deploy_server_config_pf, self.flex_pool.clients
        )

        # 2. TRAIN local models
        self.logger.debug("Starting local training round...")
        self.step_callback("Entrenamiento local (FLEX map)...", 20)
        self.flex_pool.clients.map(train_pf)

        # 3. COLLECT trees
        self.logger.debug("Collecting trees from clients...")
        self.step_callback("Recolección de pesos (FLEX run)...", 45)
        self.flex_pool.aggregators.map(
            collect_clients_trees_pf, self.flex_pool.clients
        )

        # 4. AGGREGATE
        self.logger.debug(
            f"Aggregating models using strategy: {self._get_strategy_name()}"
        )
        self.step_callback("Agregación global (FLEX aggregate)...", 60)
        strategy_name = self._get_strategy_name()
        n_estimators = self._get_config_value(
            "model", "n_estimators", default=100
        )
        t_max = self._get_config_value(
            "aggregation", "t_max", default=n_estimators
        )

        X_val_server = self.dataset_split.X_val
        y_val_server = self.dataset_split.y_val

        # Validation data as FLEX Dataset for server eval primitives
        server_val_dataset = Dataset.from_array(X_val_server, y_val_server)

        agg_kwargs = {
            "server_config": self.config,
            "X_val": X_val_server,
            "y_val": y_val_server,
            "t_max": t_max,
            "metrics_service": self.metrics_svc,
            "diversity_service": self.diversity_svc,
        }
        self.flex_pool.aggregators.map(aggregate_trees_pf, **agg_kwargs)
        self.flex_pool.aggregators.map(
            set_aggregated_trees_pf, self.flex_pool.servers
        )

        # 5. DEPLOY global model
        self.flex_pool.servers.map(
            deploy_server_model_pf, self.flex_pool.clients
        )

        # 6. EVALUATE
        self.step_callback("Evaluación (FLEX evaluate)...", 80)
        # Server-side evaluation
        server_eval = self.flex_pool.servers.map(
            evaluate_global_pf_model, test_data=server_val_dataset
        )
        # Client-side evaluation
        self.flex_pool.clients.map(evaluate_global_pf_model_at_clients)

        # 7. Collect results and build FLResults
        consolidator = ResultConsolidator(self.label_svc, self.config)
        return consolidator.consolidate(
            strategy_name=strategy_name,
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
                self.logger.debug("FlexPool terminated successfully.")
            except Exception as e:
                self.logger.error(f"Error terminating FlexPool: {e}")
            finally:
                self.flex_pool = None


__all__ = ["FLEXOrchestrator", "FLResults"]