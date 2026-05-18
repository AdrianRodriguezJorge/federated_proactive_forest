"""Aggregation strategies factory module.

Provides a unified factory to initialize, configure, and resolve tree
aggregation strategies (S1-S7, Progressive Windows) under standard interfaces.
"""

from typing import Dict, Optional, Type
from .base_strategy import IAggregationStrategy
from .strategies.s1_simple_pool import S1SimplePoolStrategy
from .strategies.s2_global_accuracy import S2GlobalAccuracyStrategy
from .strategies.s3_global_f1 import S3GlobalF1Strategy
from .strategies.s4_global_f1_pcd import S4GlobalF1PCDStrategy
from .strategies.s5_perclient_accuracy import S5PerClientAccuracyStrategy
from .strategies.s6_perclient_f1 import S6PerClientF1Strategy
from .strategies.s7_perclient_f1_pcd import S7PerClientF1PCDStrategy
from .strategies.progressive_windows.progressive_windows_strategy import (
    ProgressiveWindowsStrategy,
)
from src.domain.metrics.metrics_service import IDiversityService, IMetricsService


class AggregationFactory:
    """Factory for creating aggregation strategies based on configuration.

    Supports S1-S7 strategies and Progressive Windows (PW).
    """

    _strategies: Dict[str, Type[IAggregationStrategy]] = {
        "S1": S1SimplePoolStrategy,
        "S2": S2GlobalAccuracyStrategy,
        "S3": S3GlobalF1Strategy,
        "S4": S4GlobalF1PCDStrategy,
        "S5": S5PerClientAccuracyStrategy,
        "S6": S6PerClientF1Strategy,
        "S7": S7PerClientF1PCDStrategy,
        "PW": ProgressiveWindowsStrategy,
    }

    @classmethod
    def normalize_strategy_name(cls, raw_strategy: str) -> str:
        """Normalize strategy names to match canonical codes.

        E.g. 's4_global_f1_pcd' -> 'S4', 'pw' -> 'PW'.

        Args:
            raw_strategy (str): Raw input strategy identifier.

        Returns:
            str: Normalized strategy name.
        """
        if not raw_strategy:
            return "S1"
        if "_" in raw_strategy:
            strategy_name = raw_strategy.split("_")[0].upper()
        else:
            strategy_name = raw_strategy.upper()

        # Handle variations of Progressive Windows
        if strategy_name in ["PROGRESSIVE", "PROGRESSIVE_WINDOWS"]:
            return "PW"

        return strategy_name

    @classmethod
    def create_strategy(
        cls,
        strategy_name: str,
        metrics_service: Optional[IMetricsService] = None,
        diversity_service: Optional[IDiversityService] = None,
    ) -> IAggregationStrategy:
        """Create an aggregation strategy instance with injected services.

        Args:
            strategy_name (str): Raw strategy identifier.
            metrics_service (Optional[IMetricsService]): Performance metrics.
            diversity_service (Optional[IDiversityService]): Diversity metrics.

        Returns:
            IAggregationStrategy: Configured strategy instance.

        Raises:
            ValueError: If the normalized strategy name is unknown.
        """
        normalized_name = cls.normalize_strategy_name(strategy_name)
        if normalized_name not in cls._strategies:
            available = sorted(cls._strategies.keys())
            raise ValueError(
                f"Unknown strategy '{strategy_name}' "
                f"(normalized: '{normalized_name}'). "
                f"Available: {available}"
            )

        strategy_cls = cls._strategies[normalized_name]

        # Inject dependencies via constructor
        return strategy_cls(
            metrics_service=metrics_service,
            diversity_service=diversity_service,
        )