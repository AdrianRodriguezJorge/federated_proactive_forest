from typing import Dict, Type
from .base_strategy import IAggregationStrategy
from .strategies.s1_simple_pool import S1SimplePoolStrategy
from .strategies.s2_global_accuracy import S2GlobalAccuracyStrategy

# Import other strategies when implemented
# from .strategies.s3_global_f1 import S3GlobalF1Strategy
# etc.

class AggregationFactory:
    """
    Factory for creating aggregation strategies based on configuration.
    """

    _strategies: Dict[str, Type[IAggregationStrategy]] = {
        "S1": S1SimplePoolStrategy,
        "S2": S2GlobalAccuracyStrategy,
        # Add others as implemented
    }

    @classmethod
    def create_strategy(cls, strategy_name: str) -> IAggregationStrategy:
        """
        Create an aggregation strategy instance.

        Args:
            strategy_name: Name of the strategy (S1, S2, etc.)

        Returns:
            Instance of the requested strategy

        Raises:
            ValueError: If strategy name is not recognized
        """
        if strategy_name not in cls._strategies:
            available = list(cls._strategies.keys())
            raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {available}")

        return cls._strategies[strategy_name]()