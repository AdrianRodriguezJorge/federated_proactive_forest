"""Factory for creating FLEX pools and managing federated setup."""

from typing import Any, Dict, List
import numpy as np


class FlexPoolFactory:
    """Factory for creating and managing FLEX pools for federated learning."""

    @staticmethod
    def create_client_server_pool(
        federated_data: Any, init_model_func: Any, **kwargs: Any
    ) -> Any:
        """Create a client-server FLEX pool using the native classmethod.

        Args:
            federated_data: FedDataDistribution from FLEX.
            init_model_func: Function to initialize the model.
            **kwargs: Additional arguments passed to init_model_func.

        Returns:
            FlexPool: FlexPool instance.

        Raises:
            ImportError: If FLEX framework is not installed.
        """
        try:
            from flex.pool import FlexPool
        except ImportError:
            raise ImportError(
                "FLEX not installed. Run: pip install flex-framework"
            )

        # Native classmethod handles actor creation and model initialization
        return FlexPool.client_server_pool(
            fed_dataset=federated_data, init_func=init_model_func, **kwargs
        )

    @staticmethod
    def _create_client_server_actors(federated_data: Any) -> Any:
        """Create client-server actor configuration.

        Args:
            federated_data: Federated dataset mapping.

        Returns:
            Any: Configured actors.

        Raises:
            ImportError: If FLEX framework is not installed.
        """
        try:
            from flex.actors import client_server_architecture
        except ImportError:
            raise ImportError("FLEX not installed")

        client_ids = list(federated_data.keys())
        server_id = "server"

        actors = client_server_architecture(
            clients_ids=client_ids, server_id=server_id
        )

        return actors

    @staticmethod
    def distribute_data_to_clients(pool: Any, federated_data: Any) -> None:
        """Distribute the federated data to clients in the pool.

        Args:
            pool (FlexPool): The active FLEX pool.
            federated_data (Any): Dataset distribution mapping.
        """
        for client_id in pool.clients:
            if client_id in federated_data:
                pool[client_id]["data"] = federated_data[client_id]