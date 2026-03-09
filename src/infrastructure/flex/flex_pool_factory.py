"""Factory for creating FLEX pools and managing federated setup."""
from typing import Dict, Any, List
import numpy as np

class FlexPoolFactory:
    """
    Factory for creating and managing FLEX pools for federated learning.
    """

    @staticmethod
    def create_client_server_pool(federated_data, init_model_func):
        """
        Create a client-server FLEX pool.

        Args:
            federated_data: FedDataDistribution from FLEX
            init_model_func: Function to initialize the model

        Returns:
            FlexPool instance
        """
        try:
            from flex.pool import FlexPool
        except ImportError:
            raise ImportError("FLEX not installed. Run: pip install flex-framework")

        # Create actors
        actors = FlexPoolFactory._create_client_server_actors(federated_data)

        # Create pool
        pool = FlexPool(
            federated_data=federated_data,
            actors=actors,
            init_func=init_model_func
        )

        return pool

    @staticmethod
    def _create_client_server_actors(federated_data):
        """Create client-server actor configuration."""
        try:
            from flex.actors import client_server_architecture
        except ImportError:
            raise ImportError("FLEX not installed")

        client_ids = list(federated_data.keys())
        server_id = "server"

        actors = client_server_architecture(
            clients_ids=client_ids,
            server_id=server_id
        )

        return actors

    @staticmethod
    def distribute_data_to_clients(pool, federated_data):
        """Distribute the federated data to clients in the pool."""
        for client_id in pool.clients:
            if client_id in federated_data:
                pool[client_id]['data'] = federated_data[client_id]