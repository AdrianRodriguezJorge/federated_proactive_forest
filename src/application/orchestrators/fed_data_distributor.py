"""Federated Data Distributor.

Handles IID and non-IID allocation of client training subsets while upholding
the Enfoque B validation isolation standard.
"""

from typing import Any, Tuple
import numpy as np
from sklearn.model_selection import train_test_split

from flex.data import Dataset, FedDataDistribution, FedDatasetConfig
from src.domain.dataset.base_adapter import DatasetSplit


class FedDataDistributor:
    """Handles the distribution of dataset splits across FL clients."""

    def __init__(self, config: dict, use_flex_pool: bool = True):
        """Initializes FedDataDistributor.

        Args:
            config (dict): Server configuration mapping.
            use_flex_pool (bool): Whether to build FLEX data objects.
        """
        self.config = config
        self.use_flex_pool = use_flex_pool

    def _get_config_value(self, *keys: str, default: Any = None) -> Any:
        """Helper to extract deep config values safely."""
        val = self.config
        for key in keys:
            if isinstance(val, dict):
                val = val.get(key)
            else:
                return default
        return val if val is not None else default

    def distribute(
        self, dataset_split: DatasetSplit, seed: int = 42
    ) -> Tuple[DatasetSplit, Any]:
        """Distributes training data across federated clients.

        Uses pre-existing server validation split if available (Enfoque B)
        to ensure fitted models are never exposed to the validation target.

        Args:
            dataset_split (DatasetSplit): Target dataset adapter split.
            seed (int): Stable random seed.

        Returns:
            Tuple[DatasetSplit, Any]: Updated dataset split and FLEX
                FedDataDistribution container.
        """
        # RIGOR ENFOQUE B: If the adapter already provided X_val, use it directly
        # to ensure zero-leakage (fitted ONLY on X_train).
        if dataset_split.X_val is not None:
            X_train_fed = dataset_split.X_train
            y_train_fed = dataset_split.y_train
            X_server_val = dataset_split.X_val
            y_server_val = dataset_split.y_val
        elif len(dataset_split.X_train) > 20:
            # Fallback for legacy adapters
            try:
                (
                    X_train_fed,
                    X_server_val,
                    y_train_fed,
                    y_server_val,
                ) = train_test_split(
                    dataset_split.X_train,
                    dataset_split.y_train,
                    test_size=0.1765,
                    random_state=seed,
                    stratify=dataset_split.y_train,
                )
            except ValueError:
                (
                    X_train_fed,
                    X_server_val,
                    y_train_fed,
                    y_server_val,
                ) = train_test_split(
                    dataset_split.X_train,
                    dataset_split.y_train,
                    test_size=0.1765,
                    random_state=seed,
                )
            dataset_split.X_val = X_server_val
            dataset_split.y_val = y_server_val
            dataset_split.X_train = X_train_fed
            dataset_split.y_train = y_train_fed
        else:
            X_train_fed = dataset_split.X_train
            y_train_fed = dataset_split.y_train
            X_server_val = dataset_split.X_train
            y_server_val = dataset_split.y_train
            dataset_split.X_val = X_server_val
            dataset_split.y_val = y_server_val

        # Create FLEX Dataset
        centralized_dataset = Dataset.from_array(
            X_array=np.asarray(X_train_fed), y_array=np.asarray(y_train_fed)
        )

        n_clients = self._get_config_value(
            "federation", "n_clients"
        ) or self._get_config_value("n_clients", default=5)
        distribution_type = (
            self._get_config_value("federation", "distribution")
            or self._get_config_value("distribution", default="iid")
        ).lower()

        if distribution_type == "iid":
            federated_data = FedDataDistribution.iid_distribution(
                centralized_dataset, n_nodes=n_clients
            )
        else:
            alpha = self._get_config_value(
                "federation", "dirichlet_alpha"
            ) or self._get_config_value("alpha", default=0.5)
            y_train = centralized_dataset.y_data.to_numpy()
            n_classes = len(np.unique(y_train))
            rng = np.random.RandomState(seed)
            weights_per_label = rng.dirichlet(
                [alpha] * n_classes, size=n_clients
            )
            config = FedDatasetConfig(
                n_nodes=n_clients, weights_per_label=weights_per_label
            )
            federated_data = FedDataDistribution.from_config(
                centralized_dataset, config
            )

        return dataset_split, federated_data
