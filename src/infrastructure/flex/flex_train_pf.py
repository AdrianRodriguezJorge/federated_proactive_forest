"""FLEX Client training procedures for Proactive Forest.

Handles local training, incremental episode building, stable validation
splitting, and metadata reporting for Proactive Forest classifiers in FLEX.
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np

from flex.model import FlexModel
from flex.pool.decorators import collect_clients_weights, init_server_model


@init_server_model
def init_server_model_pf(
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Initialize server model for Proactive Forest.

    FLEX primitive function.

    Args:
        config (Optional[Dict[str, Any]]): Configuration dictionary.

    Returns:
        Dict[str, Any]: Initial dictionary used to update the FlexModel.
    """
    return {
        "model": None,
        "trees": [],
        "config": config or {},
    }


def train_pf(client_flex_model: FlexModel, client_data: Any) -> FlexModel:
    """Train Proactive Forest on client data with local validation.

    Args:
        client_flex_model (FlexModel): FLEX client model state wrapper.
        client_data (Any): Dataset local to client.

    Returns:
        FlexModel: Updated client model state.
    """
    logger = logging.getLogger("FLEX_Client")
    actor_id = getattr(client_flex_model, "actor_id", "unknown")
    logger.debug(f"Client {actor_id} starting training...")

    from src.domain.metadata.client_metadata import ClientMetadata
    from src.domain.model.proactive_forest import ProactiveForest
    from src.domain.services.label_service import SimpleLabelService

    # 1. Prepare data
    X_raw, y_raw = client_data.to_numpy()
    config = client_flex_model.get("config", {})

    class_names = config.get("class_names")
    if not class_names:
        class_names = config.get("model", {}).get("class_names", [])

    model_config = config.get("model", {})
    n_estimators = model_config.get(
        "n_estimators", config.get("n_estimators", 100)
    )
    alpha = model_config.get("alpha", config.get("alpha", 0.1))
    local_convergence_threshold = model_config.get(
        "local_convergence_threshold", 0.002
    )

    label_svc = SimpleLabelService(class_names)
    if not label_svc.classes:
        label_svc.fit(y_raw)
    y_labels = label_svc.inverse_transform(label_svc.transform(y_raw))

    n_samples = len(X_raw)
    if n_samples >= 10:
        indices = np.arange(n_samples)
        rng = np.random.default_rng(config.get("random_state", 42))
        rng.shuffle(indices)

        split_idx = int(0.9 * n_samples)
        train_idx, val_idx = indices[:split_idx], indices[split_idx:]

        X_train_local = X_raw[train_idx]
        X_val_local = X_raw[val_idx]
        y_train_local = y_labels[train_idx]
        y_val_local = y_labels[val_idx]
    else:
        X_train_local = X_raw
        X_val_local = X_raw
        y_train_local = y_labels
        y_val_local = y_labels

    pf = client_flex_model.get("model")

    if pf is None:
        # First round: instantiate and fit local model
        pf = ProactiveForest(
            n_estimators=n_estimators,
            alpha=alpha,
            verbose=config.get("verbose", False),
            class_names=class_names,
            convergence_threshold=local_convergence_threshold,
            random_state=config.get("random_state", 42),
        )
        pf.fit(
            X_train_local,
            y_train_local,
            X_val=X_val_local,
            y_val=y_val_local,
        )
        new_trees = pf.get_trees()
    else:
        # Subsequent rounds: add episode without resetting trees
        if not hasattr(pf._classifier, "_encoder_dict"):
            pf._classifier.classes_ = np.array(class_names)
            pf._classifier._encoder_dict = {
                val: idx for idx, val in enumerate(class_names)
            }
            pf._classifier._decoder_dict = {
                idx: val for idx, val in enumerate(class_names)
            }
        pf._classifier._n_classes = len(pf._classifier.classes_)

        n_prev = len(pf.get_trees())
        pf._classifier._n_estimators = n_prev + n_estimators

        pf._classifier.buildEpisode(
            X_train_local,
            y_train_local,
            X_val_local,
            y_val_local,
            EPISODE=n_estimators,
            verbose=pf.verbose,
        )

        pf._is_fitted = True
        pf.estimators_ = pf.get_trees()

        new_trees = pf.get_trees()[-n_estimators:]

    meta = ClientMetadata(client_id=actor_id, n_trees=len(new_trees))

    client_flex_model.update(
        {
            "model": pf,
            "trees": new_trees,
            "metadata": meta,
            "X_train_local": X_train_local,
            "y_train_local": y_train_local,
            "X_val_local": X_val_local,
            "y_val_local": y_val_local,
        }
    )

    return client_flex_model


@collect_clients_weights
def collect_clients_trees_pf(
    client_flex_model: FlexModel, *args: Any, **kwargs: Any
) -> Dict[str, Any]:
    """Collect trees and metadata from a single client.

    FLEX primitive function decorated with collect_clients_weights.

    Args:
        client_flex_model (FlexModel): Client model state.
        *args (Any): Variable args.
        **kwargs (Any): Keyword args.

    Returns:
        Dict[str, Any]: Mapping of client trees, id and metadata.
    """
    return {
        "client_id": getattr(client_flex_model, "actor_id", "unknown"),
        "trees": client_flex_model.get("trees", []),
        "metadata": client_flex_model.get("metadata", {}),
    }
