"""FLEX primitives for the S9 Global Attribute Roulette strategy.

These functions follow the same decorator patterns used by the existing
PF FLEX primitives (flex_train_pf, flex_aggregate_pf, flex_deploy_model_pf)
but operate on probability vectors instead of tree objects.
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional
import numpy as np

from flex.model import FlexModel
from flex.pool.decorators import (
    aggregate_weights,
    collect_clients_weights,
    deploy_server_model,
    set_aggregated_weights,
)
from src.domain.aggregation.strategies.s9_roulette_strategy import (
    create_roulette_strategy,
)


@collect_clients_weights
def collect_client_roulette(
    client_flex_model: FlexModel, *args: Any, **kwargs: Any
) -> Dict[str, Any]:
    """Extract the feature-probability vector and metadata from a client.

    Collects:
      - roulette: the local probability vector (list[float]).
      - n_samples: dataset size (for weighted aggregation).
      - macro_f1: local performance (for consensus aggregation).
      - pcd: local diversity (for proactive aggregation).
      - client_id: actor identifier.

    Args:
        client_flex_model (FlexModel): Client model state wrapper.
        *args (Any): Variable args.
        **kwargs (Any): Keyword args.

    Returns:
        Dict[str, Any]: Client roulette vector and metadata package.
    """
    model = client_flex_model.get("model")
    meta = client_flex_model.get("metadata", {})

    if model is not None and hasattr(model, "get_feature_probabilities"):
        roulette = model.get_feature_probabilities().tolist()
    else:
        roulette = []

    X_train = client_flex_model.get("X_train")
    n_samples = len(X_train) if X_train is not None else 0

    if hasattr(meta, "macro_f1"):
        macro_f1 = meta.macro_f1
    elif isinstance(meta, dict):
        macro_f1 = meta.get("macro_f1", 0.0)
    else:
        macro_f1 = 0.0

    if hasattr(meta, "pcd"):
        pcd = meta.pcd
    elif isinstance(meta, dict):
        pcd = meta.get("pcd", 0.0)
    else:
        pcd = 0.0

    return {
        "client_id": getattr(client_flex_model, "actor_id", "unknown"),
        "roulette": roulette,
        "n_samples": n_samples,
        "macro_f1": macro_f1,
        "pcd": pcd,
    }


@aggregate_weights
def aggregate_roulettes(
    weights: List[Dict[str, Any]], **kwargs: Any
) -> Dict[str, Any]:
    """Aggregate client roulette vectors into a global roulette.

    Args:
        weights (List[Dict[str, Any]]): Collected client weights.
        **kwargs (Any): Strategy configuration parameters.

    Returns:
        Dict[str, Any]: Unified global roulette and byte statistics.

    Raises:
        ValueError: If client returned empty vector or dimension mismatch
            or validation scores are missing under consensus strategies.
    """
    variant = kwargs.get("variant", "S9_MEAN")
    server_eval_f1 = kwargs.get("server_eval_f1", {})
    server_eval_pcd = kwargs.get("server_eval_pcd", {})

    strategy = create_roulette_strategy(variant)

    client_vectors: Dict[str, np.ndarray] = {}
    client_sizes: Dict[str, int] = {}
    client_f1: Dict[str, float] = {}
    client_pcd: Dict[str, float] = {}
    total_upload_bytes = 0

    for w in weights:
        cid = str(w.get("client_id", "unknown"))
        print(f"   [Aggregation] Processing roulette from client {cid}...")
        raw_vec = w.get("roulette", [])

        if not raw_vec:
            raise ValueError(
                f"CRITICAL: Client {cid} returned an empty roulette vector. "
                f"Federation cannot proceed."
            )

        vec = np.array(raw_vec, dtype=np.float64)

        if client_vectors:
            expected_shape = next(iter(client_vectors.values())).shape
            if vec.shape != expected_shape:
                raise ValueError(
                    f"CRITICAL: Dimension mismatch for client {cid}. "
                    f"Expected {expected_shape}, but got {vec.shape}. "
                    f"Check dataset consistency."
                )

        client_vectors[cid] = vec
        client_sizes[cid] = w.get("n_samples", 0)

        # Unbiased server-side evaluation priority
        val_f1 = server_eval_f1.get(cid, server_eval_f1.get(str(cid)))
        if variant == "S9_CONSENSUS" and val_f1 is None:
            raise ValueError(
                f"CRITICAL: Unbiased server-side F1-score evaluation "
                f"is missing for client {cid} under S9_CONSENSUS."
            )
        client_f1[cid] = float(val_f1) if val_f1 is not None else 0.0

        val_pcd = server_eval_pcd.get(cid, server_eval_pcd.get(str(cid)))
        if variant == "S9_PROACTIVE_PCD" and val_pcd is None:
            raise ValueError(
                f"CRITICAL: Unbiased server-side PCD evaluation "
                f"is missing for client {cid} under S9_PROACTIVE_PCD."
            )
        client_pcd[cid] = float(val_pcd) if val_pcd is not None else 0.0

        total_upload_bytes += vec.nbytes

    if not client_vectors:
        return {
            "global_roulette": [],
            "variant": variant,
            "upload_bytes": 0,
            "download_bytes": 0,
            "total_bytes": 0,
        }

    global_roulette = strategy.aggregate_vectors(
        client_vectors=client_vectors,
        client_dataset_sizes=client_sizes,
        client_f1_scores=client_f1,
        client_pcd_scores=client_pcd,
    )

    n_clients = len(client_vectors)
    total_download_bytes = global_roulette.nbytes * n_clients

    return {
        "global_roulette": global_roulette.tolist(),
        "variant": variant,
        "upload_bytes": total_upload_bytes,
        "download_bytes": total_download_bytes,
        "total_bytes": total_upload_bytes + total_download_bytes,
    }


@set_aggregated_weights
def set_global_roulette(
    server_flex_model: FlexModel, aggregated_data: Dict[str, Any], **kwargs: Any
) -> FlexModel:
    """Persist global roulette and communication statistics in server model.

    FLEX Primitive for set_aggregated_weights.

    Args:
        server_flex_model (FlexModel): Server model state wrapper.
        aggregated_data (Dict[str, Any]): Aggregation results mapping.
        **kwargs (Any): Additional options.

    Returns:
        FlexModel: Updated server model state.
    """
    server_flex_model.update(
        {
            "global_roulette": aggregated_data.get("global_roulette", []),
            "roulette_variant": aggregated_data.get("variant", "S9_MEAN"),
            "roulette_upload_bytes": aggregated_data.get("upload_bytes", 0),
            "roulette_download_bytes": aggregated_data.get("download_bytes", 0),
            "roulette_total_bytes": aggregated_data.get("total_bytes", 0),
        }
    )
    return server_flex_model


@deploy_server_model
def deploy_global_roulette(
    server_flex_model: FlexModel, *args: Any, **kwargs: Any
) -> Dict[str, Any]:
    """Send global roulette vector and local_roulette_weight parameter to client.

    FLEX Primitive for deploy_server_model.

    Args:
        server_flex_model (FlexModel): Server model state wrapper.
        *args (Any): Variable args.
        **kwargs (Any): Keyword args.

    Returns:
        Dict[str, Any]: Deployed global roulette and local_roulette_weight coefficient.
    """
    config = server_flex_model.get("config", {})
    local_roulette_weight = config.get("aggregation", {}).get("local_roulette_weight", 0.1)

    return {
        "global_roulette": deepcopy(
            server_flex_model.get("global_roulette", [])
        ),
        "local_roulette_weight": local_roulette_weight,
    }


__all__ = [
    "collect_client_roulette",
    "aggregate_roulettes",
    "set_global_roulette",
    "deploy_global_roulette",
]
