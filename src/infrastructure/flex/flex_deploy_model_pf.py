"""FLEX Server deployment primitives for Proactive Forest.

Supports distributing configuration and global model state to clients
via standard FLEX decorators.
"""

from copy import deepcopy
from typing import Any, Dict

from flex.model import FlexModel
from flex.pool.decorators import deploy_server_model


@deploy_server_model
def deploy_server_config_pf(
    server_flex_model: FlexModel, *args: Any, **kwargs: Any
) -> Dict[str, Any]:
    """Deploy server configuration to client.

    FLEX Primitive for deploy_server_model.

    Args:
        server_flex_model (FlexModel): Server model state wrapper.
        *args (Any): Variable args.
        **kwargs (Any): Keyword args.

    Returns:
        Dict[str, Any]: Configuration dictionary to deploy.
    """
    return {"config": deepcopy(server_flex_model.get("config", {}))}


@deploy_server_model
def deploy_server_model_pf(
    server_flex_model: FlexModel, *args: Any, **kwargs: Any
) -> Dict[str, Any]:
    """Deploy global server model to client.

    FLEX Primitive for deploy_server_model.

    Args:
        server_flex_model (FlexModel): Server model state wrapper.
        *args (Any): Variable args.
        **kwargs (Any): Keyword args.

    Returns:
        Dict[str, Any]: Ensemble model state to deploy.
    """
    return {
        "global_model": deepcopy(server_flex_model.get("model")),
        "global_trees": deepcopy(server_flex_model.get("trees", [])),
    }


__all__ = ["deploy_server_config_pf", "deploy_server_model_pf"]