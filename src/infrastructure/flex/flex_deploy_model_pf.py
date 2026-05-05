from copy import deepcopy
from typing import Dict, Any
from flex.model import FlexModel
from flex.pool.decorators import deploy_server_model


@deploy_server_model
def deploy_server_config_pf(server_flex_model: FlexModel, *args, **kwargs: Any) -> Dict[str, Any]:
    """
    Deploy server configuration to client.
    FLEX Primitive for @deploy_server_model.
    """
    return {
        'config': deepcopy(server_flex_model.get('config', {}))
    }


@deploy_server_model
def deploy_server_model_pf(server_flex_model: FlexModel, *args, **kwargs: Any) -> Dict[str, Any]:
    """
    Deploy global server model to client.
    FLEX Primitive for @deploy_server_model.
    """
    return {
        'global_model': deepcopy(server_flex_model.get('model')),
        'global_trees': deepcopy(server_flex_model.get('trees', []))
    }


__all__ = ['deploy_server_config_pf', 'deploy_server_model_pf']