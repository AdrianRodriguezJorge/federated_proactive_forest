"""FLEX deployment primitives for Proactive Forest."""
from copy import deepcopy
from typing import Dict, Any


def deploy_server_config_pf(server_flex_model: Dict[str, Any], 
                             client_flex_model: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deploy server configuration to client.
    Sends training/model parameters from server to clients.
    
    FLEX Primitive (called once at start).
    """
    # Copy configuration from server to client
    client_flex_model['config'] = deepcopy(server_flex_model.get('config', {}))
    return client_flex_model


def deploy_server_model_pf(server_flex_model: Dict[str, Any], 
                            client_flex_model: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deploy global server model to client.
    Sends the aggregated global forest to clients for hybrid prediction/fine-tuning.
    
    FLEX Primitive (called after aggregation each round).
    """
    client_flex_model['global_model'] = deepcopy(server_flex_model.get('model'))
    client_flex_model['global_trees'] = deepcopy(server_flex_model.get('trees', []))
    return client_flex_model


__all__ = ['deploy_server_config_pf', 'deploy_server_model_pf']