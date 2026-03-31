"""FLEX update primitives for Proactive Forest."""
from typing import Dict, Any


def update_client_with_global_pf(client_flex_model: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update client with global model after aggregation.
    This is typically handled by deploy_server_model_pf, but can be used
    if clients need additional processing/fine-tuning with global model.
    
    FLEX Primitive (optional, called instead of deploy_server_model_pf if custom logic needed).
    """
    # This is typically a no-op since deploy_server_model_pf already handles it
    # But can be extended for client-side fine-tuning, regularization, etc.
    return client_flex_model


def merge_local_global_forests_pf(client_flex_model: Dict[str, Any],
                                   local_weight: float = 0.5,
                                   global_weight: float = 0.5) -> Dict[str, Any]:
    """
    Merge local and global forests for hybrid prediction.
    Optional utility for clients that want to combine local and global models.
    
    Args:
        client_flex_model: Client model with both 'model' (local) and 'global_model'
        local_weight: Weight for local forest predictions
        global_weight: Weight for global forest predictions
    
    Returns:
        Updated client_flex_model with merged weights
    """
    client_flex_model['local_weight'] = local_weight
    client_flex_model['global_weight'] = global_weight
    return client_flex_model


__all__ = ['update_client_with_global_pf', 'merge_local_global_forests_pf']