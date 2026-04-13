from typing import List, Any
from collections import Counter
import numpy as np

def calculate_mode(predictions: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Pure Python/Numpy implementation of majority voting (mode).
    Replaces scipy.stats.mode to avoid domain layer leakage.
    
    Args:
        predictions: Array of shape (n_samples, n_predictors)
        axis: The axis along which to calculate the mode (default=1, across predictors)
        
    Returns:
        Array of shape (n_samples,) with the most frequent value per sample.
    """
    if axis != 1:
        raise NotImplementedError("Only axis=1 is currently supported for calculate_mode")
    
    n_samples = predictions.shape[0]
    modes = np.zeros(n_samples, dtype=predictions.dtype)
    
    for i in range(n_samples):
        # Count frequencies of each prediction for this sample
        counts = Counter(predictions[i])
        # Get the most common element (first one in case of ties)
        modes[i] = counts.most_common(1)[0][0]
        
    return modes
