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
    # --- OPTIMIZATION: FAST PATH FOR INTEGERS ---
    if np.issubdtype(predictions.dtype, np.integer):
        try:
            max_val = np.max(predictions)
            if max_val < 1000: # Safe assumption for classification
                counts = np.zeros((n_samples, max_val + 1), dtype=int)
                # Loop over predictors (e.g. 20) instead of samples (e.g. 13000)
                for j in range(predictions.shape[1]):
                    counts[np.arange(n_samples), predictions[:, j]] += 1
                return np.argmax(counts, axis=1)
        except Exception:
            pass # Fallback if any issue occurs

    # --- FAST PATH FOR OBJECTS/STRINGS ---
    import pandas as pd
    modes = pd.DataFrame(predictions).mode(axis=1).iloc[:, 0].values
        
    # Attempt to cast away from 'object' dtype to help downstream libraries (like sklearn)
    if modes.dtype == object:
        try:
            return np.array(modes.tolist())
        except:
            return modes
    return modes
