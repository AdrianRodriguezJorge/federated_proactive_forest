"""Prediction consolidation and voting utilities.

Provides high-performance mode calculations for multi-estimator voting arrays,
supporting fast vectorized integer paths and string category mappings.
"""

from typing import List
import numpy as np
import pandas as pd


def calculate_mode(predictions: np.ndarray, axis: int = 1) -> np.ndarray:
    """Pure Python/Numpy implementation of majority voting (mode).

    Replaces scipy.stats.mode to avoid domain layer leakage. Supports both
    integer and object/string predictions efficiently.

    Args:
        predictions (np.ndarray): Array of shape (n_samples, n_predictors).
        axis (int): Axis along which to calculate the mode (default=1).

    Returns:
        np.ndarray: Array of shape (n_samples,) with the mode prediction.

    Raises:
        NotImplementedError: If axis != 1.
    """
    if axis != 1:
        raise NotImplementedError(
            "Only axis=1 is currently supported for calculate_mode"
        )

    n_samples = predictions.shape[0]

    # --- OPTIMIZATION: FAST PATH FOR INTEGERS ---
    if np.issubdtype(predictions.dtype, np.integer):
        try:
            max_val = np.max(predictions)
            if max_val < 1000:  # Safe assumption for classification
                counts = np.zeros((n_samples, max_val + 1), dtype=int)
                # Loop over predictors instead of samples to optimize speed
                for j in range(predictions.shape[1]):
                    counts[np.arange(n_samples), predictions[:, j]] += 1
                return np.argmax(counts, axis=1)
        except Exception:
            pass  # Fallback if any issue occurs

    # --- FAST PATH FOR OBJECTS/STRINGS ---
    modes = pd.DataFrame(predictions).mode(axis=1).iloc[:, 0].values

    # Attempt to cast away from 'object' dtype to help downstream libraries
    if modes.dtype == object:
        try:
            return np.array(modes.tolist())
        except Exception:
            return modes
    return modes
