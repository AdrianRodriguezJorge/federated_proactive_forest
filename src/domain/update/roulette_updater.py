"""Roulette State Updater for S8 Federated Strategy.

Applies the fusion formula to blend a client's local roulette with the
server's global roulette:

    Roulette_new = local_roulette_weight · Roulette_local + (1 − local_roulette_weight) · Roulette_global

When local_roulette_weight = 0 (default), the client fully adopts the global roulette, ensuring
strong federation-wide coordination. When local_roulette_weight = 1, the client ignores the global
signal entirely.
"""

from typing import Optional
import numpy as np


class RouletteUpdater:
    """Fuses a local feature-probability vector with a global one."""

    def __init__(self, local_roulette_weight: float = 0.1):
        """Initializes RouletteUpdater.

        Args:
            local_roulette_weight (float): Blending coefficient in [0, 1].
                0 -> full global adoption.
                1 -> keep local roulette unchanged.

        Raises:
            ValueError: If local_roulette_weight is not in [0, 1].
        """
        if not 0.0 <= local_roulette_weight <= 1.0:
            raise ValueError(f"local_roulette_weight must be in [0, 1], got {local_roulette_weight}")
        self.local_roulette_weight = local_roulette_weight

    def fuse(
        self,
        local_roulette: np.ndarray,
        global_roulette: np.ndarray,
    ) -> np.ndarray:
        """Compute the fused roulette.

        Args:
            local_roulette (np.ndarray): Local probability array.
            global_roulette (np.ndarray): Global probability array.

        Returns:
            np.ndarray: Fused probability vector, normalized.

        Raises:
            ValueError: If vectors have different shapes.
        """
        local = np.asarray(local_roulette, dtype=np.float64)
        glob = np.asarray(global_roulette, dtype=np.float64)

        if local.shape != glob.shape:
            raise ValueError(
                f"Dimension mismatch: local has {local.shape}, "
                f"global has {glob.shape}"
            )

        fused = self.local_roulette_weight * local + (1.0 - self.local_roulette_weight) * glob

        # Re-normalise to guarantee a valid probability distribution
        total = np.sum(fused)
        if total > 0:
            fused /= total
        else:
            # Extreme edge case: fall back to uniform
            fused = np.ones_like(fused) / len(fused)

        return fused


__all__ = ["RouletteUpdater"]
