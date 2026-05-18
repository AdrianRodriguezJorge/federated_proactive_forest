"""Roulette State Updater for S9 Federated Strategy.

Applies the β-fusion formula to blend a client's local roulette with the
server's global roulette:

    Roulette_new = β · Roulette_local + (1 − β) · Roulette_global

When β = 0 (default), the client fully adopts the global roulette, ensuring
strong federation-wide coordination. When β = 1, the client ignores the global
signal entirely.
"""

from typing import Optional
import numpy as np


class RouletteUpdater:
    """Fuses a local feature-probability vector with a global one."""

    def __init__(self, beta: float = 0.0):
        """Initializes RouletteUpdater.

        Args:
            beta (float): Blending coefficient in [0, 1].
                0 -> full global adoption.
                1 -> keep local roulette unchanged.

        Raises:
            ValueError: If beta is not in [0, 1].
        """
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        self.beta = beta

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

        fused = self.beta * local + (1.0 - self.beta) * glob

        # Re-normalise to guarantee a valid probability distribution
        total = np.sum(fused)
        if total > 0:
            fused /= total
        else:
            # Extreme edge case: fall back to uniform
            fused = np.ones_like(fused) / len(fused)

        return fused


__all__ = ["RouletteUpdater"]
