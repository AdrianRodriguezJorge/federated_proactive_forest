import numpy as np
from src.domain.metrics.metrics_service import IDiversityService

class PredictionBasedDiversityService(IDiversityService):
    """
    Implementation of IDiversityService using Percentage Correct Diversity (PCD).
    This version is supervised (uses y_true).
    """

    def calculate_pcd(self, predictions_matrix: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculate Percentage Correct Diversity (PCD).
        """
        n_samples, n_classifiers = predictions_matrix.shape
        if n_classifiers == 0:
            return 0.0

        y_true_reshaped = y_true.reshape(-1, 1)
        hits_matrix = (predictions_matrix == y_true_reshaped)
        hits_per_sample = np.sum(hits_matrix, axis=1)
        
        return self.calculate_pcd_from_counts(hits_per_sample, n_classifiers)

    def calculate_marginal_pcd(self, candidate_predictions: np.ndarray, 
                               current_hits_per_sample: np.ndarray,
                               n_existing_trees: int,
                               y_true: np.ndarray) -> float:
        """
        Calculates the PCD if we added candidate_predictions to current ensemble.
        """
        # New hit (bool)
        new_hit = (candidate_predictions == y_true).astype(int)
        
        # New counts
        new_hits_per_sample = current_hits_per_sample + new_hit
        new_n_trees = n_existing_trees + 1
        
        return self.calculate_pcd_from_counts(new_hits_per_sample, new_n_trees)

    def calculate_pcd_from_counts(self, hits_per_sample: np.ndarray, n_trees: int) -> float:
        """Helper to calculate PCD from hit counts."""
        if n_trees == 0: return 0.0
        n_samples = hits_per_sample.shape[0]
        if n_samples == 0: return 0.0
        
        lower = 0.1 * n_trees
        upper = 0.9 * n_trees
        diverse = np.sum((hits_per_sample >= lower) & (hits_per_sample <= upper))
        return float(diverse / n_samples)

