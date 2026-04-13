import numpy as np
from src.domain.metrics.metrics_service import IDiversityService

class PredictionBasedDiversityService(IDiversityService):
    """
    Implementation of IDiversityService using prediction disagreement.
    This version is unsupervised (does not use y_true).
    """

    def calculate_pcd(self, predictions_matrix: np.ndarray) -> float:
        """
        Calculate Pairwise Classifier Disagreement (PCD).
        
        Args:
            predictions_matrix: Matrix of shape (n_samples, n_classifiers)
        
        Returns:
            Mean pairwise disagreement score in range [0, 1].
        """
        n_samples, n_classifiers = predictions_matrix.shape
        if n_classifiers < 2:
            return 0.0

        disagreement_sum = 0.0
        n_pairs = 0

        # Efficiently calculate pairwise disagreements
        # For small n_classifiers, a loop is fine. For very large ones, 
        # we would use a more vectorized approach.
        for i in range(n_classifiers):
            for j in range(i + 1, n_classifiers):
                # Percentage of samples where the two classifiers disagree
                diff = np.mean(predictions_matrix[:, i] != predictions_matrix[:, j])
                disagreement_sum += diff
                n_pairs += 1

        return float(disagreement_sum / n_pairs) if n_pairs > 0 else 0.0
