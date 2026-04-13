import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.infrastructure.metrics.diversity_service import PredictionBasedDiversityService

def test_pcd():
    service = PredictionBasedDiversityService()
    
    # Case 1: Identical predictions
    preds = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0]
    ])
    pcd = service.calculate_pcd(preds)
    print(f"PCD Identical: {pcd} (Expected: 0.0)")
    
    # Case 2: Total disagreement
    preds = np.array([
        [0, 1],
        [1, 0]
    ])
    pcd = service.calculate_pcd(preds)
    print(f"PCD Total Disagreement: {pcd} (Expected: 1.0)")
    
    # Case 3: Mixed
    preds = np.array([
        [0, 0, 1], # pair (0,1): agree, (0,2): disagree, (1,2): disagree
        [1, 1, 0]  # pair (0,1): agree, (0,2): disagree, (1,2): disagree
    ])
    pcd = service.calculate_pcd(preds)
    print(f"PCD Mixed: {pcd:.4f} (Expected: ~0.6667)")

if __name__ == "__main__":
    test_pcd()
