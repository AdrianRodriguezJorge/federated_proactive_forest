import pytest
import numpy as np
from src.domain.model.progressive_forest import ComparativeProgressiveForest, ProgressiveForest
from src.domain.model.proactive_forest import ProactiveForest

class MockProactiveForestClassifier:
    """Mock to simulate ProactiveForest behavior for isolated testing of ComparativeProgressiveForest."""
    def __init__(self, n_estimators=100, accuracies_to_simulate=None):
        self.n_estimators = n_estimators
        self._trees = []
        self._m_progressive_accuracy = []
        self.accuracies_to_simulate = accuracies_to_simulate or []
        self.build_calls = 0
    
    def clean_trees(self):
        self._trees = []
        self._m_progressive_accuracy = []
        self.build_calls = 0
        
    def buildEpisode(self, X, y, Xt, yt, episode_size, verbose=False):
        for _ in range(episode_size):
            if len(self._trees) < self.n_estimators:
                self._trees.append("dummy_tree")
                idx = len(self._m_progressive_accuracy)
                acc = (self.accuracies_to_simulate[idx] 
                       if idx < len(self.accuracies_to_simulate) 
                       else 0.8)
                self._m_progressive_accuracy.append(acc)
        self.build_calls += 1


def test_cpf_early_stopping_convergence():
    """Verify that ComparativeProgressiveForest stops early and truncates the forest correctly."""
    # Simulation:
    # Episode 1 (size 5):
    #   Accuracies: [0.8000, 0.8005, 0.8010, 0.8005, 0.8002] -> spread: 0.8010 - 0.8000 = 0.0010 <= 0.002
    #   This meets convergence condition (stop_counter = 1)
    # Episode 2 (size 6):
    #   Accuracies: [0.8001, 0.8002, 0.8003, 0.8002, 0.8001, 0.8000] -> spread remains 0.0010 <= 0.002
    #   Stop counter reaches 2 -> early stopping breaks!
    # Truncation:
    #   models_built = 11, new_trees = 6.
    #   best_idx_in_episode (across all 11 items) is index 2 (value 0.8010).
    #   trees_to_keep = (11 - 6) + (2 + 1) = 8 trees.
    
    simulated_accs = [
        # Episode 1 (5 items)
        0.8000, 0.8005, 0.8010, 0.8005, 0.8002,
        # Episode 2 (6 items)
        0.8001, 0.8002, 0.8003, 0.8002, 0.8001, 0.8000
    ]
    
    mock_clf = MockProactiveForestClassifier(n_estimators=100, accuracies_to_simulate=simulated_accs)
    cpf = ComparativeProgressiveForest(mock_clf, verbose=True, convergence_threshold=0.002)
    
    # Train mock
    cpf.fit(np.zeros((10, 2)), np.zeros(10), np.zeros((10, 2)), np.zeros(10))
    
    # Check stopping point and truncation
    assert mock_clf.build_calls == 2
    assert len(mock_clf._trees) == 8


def test_progressive_forest_integration():
    """Verify that ProgressiveForest integration runs successfully with ProactiveForest."""
    X_train = np.random.rand(30, 2)
    y_train = np.random.randint(0, 2, 30)
    X_val = np.random.rand(15, 2)
    y_val = np.random.randint(0, 2, 15)
    
    forest = ProactiveForest(n_estimators=20, alpha=0.1, class_names=["0", "1"])
    pf = ProgressiveForest(forest, verbose=True, convergence_threshold=0.5) # High threshold to trigger early stopping
    
    pf.fit_with_early_stopping(X_train, y_train, X_val, y_val)
    
    fitted_forest = pf.get_forest()
    assert fitted_forest is not None
    assert pf.forest_tree_size() > 0
    assert pf.forest_tree_size() < 20  # Truncated by early stopping
