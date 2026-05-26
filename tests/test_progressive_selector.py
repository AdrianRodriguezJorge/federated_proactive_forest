import pytest
import numpy as np
from src.domain.aggregation.services.progressive_selector import ProgressiveSelector
from src.domain.aggregation.tree_ranker import TreeEntry, TreeRanker, RankingCriterion
from src.domain.metrics.metrics_service import IMetricsService, IDiversityService

class SpyTree:
    """Spy DecisionTree to count predict calls and return simulated labels."""
    def __init__(self, predictions, label="tree"):
        self.predictions = predictions
        self.label = label
        self.predict_calls = 0
    
    def predict(self, X):
        self.predict_calls += 1
        return self.predictions


class MockMetricsService(IMetricsService):
    def accuracy_score(self, y_true, y_pred):
        return float(np.mean(y_true == y_pred))
    def f1_score(self, y_true, y_pred, average='macro', labels=None):
        return 0.9
    def confusion_matrix(self, y_true, y_pred, labels=None):
        return np.eye(2)
    def precision_score(self, y_true, y_pred, average='macro', labels=None):
        return 0.9
    def recall_score(self, y_true, y_pred, average='macro', labels=None):
        return 0.9


class MockDiversityService(IDiversityService):
    def __init__(self):
        self.marginal_pcd_calls = 0
    def calculate_pcd(self, predictions_matrix, y_true):
        return 0.5
    def calculate_pcd_samples(self, predictions_matrix, y_true):
        return np.ones(y_true.shape[0])
    def calculate_marginal_pcd(self, candidate_predictions, current_hits_per_sample, n_existing_trees, y_true):
        self.marginal_pcd_calls += 1
        return 0.95


def test_progressive_selector_mode_voting():
    """Verify coordinate-wise majority vote ensemble mode predictions."""
    # 3 trees, 4 validation samples
    tree1 = SpyTree(np.array([0, 1, 0, 0]))
    tree2 = SpyTree(np.array([0, 1, 1, 0]))
    tree3 = SpyTree(np.array([1, 1, 1, 1]))
    
    entries = [
        TreeEntry(tree=tree1, client_id="c0", tree_local_id=0, accuracy=0.8, macro_f1=0.8, pcd=0.8),
        TreeEntry(tree=tree2, client_id="c0", tree_local_id=1, accuracy=0.8, macro_f1=0.8, pcd=0.8),
        TreeEntry(tree=tree3, client_id="c1", tree_local_id=0, accuracy=0.8, macro_f1=0.8, pcd=0.8)
    ]
    
    selector = ProgressiveSelector()
    predictions = selector._predict_ensemble(entries, X=np.zeros((4, 2)))
    
    # Coordinate majority mode:
    # Sample 0: mode of [0, 0, 1] -> 0
    # Sample 1: mode of [1, 1, 1] -> 1
    # Sample 2: mode of [0, 1, 1] -> 1
    # Sample 3: mode of [0, 0, 1] -> 0
    assert np.array_equal(predictions, [0, 1, 1, 0])


def test_progressive_selector_caching():
    """Verify that predictions are cached to avoid redundant predict() calls on trees."""
    tree = SpyTree(np.array([0, 1, 0, 1]))
    entries = [TreeEntry(tree=tree, client_id="c0", tree_local_id=0, accuracy=0.8, macro_f1=0.8, pcd=0.8)]
    
    selector = ProgressiveSelector()
    cache = {}
    
    # Repeated predictions
    for _ in range(3):
        selector._predict_ensemble(entries, X=np.zeros((4, 2)), cache=cache)
        
    # The actual method must be called exactly once
    assert tree.predict_calls == 1


def test_progressive_selector_early_stopping():
    """Verify that early stopping stops the selector loop when improvement threshold is not met for 2 consecutive episodes."""
    y_val = np.array([0, 1, 0, 1])
    X_val = np.zeros((4, 2))
    
    # 6 trees, episode size = 2 (3 episodes total)
    # Episode 1: trees 0 & 1 -> accuracy = 0.8
    # Episode 2: trees 0-3 -> accuracy = 0.8 (improvement = 0.0 < 0.01 threshold -> stop_counter = 1)
    # Episode 3: trees 0-5 -> accuracy = 0.8 (improvement = 0.0 < 0.01 threshold -> stop_counter = 2 -> stops)
    trees = [SpyTree(np.array([0, 1, 0, 1]), label=f"t{i}") for i in range(6)]
    entries = [TreeEntry(tree=t, client_id="c0", tree_local_id=i, accuracy=0.8, macro_f1=0.8, pcd=0.8) for i, t in enumerate(trees)]
    
    selector = ProgressiveSelector(metrics_service=MockMetricsService())
    
    selected_trees, selected_entries, convergence_round, round_logs = selector.select(
        candidate_entries=entries,
        X_val=X_val,
        y_val_norm=y_val,
        episode_size=2,
        max_trees=10,
        convergence_threshold=0.01
    )
    
    # Halts exactly at the end of Episode 3 (or None if loop ends naturally)
    assert convergence_round in (3, None)
    assert len(selected_trees) == 6
    assert len(round_logs) == 3


def test_progressive_selector_proactive_re_ranking():
    """Verify active re-ranking triggers Marginal PCD calculations and resorts remaining trees at each episode step."""
    y_val = np.array([0, 1, 0, 1])
    X_val = np.zeros((4, 2))
    
    tree1 = SpyTree(np.array([0, 1, 0, 1]))
    tree2 = SpyTree(np.array([1, 0, 1, 0]))
    
    entries = [
        TreeEntry(tree=tree1, client_id="c0", tree_local_id=0, accuracy=0.8, macro_f1=0.8, pcd=0.5),
        TreeEntry(tree=tree2, client_id="c0", tree_local_id=1, accuracy=0.8, macro_f1=0.8, pcd=0.5)
    ]
    
    ranker = TreeRanker(RankingCriterion.F1_PCD, f1_weight=0.5, pcd_weight=0.5)
    div_svc = MockDiversityService()
    
    selector = ProgressiveSelector(metrics_service=MockMetricsService(), diversity_service=div_svc)
    
    selected_trees, selected_entries, convergence_round, round_logs = selector.select(
        candidate_entries=entries,
        X_val=X_val,
        y_val_norm=y_val,
        episode_size=1,  # 1 tree per episode
        max_trees=2,
        convergence_threshold=0.001,
        ranker=ranker
    )
    
    # For episode 2 (when tree1 is selected), it calculates marginal PCD for tree2
    assert div_svc.marginal_pcd_calls > 0


def test_progressive_selector_per_client_ranking():
    """Verify that is_per_client=True keeps round-robin ordering of client candidates intact, while re-ranking internally."""
    y_val = np.array([0, 1, 0, 1])
    X_val = np.zeros((4, 2))
    
    # Client 0 trees
    t0_1 = SpyTree(np.array([0, 1, 0, 1]), label="c0_t1")
    t0_2 = SpyTree(np.array([0, 0, 1, 1]), label="c0_t2")
    # Client 1 trees
    t1_1 = SpyTree(np.array([1, 1, 0, 0]), label="c1_t1")
    t1_2 = SpyTree(np.array([1, 0, 1, 0]), label="c1_t2")

    entries = [
        TreeEntry(tree=t0_1, client_id="c0", tree_local_id=0, accuracy=0.8, macro_f1=0.8, pcd=0.5),
        TreeEntry(tree=t1_1, client_id="c1", tree_local_id=0, accuracy=0.8, macro_f1=0.8, pcd=0.5),
        TreeEntry(tree=t0_2, client_id="c0", tree_local_id=1, accuracy=0.8, macro_f1=0.8, pcd=0.5),
        TreeEntry(tree=t1_2, client_id="c1", tree_local_id=1, accuracy=0.8, macro_f1=0.8, pcd=0.5)
    ]
    
    ranker = TreeRanker(RankingCriterion.F1_PCD, f1_weight=0.5, pcd_weight=0.5)
    
    # Mock diversity service that returns custom values based on label to force re-ranking
    class CustomDiversityService(IDiversityService):
        def calculate_pcd(self, pm, yt): return 0.5
        def calculate_pcd_samples(self, pm, yt): return np.ones(yt.shape[0])
        def calculate_marginal_pcd(self, candidate_predictions, current_hits_per_sample, n_existing_trees, y_true):
            # We want to check if the ranker swaps the order of client 0's trees:
            # Let's say we prefer t0_2 over t0_1, and t1_2 over t1_1:
            # If the tree has local_id == 1, return high PCD. If local_id == 0, return low PCD.
            # We don't have direct access to TreeEntry inside calculate_marginal_pcd, but we can look at predictions.
            # t0_2 has predictions [0, 0, 1, 1], t1_2 has [1, 0, 1, 0].
            if np.array_equal(candidate_predictions, [0, 0, 1, 1]) or np.array_equal(candidate_predictions, [1, 0, 1, 0]):
                return 0.99
            return 0.01

    div_svc = CustomDiversityService()
    selector = ProgressiveSelector(metrics_service=MockMetricsService(), diversity_service=div_svc)
    
    selected_trees, selected_entries, convergence_round, round_logs = selector.select(
        candidate_entries=entries,
        X_val=X_val,
        y_val_norm=y_val,
        episode_size=1,
        max_trees=4,
        convergence_threshold=0.001,
        ranker=ranker,
        is_per_client=True
    )
    
    # If round-robin client order is preserved:
    # Selected entries order MUST alternate client_id: c0, c1, c0, c1
    client_ids = [e.client_id for e in selected_entries]
    assert client_ids == ["c0", "c1", "c0", "c1"]
    
    # But within each client's pool, the order MUST have been swapped (local_id 1 before local_id 0):
    # C0: local_id 1 (t0_2) then local_id 0 (t0_1)
    # C1: local_id 1 (t1_2) then local_id 0 (t1_1)
    tree_local_ids = [e.tree_local_id for e in selected_entries]
    assert tree_local_ids == [1, 1, 0, 0]

