import pytest
import numpy as np
from src.domain.metadata.client_metadata import ClientMetadata
from src.domain.aggregation.tree_ranker import TreeRanker, TreeEntry, RankingCriterion
from src.domain.aggregation.services.tree_metric_extractor import TreeMetricExtractor
from src.domain.metrics.metrics_service import IDiversityService

class DummyTree:
    def __init__(self, predictions):
        self.predictions = predictions
    def predict(self, X):
        return self.predictions


def test_client_metadata_serialization():
    """Verify ClientMetadata field initialization and dict serialization."""
    meta = ClientMetadata(
        client_id="client_test",
        n_trees=5,
        has_converged=True
    )
    d = meta.to_dict()
    assert d["client_id"] == "client_test"
    assert d["n_trees"] == 5
    assert d["has_converged"] is True

    # Test from_dict compat and filtering of legacy fields
    data_with_legacy = {
        "client_id": "client_test",
        "n_trees": 5,
        "accuracy": 0.85,
        "macro_f1": 0.82,
        "pcd": 0.74,
        "has_converged": True,
        "stop_counter": 1,
        "prev_episode_acc": 0.1
    }
    meta_loaded = ClientMetadata.from_dict(data_with_legacy)
    assert meta_loaded.client_id == "client_test"
    assert meta_loaded.n_trees == 5
    assert meta_loaded.has_converged is True
    assert meta_loaded.stop_counter == 1
    assert meta_loaded.prev_episode_acc == 0.1
    assert not hasattr(meta_loaded, 'accuracy')


def test_tree_ranker_scoring_and_sorting():
    """Verify that TreeRanker scores and sorts candidate entries for all criteria."""
    entry1 = TreeEntry(tree="tree1", client_id="c0", tree_local_id=0, accuracy=0.8, macro_f1=0.7, pcd=0.6)
    entry2 = TreeEntry(tree="tree2", client_id="c0", tree_local_id=1, accuracy=0.7, macro_f1=0.9, pcd=0.5)
    entry3 = TreeEntry(tree="tree3", client_id="c1", tree_local_id=0, accuracy=0.9, macro_f1=0.6, pcd=0.8)
    
    entries = [entry1, entry2, entry3]
    
    # 1. Rank by ACCURACY
    ranker_acc = TreeRanker(RankingCriterion.ACCURACY)
    sorted_acc = ranker_acc.rank(entries.copy())
    assert sorted_acc[0].tree == "tree3"  # acc 0.9
    assert sorted_acc[1].tree == "tree1"  # acc 0.8
    assert sorted_acc[2].tree == "tree2"  # acc 0.7
    
    # 2. Rank by MACRO_F1
    ranker_f1 = TreeRanker(RankingCriterion.MACRO_F1)
    sorted_f1 = ranker_f1.rank(entries.copy())
    assert sorted_f1[0].tree == "tree2"  # f1 0.9
    assert sorted_f1[1].tree == "tree1"  # f1 0.7
    assert sorted_f1[2].tree == "tree3"  # f1 0.6
    
    # 3. Rank by F1_PCD (weighted scoring)
    # score = 0.7 * macro_f1 + 0.3 * pcd
    # entry1: 0.7 * 0.7 + 0.3 * 0.6 = 0.67
    # entry2: 0.7 * 0.9 + 0.3 * 0.5 = 0.78
    # entry3: 0.7 * 0.6 + 0.3 * 0.8 = 0.66
    ranker_weighted = TreeRanker(RankingCriterion.F1_PCD, f1_weight=0.7, pcd_weight=0.3)
    sorted_weighted = ranker_weighted.rank(entries.copy())
    assert sorted_weighted[0].tree == "tree2"  # score 0.78
    assert sorted_weighted[1].tree == "tree1"  # score 0.67
    assert sorted_weighted[2].tree == "tree3"  # score 0.66


def test_tree_metric_extractor_defaults():
    """Verify that TreeMetricExtractor defaults to 0.0 when validation data is missing."""
    client_trees = {
        "client_0": ["tree_0", "tree_1"]
    }
    client_metadata = {
        "client_0": ClientMetadata(
            client_id="client_0"
        )
    }
    
    extractor = TreeMetricExtractor()
    raw_entries = extractor.extract_metrics(client_trees, client_metadata)
    assert len(raw_entries) == 2
    assert raw_entries[0]["accuracy"] == 0.0
    assert raw_entries[0]["macro_f1"] == 0.0
    assert raw_entries[0]["pcd"] == 0.0


class MockDiversityService(IDiversityService):
    def calculate_pcd(self, predictions_matrix, y_true):
        return 0.5
    def calculate_pcd_samples(self, predictions_matrix, y_true):
        return np.ones(y_true.shape[0])
    def calculate_marginal_pcd(self, candidate_predictions, current_hits_per_sample, n_existing_trees, y_true):
        return 0.99


def test_tree_metric_extractor_marginal_pcd():
    """Verify Marginal PCD calculation logic based on sample correctness overlaps."""
    y_val = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # 10 samples
    X_val = np.zeros((10, 2))
    
    # tree_0 predicts y_val completely correct
    tree_0 = DummyTree(y_val.copy())
    # tree_1 predicts only 5 samples correct (samples 0-4 are correct, samples 5-9 are wrong)
    tree_1 = DummyTree(np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0]))
    
    client_trees = {
        "client_0": [tree_0, tree_1]
    }
    client_metadata = {
        "client_0": ClientMetadata(
            client_id="client_0"
        )
    }
    
    # Diverse sample threshold: hits must be between 10% (0.2) and 90% (1.8) of n_trees (2)
    # Samples 0-4: both correct -> hits = 2 (not diverse, >= 1.8)
    # Samples 5-9: only tree_0 correct -> hits = 1 (diverse, fits between 0.2 and 1.8)
    # Diverse sample ratio: 5 / 10 = 0.5
    
    extractor = TreeMetricExtractor(diversity_service=MockDiversityService())
    raw_entries = extractor.extract_metrics(client_trees, client_metadata, X_val=X_val, y_val=y_val)
    
    assert raw_entries[0]["pcd"] == 0.3
    assert raw_entries[1]["pcd"] == 0.3


def test_tree_metric_extractor_direct_evaluation():
    """Verify that when X_val and y_val are provided, individual trees are evaluated directly."""
    # 10 samples
    y_val = np.array(["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"])
    X_val = np.zeros((10, 2))
    
    # tree_0 makes 100% correct predictions
    tree_0 = DummyTree(y_val.copy())
    # tree_1 makes 60% correct predictions (first 6 correct, last 4 incorrect as "A")
    tree_1 = DummyTree(np.array(["A", "B", "A", "B", "A", "B", "A", "A", "A", "A"]))
    
    client_trees = {
        "client_0": [tree_0, tree_1]
    }
    client_metadata = {
        "client_0": ClientMetadata(
            client_id="client_0"
        )
    }
    
    extractor = TreeMetricExtractor()
    raw_entries = extractor.extract_metrics(client_trees, client_metadata, X_val=X_val, y_val=y_val)
    
    assert len(raw_entries) == 2
    
    # tree_0 must be evaluated directly on the server's global validation set: accuracy 1.0, f1 1.0
    assert raw_entries[0]["tree"] == tree_0
    assert raw_entries[0]["accuracy"] == 1.0
    assert raw_entries[0]["macro_f1"] == 1.0
    
    # tree_1 must be evaluated directly on the server's global validation set: accuracy 0.6
    # predictions: A B A B A B A A A A
    # true labels:  A B A B A B A B A B
    # matches:      1 1 1 1 1 1 1 0 1 0 -> 8 matches! Wait:
    # index 0: A == A (True)
    # index 1: B == B (True)
    # index 2: A == A (True)
    # index 3: B == B (True)
    # index 4: A == A (True)
    # index 5: B == B (True)
    # index 6: A == A (True)
    # index 7: A == B (False)
    # index 8: A == A (True)
    # index 9: A == B (False)
    # Total matches: 8 / 10 = 0.8
    assert raw_entries[1]["tree"] == tree_1
    assert raw_entries[1]["accuracy"] == 0.8
    assert raw_entries[1]["macro_f1"] > 0.7  # Macro-F1 should be computed correctly using sklearn

