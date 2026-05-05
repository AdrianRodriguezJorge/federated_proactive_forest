import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from src.domain.services.label_service import SimpleLabelService
from src.domain.aggregation.tree_ranker import TreeRanker, RankingCriterion, TreeEntry
from src.domain.aggregation.services.progressive_selector import ProgressiveSelector
from src.domain.aggregation.strategies.perclient_progressive_base import S5PerClientAccuracyStrategy, S6PerClientF1Strategy
from src.domain.prediction.hybrid_predictor import HybridPredictor
from src.domain.metrics.metrics_service import IMetricsService

# --- Tests for SimpleLabelService ---

class TestSimpleLabelService:
    def test_happy_path(self):
        service = SimpleLabelService(['apple', 'banana', 'cherry'])
        assert list(service.classes) == ['apple', 'banana', 'cherry']
        
        # Transform
        result = service.transform(['cherry', 'apple'])
        np.testing.assert_array_equal(result, [2, 0])
        
        # Inverse transform
        result_inv = service.inverse_transform([1, 0])
        np.testing.assert_array_equal(result_inv, ['banana', 'apple'])

    def test_sticky_protection(self):
        """Verify that numeric input doesn't overwrite existing name-based mapping."""
        service = SimpleLabelService(['apple', 'banana'])
        # Try to fit with numeric labels
        service.fit([0, 1])
        # Should still have apple/banana
        assert list(service.classes) == ['apple', 'banana']
        
    def test_edge_cases(self):
        service = SimpleLabelService(['A', 'B'])
        
        # Out of bounds indices should default to 0
        assert service.transform([99])[0] == 0
        
        # Unknown strings should default to 0
        assert service.transform(['unknown'])[0] == 0
        
        # Empty inputs
        assert len(service.transform([])) == 0
        assert len(service.inverse_transform([])) == 0
        
        # None inputs
        assert len(service.transform(None)) == 0
        assert len(service.inverse_transform(None)) == 0

# --- Tests for TreeRanker ---

class TestTreeRanker:
    @pytest.fixture
    def mock_trees(self):
        return [
            TreeEntry(tree=None, client_id="c1", tree_local_id=0, accuracy=0.8, macro_f1=0.7, pcd=0.5),
            TreeEntry(tree=None, client_id="c1", tree_local_id=1, accuracy=0.9, macro_f1=0.6, pcd=0.4),
            TreeEntry(tree=None, client_id="c2", tree_local_id=0, accuracy=0.7, macro_f1=0.8, pcd=0.6),
        ]

    def test_rank_by_accuracy(self, mock_trees):
        ranker = TreeRanker(criterion=RankingCriterion.ACCURACY)
        ranked = ranker.rank(mock_trees)
        assert ranked[0].accuracy == 0.9
        assert ranked[1].accuracy == 0.8
        assert ranked[2].accuracy == 0.7

    def test_rank_by_f1(self, mock_trees):
        ranker = TreeRanker(criterion=RankingCriterion.MACRO_F1)
        ranked = ranker.rank(mock_trees)
        assert ranked[0].macro_f1 == 0.8
        assert ranked[1].macro_f1 == 0.7
        assert ranked[2].macro_f1 == 0.6

# --- Tests for ProgressiveSelector ---

class TestProgressiveSelector:
    def test_convergence_logic(self):
        # Mock metrics service to simulate improvement then stagnation
        mock_metrics = MagicMock(spec=IMetricsService)
        # Round 1: 0.5, Round 2: 0.6 (improvement), Round 3: 0.601 (no improvement), Round 4: 0.602 (no improvement)
        mock_metrics.accuracy_score.side_effect = [0.5, 0.6, 0.601, 0.602]
        mock_metrics.f1_score.return_value = 0.5
        
        selector = ProgressiveSelector(metrics_service=mock_metrics)
        
        # Create 4 mock trees
        entries = []
        for i in range(4):
            tree = MagicMock()
            tree.predict.return_value = np.array([0, 1])
            entries.append(TreeEntry(tree=tree, client_id="c1", tree_local_id=i, accuracy=0.5, macro_f1=0.5, pcd=0.5))
            
        X_val = np.zeros((2, 5))
        y_val = np.array([0, 1])
        
        # episode_size = 1, convergence_threshold = 0.05
        # Stop after 2 consecutive rounds with improvement < 0.05
        # Round 1: 0.5 -> ok
        # Round 2: 0.6 (imp 0.1) -> ok
        # Round 3: 0.601 (imp 0.001 < 0.05) -> stop_counter = 1
        # Round 4: 0.602 (imp 0.001 < 0.05) -> stop_counter = 2 -> STOP
        
        selected_trees, selected_entries, conv_round, logs = selector.select(
            candidate_entries=entries,
            X_val=X_val,
            y_val_norm=y_val,
            episode_size=1,
            t_max=10,
            convergence_threshold=0.05
        )
        
        assert len(selected_trees) == 4
        assert conv_round == 4
        assert len(logs) == 4

    def test_caching(self):
        selector = ProgressiveSelector()
        tree = MagicMock()
        tree.predict.return_value = np.array([1, 1])
        entry = TreeEntry(tree=tree, client_id="c1", tree_local_id=0, accuracy=0.5, macro_f1=0.5, pcd=0.5)
        
        X = np.zeros((2, 5))
        cache = {}
        
        # First call should call tree.predict
        selector._predict_ensemble([entry], X, cache=cache)
        assert tree.predict.call_count == 1
        assert id(tree) in cache
        
        # Second call should use cache
        selector._predict_ensemble([entry], X, cache=cache)
        assert tree.predict.call_count == 1

# --- Tests for PerClientProgressiveStrategy ---

class TestStrategies:
    def test_round_robin_interleaving(self):
        strategy = S5PerClientAccuracyStrategy()
        
        # Create trees for 2 clients
        c1_trees = [
            TreeEntry(tree="t1_c1", client_id="c1", tree_local_id=0, accuracy=0.9, macro_f1=0.5, pcd=0.5),
            TreeEntry(tree="t2_c1", client_id="c1", tree_local_id=1, accuracy=0.8, macro_f1=0.5, pcd=0.5),
        ]
        c2_trees = [
            TreeEntry(tree="t1_c2", client_id="c2", tree_local_id=0, accuracy=0.85, macro_f1=0.5, pcd=0.5),
        ]
        
        client_ranked = {"c1": c1_trees, "c2": c2_trees}
        interleaved = strategy._interleave_round_robin(client_ranked, ["c1", "c2"])
        
        # Expected: c1[0], c2[0], c1[1]
        assert interleaved[0].tree == "t1_c1"
        assert interleaved[1].tree == "t1_c2"
        assert interleaved[2].tree == "t2_c1"

# --- Tests for HybridPredictor ---

class TestHybridPredictor:
    def test_weight_validation(self):
        # Should raise AssertionError if weights don't sum to 1.0
        with pytest.raises(AssertionError):
            HybridPredictor(local_weight=0.8, global_weight=0.8, n_classes=2)
        
        # Valid weights should work
        predictor = HybridPredictor(local_weight=1.0, global_weight=0.0, n_classes=2)
        assert predictor.lw == 1.0
        assert predictor.gw == 0.0

    def test_prediction_logic(self):
        # Mock trees that return fixed predictions
        tree_local = MagicMock()
        tree_local.predict.return_value = np.array([0, 0])
        tree_global = MagicMock()
        tree_global.predict.return_value = np.array([1, 1])
        
        # 100% local -> result should be class 0
        predictor = HybridPredictor(local_weight=1.0, global_weight=0.0, n_classes=2)
        preds = predictor.predict(np.zeros((2, 5)), [tree_local], [tree_global])
        np.testing.assert_array_equal(preds, [0, 0])
        
        # 100% global -> result should be class 1
        predictor = HybridPredictor(local_weight=0.0, global_weight=1.0, n_classes=2)
        preds = predictor.predict(np.zeros((2, 5)), [tree_local], [tree_global])
        np.testing.assert_array_equal(preds, [1, 1])

        # 50/50 mix, but local has more trees (higher weight per tree)? 
        # No, lw is total weight for all local trees.
        # If local says 0 and global says 1, and lw=0.6, gw=0.4, result is 0.
        predictor = HybridPredictor(local_weight=0.6, global_weight=0.4, n_classes=2)
        preds = predictor.predict(np.zeros((2, 5)), [tree_local], [tree_global])
        np.testing.assert_array_equal(preds, [0, 0])
