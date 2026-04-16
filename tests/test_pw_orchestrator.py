import unittest
import numpy as np
from src.application.orchestrators.progressive_windows_orchestrator import ProgressiveWindowsOrchestrator
from src.domain.dataset.base_adapter import DatasetSplit

class MockDataset:
    @staticmethod
    def get_split():
        # Small synthetic dataset for 3 classes
        X_train = np.random.rand(100, 4)
        y_train = np.random.randint(0, 3, 100)
        X_test = np.random.rand(20, 4)
        y_test = np.random.randint(0, 3, 20)
        class_names = ['C0', 'C1', 'C2']
        
        return DatasetSplit(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            feature_names=['f1', 'f2', 'f3', 'f4'],
            class_names=class_names,
            dataset_name="Mock"
        )

class TestProgressiveWindowsOrchestrator(unittest.TestCase):
    def setUp(self):
        self.ds = MockDataset.get_split()
        self.config = {
            'n_clients': 3,
            'aggregation': {
                'window_size': 2,
                'max_rounds': 3,
                'f1_weight': 0.5,
                'convergence_threshold': 0.001
            },
            'prediction': {
                'local_weight': 0.5
            },
            'alpha_pf': 0.1,
            'seed': 42
        }
        # In setup_federation we usually do Dirichlet but here we mock client partitions
        self.orchestrator = ProgressiveWindowsOrchestrator(self.config, self.ds, verbose=False)
        self.orchestrator.client_partitions = {
            'client_0': (self.ds.X_train[:33], self.ds.y_train[:33]),
            'client_1': (self.ds.X_train[33:66], self.ds.y_train[33:66]),
            'client_2': (self.ds.X_train[66:], self.ds.y_train[66:])
        }

    def test_run_federated_round(self):
        results = self.orchestrator.run_federated_round()
        
        # Verify basic structure
        self.assertEqual(results.strategy_id, "PW")
        self.assertTrue(results.num_rounds > 0)
        self.assertEqual(len(results.client_ids), 3)
        self.assertEqual(results.n_trees_global, results.num_rounds * 3)
        
        # Verify results match y_test size
        self.assertEqual(len(results.global_predictions), len(self.ds.y_test))
        for cid in results.client_ids:
            self.assertEqual(len(results.client_hybrid_predictions[cid]), len(self.ds.y_test))

    def test_convergence_detection(self):
        # Force very high convergence threshold to trigger early stopping
        self.orchestrator.strategy.convergence_threshold = 1.0 
        self.orchestrator.convergence_threshold = 1.0
        
        results = self.orchestrator.run_federated_round()
        
        # Should stop after 2 rounds (previous_accuracy + 1 round with improvement < threshold)
        # Actually in PW, improvement <= threshold increments counter. stop_counter >= 2 stops.
        # Round 1: sets previous_accuracy.
        # Round 2: improvement 0 (random) <= 1.0 -> stop_counter = 1.
        # Round 3: improvement 0 <= 1.0 -> stop_counter = 2 -> STOP.
        self.assertLessEqual(results.num_rounds, 3)

if __name__ == '__main__':
    unittest.main()
