import unittest
import numpy as np
import warnings
from src.domain.services.label_service import SimpleLabelService
from src.domain.metrics.forest_evaluator import ForestEvaluator
from src.domain.model.proactive_forest import ProactiveForest
from src.infrastructure.metrics.sklearn_metrics_service import SklearnMetricsService

class TestLabelNormalizationSystem(unittest.TestCase):
    def setUp(self):
        self.metrics_svc = SklearnMetricsService()

    def test_out_of_range_labels_silence(self):
        """Verify that out-of-range labels do not trigger UserWarnings and are handled by the system."""
        class_names = ["A", "B", "C"]
        
        # Simulating illegal labels (index 10 is out of range for [0, 2])
        y_true_illegal = np.array([0, 1, 2, 10, -5]) 
        # Simulating illegal predictions
        y_pred_illegal = np.array([2, 1, 0, 5, 100])
        
        # Create a mock forest that 'predicts' those illegal indices
        class MockForest:
            def predict(self, X): return y_pred_illegal
            def get_trees(self): return [1, 2, 3] # Dummy trees
            def diversity_measure(self, X, y, diversity): return 0.5

        forest = MockForest()
        X_dummy = np.zeros((5, 10))

        # We catch warnings to see if any are raised
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            report = ForestEvaluator.evaluate(
                forest, X_dummy, y_true_illegal, class_names, metrics_svc=self.metrics_svc
            )
            
            # Check if our specific UserWarning about "indices out of range" was triggered
            range_warnings = [str(warning.message) for warning in w if "indices out of range" in str(warning.message)]
            
            self.assertEqual(len(range_warnings), 0, f"Detected range warnings: {range_warnings}")
            
            # Check that the report was actually generated and used class names
            self.assertEqual(len(report.per_class_f1), 3)
            self.assertIn("A", report.per_class_f1)
            self.assertIn("B", report.per_class_f1)
            self.assertIn("C", report.per_class_f1)

    def test_proactive_forest_notebook_compat(self):
        """Verify that ProactiveForest handles manual loading (integer labels) safely."""
        class_names = ["Setosa", "Versicolor", "Virginica"]
        X = np.random.rand(20, 4)
        # Using integers manually (like in some notebooks)
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0] * 2)
        
        forest = ProactiveForest(n_estimators=5, class_names=class_names)
        
        # This should not raise any 'range' error and should map correctly
        forest.fit(X, y)
        
        preds = forest.predict(X[:5])
        
        # Predictions should be strings from class_names, not raw integers
        for p in preds:
            self.assertIsInstance(p, (str, np.str_))
            self.assertIn(p, class_names)

if __name__ == '__main__':
    unittest.main()
