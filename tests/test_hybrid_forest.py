import pytest
import numpy as np
from src.domain.model.cpf_implementation.estimator import ProactiveForestClassifier
from src.domain.model.hybrid_forest import HybridForest
from src.domain.services.label_service import SimpleLabelService

class DummyTree:
    """Mock tree class to simulate tree behavior during weighted voting and diversity checks."""
    def __init__(self, predictions):
        self.predictions = np.asarray(predictions)
    
    def predict(self, X):
        return self.predictions


def test_hybrid_forest_predict():
    """Verify that HybridForest correctly performs weighted majority voting."""
    # 3 samples, 2 classes
    X = np.zeros((3, 2))
    
    # Setup dummy trees
    # local tree: predicts [0, 0, 1]
    # global tree: predicts [1, 0, 1]
    local_tree = DummyTree(["0", "0", "1"])
    global_tree = DummyTree(["1", "0", "1"])
    
    label_svc = SimpleLabelService(["0", "1"])
    
    # Initialize hybrid forest
    # Weight local = 0.4, weight global = 0.6
    # Sample 0:
    #   local: predicts 0. Vote: local_weight = 0.4
    #   global: predicts 1. Vote: global_weight = 0.6
    #   Winner: 1 (global has higher weight!)
    # Sample 1:
    #   local: predicts 0. Vote: 0.4
    #   global: predicts 0. Vote: 0.6
    #   Winner: 0
    # Sample 2:
    #   local: predicts 1. Vote: 0.4
    #   global: predicts 1. Vote: 0.6
    #   Winner: 1
    hybrid = HybridForest(
        local_trees=[local_tree],
        global_trees=[global_tree],
        local_weight=0.4,
        global_weight=0.6,
        n_classes=2,
        class_names=["0", "1"],
        label_service=label_svc
    )
    
    preds = hybrid.predict(X)
    assert np.array_equal(preds, [1, 0, 1])


def test_hybrid_forest_diversity_measure():
    """Verify that HybridForest calculates diversity using PCD correctly."""
    # 4 samples, 2 classes
    X = np.zeros((4, 2))
    y = np.array(["0", "0", "0", "0"])
    
    # 10 trees total (5 local, 5 global)
    # n_trees = 10. lower = 1.0 (10%), upper = 9.0 (90%)
    # For a sample to be "diverse", the number of tree hits must be between [1, 9] inclusive.
    # Sample 0: 0 trees hit -> not diverse (hits = 0)
    # Sample 1: 5 trees hit -> diverse (hits = 5)
    # Sample 2: 10 trees hit -> not diverse (hits = 10)
    # Sample 3: 1 tree hits -> diverse (hits = 1)
    # diverse samples = 2 out of 4 -> diversity = 0.5
    
    local_trees = [
        DummyTree(["1", "0", "0", "0"]), # hits: [no, yes, yes, yes]
        DummyTree(["1", "0", "0", "1"]), # hits: [no, yes, yes, no]
        DummyTree(["1", "0", "0", "1"]), # hits: [no, yes, yes, no]
        DummyTree(["1", "0", "0", "1"]), # hits: [no, yes, yes, no]
        DummyTree(["1", "0", "0", "1"]), # hits: [no, yes, yes, no]
    ]
    
    global_trees = [
        DummyTree(["1", "1", "0", "1"]), # hits: [no, no, yes, no]
        DummyTree(["1", "1", "0", "1"]), # hits: [no, no, yes, no]
        DummyTree(["1", "1", "0", "1"]), # hits: [no, no, yes, no]
        DummyTree(["1", "1", "0", "1"]), # hits: [no, no, yes, no]
        DummyTree(["1", "1", "0", "1"]), # hits: [no, no, yes, no]
    ]
    
    # Total hits per sample:
    # Sample 0: 0 hits (all say 1)
    # Sample 1: 5 hits (local trees say 0)
    # Sample 2: 10 hits (all say 0)
    # Sample 3: 1 hit (only local_trees[0] says 0)
    # Diverse samples: Sample 1 (5 hits) and Sample 3 (1 hit) => 2/4 = 0.5
    
    label_svc = SimpleLabelService(["0", "1"])
    
    hybrid = HybridForest(
        local_trees=local_trees,
        global_trees=global_trees,
        n_classes=2,
        class_names=["0", "1"],
        label_service=label_svc
    )
    
    div = hybrid.diversity_measure(X, y, diversity='pcd')
    assert div == 0.5
    
    # Verify invalid diversity metric raises ValueError
    with pytest.raises(ValueError):
        hybrid.diversity_measure(X, y, diversity='invalid')


def test_proactive_forest_classifier_diversity_handles_numeric_labels_with_string_encoder():
    """Verify ProactiveForestClassifier.diversity_measure handles numeric y labels.

    This guards against the case where encoder keys are stored as string values
    while the target labels are numeric integers.
    """
    class NumericDummyTree:
        def __init__(self, predictions):
            self.predictions = np.asarray(predictions, dtype=np.int64)

        def predict(self, X):
            return self.predictions

    classifier = ProactiveForestClassifier(n_estimators=2, alpha=0.1)
    classifier._trees = [
        NumericDummyTree([0, 1, 0]),
        NumericDummyTree([1, 1, 0]),
    ]
    classifier._encoder_dict = {"0": 0, "1": 1}

    X = np.zeros((3, 1))
    y = np.array([0, 1, 0], dtype=np.int64)

    diversity = classifier.diversity_measure(X, y, diversity="pcd")
    assert diversity > 0.0
    assert diversity <= 1.0
