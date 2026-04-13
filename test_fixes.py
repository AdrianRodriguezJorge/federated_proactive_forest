"""Test to verify label encoding and tree prediction fixes."""
import numpy as np
from pathlib import Path
from src.infrastructure.dataset.dataset_factory import DatasetFactory
from src.domain.services.label_service import SimpleLabelService
from src.domain.model.proactive_forest import ProactiveForest

def test_label_service_with_integers():
    """Test that label_service can handle both string and integer labels."""
    print("=" * 60)
    print("TEST 1: Label Service with Integer Predictions")
    print("=" * 60)
    
    # Create label service with class names
    class_names = ['hAd', 'hEd', 'hId', 'hOd', 'hUd']
    label_svc = SimpleLabelService(class_names=class_names)
    
    # Test with string labels
    string_labels = ['hAd', 'hEd', 'hId', 'hOd', 'hUd']
    encoded = label_svc.transform(string_labels)
    print(f"String labels: {string_labels}")
    print(f"Encoded: {encoded}")
    assert np.array_equal(encoded, [0, 1, 2, 3, 4]), "String encoding failed"
    
    # Test with integer labels (simulating tree predictions)
    int_labels = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    encoded_int = label_svc.transform(int_labels)
    print(f"Integer labels: {int_labels}")
    print(f"Encoded (should be same): {encoded_int}")
    assert np.array_equal(encoded_int, [0, 1, 2, 3, 4]), "Integer passthrough failed"
    
    print("✅ Label service test passed!\n")


def test_tree_predictions_are_integers():
    """Test that tree.predict() returns integer class indices."""
    print("=" * 60)
    print("TEST 2: Tree Predictions are Integers")
    print("=" * 60)
    
    # Load vowel dataset
    ds = DatasetFactory.load_from_config(
        {'type': 'Vowel', 'file_path': 'data/vowel.csv', 'target_column': 'Class', 
         'test_size': 0.2, 'scale': True, 'scaler_type': 'standard', 'sep': ','},
        project_root=Path('.')
    )
    
    print(f"Dataset: {ds.dataset_name}")
    print(f"Classes: {ds.class_names}")
    print(f"Train shape: {ds.X_train.shape}, Test shape: {ds.X_test.shape}")
    print(f"y_train sample: {ds.y_train[:5]}")
    print(f"y_train dtype: {ds.y_train.dtype}")
    
    # Train a small forest
    forest = ProactiveForest(
        n_estimators=5,
        alpha=0.1,
        random_state=42,
        verbose=False,
        class_names=ds.class_names
    )
    
    print("\nTraining forest...")
    forest.fit(ds.X_train, ds.y_train)
    
    # Get trees and test predictions
    trees = forest.get_trees()
    print(f"\nTrained {len(trees)} trees")
    
    # Test predictions from individual tree
    test_X = ds.X_test[:10]
    for i, tree in enumerate(trees[:2]):  # Test first 2 trees
        preds = tree.predict(test_X)
        print(f"\nTree {i} predictions:")
        print(f"  Raw predictions: {preds}")
        print(f"  dtype: {preds.dtype}")
        print(f"  First pred type: {type(preds[0])}")
        
        # Verify they're integers
        assert preds.dtype == 'O' or np.issubdtype(preds.dtype, np.integer), \
            f"Expected integer predictions, got {preds.dtype}"
        
        # Check values are valid class indices
        for j, pred in enumerate(preds):
            pred_int = int(pred)
            assert 0 <= pred_int < len(ds.class_names), \
                f"Prediction {pred_int} out of range [0, {len(ds.class_names)})"
    
    print("\n✅ Tree predictions test passed!\n")


def test_label_service_integration():
    """Test label service integration with tree predictions."""
    print("=" * 60)
    print("TEST 3: Label Service + Tree Predictions Integration")
    print("=" * 60)
    
    # Load dataset
    ds = DatasetFactory.load_from_config(
        {'type': 'Vowel', 'file_path': 'data/vowel.csv', 'target_column': 'Class',
         'test_size': 0.2, 'scale': True, 'scaler_type': 'standard', 'sep': ','},
        project_root=Path('.')
    )
    
    # Create label service
    label_svc = SimpleLabelService(class_names=ds.class_names)
    
    # Train forest
    forest = ProactiveForest(
        n_estimators=3,
        alpha=0.1,
        random_state=42,
        verbose=False,
        class_names=ds.class_names
    )
    forest.fit(ds.X_train, ds.y_train)
    
    # Get a tree and make predictions
    trees = forest.get_trees()
    tree = trees[0]
    test_X = ds.X_test[:5]
    
    # Raw predictions from tree (should be integers)
    raw_preds = tree.predict(test_X)
    print(f"Raw tree predictions: {raw_preds}")
    print(f"Raw predictions dtype: {raw_preds.dtype}")
    
    # Transform using label service (should handle integers correctly)
    transformed_preds = label_svc.transform(raw_preds)
    print(f"Transformed predictions: {transformed_preds}")
    
    # Verify they're the same (integers should pass through)
    assert np.array_equal(raw_preds.astype(int), transformed_preds), \
        "Label service should pass through integer predictions unchanged"
    
    # Now test with string labels
    string_labels = ds.y_test[:5]
    print(f"\nString labels: {string_labels}")
    encoded_labels = label_svc.transform(string_labels)
    print(f"Encoded string labels: {encoded_labels}")
    
    # Verify encoding works
    for i, label in enumerate(string_labels):
        expected_idx = ds.class_names.index(label)
        assert encoded_labels[i] == expected_idx, \
            f"Label '{label}' should encode to {expected_idx}, got {encoded_labels[i]}"
    
    print("\n✅ Integration test passed!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RUNNING COMPREHENSIVE TESTS")
    print("=" * 60 + "\n")
    
    try:
        test_label_service_with_integers()
        test_tree_predictions_are_integers()
        test_label_service_integration()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
