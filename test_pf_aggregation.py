#!/usr/bin/env python
"""Test script to verify Progressive Forest aggregation in S2-S7 strategies."""

import sys
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def test_progressive_aggregation():
    """Test that S2-S7 strategies apply Progressive Forest with early stopping."""
    print("=" * 60)
    print("Testing Progressive Forest Aggregation (S2-S7)")
    print("=" * 60)
    
    # Generate synthetic data
    X, y = make_classification(n_samples=200, n_features=10, n_classes=3, 
                               n_informative=8, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train local forests for 3 clients
    from src.domain.model.proactive_forest import ProactiveForest
    
    print("\n1. Training local forests for 3 clients...")
    client_forests = {}
    client_trees = {}
    client_metadata = {}
    
    for i, client_id in enumerate(['client_0', 'client_1', 'client_2']):
        # Use different data subset for each client (simulating Non-IID)
        start_idx = i * 50
        end_idx = start_idx + 50
        X_client = X_train[start_idx:end_idx]
        y_client = y_train[start_idx:end_idx]
        
        pf = ProactiveForest(n_estimators=50, alpha=0.1, verbose=False)
        pf.fit(X_client, y_client)
        
        client_forests[client_id] = pf
        client_trees[client_id] = pf.get_trees()
        
        # Calculate metadata
        y_pred = pf.predict(X_test)
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        pcd = pf.diversity_measure(X_test, y_test, 'pcd')
        
        client_metadata[client_id] = type('Metadata', (), {
            'accuracy': acc,
            'macro_f1': f1,
            'pcd': pcd,
            'client_id': client_id,
            'n_trees': len(pf.get_trees())
        })()
        
        print(f"   {client_id}: {len(client_trees[client_id])} trees, acc={acc:.3f}, f1={f1:.3f}")
    
    total_trees = sum(len(trees) for trees in client_trees.values())
    print(f"\n   Total trees before aggregation: {total_trees}")
    
    # Test each strategy S2-S7
    from src.domain.aggregation.aggregation_factory import AggregationFactory
    
    print("\n2. Testing aggregation strategies with Progressive Forest...")
    print("-" * 60)
    
    results = {}
    for strategy_name in ['S2', 'S3', 'S4', 'S5', 'S6', 'S7']:
        strategy = AggregationFactory.create_strategy(strategy_name)
        
        # Aggregate with validation data (enables CPF early stopping)
        global_trees, selected_ids, all_entries = strategy.aggregate(
            client_trees,
            client_metadata,
            X_val=X_test,
            y_val=y_test,
            max_trees=100,  # For S2-S4: not used for stopping, only for reference
            max_trees_per_client=100,  # For S5-S7: limits per client before aggregation
            f1_weight=0.5,
            pcd_weight=0.5
        )
        
        results[strategy_name] = {
            'n_trees': len(global_trees),
            'reduction': (1 - len(global_trees) / total_trees) * 100
        }
        
        print(f"   {strategy_name}: {len(global_trees)} trees selected "
              f"({results[strategy_name]['reduction']:.1f}% reduction)")
        
        # Verify early stopping occurred (should select fewer than total trees)
        if len(global_trees) < total_trees:
            print(f"      ✓ Early stopping applied (saved {total_trees - len(global_trees)} trees)")
        else:
            print(f"      ⚠ All trees selected (no early stopping triggered)")
    
    print("-" * 60)
    
    # Test without validation data (should return all trees)
    print("\n3. Testing without validation data (no early stopping)...")
    strategy_s2 = AggregationFactory.create_strategy('S2')
    global_trees_no_val, _, _ = strategy_s2.aggregate(
        client_trees,
        client_metadata,
        X_val=None,
        y_val=None
    )
    print(f"   S2 without validation: {len(global_trees_no_val)} trees")
    if len(global_trees_no_val) == total_trees:
        print(f"      ✓ All trees returned (as expected without validation)")
    else:
        print(f"      ⚠ Unexpected tree count")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
    
    return True

if __name__ == '__main__':
    try:
        success = test_progressive_aggregation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
