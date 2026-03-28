#!/usr/bin/env python
"""Test script to verify all strategy imports work correctly."""

import sys

def test_imports():
    print("Testing strategy imports...")
    
    try:
        from src.domain.aggregation.aggregation_factory import AggregationFactory
        print("✓ AggregationFactory imported")
    except Exception as e:
        print(f"✗ Failed to import AggregationFactory: {e}")
        return False
    
    strategies = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']
    all_ok = True
    
    for s in strategies:
        try:
            strategy = AggregationFactory.create_strategy(s)
            print(f"✓ {s}: {strategy.strategy_id} - OK")
        except Exception as e:
            print(f"✗ {s}: FAILED - {e}")
            all_ok = False
    
    if all_ok:
        print("\n✓ All strategy imports successful!")
    else:
        print("\n✗ Some strategies failed to import")
    
    return all_ok

if __name__ == '__main__':
    success = test_imports()
    sys.exit(0 if success else 1)
