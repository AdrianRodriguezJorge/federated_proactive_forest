import sys
import os
import numpy as np
import traceback
from sklearn.datasets import load_iris

# Add src to python path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.domain.dataset.base_adapter import DatasetSplit
from src.application.orchestrators.fl_orchestrator import FLEXOrchestrator
from src.application.orchestrators.progressive_windows_orchestrator import ProgressiveWindowsOrchestrator
from src.application.orchestrators.roulette_orchestrator import RouletteOrchestrator

def run_strategies_iris():
    print("Starting guided testing of all 13 federated strategies on Iris dataset...")
    
    # 1. Load Iris
    iris = load_iris()
    X = iris.data
    y = iris.target.astype(str) # SimpleLabelService needs string labels
    
    # Shuffle and split
    np.random.seed(42)
    shuffled_idx = np.random.permutation(len(X))
    X, y = X[shuffled_idx], y[shuffled_idx]
    
    # Split: 100 training, 25 validation, 25 test
    X_train, y_train = X[:100], y[:100]
    X_val, y_val = X[100:125], y[100:125]
    X_test, y_test = X[125:], y[125:]
    
    split = DatasetSplit(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        feature_names=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        class_names=["0", "1", "2"],
        dataset_name="Iris"
    )
    
    # List of all 13 strategies:
    strategies = [
        # FLEX-based strategies S1 - S7
        ("S1_simple_pool", "S1"),
        ("S2_global_accuracy", "S2"),
        ("S3_global_macro_f1", "S3"),
        ("S4_global_f1_pcd", "S4"),
        ("S5_perclient_accuracy", "S5"),
        ("S6_perclient_f1", "S6"),
        ("S7_perclient_f1_pcd", "S7"),
        # Progressive Windows Strategy
        ("PW_progressive_windows", "PW"),
        # Roulette-based strategies S9 variants
        ("S9_WEIGHTED", "S9_WEIGHTED"),
        ("S9_MEAN", "S9_MEAN"),
        ("S9_MEDIAN", "S9_MEDIAN"),
        ("S9_CONSENSUS", "S9_CONSENSUS"),
        ("S9_PROACTIVE_PCD", "S9_PROACTIVE_PCD"),
    ]
    
    results_summary = {}
    
    for name, strategy_id in strategies:
        print(f"\n" + "=" * 60)
        print(f" RUNNING: {name} (ID: {strategy_id})")
        print("=" * 60)
        
        config = {
            "federation": {"n_clients": 3, "distribution": "iid"},
            "model": {
                "n_estimators": 6, 
                "alpha": 0.1, 
                "voting": "soft", 
                "class_names": ["0", "1", "2"],
                "local_convergence_threshold": 0.0001
            },
            "aggregation": {
                "strategy": strategy_id if not strategy_id.startswith("S9_") else "S9",
                "variant": strategy_id,
                "window_size": 2,
                "max_rounds": 3,
                "convergence_threshold": -1.0,
                "beta": 0.5
            }
        }
        
        # Instantiate orchestrator
        if strategy_id == "PW":
            orch = ProgressiveWindowsOrchestrator(config)
        elif strategy_id.startswith("S9_"):
            orch = RouletteOrchestrator(config)
        else:
            orch = FLEXOrchestrator(config)
            
        orch.setup_federation(split)
        
        try:
            res = orch.run_federated_round(n_bootstrap=0)
            
            print(f"[{name}] Successfully completed execution!")
            print(f"Global Accuracy: {res.global_accuracy:.4f}")
            print(f"Global Macro F1: {res.global_macro_f1:.4f}")
            print(f"Number of Trees Global: {res.n_trees_global}")
            print(f"Convergence Round: {res.convergence_round}")
            
            results_summary[name] = {
                "accuracy": res.global_accuracy,
                "macro_f1": res.global_macro_f1,
                "n_trees_global": res.n_trees_global,
                "convergence_round": res.convergence_round,
                "client_accuracies": res.client_accuracies,
                "client_f1_scores": res.client_f1_scores
            }
            
        except Exception as e:
            print(f"[ERROR IN {name}]: {str(e)}")
            traceback.print_exc()
            results_summary[name] = {"error": str(e)}
        finally:
            if hasattr(orch, 'cleanup'):
                orch.cleanup()
                
    # Comparative table
    print("\n\n" + "=" * 60)
    print(" SUMMARY OF COMPARATIVE RESULTS ON IRIS")
    print("=" * 60)
    print(f"{'Strategy Name':<25} | {'Accuracy':<10} | {'Macro-F1':<10} | {'Global Trees':<12}")
    print("-" * 65)
    for name, metrics in results_summary.items():
        if "error" in metrics:
            print(f"{name:<25} | ERROR: {metrics['error'][:30]}")
        else:
            print(f"{name:<25} | {metrics['accuracy']:<10.4f} | {metrics['macro_f1']:<10.4f} | {metrics['n_trees_global']:<12}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    run_strategies_iris()
