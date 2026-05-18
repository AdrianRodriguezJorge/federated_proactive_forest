"""Strategy flow analyzer to extract detailed behavior for the report."""

import logging
import sys
import numpy as np

from src.domain.dataset.base_adapter import DatasetSplit
from src.application.orchestrators.fl_orchestrator import FLEXOrchestrator
from src.application.orchestrators.progressive_windows_orchestrator import ProgressiveWindowsOrchestrator
from src.application.orchestrators.roulette_orchestrator import RouletteOrchestrator

# Setup logging to stdout with INFO level to capture all detailed flow logs
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='[%(levelname)s] %(message)s')

ALL_STRATEGIES = [
    ("S1_simple_pool", "S1"),
    ("S2_global_accuracy", "S2"),
    ("S3_global_macro_f1", "S3"),
    ("S4_global_f1_pcd", "S4"),
    ("S5_perclient_accuracy", "S5"),
    ("S6_perclient_f1", "S6"),
    ("S7_perclient_f1_pcd", "S7"),
    ("PW_progressive_windows", "PW"),
    ("S9_WEIGHTED", "S9_WEIGHTED"),
]

def generate_data():
    np.random.seed(42)
    X = np.random.rand(40, 2)
    y = np.random.choice(["0", "1"], size=40)
    return DatasetSplit(
        X_train=X[:20], X_val=X[20:30], X_test=X[30:],
        y_train=y[:20], y_val=y[20:30], y_test=y[30:],
        feature_names=["f1", "f2"],
        class_names=["0", "1"],
        dataset_name="AnalyzerData"
    )

def main():
    split = generate_data()
    
    for name, strategy_id in ALL_STRATEGIES:
        print("\n" + "="*80)
        print(f"=== STRATEGY DETAILED FLOW: {name} ({strategy_id}) ===")
        print("="*80)
        
        config = {
            "federation": {"n_clients": 2, "distribution": "iid"},
            "model": {
                "n_estimators": 5, 
                "alpha": 0.1, 
                "voting": "soft", 
                "class_names": ["0", "1"],
                "local_convergence_threshold": 0.001
            },
            "aggregation": {
                "strategy": strategy_id if not strategy_id.startswith("S9_") else "S9",
                "variant": strategy_id,
                "window_size": 2,
                "max_rounds": 1,
                "convergence_threshold": 0.01,
                "f1_weight": 0.5,
                "pcd_weight": 0.5
            }
        }
        
        if strategy_id == "PW":
            orch = ProgressiveWindowsOrchestrator(config)
        elif strategy_id.startswith("S9_"):
            orch = RouletteOrchestrator(config)
        else:
            orch = FLEXOrchestrator(config)
            
        orch.setup_federation(split)
        
        print("-> Running 1 round of FL...")
        res = orch.run_federated_round(n_bootstrap=0)
        
        print(f"\n-> Server global pool size: {res.n_trees_global}")
        print(f"-> Global Metrics: Accuracy={res.global_accuracy:.4f}, Macro-F1={res.global_macro_f1:.4f}")
        if hasattr(orch, 'cleanup'):
            orch.cleanup()

if __name__ == "__main__":
    main()
