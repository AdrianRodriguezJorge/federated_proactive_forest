"""End-to-end test: run a quick experiment and verify the CSV log is created."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.infrastructure.dataset.dataset_factory import DatasetFactory
from src.application.orchestrators.fl_orchestrator import FLEXOrchestrator
from src.interfaces.streamlit.state.experiment_logger import save_experiment_log

config = {
    "dataset": {"type": "Iris", "file_path": "data/iris.csv", "target_column": "class",
                "test_size": 0.2, "scale": True, "sep": ","},
    "federation": {"n_clients": 2, "distribution": "iid", "dirichlet_alpha": 0.5},
    "model": {"n_estimators": 20, "alpha": 0.1, "split_criterion": "entropy",
              "feature_selection": "prob", "use_progressive_stopping": True,
              "convergence": 0.002, "episode_size": 5},
    "aggregation": {"strategy": "s3_global_f1", "f1_weight": 0.5, "pcd_weight": 0.5,
                    "convergence": 0.002, "episode_size": 5,
                    "window_size": 5, "max_rounds": 20},
    "prediction": {"local_weight": 0.4, "global_weight": 0.6, "use_weighted": True},
    "verbose": False, "seed": 42,
}

print("1. Loading dataset...")
ds = DatasetFactory.load_from_config(config["dataset"], project_root=ROOT)

print("2. Running federated round...")
orch = FLEXOrchestrator(config)
orch.setup_federation(ds, seed=42)
results = orch.run_federated_round()
print(f"   Global: acc={results.global_accuracy:.4f}, f1={results.global_macro_f1:.4f}")

print("3. Saving experiment log...")
log_path = save_experiment_log(results, config)

if log_path and log_path.exists():
    print(f"   CSV saved: {log_path}")
    import pandas as pd
    df = pd.read_csv(log_path)
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    print()
    print(df.to_string(index=False))
    print("\n=== LOG TEST PASSED ===")
else:
    print("   FAILED: log not saved!")
    sys.exit(1)
