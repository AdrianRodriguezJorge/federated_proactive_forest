import os
import sys
import optuna
from pathlib import Path

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.infrastructure.persistence.results_logger import OptimizationResultsLogger

def main():
    output_dir = "results/unified_optimization"
    db_path = Path(output_dir) / "unified_study.db"
    
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return

    storage_url = f"sqlite:///{db_path.resolve().as_posix()}"
    print(f"Loading study from: {storage_url}")
    study = optuna.load_study(study_name="unified_hpo", storage=storage_url)
    
    print("Initializing results logger...")
    logger = OptimizationResultsLogger(output_dir)
    
    metadata = {
        "strategy": "UNIFIED_BAYESIAN",
        "dataset": "sonar_vowel_spambase_nursery",
        "metric": "mean_normalized_accuracy",
        "n_trials": len(study.trials),
        "n_clients": 3,
        "seed": 42,
        "duration_seconds": 0,
    }
    
    print("Generating files (YAML, CSV, optimization.db, study.pkl)...")
    logger.save_study(study, metadata=metadata)
    
    print("Generating HTML visualization plots...")
    logger.generate_plots(study)
    
    print("Generating text summary report...")
    try:
        report_path = logger.generate_summary_report(study, metadata=metadata)
        print(f"Summary report generated at: {report_path}")
    except Exception as e:
        print(f"Warning: Could not generate text summary report: {e}")
        
    print("\nDone! All files generated successfully in results/unified_optimization/")

if __name__ == "__main__":
    main()
