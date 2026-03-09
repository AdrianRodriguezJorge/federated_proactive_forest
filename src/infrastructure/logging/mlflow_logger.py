"""MLflow logging integration."""
import mlflow
import mlflow.sklearn
from typing import Dict, Any, Optional
from src.domain.model.base_forest import ABCForest


class MLflowLogger:
    """
    Logger for MLflow experiment tracking.
    """

    def __init__(self, experiment_name: str = "federated_proactive_forest"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start a new MLflow run."""
        mlflow.start_run(run_name=run_name)

    def end_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics."""
        mlflow.log_metrics(metrics, step=step)

    def log_forest(self, forest: ABCForest, name: str = "forest") -> None:
        """Log forest model."""
        # Convert to sklearn-compatible format for logging
        from sklearn.ensemble import RandomForestClassifier
        trees = forest.get_trees()

        if trees:
            # Create a dummy RandomForest for logging purposes
            rf = RandomForestClassifier(n_estimators=len(trees))
            rf.estimators_ = trees
            mlflow.sklearn.log_model(rf, name)

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log complete configuration."""
        # Flatten nested config for MLflow
        flat_config = self._flatten_config(config)
        mlflow.log_params(flat_config)

    def _flatten_config(self, config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested dictionary for MLflow."""
        flat = {}
        for key, value in config.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flat.update(self._flatten_config(value, new_key))
            else:
                flat[new_key] = value
        return flat