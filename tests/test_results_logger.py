"""Tests for OptimizationResultsLogger."""

import os
from pathlib import Path

import pytest

optuna = pytest.importorskip("optuna")

from src.infrastructure.persistence.results_logger import OptimizationResultsLogger


def test_results_logger_initialization(tmp_path):
    """Test that the logger creates the output directory."""
    logger_dir = tmp_path / "opt_results"
    logger = OptimizationResultsLogger(str(logger_dir))

    assert logger_dir.exists()
    assert logger_dir.is_dir()
    assert logger.output_path == logger_dir


def test_save_study(tmp_path):
    """Test saving an optuna study to various formats."""
    # Create a simple study
    study = optuna.create_study(direction="maximize")

    def objective(trial):
        x = trial.suggest_float("x", -10, 10)
        return -((x - 2) ** 2)

    study.optimize(objective, n_trials=3)

    logger_dir = tmp_path / "opt_results"
    logger = OptimizationResultsLogger(str(logger_dir))

    metadata = {"strategy": "S1", "dataset": "iris", "metric": "macro_f1"}
    saved_files = logger.save_study(study, metadata=metadata)

    assert "csv" in saved_files
    assert "yaml" in saved_files
    assert "sqlite" in saved_files
    assert "study_pickle" in saved_files

    assert os.path.exists(saved_files["csv"])
    assert os.path.exists(saved_files["yaml"])
    assert os.path.exists(saved_files["sqlite"])
    assert os.path.exists(saved_files["study_pickle"])


def test_generate_summary_report(tmp_path):
    """Test generating a summary report text file."""
    study = optuna.create_study(direction="maximize")

    def objective(trial):
        return 1.0

    study.optimize(objective, n_trials=2)

    logger_dir = tmp_path / "opt_results"
    logger = OptimizationResultsLogger(str(logger_dir))

    report_path = logger.generate_summary_report(
        study, metadata={"metric": "acc"}
    )

    assert os.path.exists(report_path)
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "OPTIMIZATION SUMMARY REPORT" in content
        assert "Total trials: 2" in content
        assert "Metric: acc" in content
