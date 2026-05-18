"""Tests for Pydantic configuration models."""

import pytest
from pydantic import ValidationError

from src.domain.config.config_models import (
    DatasetConfig,
    FederationConfig,
    ModelConfig,
    AggregationConfig,
    PredictionConfig,
    FLConfig,
)


def test_dataset_config_defaults():
    """Verify default values and type validation for DatasetConfig."""
    config = DatasetConfig(type="csv")
    assert config.type == "csv"
    assert config.target_column == "target"
    assert config.test_size == 0.15
    assert config.scale is True
    assert config.scaler_type == "standard"
    assert config.sep == ","
    assert config.columns_to_drop == []


def test_dataset_config_invalid_type():
    """Verify that omitting required fields raises ValidationError."""
    with pytest.raises(ValidationError):
        DatasetConfig()


def test_federation_config_defaults():
    """Verify default values for FederationConfig."""
    config = FederationConfig()
    assert config.n_clients == 5
    assert config.distribution == "iid"
    assert config.dirichlet_alpha == 0.5
    assert config.seed == 42


def test_model_config_defaults():
    """Verify default values for ModelConfig."""
    config = ModelConfig()
    assert config.n_estimators == 100
    assert config.alpha == 0.1
    assert config.bootstrap is True
    assert config.split_criterion == "entropy"


def test_fl_config_initialization():
    """Verify full FLConfig initialization with nested sub-configs."""
    ds_conf = DatasetConfig(type="synthesized")
    fed_conf = FederationConfig()
    mod_conf = ModelConfig()
    agg_conf = AggregationConfig()
    pred_conf = PredictionConfig()

    fl_config = FLConfig(
        dataset=ds_conf,
        federation=fed_conf,
        model=mod_conf,
        aggregation=agg_conf,
        prediction=pred_conf,
    )

    assert fl_config.verbose is True
    assert fl_config.seed == 42
    assert fl_config.dataset.type == "synthesized"
    assert fl_config.aggregation.strategy == "S1"
