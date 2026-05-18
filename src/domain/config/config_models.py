"""Configuration models for the Federated Proactive Forest.

This module defines Pydantic schemas for datasets, federation settings,
model hyperparameters, aggregation strategies, and global orchestration.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    """Configuration settings for dataset loading and preprocessing.

    Attributes:
        type (str): Type of the dataset (e.g., 'csv', 'synthesized').
        file_path (Optional[str]): Path to the data file. Defaults to None.
        target_column (Optional[str]): Name of the target variable.
            Defaults to "target".
        test_size (float): Proportion of dataset to reserve for testing.
            Defaults to 0.15.
        scale (bool): Whether to scale numerical features. Defaults to True.
        scaler_type (str): Type of scaling ('standard' or 'minmax').
            Defaults to "standard".
        sep (str): Delimiter character for text file reading. Defaults to ",".
        columns_to_drop (List[str]): List of columns to drop from dataset.
            Defaults to an empty list.
    """

    type: str
    file_path: Optional[str] = None
    target_column: Optional[str] = "target"
    test_size: float = 0.15
    scale: bool = True
    scaler_type: str = "standard"
    sep: str = ","
    columns_to_drop: List[str] = Field(default_factory=list)


class FederationConfig(BaseModel):
    """Configuration settings for the simulated federated environment.

    Attributes:
        n_clients (int): Number of simulated clients. Defaults to 5.
        distribution (str): Data partitioning strategy (e.g., 'iid',
            'non-iid', 'dirichlet'). Defaults to "iid".
        dirichlet_alpha (float): Alpha parameter for Dirichlet partitioner.
            Defaults to 0.5.
        seed (int): Random seed for reproducible partitioning. Defaults to 42.
    """

    n_clients: int = 5
    distribution: str = "iid"
    dirichlet_alpha: float = 0.5
    seed: int = 42


class ModelConfig(BaseModel):
    """Configuration settings for the Proactive Forest model.

    Attributes:
        n_estimators (int): Number of trees in the forest. Defaults to 100.
        alpha (float): Tolerance/Proactivity parameter (alpha).
            Defaults to 0.1.
        bootstrap (bool): Whether to use bootstrap sampling. Defaults to True.
        max_depth (Optional[int]): Maximum depth of each decision tree.
            Defaults to None.
        split_criterion (str): Criterion used to determine best split
            ('entropy' or 'gini'). Defaults to "entropy".
    """

    n_estimators: int = 100
    alpha: float = 0.1
    bootstrap: bool = True
    max_depth: Optional[int] = None
    split_criterion: str = "entropy"


class AggregationConfig(BaseModel):
    """Configuration settings for federated aggregation strategies.

    Attributes:
        strategy (str): Strategy identifier (e.g., 'S1', 'S9').
            Defaults to "S1".
        f1_weight (float): Weight for F1 score in multi-metric strategies.
            Defaults to 0.7.
        pcd_weight (float): Weight for PCD in multi-metric strategies.
            Defaults to 0.3.
        t_max (Optional[int]): Maximum number of trees to retain in pool.
            Defaults to None.
        window_size (int): Size of the evaluation sliding window.
            Defaults to 5.
        max_rounds (int): Maximum communication rounds. Defaults to 20.
        convergence_threshold (float): Convergence criteria.
            Defaults to 0.002.
    """

    strategy: str = "S1"
    f1_weight: float = 0.7
    pcd_weight: float = 0.3
    t_max: Optional[int] = None
    window_size: int = 5
    max_rounds: int = 20
    convergence_threshold: float = 0.002


class PredictionConfig(BaseModel):
    """Configuration settings for prediction routing and weighting.

    Attributes:
        local_weight (float): Weight assigned to local client predictions.
            Defaults to 0.4.
        global_weight (float): Weight assigned to aggregated global predictions.
            Defaults to 0.6.
        use_weighted (bool): Whether to perform weighted ensemble voting.
            Defaults to True.
    """

    local_weight: float = 0.4
    global_weight: float = 0.6
    use_weighted: bool = True


class FLConfig(BaseModel):
    """Global configuration settings for Federated Learning.

    Attributes:
        dataset (DatasetConfig): Dataset loading parameters.
        federation (FederationConfig): Federated environment parameters.
        model (ModelConfig): Proactive Forest parameters.
        aggregation (AggregationConfig): Aggregation strategy parameters.
        prediction (PredictionConfig): Prediction logic parameters.
        verbose (bool): Whether to enable detailed logs. Defaults to True.
        seed (int): Global random seed. Defaults to 42.
    """

    dataset: DatasetConfig
    federation: FederationConfig
    model: ModelConfig
    aggregation: AggregationConfig
    prediction: PredictionConfig
    verbose: bool = True
    seed: int = 42

