from typing import List, Optional, Union
from pydantic import BaseModel, Field

class DatasetConfig(BaseModel):
    type: str
    file_path: Optional[str] = None
    target_column: Optional[str] = "target"
    test_size: float = 0.15
    scale: bool = True
    scaler_type: str = "standard"
    sep: str = ","
    columns_to_drop: List[str] = Field(default_factory=list)

class FederationConfig(BaseModel):
    n_clients: int = 5
    distribution: str = "iid"
    dirichlet_alpha: float = 0.5
    seed: int = 42

class ModelConfig(BaseModel):
    n_estimators: int = 100
    alpha: float = 0.1
    bootstrap: bool = True
    max_depth: Optional[int] = None
    split_criterion: str = "entropy"

class AggregationConfig(BaseModel):
    strategy: str = "S1"
    f1_weight: float = 0.7
    pcd_weight: float = 0.3
    t_max: Optional[int] = None
    window_size: int = 5
    max_rounds: int = 20
    convergence_threshold: float = 0.002

class PredictionConfig(BaseModel):
    local_weight: float = 0.4
    global_weight: float = 0.6
    use_weighted: bool = True

class FLConfig(BaseModel):
    dataset: DatasetConfig
    federation: FederationConfig
    model: ModelConfig
    aggregation: AggregationConfig
    prediction: PredictionConfig
    verbose: bool = True
    seed: int = 42
