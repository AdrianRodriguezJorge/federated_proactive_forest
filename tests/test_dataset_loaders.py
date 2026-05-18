import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.infrastructure.dataset.csv_adapter import GenericCsvAdapter
from src.infrastructure.dataset.dataset_factory import DatasetFactory

def test_generic_csv_adapter_atomic_split_and_scale(tmp_path):
    """Verify that GenericCsvAdapter correctly performs Atomic Triple Split and scales correctly."""
    # 1. Create a dummy CSV file inside the temp folder
    csv_file = tmp_path / "test_data.csv"
    
    # 100 rows, 5 columns (2 numerical, 1 categorical, 1 target, 1 column to drop)
    data = {
        "num1": np.arange(100, dtype=np.float64),
        "num2": np.arange(100, dtype=np.float64) * 2,
        "cat1": ["A" if i % 2 == 0 else "B" for i in range(100)],
        "target": ["class0" if i < 50 else "class1" for i in range(100)],
        "drop_me": np.ones(100)
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    
    # 2. Instantiate the adapter with Standard Scaling
    adapter = GenericCsvAdapter(
        name="test_standard",
        train_path=str(csv_file),
        target_column="target",
        categorical_features=["cat1"],
        columns_to_drop=["drop_me"],
        scale=True,
        scaler_type="standard",
        test_size=0.20, # 20% test (20 samples)
        seed=42
    )
    
    split = adapter.load()
    
    # 3. Assertions on the atomic 3-way split sizes
    # First split: Test size = 0.20 * 100 = 20 samples. Remaining = 80 samples.
    # Second split: Val relative size = 0.15 / (1.0 - 0.20) = 0.1875 of remaining.
    # Val size = 0.1875 * 80 = 15 samples (15% of total dataset).
    # Train size = 80 - 15 = 65 samples.
    assert split.X_train.shape == (65, 3)
    assert split.X_val.shape == (15, 3)
    assert split.X_test.shape == (20, 3)
    
    assert split.y_train.shape == (65,)
    assert split.y_val.shape == (15,)
    assert split.y_test.shape == (20,)
    
    # Verify dropped column
    assert "drop_me" not in split.feature_names
    assert len(split.feature_names) == 3
    assert split.feature_names == ["num1", "num2", "cat1"]
    
    # Verify class names
    assert sorted(split.class_names) == ["class0", "class1"]
    
    # Verify categorical features are ordinal encoded (cat1 is the last feature)
    # The categories "A" and "B" should be encoded to numerical floats (0.0 and 1.0)
    assert np.all(np.isin(split.X_train[:, 2], [0.0, 1.0]))
    
    # Verify numerical columns are standardized (mean approx 0, std approx 1)
    # Fit stats must be calculated ONLY on the training subset!
    train_num1_mean = np.mean(split.X_train[:, 0])
    train_num1_std = np.std(split.X_train[:, 0])
    assert np.isclose(train_num1_mean, 0.0, atol=1e-7)
    assert np.isclose(train_num1_std, 1.0, atol=1e-7)


def test_generic_csv_adapter_minmax_scaling(tmp_path):
    """Verify that GenericCsvAdapter correctly performs MinMax scaling."""
    csv_file = tmp_path / "test_data.csv"
    data = {
        "num": np.arange(10, dtype=np.float64), # min = 0, max = 9
        "target": ["0"] * 5 + ["1"] * 5
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    
    adapter = GenericCsvAdapter(
        name="test_minmax",
        train_path=str(csv_file),
        target_column="target",
        scale=True,
        scaler_type="minmax",
        test_size=0.2, # 2 samples test, 1.5 rounded to 1 or 2 samples val, rest train
        seed=42
    )
    
    split = adapter.load()
    
    # Verify MinMax scaling (bounds [0, 1] on train)
    assert np.min(split.X_train) == 0.0
    assert np.max(split.X_train) == 1.0


def test_generic_csv_adapter_missing_target(tmp_path):
    """Verify that GenericCsvAdapter raises RuntimeError when the target column is missing."""
    csv_file = tmp_path / "test_data.csv"
    data = {
        "num": np.arange(10, dtype=np.float64),
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    
    adapter = GenericCsvAdapter(
        name="test_missing_target",
        train_path=str(csv_file),
        target_column="target",
        test_size=0.2,
        seed=42
    )
    
    with pytest.raises(RuntimeError) as exc_info:
        adapter.load()
    
    assert "target" in str(exc_info.value)


def test_dataset_factory_presets():
    """Verify that DatasetFactory correctly resolves metadata presets for registered datasets."""
    cfg = {
        "type": "iris",
        "test_size": 0.15,
        "scale": True,
        "scaler_type": "standard"
    }
    
    # Resolves Preset metadata
    adapter = DatasetFactory.get_adapter_from_config(cfg)
    assert adapter.name == "iris"
    assert adapter.target_column == "class"
    assert adapter.sep == ","
