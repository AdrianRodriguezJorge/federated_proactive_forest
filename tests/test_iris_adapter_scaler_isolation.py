import pytest
import numpy as np
from src.infrastructure.dataset.iris_adapter import IrisAdapter

def test_iris_adapter_scaler_isolation():
    """
    Verify that two IrisAdapter instances maintain independent scalers.
    By using different train_test_split_ratios, they will have different 
    training sets, resulting in different fitted scaler parameters.
    """
    # Create two adapters with different split ratios so they fit on different data
    adapter1 = IrisAdapter(scale=True, scaler_type="standard", train_test_split_ratio=0.5)
    adapter2 = IrisAdapter(scale=True, scaler_type="standard", train_test_split_ratio=0.8)
    
    # Ensure scalers are different objects
    assert adapter1._scaler is not adapter2._scaler, "Scalers must be separate instances"

    # Initially, scalers are not fitted
    assert not hasattr(adapter1._scaler, "mean_")
    assert not hasattr(adapter2._scaler, "mean_")

    # Load data which also fits the scaler
    split1 = adapter1.load()
    split2 = adapter2.load()

    # Verify scalers are now fitted
    assert hasattr(adapter1._scaler, "mean_")
    assert hasattr(adapter2._scaler, "mean_")

    # Verify scaling parameters differ because of different data sizes/values
    # (Since random seeds are the same but sizes differ, the means will be slightly different)
    assert not np.allclose(adapter1._scaler.mean_, adapter2._scaler.mean_), \
        "Fitted scaler means should differ for different training sets"
    assert not np.allclose(adapter1._scaler.scale_, adapter2._scaler.scale_), \
        "Fitted scaler scales should differ for different training sets"

    print("Test passed: IrisAdapters have independent scalers.")
