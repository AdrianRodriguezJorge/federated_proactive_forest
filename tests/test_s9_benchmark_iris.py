import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from src.domain.dataset.base_adapter import DatasetSplit
from src.infrastructure.dataset.dataset_factory import DATASET_METADATA


def load_module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location("final_benchmark_s9_only", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_s9_iris_pcd_nonzero():
    # Locate script
    script_path = Path("scripts") / "final_benchmark_s9_only.py"
    module = load_module_from_path(script_path)

    # Prepare Iris split following the script's preprocessing
    preset = DATASET_METADATA["iris"]
    df = pd.read_csv(preset["file_path"], sep=preset["sep"])
    cols_to_drop = preset.get("columns_to_drop", [])
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    target = preset["target_column"]
    X_raw = df.drop(columns=[target])
    y_raw = df[target].astype(str).values

    cat_cols = list(preset.get("categorical_features", []))
    for col in X_raw.columns:
        is_obj = X_raw[col].dtype == "object"
        if (is_obj or pd.api.types.is_string_dtype(X_raw[col])) and col not in cat_cols:
            cat_cols.append(col)

    le = LabelEncoder()
    le.fit(y_raw)

    X_train_val_raw, X_test_raw, y_train_val_raw, y_test_raw = train_test_split(
        X_raw,
        y_raw,
        test_size=0.1,
        random_state=42,
        stratify=y_raw if len(np.unique(y_raw)) > 1 else None,
    )

    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
        X_train_val_raw,
        y_train_val_raw,
        test_size=0.1111,
        random_state=43,
        stratify=y_train_val_raw if len(np.unique(y_train_val_raw)) > 1 else None,
    )

    X_train = X_train_raw.values.copy()
    X_val = X_val_raw.values.copy()
    X_test = X_test_raw.values.copy()

    cat_cols_idx = [X_raw.columns.get_loc(c) for c in cat_cols] if cat_cols else []
    num_cols_idx = [X_raw.columns.get_loc(c) for c in X_raw.columns if c not in cat_cols]

    if cat_cols_idx:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train[:, cat_cols_idx] = enc.fit_transform(X_train[:, cat_cols_idx].astype(str))
        X_val[:, cat_cols_idx] = enc.transform(X_val[:, cat_cols_idx].astype(str))
        X_test[:, cat_cols_idx] = enc.transform(X_test[:, cat_cols_idx].astype(str))

    if num_cols_idx:
        scaler = StandardScaler()
        X_train[:, num_cols_idx] = scaler.fit_transform(X_train[:, num_cols_idx])
        X_val[:, num_cols_idx] = scaler.transform(X_val[:, num_cols_idx])
        X_test[:, num_cols_idx] = scaler.transform(X_test[:, num_cols_idx])

    y_train = le.transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    y_test = le.transform(y_test_raw)

    split = DatasetSplit(
        X_train=X_train.astype(np.float64),
        X_val=X_val.astype(np.float64),
        X_test=X_test.astype(np.float64),
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=list(X_raw.columns),
        class_names=[str(c) for c in le.classes_],
        dataset_name="Iris",
    )

    precomputed_splits = [split]

    pcd_values = []
    for strat in module.S9_STRATEGIES:
        r = module.run_single_strategy(strat, "Iris", 1, precomputed_splits)
        pcd_values.append((strat, float(r["pcd_mean"])))

    # Print values for user inspection (pytest captures stdout)
    for strat, pcd in pcd_values:
        print(f"Strategy={strat} | PCD={pcd}")

    # Fail if all PCDs are zero (indicating the prior bug still present)
    assert any(pcd > 0.0 for _, pcd in pcd_values), "All S9 PCD values are 0.0 for Iris" 
