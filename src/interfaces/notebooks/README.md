# Notebooks Organization

This directory contains Jupyter notebooks organized by model type and proposal type.

## 📁 Directory Structure

### 🔗 `federated_strategies/`
Notebooks for **federated learning strategies** (S1-S7 + Progressive Windows):
- **S1-S7**: Cepero (2023) federated aggregation strategies with FL extensions
- **Progressive Windows (PW)**: Window-based training with dynamic scoring
- **Benchmarking**: Comparative analysis across strategies

**Notebooks:**
- `s1_simple_pool.ipynb` - Simple Pool strategy (all trees, no ordering)
- `s2_global_accuracy.ipynb` - Global ordering by Accuracy + Progressive
- `s3_global_f1.ipynb` - Global ordering by Macro-F1 + Progressive
- `s4_global_f1_pcd.ipynb` - Global ordering by α·F1 + β·PCD + Progressive
- `s5_perclient_accuracy.ipynb` - Per-Client ordering by Accuracy + Progressive
- `s6_perclient_f1.ipynb` - Per-Client ordering by Macro-F1 + Progressive
- `s7_perclient_f1_pcd.ipynb` - Per-Client ordering by α·F1 + β·PCD + Progressive
- `progressive_windows.ipynb` - Progressive Windows strategy (windows + dynamic scoring)
- `benchmarking_7_estrategias_nsl.ipynb` - Benchmarking on NSL-KDD dataset
- `benchmarking_7_estrategias_students.ipynb` - Benchmarking on Students Dropout dataset

### 🌲 `baselines/`
**Baseline models** for comparison (non-federated):
- **Random Forest**: Traditional RF implementations
- **Proactive Forest**: Standalone PF implementations

**Notebooks:**
- `simple_random_forest_iris.ipynb` - Random Forest on Iris dataset
- `simple_random_forest_nsl_kdd.ipynb` - Random Forest on NSL-KDD dataset
- `simple_random_forest_students.ipynb` - Random Forest on Students Dropout dataset
- `simple_proactive_forest_iris.ipynb` - Proactive Forest on Iris dataset
- `simple_proactive_forest_nsl.ipynb` - Proactive Forest on NSL-KDD dataset
- `simple_proactive_forest_students.ipynb` - Proactive Forest on Students Dropout dataset

### 🎯 `standalone_models/`
**Standalone model implementations** (single-model training, non-federated):

**Notebooks:**
- `proactive_forest_students_training.ipynb` - Complete Proactive Forest training workflow on Students Dropout

### 🧪 `tests/`
**Test notebooks** for development and integration validation:

**Notebooks:**
- `test_feature_selection_s6.ipynb` - Feature selection testing for S6 strategy
- `test_flex_integration_s6.ipynb` - FLEX framework integration testing for S6

## 📊 Dataset Reference

| Dataset | Description | Location |
|---------|-------------|----------|
| **Iris** | Classic classification dataset | `baselines/` |
| **NSL-KDD** | Network intrusion detection | `baselines/`, `federated_strategies/` |
| **Students Dropout** | Student dropout prediction | `baselines/`, `federated_strategies/`, `standalone_models/` |

## 🚀 Usage

1. **For Federated Learning**: Start with notebooks in `federated_strategies/`
2. **For Baseline Comparison**: Check `baselines/` for traditional implementations
3. **For Single Model Training**: See `standalone_models/`
4. **For Development/Testing**: Refer to `tests/`

## 📝 Notes

- Notebooks in the same folder share the same model/proposal type
- Different datasets for the same strategy are grouped together
- Benchmarking notebooks compare multiple strategies on specific datasets
