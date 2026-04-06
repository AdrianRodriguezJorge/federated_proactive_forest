# 🌲 Federated Proactive Forest

**Federated Proactive Forest** is an advanced **horizontal federated learning** system that implements and compares **8 aggregation strategies** based on **Proactive Forest** (Cepero, 2023). The project combines a pure FL framework (no external FL library dependencies) with modern interactive interfaces, enabling systematic investigation of how different tree selection and ranking criteria affect performance in distributed non-IID environments.

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 Main Features

- ✅ **8 aggregation strategies** (S1-S7 + Progressive Windows) with varied tree selection criteria
- ✅ **Horizontal Federated Learning** with configurable N clients
- ✅ **Complete non-IID heterogeneity support** (Dirichlet distributions)
- ✅ **Modern Web UI** with Streamlit (4 interactive pages)
- ✅ **Full CLI** with YAML configuration for reproducible experimentation
- ✅ **Clean Hexagonal Architecture** (Domain → Application → Infrastructure)
- ✅ **Multiple datasets**: NSL-KDD, Iris, Students Dropout, and 7 additional CSV datasets
- ✅ **Complete metrics**: Accuracy, F1, Macro-F1, PCD, confusion matrices
- ✅ **Ranking visualization** with per-client and strategy-based selection
- ✅ **Extensible**: Easily add new datasets and strategies
- ✅ **Hyperparameter optimization** with Optuna integration
- ✅ **Hybrid prediction** with configurable local/global weights

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Supported Datasets](#-supported-datasets)
- [Aggregation Strategies](#-aggregation-strategies)
- [System Architecture](#-system-architecture)
- [Streamlit Interface](#-streamlit-interface)
- [CLI](#-cli)
- [Advanced Configuration](#-advanced-configuration)
- [Adding New Datasets](#-adding-new-datasets)
- [Contributing](#-contributing)
- [License](#-license)
- [Citations](#-citations)

## 🚀 Installation

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB, recommended 8GB+ for large datasets
- **Space**: 2GB free for datasets and models
- **OS**: Windows 10+, macOS 10.15+, Ubuntu 18.04+

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/federated-proactive-forest.git
   cd federated-proactive-forest
   ```

2. **Create virtual environment** (highly recommended)
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/macOS
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install with optional dependencies:
   ```bash
   # With UI support
   pip install -e ".[ui]"
   
   # With development tools
   pip install -e ".[dev]"
   
   # With hyperparameter optimization
   pip install -e ".[opt]"
   
   # With FLEX Framework (recommended for full FL orchestration)
   pip install -e ".[flex]"
   ```

4. **Verify installation**
   ```bash
   python -c "from src.domain.model.proactive_forest import ProactiveForest; print('✅ Installation successful')"
   ```

## ⚡ Quick Start

### Option 1: Web Interface (Recommended for beginners)

```bash
# Run the complete web application
streamlit run src/interfaces/streamlit/app.py
```

Open your browser at `http://localhost:8501` and follow these steps:

1. **⚙️ Configuration**: Select Iris dataset, 3 clients, strategy S1
2. **▶️ Run**: Click "Run federated round"
3. **🏆 Ranking**: Visualize which trees were selected
4. **📊 Metrics**: Review global and per-client accuracy

### Option 2: CLI (For advanced experimentation)

```bash
# Run complete experiment from YAML
python -m src.interfaces.cli.main --config configs/experiments/exp_s1_simple_pool.yaml
```

### Option 3: Programmatic Usage

```python
from src.application.orchestrators.fl_orchestrator import FLOrchestrator

# Basic configuration
config = {
    "dataset": {"type": "Iris", "test_size": 0.2},
    "federation": {"n_clients": 5, "distribution": "iid"},
    "model": {"n_estimators": 50, "alpha": 0.1},
    "aggregation": {"strategy": "s1_simple_pool"}
}

# Run federated round
orchestrator = FLOrchestrator(config)
results = orchestrator.run_federated_round()
print(f"Global accuracy: {results.global_accuracy:.4f}")
```

## 📊 Supported Datasets

| Dataset | Samples | Features | Classes | Type | Adapter | FL Time |
|---------|----------|----------|--------|------|-----------|-----------|
| **NSL-KDD** | ~150K | 41 (3 cat) | 5 | Security | `nslkdd_adapter.py` | 4-8 min |
| **Iris** | 150 | 4 (num) | 3 | Classification | `iris_adapter.py` | 1-2 min |
| **Students Dropout** | ~4K | 36 (mix) | 3 | Education | `csv_adapter.py` | 2-4 min |
| **Car Evaluation** | 1.7K | 6 (cat) | 4 | Classification | `csv_adapter.py` | 1-2 min |
| **Letter Recognition** | 20K | 16 (num) | 26 | Classification | `csv_adapter.py` | 3-5 min |
| **Nursery** | 13K | 8 (cat) | 5 | Classification | `csv_adapter.py` | 2-4 min |
| **Optdigits** | 5.6K | 64 (num) | 10 | Classification | `csv_adapter.py` | 3-6 min |
| **Sonar** | 208 | 60 (num) | 2 | Classification | `csv_adapter.py` | 1-2 min |
| **Spambase** | 4.6K | 57 (num) | 2 | Classification | `csv_adapter.py` | 2-4 min |
| **Vowel** | 990 | 10 (num) | 11 | Classification | `csv_adapter.py` | 1-3 min |
| **Generic CSV** | Variable | Variable | Variable | Any | `csv_adapter.py` | Variable |

### Dataset Preparation

#### NSL-KDD (Intrusion Detection)
```bash
# Download from official source
# https://www.unb.ca/cic/datasets/nsl.html
# Extract and place in data/:
# - NSL-KDD_train.csv
# - NSL-KDD_test.csv
```

#### Iris & Other Datasets
- Most datasets are already included in the `data/` directory
- Iris can also be loaded from scikit-learn if needed
- No additional download required for included datasets

## 🏆 Aggregation Strategies

The project implements **8 strategies** for tree selection in federated environments:

| Strategy | Scope | Criterion | Description |
|----------|-------|-----------|-------------|
| **S1** | Global | None | Simple pool - all trees |
| **S2** | Global | Accuracy | Global ranking by accuracy |
| **S3** | Global | Macro-F1 | Global ranking by F1-score |
| **S4** | Global | F1 + PCD | Global ranking by α·F1 + β·PCD |
| **S5** | Per-Client | Accuracy | Individual ranking per client |
| **S6** | Per-Client | Macro-F1 | Individual ranking per client |
| **S7** | Per-Client | F1 + PCD | Individual ranking with diversity |
| **PW** | Progressive Windows | Adaptive | Progressive windows with adaptive stopping |

### Weight Configuration (S4, S7)
```yaml
aggregation:
  strategy: s7_perclient_f1_pcd
  f1_weight: 0.7    # Weight for F1-score
  pcd_weight: 0.3   # Weight for PCD diversity
```

### Progressive Windows Strategy (PW)
The PW strategy implements a novel approach with:
- **Window-based aggregation**: Processes trees in windows
- **Adaptive stopping**: Automatically stops when convergence is detected
- **Configurable parameters**: window_size, max_rounds, alpha
- **Round-robin selection**: Ensures diversity across clients

```yaml
aggregation:
  strategy: pw
  window_size: 5
  max_rounds: 20
  alpha: 1.0
  convergence: 0.002
  episode_size: 5
```

## 🏗️ System Architecture

### Architectural Pattern: Hexagonal (Ports & Adapters)

```
src/
├── domain/                    # 📦 Business core (no external dependencies)
│   ├── model/                # ML models: ProactiveForest, strategies
│   │   ├── cpf_implementation/  # CPF algorithm components
│   │   ├── proactive_forest.py
│   │   ├── progressive_forest.py
│   │   └── random_forest.py
│   ├── aggregation/          # Aggregation logic S1-S7 + PW
│   │   ├── strategies/       # Individual strategy implementations
│   │   │   ├── progressive_windows/
│   │   │   ├── global_progressive_base.py
│   │   │   └── perclient_progressive_base.py
│   │   ├── tree_ranker.py    # Tree ranking logic
│   │   ├── cpf_stopper.py    # Progressive stopping
│   │   └── aggregation_factory.py
│   ├── dataset/              # Dataset interfaces
│   ├── metrics/              # Model evaluation
│   ├── metadata/             # Client metadata management
│   ├── prediction/           # Hybrid prediction (local + global)
│   └── update/               # Client update logic (No-Repeat Merge)
│
├── application/              # 🎯 Use cases and orchestration
│   ├── commands/             # FL commands (train, aggregate, predict, update)
│   ├── orchestrators/        # Orchestrators: FL, Progressive Windows
│   └── hyperparam_optimizer.py  # Hyperparameter optimization with Optuna
│
├── infrastructure/           # 🔌 Concrete adapters
│   └── dataset/              # Adapters: NSL-KDD, Iris, CSV
│
└── interfaces/               # 🎨 User interfaces
    ├── cli/                  # Command-line interface
    └── streamlit/            # Modern web interface
        ├── app.py            # Entry point
        ├── components/       # Reusable UI components
        ├── pages_manual/     # Page implementations
        └── state/            # Session state management
```

### Key Components

- **Proactive Forest (CPF)**: Ensemble algorithm with dynamic probability adjustment
- **FLOrchestrator**: Main orchestrator for federated learning rounds
- **ProgressiveWindowsOrchestrator**: Specialized orchestrator for PW strategy
- **Early Stopping**: Automatic convergence detection in progressive training
- **PCD Diversity**: Pairwise Classifier Disagreement diversity measure
- **Hybrid Prediction**: Configurable local/global model weighting
- **No-Repeat Merge**: Prevents duplicate tree selection across rounds

## 🌐 Streamlit Interface

The web interface provides a complete experimentation experience:

### Page 1: ⚙️ Configuration
- Dataset selection and parameters
- Federation configuration (clients, distribution)
- Proactive Forest model parameters
- Aggregation strategy configuration
- Hybrid prediction weights (local/global)

### Page 2: ▶️ Run Experiment
- Real-time progress bar
- Detailed execution logs
- Complete error handling with tracebacks
- Results summary with accuracy, F1, and tree count

### Page 3: 🏆 Tree Ranking
- Visualization of selected vs discarded trees
- Grouping by client (S5-S7) or global (S1-S4)
- Accuracy and PCD information per tree
- Interactive filtering by client and strategy

### Page 4: 📊 Complete Metrics
- Global Accuracy, F1, Macro-F1
- Interactive confusion matrices
- Per-client metrics
- Strategy comparison

## 💻 CLI

### Basic Usage
```bash
# Run experiment from YAML configuration
python -m src.interfaces.cli.main --config configs/experiments/exp_s1_simple_pool.yaml

# Run with custom parameters
python -m src.interfaces.cli.main \
  --dataset iris \
  --clients 5 \
  --strategy s7_perclient_f1_pcd \
  --trees 100
```

### YAML Configuration
```yaml
# configs/experiments/exp_s7_perclient_f1_pcd.yaml
dataset:
  type: "NSL-KDD"
  train_path: "data/NSL-KDD_train.csv"
  test_path: "data/NSL-KDD_test.csv"

federation:
  n_clients: 5
  distribution: "noniid_dirichlet"
  dirichlet_alpha: 0.5

model:
  n_estimators: 100
  alpha: 0.1

aggregation:
  strategy: "s7_perclient_f1_pcd"
  f1_weight: 0.7
  pcd_weight: 0.3
```

### Available Experiment Configs
- `exp_s1_simple_pool.yaml` - Simple pool baseline
- `exp_s2_global_accuracy.yaml` - Global accuracy ranking
- `exp_s3_global_f1.yaml` - Global F1 ranking
- `exp_s4_global_f1_pcd.yaml` - Global F1 + PCD
- `exp_s5_perclient_accuracy.yaml` - Per-client accuracy
- `exp_s6_perclient_f1.yaml` - Per-client F1
- `exp_s7_perclient_f1_pcd.yaml` - Per-client F1 + PCD
- `exp_pw_progressive_windows.yaml` - Progressive Windows strategy

## ⚙️ Advanced Configuration

### Data Distributions

#### IID (Uniform)
```yaml
federation:
  distribution: "iid"  # Uniformly distributed data
```

#### Non-IID Dirichlet
```yaml
federation:
  distribution: "noniid_dirichlet"
  dirichlet_alpha: 0.5  # Lower α = higher heterogeneity
```

### Proactive Forest Parameters
```yaml
model:
  n_estimators: 100      # Trees per local forest
  alpha: 0.1            # Diversity rate (0-1)
  bootstrap: true       # Bagging for diversity
  max_depth: null       # Maximum depth (null = unlimited)
  split_criterion: "entropy"  # "gini" or "entropy"
  feature_selection: "prob"   # Feature selection method
  use_progressive_stopping: true  # Enable early stopping
  convergence: 0.002    # Convergence threshold
  episode_size: 5       # Trees per episode
```

### Hybrid Prediction
```yaml
prediction:
  local_weight: 0.0     # Weight for local model
  global_weight: 1.0    # Weight for global model
```

### Progressive Windows Strategy
```yaml
aggregation:
  strategy: "pw"
  window_size: 5        # Trees per window
  max_rounds: 20        # Maximum aggregation rounds
  alpha: 1.0            # Learning rate
  convergence: 0.002    # Convergence threshold
  episode_size: 5       # Trees per episode
```

## 🔧 Adding New Datasets

### Step 1: Create Adapter
```python
# src/infrastructure/dataset/new_dataset_adapter.py
from src.domain.dataset.base_adapter import IDatasetAdapter, DatasetSplit

class NewDatasetAdapter(IDatasetAdapter):
    def __init__(self, data_path, scale=True, scaler_type="standard"):
        # Implement constructor

    @property
    def name(self) -> str:
        return "new_dataset"

    @property
    def n_classes(self) -> int:
        return len(self._class_names_)

    def load(self) -> DatasetSplit:
        # Implement loading and preprocessing
        # Return DatasetSplit with X_train, X_test, y_train, y_test
```

### Step 2: Register in Configuration
Update the dataset configuration YAML:
```yaml
# configs/datasets/new_dataset.yaml
dataset:
  type: "NewDataset"
  file_path: "data/new_dataset.csv"
  target_column: "target"
  test_size: 0.2
  scale: true
  scaler_type: "standard"
```

### Step 3: Update UI (if needed)
```python
# src/interfaces/streamlit/pages_manual/page_config.py
def create_dataset_adapter(config: dict):
    if config["type"] == "NewDataset":
        return NewDatasetAdapter(...)
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the project
2. Create a branch for your feature (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines
- Follow the hexagonal architecture pattern
- Add tests for new functionality
- Update documentation
- Use type hints and docstrings
- Ensure all tests pass before submitting PR

## 📄 License

This project is under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📖 Citations

If you use this code in your research, please cite:

```bibtex
@article{cepero2023proactive,
  title={Proactive Forest: Hybrid Intelligence for Heterogeneous Federated Learning},
  author={Cepero, Mario and others},
  journal={IEEE Access},
  year={2023},
  publisher={IEEE}
}

@article{federated_proactive_forest,
  title={Federated Proactive Forest: Comparative Analysis of Tree Aggregation Strategies},
  author={Your Name},
  year={2026},
  note={GitHub repository: https://github.com/your-username/federated-proactive-forest}
}
```

---

**⭐ If you find this project useful, please give it a star on GitHub!**

## 📈 Decision Guide: Which Strategy to Use?

| Use Case | Recommendation | Reason |
|----------|----------------|--------|
| Pure research | S7 (Per-Client F1+PCD) | Maximum diversity + balanced metric |
| Maximum performance | S4 (Global F1+PCD) | Global ranking + PCD |
| Maximum speed | S1 (Simple Pool) | No ranking overhead |
| Small datasets | S5-S7 | Ensures all clients contribute |
| Large datasets | S2-S4 | More efficient |
| Adaptive stopping | PW (Progressive Windows) | Automatic convergence detection |

## 🐛 Troubleshooting

### Error: `ModuleNotFoundError: No module named 'src'`
```bash
# Make sure you're in the root directory
cd federated-proactive-forest
pip install -e .
```

### Error: Dataset doesn't load
- Verify paths in configuration
- Ensure column names match
- Check CSV encoding (UTF-8 recommended)

### Streamlit is slow
- Reduce `n_estimators` (100 → 50)
- Reduce `n_clients` (5 → 3)
- Use `convergence=0.005` (more aggressive)

### Out of memory
- Use smaller datasets for testing
- Reduce `max_depth` in model parameters
- Reduce number of clients

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_hybrid_inference.py -v
```

### Available Tests
- `test_hybrid_inference.py` - Hybrid prediction tests
- `test_rr_ds_explicit.py` - Round-robin dataset selection tests
- `test_rr_ds_rounds.py` - Round-robin rounds tests

## 🔌 FLEX Framework Integration

The system integrates with the **FLEX Framework** for advanced federated learning orchestration:

### What is FLEX?

FLEX is a federated learning framework that provides:
- **FlexPool**: Client-server and P2P pool management
- **FedDataDistribution**: IID and Non-IID (Dirichlet) data partitioning
- **Decorators**: `@aggregate_weights`, `@send_to_server`, etc.
- **Actors**: Client-server architecture components

### Installation

```bash
# Install with FLEX support
pip install -e ".[flex]"

# Or manually
pip install flex-framework flex-trees
```

### FLEX Components Used

- **FlexPool**: Manages client-server communication
- **FedDataDistribution**: Handles data partitioning across clients
- **Decorators**: Simplifies FL task orchestration
- **Actors**: Defines client and server roles

### Without FLEX

The system can run **without FLEX** using simplified orchestration, but you'll miss:
- Advanced data distribution (Dirichlet)
- FlexPool decorators
- Some deployment features

The code gracefully handles missing FLEX with fallbacks and warnings.

## 📜 Scripts

### Hyperparameter Optimization
```bash
# Optimize S6 strategy alpha parameter
python scripts/optimize_s6_alpha_pf.py

# Run optimization pipeline
python scripts/run_optimization.py
```

## 🛠️ Technologies

| Tech | Version | Role |
|------|---------|------|
| Python | 3.8+ | Language |
| NumPy | ≥1.24 | Numerical computation |
| Pandas | ≥2.0 | Data manipulation |
| scikit-learn | ≥1.3 | ML: DecisionTree, metrics |
| PyYAML | ≥6.0 | YAML configuration |
| Streamlit | ≥1.35 | Web UI |
| Plotly | ≥5.20 | Visualizations |
| Optuna | ≥3.5 | Hyperparameter optimization |
| pytest | ≥7.4 | Testing framework |
| **FLEX Framework** | ≥0.1.0 | FL orchestration (FlexPool, decorators) |
| **FLEX Trees** | ≥0.1.0 | Additional dataset adapters |

## 👤 Author

- **Adrián Rodríguez** — Development, integration, Streamlit UI

**Based on:** Proactive Forest (Cepero, 2023)

## 📧 Contact

For questions or suggestions: [Open an issue on GitHub]

---

**Made with ❤️ for Federated Learning research**
