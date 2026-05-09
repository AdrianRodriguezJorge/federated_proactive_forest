# 🌲 Federated Proactive Forest

**Federated Proactive Forest** is an advanced **horizontal federated learning** system that implements and compares **9 aggregation strategies** based on **Proactive Forest** (Cepero, 2023). The project combines a pure FL framework (no external FL library dependencies) with modern interactive interfaces, enabling systematic investigation of how different tree selection and ranking criteria affect performance in distributed non-IID environments.

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 Main Features

- ✅ **9 aggregation strategies** (S1-S7, PW, S9) with varied tree selection and feature exploration criteria
- ✅ **Horizontal Federated Learning** with configurable N clients and unified synchronization
- ✅ **Complete non-IID heterogeneity support** (Dirichlet distributions)
- ✅ **Robust Label Handling**: Unified encoding via `LabelService` to ensure consistency across clients
- ✅ **Weighted Hybrid Prediction**: Optimized local/global voting logic with biased-prediction fixes
- ✅ **Statistical Validation**: Built-in Friedman and Wilcoxon tests for rigorous performance comparison
- ✅ **Modern Web UI**: Streamlit interface with real-time ranking and confusion matrices
- ✅ **Clean Hexagonal Architecture**: Strict separation between Domain, Application, and Infrastructure
- ✅ **Hyperparameter Optimization**: Integrated Optuna support with YAML-based search spaces
- ✅ **Interactive Notebooks**: Complete pipelines for baseline comparison and strategy optimization
- ✅ **FLEX Framework Integration**: Advanced FL orchestration support
- ✅ **Professional Documentation**: Auto-generated documentation site with MkDocs

## 📖 Documentation

The project includes comprehensive documentation generated from the code's docstrings.

*   **Online/Local Page**: Run `mkdocs serve` to view the documentation site.
*   **API Reference**: Detailed description of strategies, models, and services.
*   **Hexagonal Design**: Explanation of the architectural patterns used.
*   **Notebooks**: See `src/interfaces/notebooks/` for interactive experimentation.

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Supported Datasets](#-supported-datasets)
- [Aggregation Strategies](#-aggregation-strategies)
- [System Architecture](#-system-architecture)
- [Streamlit Interface](#-streamlit-interface)
- [CLI](#-cli)
- [Jupyter Notebooks](#-jupyter-notebooks)
- [Advanced Configuration](#-advanced-configuration)
- [Adding New Datasets](#-adding-new-datasets)
- [Statistical Validation](#-statistical-validation)
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

You can use the `FLEXOrchestrator` to run experiments directly from Python code. This is ideal for integration into larger pipelines or automated scripts.

```python
from src.application.orchestrators.fl_orchestrator import FLEXOrchestrator
from src.infrastructure.dataset.dataset_factory import DatasetFactory
from src.domain.services.label_service import LabelService

# 1. Load and prepare dataset
adapter = DatasetFactory.create_adapter({"type": "Iris"})
dataset_split = adapter.load()

# 2. Define experimental configuration
config = {
    "n_clients": 5,
    "distribution": "noniid_dirichlet",
    "alpha": 0.5,
    "strategy": "S7",  # Per-client F1 + PCD
    "model": {
        "n_estimators": 50,
        "alpha": 0.1,
        "bootstrap": True
    },
    "prediction": {
        "local_weight": 0.4,
        "global_weight": 0.6
    }
}

# 3. Initialize and run orchestrator
orchestrator = FLEXOrchestrator(config)
orchestrator.setup_federation(dataset_split)
results = orchestrator.run_federated_round()

# 4. Access results
print(f"✅ Global accuracy: {results.global_accuracy:.4f}")
print(f"✅ Total trees in global model: {results.n_trees_global}")
```

## 📊 Supported Datasets

| Dataset | Samples | Features | Classes | Type | Adapter | FL Time |
|---------|----------|----------|--------|------|-----------|-----------|

| **Iris** | 150 | 4 (num) | 3 | Classification | `iris_adapter.py` | 1-2 min |
| **Car Evaluation** | 1.7K | 6 (cat) | 4 | Classification | `csv_adapter.py` | 1-2 min |
| **Letter Recognition** | 20K | 16 (num) | 26 | Classification | `csv_adapter.py` | 3-5 min |
| **Nursery** | 13K | 8 (cat) | 5 | Classification | `csv_adapter.py` | 2-4 min |
| **Optdigits** | 5.6K | 64 (num) | 10 | Classification | `csv_adapter.py` | 3-6 min |
| **Sonar** | 208 | 60 (num) | 2 | Classification | `csv_adapter.py` | 1-2 min |
| **Spambase** | 4.6K | 57 (num) | 2 | Classification | `csv_adapter.py` | 2-4 min |
| **Vowel** | 990 | 10 (num) | 11 | Classification | `csv_adapter.py` | 1-3 min |
| **Generic CSV** | Variable | Variable | Variable | Any | `csv_adapter.py` | Variable |

### Dataset Preparation


#### Iris & Other Datasets
- Most datasets are already included in the `data/` directory
- Iris can also be loaded from scikit-learn if needed
- No additional download required for included datasets

## 🏆 Aggregation Strategies

The project implements **9 strategies** for tree selection and feature exploration in federated environments:

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
| **S9** | Global Roulette | Statistical | Global attribute roulette (S9_MEAN, S9_WEIGHTED, S9_MEDIAN, S9_CONSENSUS, S9_PROACTIVE_PCD) |

### Weight Configuration (S4, S7, PW)
```yaml
aggregation:
  strategy: s7_perclient_f1_pcd
  f1_weight: 0.7    # Weight for F1-score performance
  pcd_weight: 0.3   # Weight for PCD diversity (automatically 1-f1_weight)
```

### Tree Selection & Synchronization Fixes
The recent refactoring addressed critical issues in tree aggregation:
- **Unified Label Mapping**: Using `LabelService` to ensure all clients and the orchestrator share the same index-to-class mapping, preventing prediction "shifting".
- **Biased Fallback Fix**: Removed fallback logic that caused trees to vote for the majority class of the first client when a label was unknown.
- **Correct Mapping**: Verified tree-to-client origins to ensure `HybridPredictor` correctly excludes a client's own trees from the "global" pool it receives, preventing over-representation.

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

### Global Attribute Roulette (S9)
A radically different approach for low-bandwidth environments:
- **Vector-based aggregation**: Instead of trees, clients exchange **probability vectors** of attributes.
- **Privacy-preserving**: No model weights or trees are sent, only statistical feature importance.
- **Fused Learning**: Clients combine the Global Roulette with their Local Roulette via a $\beta$ balance parameter.
- **Extreme Efficiency**: Reduces communication cost by **>99%** (typical payload < 2KB per round).

```yaml
aggregation:
  strategy: s9_roulette
  variant: S9_CONSENSUS  # MEAN, WEIGHTED, MEDIAN, CONSENSUS, PROACTIVE_PCD
  beta: 0.1             # 0=Adopt global, 1=Stay local
  window_size: 5        # Trees per window
  max_rounds: 20
```
```

## 🏗️ System Architecture

### Architectural Pattern: Hexagonal (Ports & Adapters)

```
src/
├── domain/                    # 📦 Business core (no external dependencies)
│   ├── model/                # ML models: ProactiveForest, strategies
│   │   ├── cpf_implementation/  # CPF algorithm tree components
│   │   ├── proactive_forest.py
│   │   └── progressive_forest.py
│   ├── aggregation/          # Aggregation logic S1-S7 + PW
│   │   ├── strategies/       # S1-S7 and Progressive Windows
│   │   ├── tree_ranker.py    # Tree ranking for S2-S7
│   │   └── aggregation_factory.py
│   ├── metrics/              # Model evaluation (ForestEvaluator)
│   ├── services/             # LabelService (Unified encoding)
│   ├── prediction/           # Weighted Hybrid Prediction
│   ├── update/               # No-Repeat Merge logic
│   ├── config/               # Domain-specific configuration
│   └── dataset/              # Domain dataset abstractions
│
├── application/              # 🎯 Use cases and orchestration
│   ├── orchestrators/        # FLEXOrchestrator, PWOrchestrator
│   ├── commands/             # Command pattern implementations
│   └── hyperparam_optimizer.py  # Optuna-based optimization
│
├── infrastructure/           # 🔌 Concrete adapters
│   ├── dataset/              # Adapters: CSV, FlexTrees, DatasetFactory
│   ├── flex/                 # FLEX Framework primitives
│   ├── persistence/          # Results logging and CSV storage
│   ├── models/               # Model adapters (RandomForestAdapter)
│   ├── logging/              # System logging infrastructure
│   ├── metrics/              # Infrastructure-level metric collection
│   ├── privacy/              # Privacy-preserving mechanisms
│   └── serialization/        # State serialization/deserialization
│
└── interfaces/               # 🎨 User interfaces
    ├── cli/                  # Command-line interface (main.py)
    └── streamlit/            # Streamlit multi-page application
```

### Key Components

- **Proactive Forest (CPF)**: Ensemble algorithm with dynamic probability adjustment
- **FLEXOrchestrator**: Main orchestrator for federated learning rounds with FLEX support
- **PWOrchestrator**: Specialized orchestrator for Progressive Windows strategy
- **LabelService**: Unified label service for consistent encoding across federation
- **Early Stopping**: Automatic convergence detection in progressive training
- **IDiversityService**: Standardized PCD (Pairwise Classifier Disagreement) measure
- **Hybrid Prediction**: Configurable weighted voting between local and global models
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
- **🎰 Roulette Evolution**: Dynamic heatmap of attribute importance over rounds (S9 only)

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
  type: "Iris"
  file_path: "data/iris.csv"

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
- `exp_s9_consensus.yaml` - Global Attribute Roulette (Consensus variant)
- `exp_s9_proactive_pcd.yaml` - Diversity-weighted Roulette (Proactive PCD)
- `exp_s9_mean.yaml`, `exp_s9_median.yaml`, `exp_s9_weighted.yaml` - Other S9 variants

## 📓 Jupyter Notebooks

The project includes interactive notebooks for detailed analysis and custom optimization located in `src/interfaces/notebooks/`:

### Baseline Comparison
- `baselines/centralized_baselines_comparison.ipynb`: Compares federated strategies against centralized Random Forest and Proactive Forest baselines.

### Strategy Optimization
Dedicated notebooks for tuning each strategy using Optuna:
- `optimization/optuna_s1.ipynb` to `optuna_s7.ipynb`
- `optimization/optuna_pw.ipynb` (Progressive Windows)
- `optimization/optuna_s9.ipynb` (Global Roulette variants)

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

### Step 2: Register in Dataset Factory
Update `DATASET_METADATA` in `src/infrastructure/dataset/dataset_factory.py`:
```python
"new_dataset": {
    "target_column": "target",
    "sep": ",",
    "file_path": "data/new_dataset.csv"
}
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
| **Low Bandwidth** | **S9 (Roulette)** | **Extreme communication efficiency** |
| High Heterogeneity | S9_CONSENSUS | Weights clients by performance |
| Maximum Selection Diversity | S9_PROACTIVE_PCD | Weights clients by PCD diversity |

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
- `test_exhaustive_domain_pytest.py` - Exhaustive validation for domain core logic
- `test_label_normalization_system.py` - Cross-client label normalization and translation
- `test_label_service.py` - Unit tests for LabelService
- `test_pw_orchestrator.py` - Progressive Windows orchestrator testing
- `test_iris_adapter_scaler_isolation.py` - Scaler isolation tests for Dataset Adapters

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

## 📊 Statistical Validation

To ensure the reliability of the comparative analysis, the project includes a specialized script for non-parametric statistical testing.

### Friedman & Wilcoxon Tests
Located in `scripts/Friedman_test_new_results.py`, this script performs:
1.  **Friedman Test**: Determines if there are globally significant differences between the 8 strategies and the standalone Proactive Forest (PF) baseline.
2.  **Wilcoxon Post-hoc**: Conducts pairwise comparisons (PF vs each S1-S7/PW strategy) with **Bonferroni correction**.
3.  **Maximum Impact Ranking**: Identifies which datasets show the largest performance gain when moving from standalone to federated models.

```bash
# Run statistical analysis on current results
python scripts/Friedman_test_new_results.py
```

## 📜 Scripts

The project includes several utilities in the `scripts/` directory for experimentation and analysis:

### Benchmarking & Master Experiments
Run comprehensive benchmarks across all datasets and strategies.
```bash
# Run the Master Experiment (compares all 8 strategies + S9 variants across all datasets)
python master_experiment.py
```
*Results are saved automatically to `results_master.csv` with atomic progress tracking.*

```bash
# Run S9 specific benchmark (Nursery/Iris)
python scripts/run_s9_benchmark.py
```

### Hyperparameter Optimization
Using **Optuna**, you can optimize strategy-specific parameters. The search spaces are defined in `configs/optimization/search_spaces.yaml`.
```bash
# Run optimization via script
python scripts/run_optimization.py --strategy S7 --dataset letter --n_trials 50
```
*You can also use the specialized notebooks in `src/interfaces/notebooks/optimization/` for interactive tuning.*

### Analysis Utilities
```bash
# Run statistical analysis (Friedman & Wilcoxon) on results
python scripts/Friedman_test_new_results.py

# Compare weighted vs uniform voting logic
python scripts/weighted_vs_uniform.py

# Rerun CLI with last used configuration (from configs/last_config.json)
python scripts/run_cli_last_config.py
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
