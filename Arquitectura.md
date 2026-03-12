
ARQUITECTURA DE PROYECTO — v3.0
Federated Proactive Forest con FLEX

v3: Dataset NSL-KDD, DatasetAdapter intercambiable, interfaz Streamlit
con visualización de rankings, métricas completas y configuración total

Framework FL	Modelo	Dataset	Estrategias	UI
FLEX+FLEX-Trees	Proactive Forest	NSL-KDD	7 estrategias	Streamlit
Ciclo vida	Predicción	Datasets alt.	Lenguaje	Tipo FL
Ronda única	Híbrida pond.	Plug-and-play	Python 3.10+	Horizontal HFL

Arquitectura Senior — Sistemas Distribuidos & Federated Learning
 
🔄 Resumen de Cambios v2 → v3
Dos incorporaciones independientes que no alteran la lógica FL ya definida en v2:

Dimensión	v2 (anterior)	v3 (actual)
Dataset	ILDP embebido de flex-trees — cambiar requería modificar código	DatasetAdapter intercambiable: cualquier CSV con config YAML; NSL-KDD como implementación de referencia
Preprocesamiento	Implícito dentro del código de entrenamiento	DatasetAdapter.load() retorna X_train, X_test, y_train, y_test ya preprocesados; encoders y scalers configurables por YAML
Soporte multi-dataset	No existía abstraído	IDatasetAdapter como puerto hexagonal; NslKddAdapter y FlexTreesAdapter como implementaciones concretas
Interfaz de usuario	Sólo CLI y notebooks	Interfaz Streamlit completa: configuración de experimento, ejecución, ranking de árboles, métricas de modelos
Ranking visual	No existía	Panel de ranking con marcado de seleccionados, color por cliente, agrupación per-client para S5–S7
Panel de métricas	Sólo logs MLflow	Sección de métricas por cliente o modelo global: accuracy, matriz de confusión, precisión, recall, F1, Macro-F1, PCD, tamaño del bosque
Configuración UI	No existía	Streamlit sidebar expone todos los hiperparámetros: n_clients, strategy, n_estimators, α, pesos, distribution, etc.

🔍 FASE 1 — Análisis del Contexto (v3)

Dimensión	Valor / Decisión
Tipo de FL	Horizontal (HFL) — mismos features en todos los clientes, datos locales distintos
Dataset primario	NSL-KDD (intrusion detection): 151 165 instancias, 41 features (3 categóricas), target float64 class
Cambio de dataset	Implementar IDatasetAdapter → DatasetConfig en YAML → sin tocar código FL
Entorno	Simulado con FLEX FlexPool; adaptadores listos para migración a gRPC real
Escala	2–50 clientes, ronda única, 50–500 árboles por bosque local
Estrategias de agregación	7 estrategias: S1 simple + S2/S3/S4 global + S5/S6/S7 per-client
UI	Streamlit: sidebar de configuración, ejecución, ranking de árboles, métricas completas
Ciclo	Ronda única con reinicio total de modelos — stateless entre ejecuciones
Lenguaje	Python 3.10+; scikit-learn para árboles; pandas/numpy para preprocesamiento
Privacidad	Fase 1: sin mecanismos. DP sobre metadatos planificado para Fase 2

 
📁 FASE 3 — Estructura de Carpetas (v3)

Cambios estructurales v3: (1) domain/dataset/ con IDatasetAdapter; (2) infrastructure/dataset/ con NslKddAdapter y FlexTreesAdapter; (3) domain/metrics/ con ForestEvaluator; (4) interfaces/streamlit/ con 4 páginas y componentes reutilizables; (5) configs/datasets/ con YAML por dataset; (6) data/nslkdd/ para el CSV.

federated-proactive-forest/
├── src/
│   ├── domain/                              # Núcleo — NUNCA importa FLEX ni Streamlit
│   │   ├── model/
│   │   │   ├── base_forest.py               # ABCForest: fit/predict/get_trees/from_trees
│   │   │   ├── proactive_forest.py          # Algoritmo α-ruleta (Cepero 2023)
│   │   │   ├── progressive_forest.py        # Detención por episodios (convergencia 0.002)
│   │   │   └── random_forest.py             # Wrapper intercambiable
│   │   ├── aggregation/
│   │   │   ├── base_strategy.py             # IAggregationStrategy
│   │   │   ├── strategies/
│   │   │   │   ├── s1_simple_pool.py
│   │   │   │   ├── s2_global_accuracy.py
│   │   │   │   ├── s3_global_f1.py
│   │   │   │   ├── s4_global_f1_pcd.py
│   │   │   │   ├── s5_perclient_accuracy.py
│   │   │   │   ├── s6_perclient_f1.py
│   │   │   │   └── s7_perclient_f1_pcd.py
│   │   │   ├── aggregation_factory.py       # Config.strategy → clase concreta
│   │   │   └── tree_ranker.py               # 3 criterios: ACCURACY | MACRO_F1 | F1_PCD
│   │   ├── dataset/                         # ★ NUEVO en v3
│   │   │   ├── __init__.py
│   │   │   ├── base_adapter.py              # IDatasetAdapter (puerto hexagonal)
│   │   │   └── dataset_config.py            # DatasetConfig: nombre, ruta, features, target...
│   │   ├── metadata/
│   │   │   ├── client_metadata.py           # Dataclass: accuracy, macro_f1, pcd, n_trees
│   │   │   └── pcd_calculator.py            # Pair Classifier Disagreement
│   │   ├── prediction/
│   │   │   └── hybrid_predictor.py          # Votación ponderada local + global
│   │   ├── update/
│   │   │   └── client_updater.py            # No-Repeat Merge
│   │   ├── metrics/                         # ★ NUEVO en v3 (métricas para Streamlit)
│   │   │   ├── __init__.py
│   │   │   └── forest_evaluator.py          # accuracy, conf_matrix, prec, recall, F1, Macro-F1, PCD
│   │   ├── meta_learning/
│   │   │   ├── dataset_descriptor.py
│   │   │   └── alpha_selector.py
│   │   └── ports/
│   │       ├── forest_repository.py
│   │       ├── metadata_repository.py
│   │       ├── data_repository.py
│   │       └── metrics_port.py
│   │
│   ├── application/
│   │   ├── commands/
│   │   │   ├── train_command.py
│   │   │   ├── aggregate_command.py
│   │   │   ├── update_client_command.py
│   │   │   └── predict_command.py
│   │   └── fl_orchestrator.py               # Ronda única: train→collect→agg→update→predict
│   │
│   ├── infrastructure/
│   │   ├── dataset/                         # ★ NUEVO en v3
│   │   │   ├── __init__.py
│   │   │   ├── nslkdd_adapter.py            # NslKddAdapter — dataset de referencia
│   │   │   └── flextrees_adapter.py         # FlexTreesAdapter (ILDP, Adult, etc.)
│   │   ├── flex/
│   │   │   ├── flex_pool_factory.py
│   │   │   ├── flex_train_pf.py
│   │   │   ├── flex_collect_trees_pf.py
│   │   │   ├── flex_aggregate_pf.py
│   │   │   ├── flex_deploy_model_pf.py
│   │   │   ├── flex_update_client_pf.py
│   │   │   └── flex_evaluate_pf.py
│   │   ├── serialization/
│   │   │   └── tree_serializer.py
│   │   ├── persistence/
│   │   │   ├── local_forest_repo.py
│   │   │   └── local_metadata_repo.py
│   │   ├── logging/
│   │   │   ├── structured_logger.py
│   │   │   └── mlflow_logger.py
│   │   └── privacy/
│   │       └── dp_noise_injector.py
│   │
│   └── interfaces/
│       ├── cli/
│       │   └── main.py
│       ├── streamlit/                       # ★ NUEVO en v3 — independiente del resto
│       │   ├── __init__.py
│       │   ├── app.py                       # Entrypoint: streamlit run src/interfaces/streamlit/app.py
│       │   ├── pages/
│       │   │   ├── 01_config.py             # Sidebar: todos los hiperparámetros
│       │   │   ├── 02_run.py                # Ejecutar experimento + progress bar
│       │   │   ├── 03_ranking.py            # Ranking de árboles con marcado de seleccionados
│       │   │   └── 04_metrics.py            # Métricas por cliente o modelo global
│       │   ├── components/
│       │   │   ├── ranking_table.py         # Componente: tabla de ranking con colores
│       │   │   ├── metrics_panel.py         # Componente: métricas + matriz de confusión
│       │   │   └── forest_summary.py        # Componente: resumen del bosque (tamaño, PCD)
│       │   └── state/
│       │       └── session_state.py         # Gestión de st.session_state centralizada
│       └── notebooks/
│           ├── 01_data_exploration.ipynb
│           ├── 02_local_proactive_forest.ipynb
│           ├── 03_strategy_comparison.ipynb
│           └── 04_full_fl_experiment.ipynb
│
├── configs/
│   ├── base.yaml
│   ├── datasets/                            # ★ NUEVO en v3
│   │   ├── nslkdd.yaml                      # Config completa del dataset NSL-KDD
│   │   ├── ildp.yaml
│   │   └── adult.yaml
│   └── experiments/
│       ├── exp001_s1_simple.yaml
│       ├── exp002_s2_global_accuracy.yaml
│       ├── exp003_s3_global_f1.yaml
│       ├── exp004_s4_global_f1_pcd.yaml
│       ├── exp005_s5_perclient_accuracy.yaml
│       ├── exp006_s6_perclient_f1.yaml
│       └── exp007_s7_perclient_f1_pcd.yaml
│
├── data/                                    # ★ NUEVO en v3
│   └── nslkdd/
│       ├── KDDTrain+.csv                    # Dataset de entrenamiento NSL-KDD
│       ├── KDDTest+.csv                     # Dataset de test NSL-KDD
│       └── README.md                        # Fuente, licencia, descripción de columnas
│
├── tests/
│   ├── unit/domain/
│   │   ├── test_proactive_forest.py
│   │   ├── test_progressive_forest.py
│   │   ├── test_pcd_calculator.py
│   │   ├── test_tree_ranker.py
│   │   ├── test_all_7_strategies.py
│   │   ├── test_client_updater.py
│   │   ├── test_hybrid_predictor.py
│   │   ├── test_forest_evaluator.py         # ★ NUEVO en v3
│   │   └── test_nslkdd_adapter.py           # ★ NUEVO en v3
│   ├── integration/
│   │   ├── test_flex_adapters.py
│   │   └── test_full_fl_round.py
│   └── fixtures/
│       ├── sample_datasets.py
│       └── nslkdd_sample.csv               # 500 filas para tests rápidos
│
├── scripts/
│   ├── run_experiment.sh
│   ├── compare_all_strategies.py
│   └── download_nslkdd.sh                  # Descarga NSL-KDD de la fuente oficial
│
├── docker/
│   ├── Dockerfile.server
│   ├── Dockerfile.client
│   ├── Dockerfile.streamlit                 # ★ NUEVO en v3
│   └── docker-compose.yml
│
├── .github/workflows/ci.yml
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md

 ⚙️ FASE 4 — Archivos Clave con Contenido Base (v3)
Esta sección documenta únicamente los archivos nuevos o modificados en v3. Los archivos ya documentados en v2 (ProactiveForest, TreeRanker, IAggregationStrategy, S1–S7, ClientUpdater, HybridPredictor) siguen vigentes sin cambios.

4.1 IDatasetAdapter — Puerto hexagonal para datasets
# src/domain/dataset/base_adapter.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

@dataclass
class DatasetSplit:
    """Resultado estandarizado de cualquier DatasetAdapter."""
    X_train: np.ndarray
    X_test:  np.ndarray
    y_train: np.ndarray
    y_test:  np.ndarray
    feature_names: List[str]
    class_names: List[str]
    dataset_name: str

class IDatasetAdapter(ABC):
    """
    Puerto hexagonal para datasets.
    Implementar este puerto = el dataset es compatible con todo el proyecto.
    El adaptador es responsable de:
      - Cargar el CSV / fuente
      - Codificar variables categóricas
      - Escalar features numéricas (si aplica)
      - Retornar X_train, X_test, y_train, y_test listos para sklearn
    """
    @abstractmethod
    def load(self) -> DatasetSplit: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def n_classes(self) -> int: ...

4.2 DatasetConfig YAML — NSL-KDD
# configs/datasets/nslkdd.yaml
dataset:
  name: nslkdd
  adapter: nslkdd                  # Mapea a NslKddAdapter en infra/dataset/
  train_path: "data/nslkdd/KDDTrain+.csv"
  test_path:  "data/nslkdd/KDDTest+.csv"
  target_column: "class"
  # Columnas categóricas a codificar con OrdinalEncoder
  categorical_features:
    - protocol_type                # tcp, udp, icmp
    - service                      # http, ftp, smtp, ...
    - flag                         # SF, S0, REJ, ...
  # Columnas numéricas a escalar (StandardScaler)
  scale_features: true
  scaler: standard                 # standard | minmax | none
  # Semilla para split reproducible (si el CSV no tiene split propio)
  split_seed: 42
  test_size: 0.2                   # Sólo si el CSV no tiene test separado
  # NSL-KDD ya tiene train/test separados → ignorar test_size

4.3 NslKddAdapter — Adaptador concreto NSL-KDD
# src/infrastructure/dataset/nslkdd_adapter.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from ...domain.dataset.base_adapter import IDatasetAdapter, DatasetSplit

# Columnas en el orden del CSV NSL-KDD (41 features + target)
COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","class"
]
CAT_COLS  = ["protocol_type", "service", "flag"]
FEAT_COLS = [c for c in COLUMNS if c != "class"]

class NslKddAdapter(IDatasetAdapter):
    """
    Adaptador para NSL-KDD (KDDTrain+.csv / KDDTest+.csv).
    Maneja: OrdinalEncoder para 3 cols categóricas + StandardScaler.
    """
    def __init__(self, train_path: str, test_path: str, scale: bool = True):
        self.train_path = train_path
        self.test_path  = test_path
        self.scale      = scale
        self._encoder   = OrdinalEncoder(handle_unknown="use_encoded_value",
                                          unknown_value=-1)
        self._scaler    = StandardScaler() if scale else None

    @property
    def name(self) -> str: return "nslkdd"

    @property
    def n_classes(self) -> int: return len(self._class_names_)

    def _read(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, header=None, names=COLUMNS)
        # El CSV original incluye columna extra de dificultad — eliminar si existe
        if df.shape[1] == 43:
            df = df.iloc[:, :42]
        return df

    def load(self) -> DatasetSplit:
        train_df = self._read(self.train_path)
        test_df  = self._read(self.test_path)

        # Encode categóricas
        train_df[CAT_COLS] = self._encoder.fit_transform(train_df[CAT_COLS])
        test_df[CAT_COLS]  = self._encoder.transform(test_df[CAT_COLS])

        X_train = train_df[FEAT_COLS].values.astype(np.float64)
        X_test  = test_df[FEAT_COLS].values.astype(np.float64)
        y_train = train_df["class"].values.astype(np.int64)
        y_test  = test_df["class"].values.astype(np.int64)

        # Escalar
        if self._scaler:
            X_train = self._scaler.fit_transform(X_train)
            X_test  = self._scaler.transform(X_test)

        classes = sorted(set(y_train) | set(y_test))
        self._class_names_ = [str(c) for c in classes]

        return DatasetSplit(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            feature_names=FEAT_COLS,
            class_names=self._class_names_,
            dataset_name=self.name,
        )

4.4 ForestEvaluator — Todas las métricas para Streamlit
# src/domain/metrics/forest_evaluator.py
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score,
)
from ..metadata.pcd_calculator import PCDCalculator

@dataclass
class ForestReport:
    """Todas las métricas que muestra el panel Streamlit de métricas."""
    accuracy:          float
    macro_f1:          float
    macro_precision:   float
    macro_recall:      float
    per_class_f1:      Dict[str, float]   # {class_name: f1}
    per_class_prec:    Dict[str, float]
    per_class_recall:  Dict[str, float]
    confusion_matrix:  np.ndarray
    pcd:               float              # Pair Classifier Disagreement
    forest_size:       int                # Número de árboles
    class_names:       List[str]

class ForestEvaluator:
    """
    Calcula todas las métricas para el panel de métricas de Streamlit.
    Funciona para cualquier bosque (local o global) sobre cualquier split.
    """
    @staticmethod
    def evaluate(forest, X: np.ndarray, y: np.ndarray,
                 class_names: List[str]) -> ForestReport:
        trees = forest.get_trees()
        y_pred = forest.predict(X)

        labels = list(range(len(class_names)))

        per_f1   = f1_score(y, y_pred, labels=labels, average=None, zero_division=0)
        per_prec = precision_score(y, y_pred, labels=labels, average=None, zero_division=0)
        per_rec  = recall_score(y, y_pred, labels=labels, average=None, zero_division=0)

        pcd = PCDCalculator.forest_pcd(trees, X) if len(trees) > 1 else 0.0

        return ForestReport(
            accuracy=float(accuracy_score(y, y_pred)),
            macro_f1=float(f1_score(y, y_pred, average="macro", zero_division=0)),
            macro_precision=float(precision_score(y, y_pred, average="macro", zero_division=0)),
            macro_recall=float(recall_score(y, y_pred, average="macro", zero_division=0)),
            per_class_f1={cn: float(v) for cn, v in zip(class_names, per_f1)},
            per_class_prec={cn: float(v) for cn, v in zip(class_names, per_prec)},
            per_class_recall={cn: float(v) for cn, v in zip(class_names, per_rec)},
            confusion_matrix=confusion_matrix(y, y_pred, labels=labels),
            pcd=pcd,
            forest_size=len(trees),
            class_names=class_names,
        )

4.5 Streamlit app.py — Entrypoint
# src/interfaces/streamlit/app.py
"""
Entrypoint de la interfaz Streamlit.
Ejecutar: streamlit run src/interfaces/streamlit/app.py
"""
import streamlit as st
from .state.session_state import init_session_state

st.set_page_config(
    page_title="Federated Proactive Forest",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()

st.title("🌲 Federated Proactive Forest — Panel de Control")
st.markdown("""
Sistema de Aprendizaje Federado basado en **Proactive Forest** (Cepero, 2023)
con **7 estrategias de agregación**, dataset **NSL-KDD** y análisis completo
de rankings y métricas.

Usa el menú lateral para navegar entre las secciones.
""")

4.6 Página 01_config.py — Configuración de hiperparámetros
# src/interfaces/streamlit/pages/01_config.py
"""
Página 1 — Configuración del experimento.
Todos los hiperparámetros importantes expuestos como widgets Streamlit.
"""
import streamlit as st
from ..state.session_state import save_config

st.header("⚙️ Configuración del Experimento")

# ── Dataset ────────────────────────────────────────────────────────────────────
st.subheader("Dataset")
col1, col2 = st.columns(2)
with col1:
    dataset = st.selectbox("Dataset", ["nslkdd", "ildp", "adult", "custom_csv"])
    if dataset == "custom_csv":
        train_path = st.text_input("Ruta CSV de entrenamiento")
        test_path  = st.text_input("Ruta CSV de test (dejar vacío = split automático)")
        target_col = st.text_input("Columna target", value="class")
with col2:
    scale_features = st.checkbox("Escalar features (StandardScaler)", value=True)
    scaler_type    = st.selectbox("Tipo de scaler", ["standard", "minmax", "none"],
                                  disabled=not scale_features)

# ── Federación ─────────────────────────────────────────────────────────────────
st.subheader("Federación")
col3, col4 = st.columns(2)
with col3:
    n_clients    = st.slider("Número de clientes", 2, 20, 5)
    distribution = st.selectbox("Distribución de datos",
                                ["iid", "noniid_dirichlet", "noniid_manual"])
    if distribution == "noniid_dirichlet":
        dirichlet_alpha = st.slider("Parámetro Dirichlet α", 0.1, 5.0, 0.5, 0.1)
with col4:
    client_fraction = st.slider("Fracción de clientes por ronda", 0.1, 1.0, 1.0, 0.1)

# ── Modelo ─────────────────────────────────────────────────────────────────────
st.subheader("Modelo (ProactiveForest)")
col5, col6 = st.columns(2)
with col5:
    model_type     = st.selectbox("Tipo de modelo", ["proactive_forest","random_forest"])
    n_estimators   = st.slider("Árboles por cliente (n_estimators)", 10, 500, 100, 10)
    alpha_pf       = st.slider("Parámetro diversidad α (Proactive)", 0.1, 0.5, 0.1, 0.05,
                               disabled=(model_type != "proactive_forest"))
with col6:
    use_progressive = st.checkbox("Activar Progressive Forest stopping", value=True)
    if use_progressive:
        episode_size  = st.number_input("Tamaño del episodio", 3, 20, 5)
        conv_thresh   = st.number_input("Umbral de convergencia", 0.0001, 0.01, 0.002,
                                        format="%.4f")
        consec_ep     = st.number_input("Episodios consecutivos para parar", 1, 5, 2)
    use_meta_learning = st.checkbox("Meta-aprendizaje para α automático", value=False)

# ── Agregación ─────────────────────────────────────────────────────────────────
st.subheader("Estrategia de Agregación")
strategy = st.selectbox("Estrategia", {
    "s1_simple_pool":        "S1 — Simple Pool (todos los árboles)",
    "s2_global_accuracy":    "S2 — Global, orden por Accuracy",
    "s3_global_f1":          "S3 — Global, orden por Macro-F1",
    "s4_global_f1_pcd":      "S4 — Global, orden por α·F1 + β·PCD",
    "s5_perclient_accuracy": "S5 — Per-Client, orden por Accuracy",
    "s6_perclient_f1":       "S6 — Per-Client, orden por Macro-F1",
    "s7_perclient_f1_pcd":   "S7 — Per-Client, orden por α·F1 + β·PCD",
}.keys(), format_func=lambda x: {...}[x])  # ver implementación completa

if strategy in ["s4_global_f1_pcd", "s7_perclient_f1_pcd"]:
    col7, col8 = st.columns(2)
    with col7:
        f1_weight  = st.slider("Peso F1 (α)", 0.0, 1.0, 0.5, 0.05)
    with col8:
        pcd_weight = 1.0 - f1_weight
        st.metric("Peso PCD (β)", f"{pcd_weight:.2f}")

# ── Predicción ────────────────────────────────────────────────────────────────
st.subheader("Predicción Híbrida")
col9, col10 = st.columns(2)
with col9:
    local_w  = st.slider("Peso votos locales", 0.0, 1.0, 0.4, 0.05)
with col10:
    global_w = 1.0 - local_w
    st.metric("Peso votos globales", f"{global_w:.2f}")

# ── Reproducibilidad ──────────────────────────────────────────────────────────
st.subheader("Reproducibilidad")
seed = st.number_input("Semilla global", 0, 99999, 42)

if st.button("💾 Guardar configuración", type="primary"):
    save_config(locals())
    st.success("Configuración guardada. Ve a '▶️ Ejecutar' para correr el experimento.")

4.7 Página 02_run.py — Ejecución del experimento
# src/interfaces/streamlit/pages/02_run.py
"""
Página 2 — Ejecutar experimento y mostrar progreso.
"""
import streamlit as st
from ..state.session_state import get_config, save_results
from ....application.fl_orchestrator import FLEXOrchestrator

st.header("▶️ Ejecutar Experimento")

cfg = get_config()
if not cfg:
    st.warning("Primero configura el experimento en la página '⚙️ Configuración'.")
    st.stop()

# Mostrar resumen de config antes de ejecutar
with st.expander("📋 Configuración activa"):
    st.json(cfg)

if st.button("🚀 Ejecutar ronda federada", type="primary"):
    progress = st.progress(0, text="Inicializando...")
    status   = st.empty()
    log_area = st.expander("📜 Log de ejecución", expanded=True)
    logs     = []

    def on_step(step: str, pct: int, detail: str = ""):
        progress.progress(pct, text=step)
        status.info(f"**{step}** {detail}")
        logs.append(f"[{pct}%] {step} {detail}")
        log_area.code("\n".join(logs))

    try:
        orchestrator = FLOrchestrator.from_config(cfg, step_callback=on_step)
        results      = orchestrator.run()
        save_results(results)
        progress.progress(100, text="✅ Completado")
        st.success("Ronda federada completada. Explora los resultados en las pestañas 3 y 4.")
        # Métricas rápidas
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy global", f"{results.global_accuracy:.4f}")
        col2.metric("Macro-F1 global", f"{results.global_macro_f1:.4f}")
        col3.metric("Árboles en bosque global", results.n_trees_global)
    except Exception as e:
        st.error(f"Error durante la ejecución: {e}")
        raise

4.8 Página 03_ranking.py — Ranking de árboles
# src/interfaces/streamlit/pages/03_ranking.py
"""
Página 3 — Ranking de árboles.
Muestra todos los árboles, marcando los seleccionados y el cliente de origen.
"""
import streamlit as st
import pandas as pd
from ..state.session_state import get_results
from ..components.ranking_table import render_ranking_table

st.header("🏆 Ranking de Árboles")

results = get_results()
if not results:
    st.warning("Ejecuta primero el experimento."); st.stop()

strategy_id = results.strategy_id
is_per_client = strategy_id.startswith("s5") or                 strategy_id.startswith("s6") or                 strategy_id.startswith("s7")

st.info(f"Estrategia activa: **{strategy_id}** | "
        f"Total árboles candidatos: **{len(results.all_tree_entries)}** | "
        f"Árboles seleccionados: **{results.n_trees_global}**")

# ── Leyenda ───────────────────────────────────────────────────────────────────
st.markdown("**Leyenda:**")
col1, col2, col3 = st.columns(3)
col1.success("🟢 Seleccionado — incluido en el bosque global")
col2.error("🔴 No seleccionado — descartado por Progressive Forest")
col3.info("Cliente de origen — identificado por color")

# ── Filtros ──────────────────────────────────────────────────────────────────
clients = sorted(set(e.client_id for e in results.all_tree_entries))
selected_clients = st.multiselect("Filtrar por cliente", clients, default=clients)
show_only_selected = st.checkbox("Mostrar sólo árboles seleccionados", value=False)

# ── Tabla de ranking ──────────────────────────────────────────────────────────
render_ranking_table(
    entries=results.all_tree_entries,
    selected_local_ids=results.selected_ids,
    selected_clients=selected_clients,
    show_only_selected=show_only_selected,
    is_per_client=is_per_client,
    n_clients=len(clients),
)

4.9 Componente ranking_table.py — Tabla con colores y selección
# src/interfaces/streamlit/components/ranking_table.py
"""
Componente reutilizable: tabla de ranking con colores por cliente
y marcado de árboles seleccionados.
"""
import streamlit as st
import pandas as pd
from typing import Dict, List

# Paleta de colores para clientes (hasta 20)
CLIENT_COLORS = [
    "#AED6F1","#A9DFBF","#F9E79F","#F5CBA7","#D7BDE2",
    "#FADBD8","#D5F5E3","#FCF3CF","#D6EAF8","#FDFEFE",
]

def render_ranking_table(entries, selected_local_ids: Dict[str, List[int]],
                         selected_clients, show_only_selected,
                         is_per_client: bool, n_clients: int):
    """
    Renderiza la tabla de ranking.
    - Columnas: Rank | ID_árbol | Cliente | Accuracy | Macro-F1 | PCD | Score | Estado
    - Verde = seleccionado; rojo = no seleccionado
    - Para is_per_client: agrupa por cliente, luego ordena por score dentro de cada grupo
    """
    client_list  = sorted(set(e.client_id for e in entries))
    color_map    = {cid: CLIENT_COLORS[i % len(CLIENT_COLORS)]
                    for i, cid in enumerate(client_list)}

    def is_selected(entry) -> bool:
        ids = selected_local_ids.get(entry.client_id, [])
        return entry.tree_local_id in ids

    rows = []
    for rank, entry in enumerate(entries, 1):
        if entry.client_id not in selected_clients:
            continue
        sel = is_selected(entry)
        if show_only_selected and not sel:
            continue
        rows.append({
            "Rank":      rank,
            "ID árbol":  f"{entry.client_id}_tree{entry.tree_local_id}",
            "Cliente":   entry.client_id,
            "Accuracy":  f"{entry.accuracy:.4f}",
            "Macro-F1":  f"{entry.macro_f1:.4f}",
            "PCD":       f"{entry.pcd_contribution:.4f}",
            "Score":     f"{entry.score:.4f}" if hasattr(entry, "score") else "—",
            "Estado":    "✅ Seleccionado" if sel else "❌ Descartado",
            "_selected": sel,
            "_client":   entry.client_id,
        })

    if not rows:
        st.warning("No hay árboles que mostrar con los filtros actuales.")
        return

    df = pd.DataFrame(rows)

    # Si es per-client, reordenar: primero por cliente, luego por rank original
    if is_per_client:
        df = df.sort_values(["Cliente", "Rank"]).reset_index(drop=True)

    def style_row(row):
        bg_client = color_map.get(row["_client"], "#FFFFFF")
        bg_status = "#D5F5E3" if row["_selected"] else "#FADBD8"
        # Mezcla: fondo del cliente en columna Cliente; fondo de estado en columna Estado
        n = len(row)
        styles = [f"background-color: {bg_client}; color: #1a1a1a"] * n
        estado_idx = list(df.columns).index("Estado")
        cliente_idx = list(df.columns).index("Cliente")
        styles[estado_idx]  = f"background-color: {bg_status}; font-weight: bold"
        styles[cliente_idx] = f"background-color: {bg_client}; font-weight: bold"
        return styles

    display_cols = ["Rank","ID árbol","Cliente","Accuracy","Macro-F1","PCD","Score","Estado"]
    styled = df[display_cols + ["_selected","_client"]] \
               .style.apply(style_row, axis=1)
    st.dataframe(styled[display_cols], use_container_width=True, height=600)

    # Estadísticas rápidas
    n_sel = sum(1 for r in rows if r["_selected"])
    n_tot = len(rows)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total mostrados", n_tot)
    col2.metric("Seleccionados", n_sel, delta=f"{100*n_sel/n_tot:.1f}%")
    col3.metric("Descartados por Progressive", n_tot - n_sel)

4.10 Página 04_metrics.py — Panel de métricas
# src/interfaces/streamlit/pages/04_metrics.py
"""
Página 4 — Métricas del modelo.
Seleccionar cliente o modelo global; ver todas las métricas.
"""
import streamlit as st
import pandas as pd
import numpy as np
from ..state.session_state import get_results
from ..components.metrics_panel import render_metrics_panel

st.header("📊 Métricas del Modelo")

results = get_results()
if not results:
    st.warning("Ejecuta primero el experimento."); st.stop()

# Selector de modelo
options  = ["🌐 Modelo Global"] + [f"👤 Cliente {c}" for c in results.client_ids]
selected = st.selectbox("Seleccionar modelo a analizar", options)

# Determinar qué ForestReport mostrar
if selected == "🌐 Modelo Global":
    report = results.global_report
    title  = "Modelo Global (Bosque Federado)"
else:
    client_id = selected.replace("👤 Cliente ", "")
    report    = results.client_reports[client_id]
    title     = f"Modelo Local — Cliente {client_id}"

st.subheader(f"📋 {title}")
render_metrics_panel(report)

4.11 Componente metrics_panel.py — KPIs + matriz + barras por clase
# src/interfaces/streamlit/components/metrics_panel.py
"""
Componente: panel de métricas completo para un ForestReport.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from ....domain.metrics.forest_evaluator import ForestReport

def render_metrics_panel(report: ForestReport):
    """Renderiza todas las métricas de un ForestReport."""

    # ── KPIs principales ─────────────────────────────────────────────────────
    st.markdown("#### Métricas Globales")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy",       f"{report.accuracy:.4f}")
    col2.metric("Macro-F1",       f"{report.macro_f1:.4f}")
    col3.metric("Macro Precision",f"{report.macro_precision:.4f}")
    col4.metric("Macro Recall",   f"{report.macro_recall:.4f}")
    col5.metric("PCD (diversidad)",f"{report.pcd:.4f}")

    col6, col7 = st.columns(2)
    col6.metric("Tamaño del bosque", report.forest_size,
                help="Número de árboles en el bosque (tras Progressive stopping si aplica)")
    col7.metric("Número de clases",  len(report.class_names))

    st.divider()

    # ── Matriz de confusión ───────────────────────────────────────────────────
    st.markdown("#### Matriz de Confusión")
    cm = report.confusion_matrix
    fig_cm = px.imshow(
        cm, text_auto=True, aspect="auto",
        x=report.class_names, y=report.class_names,
        labels={"x":"Predicho","y":"Real","color":"Instancias"},
        color_continuous_scale="Blues",
        title="Matriz de Confusión"
    )
    fig_cm.update_layout(height=400)
    st.plotly_chart(fig_cm, use_container_width=True)

    # ── Métricas por clase ───────────────────────────────────────────────────
    st.markdown("#### Métricas por Clase")
    per_class_df = pd.DataFrame({
        "Clase":     report.class_names,
        "Precision": [report.per_class_prec[c] for c in report.class_names],
        "Recall":    [report.per_class_recall[c] for c in report.class_names],
        "F1-Score":  [report.per_class_f1[c] for c in report.class_names],
    })
    fig_bar = px.bar(
        per_class_df.melt(id_vars="Clase", var_name="Métrica", value_name="Valor"),
        x="Clase", y="Valor", color="Métrica", barmode="group",
        title="Precision / Recall / F1 por clase",
        color_discrete_map={"Precision":"#2E75B6","Recall":"#00708F","F1-Score":"#375623"}
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.dataframe(per_class_df.set_index("Clase"), use_container_width=True)

4.12 session_state.py — Estado centralizado de Streamlit
# src/interfaces/streamlit/state/session_state.py
"""
Gestión centralizada de st.session_state.
Evita accesos directos a session_state dispersos por la aplicación.
"""
import streamlit as st
from typing import Any, Dict, Optional

_CONFIG_KEY  = "fl_config"
_RESULTS_KEY = "fl_results"

def init_session_state():
    if _CONFIG_KEY  not in st.session_state: st.session_state[_CONFIG_KEY]  = None
    if _RESULTS_KEY not in st.session_state: st.session_state[_RESULTS_KEY] = None

def save_config(cfg: Dict[str, Any]):
    st.session_state[_CONFIG_KEY] = cfg

def get_config() -> Optional[Dict[str, Any]]:
    return st.session_state.get(_CONFIG_KEY)

def save_results(results: Any):
    st.session_state[_RESULTS_KEY] = results

def get_results() -> Any:
    return st.session_state.get(_RESULTS_KEY)

 
📂 FASE 5 — Dataset NSL-KDD: Descripción y Decisiones
5.1 Descripción del dataset
NSL-KDD es una versión corregida del clásico KDD Cup 1999 para detección de intrusiones en redes. Elimina registros duplicados y equilibra mejor las clases respecto a su predecesor.

Propiedad	Train (KDDTrain+.csv)	Test (KDDTest+.csv)
Instancias	125 973	22 544
Features	41 (3 categóricas + 38 numéricas)	Mismas 41 features
Target	class (float64 en el CSV)	Mismas clases
Columna extra	Columna 43 = nivel de dificultad (eliminar)	Ídem

5.2 Features categóricas y su tratamiento
Feature	Valores posibles	Tratamiento en NslKddAdapter
protocol_type	tcp, udp, icmp	OrdinalEncoder — fit en train, transform en test
service	http, ftp, smtp, ssh, …(~70 valores)	OrdinalEncoder — unknown_value=-1 para servicios no vistos en test
flag	SF, S0, REJ, RSTO, …(11 valores)	OrdinalEncoder — ídem

5.3 Decisiones de preprocesamiento
•	OrdinalEncoder con handle_unknown='use_encoded_value' y unknown_value=-1: evita error en test si aparece un servicio no visto en train
•	StandardScaler aplicado después del encoding — fit sólo en train, transform en test (evita data leakage)
•	Columna 43 (difficulty level) eliminada automáticamente en _read() si el CSV tiene 43 columnas
•	El target 'class' es float64 en el CSV → se castea a int64 en load()
•	Los splits train/test ya están separados en los CSVs oficiales → no se aplica train_test_split

5.4 Cómo cambiar de dataset (plug-and-play)
1.	Crear configs/datasets/mi_dataset.yaml con adapter: mi_dataset, rutas y columna target
2.	Crear src/infrastructure/dataset/mi_dataset_adapter.py implementando IDatasetAdapter.load()
3.	Registrar el adaptador en DatasetAdapterFactory (1 línea)
4.	Seleccionar el dataset en el YAML del experimento: dataset: mi_dataset
5.	Opcionalmente: añadirlo al selectbox de Streamlit en 01_config.py
No se modifica ningún código FL, de agregación ni de predicción.

🖥️ FASE 6 — Interfaz Streamlit: Arquitectura y Flujo
6.1 Principio de diseño: independencia total
La interfaz Streamlit es un módulo independiente que NO modifica ningún componente del sistema FL. Se comunica únicamente a través de FLOrchestrator.from_config() y lee los resultados de session_state. Puede eliminarse sin afectar CLI, notebooks ni lógica de negocio.

6.2 Flujo de pantallas
Pág.	Nombre	Contenido principal	Componentes usados
1	⚙️ Configuración	Dataset, n_clients, distribution, model_type, n_estimators, α, use_progressive, strategy, f1_weight, pcd_weight, local_weight, global_weight, seed	Widgets nativos Streamlit — st.slider, st.selectbox, st.checkbox
2	▶️ Ejecutar	Botón de ejecución, progress bar, log en tiempo real, KPIs rápidos post-ejecución (accuracy, F1, n_trees)	FLOrchestrator con step_callback
3	🏆 Ranking	Tabla de todos los árboles ordenada por la estrategia activa, coloreada por cliente, marcando seleccionados. Filtros por cliente y estado.	ranking_table.py, forest_summary.py
4	📊 Métricas	Selector cliente/global → accuracy, Macro-F1, precisión, recall, F1 por clase, matriz de confusión (Plotly), PCD, tamaño del bosque	metrics_panel.py (Plotly)

6.3 Ranking de árboles — Comportamiento detallado
Estrategia	Orden de la tabla	Indicación de selección
S1 — Simple Pool	Sin orden específico (todos seleccionados)	Todos en verde — no hay descartados
S2/S3/S4 — Global	Ordenado globalmente por el criterio (Accuracy / F1 / F1+PCD) de mayor a menor	Verde hasta la línea de corte de Progressive Forest; rojo después
S5/S6/S7 — Per-Client	Agrupado por cliente; dentro de cada grupo, ordenado por criterio de mayor a menor	Verde = top de su lista de cliente seleccionado en esa iteración; rojo = no alcanzado antes del stopping

Columnas en la tabla de ranking: Rank | ID árbol | Cliente | Accuracy | Macro-F1 | PCD | Score | Estado
•	ID árbol: formato {client_id}_tree{local_id} — este ID es el que usa ClientUpdater para el No-Repeat Merge
•	Score: valor del criterio de ordenamiento (— para S1 que no ordena)
•	Estado: ✅ Seleccionado / ❌ Descartado
•	Color de fondo por columna Cliente: paleta de 20 colores, uno por cliente
•	Para S5/S6/S7: subtítulo por grupo de cliente con n_seleccionados/n_total

6.4 Panel de métricas — Visualizaciones
•	KPIs: 5 métricas numéricas en columnas (Accuracy, Macro-F1, Macro Precision, Macro Recall, PCD)
•	Tamaño del bosque y número de clases como métricas adicionales
•	Matriz de confusión: heatmap Plotly interactivo con valores absolutos
•	Barras agrupadas: Precision / Recall / F1 por clase — comparación visual
•	Tabla: métricas por clase en formato tabular descargable
•	Disponible para: modelo global + cada cliente individual

6.5 Dependencias adicionales para Streamlit
# pyproject.toml — dependencias opcionales para la UI
[project.optional-dependencies]
ui = [
    "streamlit>=1.35",
    "plotly>=5.20",
    "pandas>=2.0",
]

# Instalación:
pip install -e '.[ui]'

# Ejecución:
streamlit run src/interfaces/streamlit/app.py

 
🔒 FASE 7 — Privacidad y Seguridad
7.1 NSL-KDD y privacidad
NSL-KDD contiene tráfico de red sintético/anonimizado — bajo riesgo de privacidad inherente al dataset. Los metadatos FL (accuracy, F1, PCD) son métricas de comportamiento del modelo, no atributos de individuos.

7.2 Plan de privacidad por fases
Fase	Mecanismo	Dónde en el código
1 (actual)	Sin protección — simulación académica	N/A
2	DP Gaussian sobre accuracy, macro_f1 y pcd antes de enviar al servidor	infra/privacy/dp_noise_injector.py → activar en flex_collect_trees_pf.py
3	TLS mutuamente autenticado cuando se migre a gRPC real	infra/flex/ → reemplazar FlexPool simulado
4	SecAgg sobre árboles completos si el contexto lo requiere	Nueva primitiva infra/flex/flex_secagg.py

7.3 Streamlit y seguridad
•	La UI sólo corre localmente en la fase de investigación — no exponer a internet sin autenticación
•	No almacena credentials ni datos sensibles en session_state
•	Si se despliega con Docker, usar red interna (no exponer puerto 8501 externamente en producción)

📊 FASE 8 — Observabilidad y Experimentos
8.1 Convención de nombres MLflow
Experimento:  federated_proactive_forest_nslkdd
Run:          {strategy_id}_{distribution}_{n_clients}c_{seed}s
Ejemplo:      s4_global_f1_pcd_noniid_5c_42s
Tags:         dataset=nslkdd, model=proactive_forest, alpha=0.1

8.2 Métricas obligatorias por ejecución
Categoría	Métrica	Fuente
Global	global_accuracy, global_macro_f1	ForestEvaluator sobre bosque global + X_test
Global	global_pcd	PCDCalculator sobre bosque global
Global	n_trees_global	len(global_forest.get_trees())
Por cliente	client_{id}_accuracy, client_{id}_macro_f1	ForestEvaluator por cliente
Por cliente	client_{id}_pcd	PCDCalculator por cliente
Estrategia	n_trees_selected, n_trees_discarded	len(selected_ids) vs total enviado
Estrategia	trees_per_client_selected	len(selected_ids[c]) por cada c
Fairness	std_accuracy_clients, min_accuracy_client	std/min sobre accuracy de clientes

 
⚠️ FASE 9 — Anti-patrones a Evitar (v3)
Anti-patrón	Riesgo	Mitigación
Data leakage en preprocesamiento	StandardScaler o OrdinalEncoder fit en train+test → métricas infladas artificialmente	NslKddAdapter: _scaler.fit_transform(X_train) y _scaler.transform(X_test) estrictamente separados
Hardcode de columnas NSL-KDD	Columnas hardcoded en el orchestrator → cambiar dataset requiere tocar código FL	COLUMNS, CAT_COLS, FEAT_COLS sólo en NslKddAdapter; el dominio sólo ve DatasetSplit
Streamlit acoplado al dominio	Importar ProactiveForest directamente en páginas Streamlit → cambiar modelo rompe UI	UI llama sólo FLOrchestrator.from_config(); nunca importa clases de dominio directamente
session_state disperso	st.session_state['clave'] en múltiples archivos → inconsistencia entre páginas	session_state.py centraliza todos los accesos; páginas usan get_config()/get_results()
Métricas calculadas en UI	ForestEvaluator llamado dentro de metrics_panel.py → recalculo al recargar página	ForestReport se calcula en FLOrchestrator y se guarda en results; UI sólo lee y visualiza
Ranking recalculado en UI	Reordenar entries en ranking_table.py en vez de usar el orden del orchestrador	El orchestrador guarda results.all_tree_entries ya ordenados; la UI sólo aplica filtros de display
Test sin fixture NSL-KDD	Tests de integración que descargan el CSV en CI → tests lentos y frágiles	tests/fixtures/nslkdd_sample.csv (500 filas) para tests rápidos; CI nunca descarga datos reales
Scaler no persistido	Scaler fit en entrenamiento no guardado → imposible reproducir preprocesamiento en predicción	NslKddAdapter.load() retorna el scaler como atributo; se serializa junto al bosque en LocalForestRepo

 
📋 Entregable Final — Decisiones Arquitectónicas v3
Decisión	Alternativa descartada	Razón
IDatasetAdapter como puerto hexagonal	Lógica NSL-KDD directamente en el orchestrador	DIP: el dominio no conoce el dataset; cambiar a CICIDS2017 o CIC-IDS requiere sólo un nuevo adaptador
NslKddAdapter con OrdinalEncoder (no OneHot)	OneHotEncoder para categóricas	Los árboles de decisión no requieren one-hot; OrdinalEncoder reduce dimensionalidad sin pérdida de información para este modelo
Splits train/test del CSV oficial NSL-KDD	train_test_split aleatorio del CSV	El benchmark NSL-KDD tiene split oficial; usarlo garantiza comparabilidad con la literatura
ForestEvaluator en domain/metrics/	Cálculo de métricas dentro de las páginas Streamlit	SRP: el dominio sabe evaluar; la UI sólo visualiza. Reutilizable desde CLI y notebooks
Streamlit multi-page (pages/ directory)	Una sola página con tabs	Multi-page da URLs independientes, historial de navegación y carga lazy por página
session_state.py centralizado	Acceso directo a st.session_state en cada página	DRY + testabilidad: las páginas no conocen las claves del estado; fácil de mockear en tests
ranking_table.py como componente puro	Lógica de ranking en la página 03_ranking.py	Reutilizable desde notebooks y tests; separación visual/lógica
Plotly para visualizaciones	Matplotlib o Altair	Plotly es interactivo en Streamlit sin configuración extra; zoom, hover y descarga de gráficos gratis
Dependencias UI opcionales [ui]	UI en dependencias principales	No instalar Streamlit/Plotly en entornos de producción FL que sólo usan CLI
data/ ignorado por git (sólo README)	CSV en el repositorio	NSL-KDD es ~24MB comprimido; download_nslkdd.sh + README.md es la práctica estándar

Próximos pasos recomendados (en orden)
6.	Descargar NSL-KDD con scripts/download_nslkdd.sh y verificar con test_nslkdd_adapter.py
7.	Implementar NslKddAdapter completo y su fixture nslkdd_sample.csv
8.	Implementar ForestEvaluator y test_forest_evaluator.py
9.	Conectar FLOrchestrator para que retorne FLResults con global_report y client_reports
10.	Implementar session_state.py y app.py — smoke test de navegación
11.	Implementar 01_config.py con todos los widgets y save_config()
12.	Implementar 02_run.py conectando al orchestrador con step_callback
13.	Implementar ranking_table.py — probar con datos mock antes de conectar resultados reales
14.	Implementar metrics_panel.py con Plotly — probar con ForestReport mock
15.	Integración completa: ejecutar experimento S4 desde Streamlit y verificar ranking + métricas

Arquitectura v3: el sistema FL es completamente independiente de la UI y del dataset. Cambiar cualquiera de los tres (modelo, dataset, interfaz) no afecta a los otros dos.
