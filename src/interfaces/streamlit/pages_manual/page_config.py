"""Página 1 — Configuración completa del experimento."""
import streamlit as st
import json
import os
from pathlib import Path

# Importar adaptadores de dataset
from src.infrastructure.dataset.iris_adapter import IrisAdapter
from src.infrastructure.dataset.nslkdd_adapter import NslKddAdapter
from src.infrastructure.dataset.csv_adapter import GenericCsvAdapter


# Configuración de persistencia - usar ruta absoluta del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[4]  # federated_proactive_forest/
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_FILE = CONFIG_DIR / "last_config.json"


def save_config_to_file(config: dict):
    """Guarda la configuración en un archivo JSON."""
    try:
        CONFIG_DIR.mkdir(exist_ok=True)
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            # No guardar el dataset_split ya que contiene arrays numpy
            config_to_save = {k: v for k, v in config.items() if k != "_dataset_split"}
            json.dump(config_to_save, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Error al guardar configuración: {e}")
        import traceback
        with st.expander("Traceback completo"):
            st.code(traceback.format_exc())
        return False


def load_config_from_file() -> dict:
    """Carga la configuración desde un archivo JSON."""
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"No se pudo cargar la configuración guardada: {e}")
        import traceback
        with st.expander("Traceback completo"):
            st.code(traceback.format_exc())
    return {}


def create_dataset_adapter(config: dict):
    """Crea el adaptador de dataset basado en la configuración."""
    dataset_config = config["dataset"]

    if dataset_config["type"] == "Iris":
        # Iris se carga desde sklearn, el path se ignora
        return IrisAdapter(
            data_path=None,
            train_test_split_ratio=1 - dataset_config.get("test_size", 0.2),
            scale=dataset_config.get("scale", True),
            scaler_type=dataset_config.get("scaler_type", "standard")
        )
    elif dataset_config["type"] == "NSL-KDD":
        return NslKddAdapter(
            train_path=str(PROJECT_ROOT / "data" / "NSL-KDD_train.csv"),
            test_path=str(PROJECT_ROOT / "data" / "NSL-KDD_test.csv"),
            scale=dataset_config.get("scale", True),
            scaler_type=dataset_config.get("scaler_type", "standard")
        )
    elif dataset_config["type"] == "Students Dropout":
        # Usar GenericCsvAdapter con configuración específica
        categorical_cols = [
            "Marital status", "Application mode", "Application order", "Course",
            "Daytime/evening attendance", "Previous qualification", "Nacionality",
            "Mother's qualification", "Father's qualification", "Mother's occupation",
            "Father's occupation", "Displaced", "Educational special needs", "Debtor",
            "Tuition fees up to date", "Gender", "Scholarship holder", "International",
            "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
            "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
            "Curricular units 1st sem (without evaluations)", "Curricular units 2nd sem (credited)",
            "Curricular units 2nd sem (enrolled)", "Curricular units 2nd sem (evaluations)",
            "Curricular units 2nd sem (approved)", "Curricular units 2nd sem (without evaluations)"
        ]
        return GenericCsvAdapter(
            name="students_dropout",
            train_path=dataset_config.get("file_path", "data/students_dropout.csv"),
            target_column="Target",
            test_size=dataset_config.get("test_size", 0.2),
            categorical_features=categorical_cols,
            scale=dataset_config.get("scale", True),
            scaler_type=dataset_config.get("scaler_type", "standard"),
            seed=config.get("seed", 42),
            sep=";"
        )
    else:
        raise ValueError(f"Tipo de dataset no soportado: {dataset_config['type']}")


def get_default_config():
    """Retorna configuración por defecto."""
    return {
        "dataset": {
            "type": "Iris",
            "file_path": "",
            "target_column": "class",
            "test_size": 0.2,
            "scale": True,
            "scaler_type": "standard",
        },
        "federation": {
            "n_clients": 5,
            "distribution": "iid",
            "dirichlet_alpha": 0.5,
        },
        "model": {
            "n_estimators": 100,
            "alpha": 0.1,
            "split_criterion": "entropy",
            "feature_selection": "prob",
            "use_progressive_stopping": True,
            "convergence": 0.002,
            "episode_size": 5,
        },
        "aggregation": {
            "strategy": "s1_simple_pool",
            "f1_weight": 0.5,
            "pcd_weight": 0.5,
            "convergence": 0.002,
            "episode_size": 5,
        },
        "prediction": {
            "local_weight": 0.4,
            "global_weight": 0.6,
        },
        "metadata": {
            "validation_split": 0.2,
        },
        "verbose": False,
        "seed": 42,
    }


STRATEGY_LABELS = {
    "s1_simple_pool":        "S1 — Simple Pool (todos los árboles, sin ordenar)",
    "s2_global_accuracy":    "S2 — Global, orden por Accuracy + Progressive",
    "s3_global_f1":          "S3 — Global, orden por Macro-F1 + Progressive",
    "s4_global_f1_pcd":      "S4 — Global, orden por α·F1 + β·PCD + Progressive",
    "s5_perclient_accuracy": "S5 — Per-Client, orden por Accuracy + Progressive",
    "s6_perclient_f1":       "S6 — Per-Client, orden por Macro-F1 + Progressive",
    "s7_perclient_f1_pcd":   "S7 — Per-Client, orden por α·F1 + β·PCD + Progressive",
}


def render():
    st.header("⚙️ Configuración del Experimento")

    # Cargar configuración guardada o usar valores por defecto
    saved_config = load_config_from_file()
    default_config = get_default_config()
    current_config = {**default_config, **saved_config}

    # ── Dataset ───────────────────────────────────────────────────────────────
    st.subheader("📂 Dataset")
    col1, col2 = st.columns(2)
    with col1:
        dataset_type = st.selectbox("Tipo de dataset",
                                    ["Iris", "NSL-KDD", "Students Dropout", "CSV personalizado"],
                                    index=["Iris", "NSL-KDD", "Students Dropout", "CSV personalizado"].index(
                                        current_config["dataset"]["type"]))
    with col2:
        scale = st.checkbox("Escalar features",
                           value=current_config["dataset"].get("scale", True))

    # ──────────────────────────────────────────────────────────────────────────
    # CONFIGURACIÓN ESPECÍFICA POR DATASET
    # ──────────────────────────────────────────────────────────────────────────

    if dataset_type == "Iris":
        st.markdown("##### 🌸 Configuración Iris")
        st.success("✓ Dataset Iris cargado automáticamente desde CSV local")
        st.markdown("""
        **📊 Características:**
        - Muestras: 150
        - Features: 4 (sepallength, sepalwidth, petallength, petalwidth)
        - Clases: 3 (setosa, versicolor, virginica)
        - Split: Automático Train/Test
        """)
        test_size = st.slider("Tamaño del conjunto de test", 0.1, 0.5,
                             value=current_config["dataset"].get("test_size", 0.2),
                             step=0.05, key="iris_test_size")
        # Iris se carga desde sklearn, el path es opcional y se ignora
        file_path = current_config["dataset"].get("file_path", None)
        target_column = "class"

    elif dataset_type == "NSL-KDD":
        st.markdown("##### 🛡️ Configuración NSL-KDD")
        st.success("✓ Dataset NSL-KDD para detección de intrusiones")
        st.markdown("""
        **📊 Características:**
        - Muestras: ~148K train, ~22K test
        - Features: 41 (numéricas + categóricas)
        - Clases: 5 (normal, dos, probe, r2l, u2r)
        - Archivos separados: Train y Test
        """)
        file_path = ""  # No se usa
        target_column = "class"
        test_size = 0.0  # No se usa, archivos separados

    elif dataset_type == "Students Dropout":
        st.markdown("##### 🎓 Configuración Students Dropout")
        st.success("✓ Dataset Students Dropout cargado automáticamente desde CSV local")
        st.markdown("""
        **📊 Características:**
        - Muestras: ~4K
        - Features: 36 (demográficas, académicas, socioeconómicas)
        - Clases: 3 (Dropout, Graduate, Enrolled)
        - Formato: CSV con separador ';'
        """)
        file_path = "data/students_dropout.csv"  # Ruta fija
        target_column = "Target"
        test_size = st.slider("Tamaño del conjunto de test", 0.1, 0.5,
                             value=current_config["dataset"].get("test_size", 0.2),
                             step=0.05, key="dropout_test_size")

    else:  # CSV personalizado
        st.markdown("##### ⚙️ Configuración CSV Personalizado")
        col3, col4 = st.columns(2)
        with col3:
            file_path = st.text_input("📄 Ruta del archivo CSV",
                                     value=current_config["dataset"].get("file_path", ""),
                                     key="csv_file")
        with col4:
            target_column = st.text_input("🎯 Columna objetivo",
                                         value=current_config["dataset"].get("target_column", "class"),
                                         key="csv_target")
        test_size = st.slider("Tamaño del conjunto de test", 0.1, 0.5,
                             value=current_config["dataset"].get("test_size", 0.2),
                             step=0.05, key="csv_test_size")

    st.divider()

    # ── Federación ────────────────────────────────────────────────────────────
    st.subheader("🔗 Federación")
    col5, col6 = st.columns(2)
    with col5:
        n_clients    = st.slider("Número de clientes", 2, 20,
                                value=current_config["federation"].get("n_clients", 5))
        distribution_options = ["iid", "noniid_dirichlet"]
        distribution = st.selectbox("Distribución de datos", distribution_options,
                                   index=distribution_options.index(
                                       current_config["federation"].get("distribution", "iid")))
    with col6:
        dirichlet_alpha = current_config["federation"].get("dirichlet_alpha", 0.5)
        if distribution == "noniid_dirichlet":
            dirichlet_alpha = st.slider("Parámetro Dirichlet α", 0.1, 5.0,
                                       value=dirichlet_alpha, step=0.1)

    st.divider()

    # ── Modelo ────────────────────────────────────────────────────────────────
    st.subheader("🌲 Modelo — Proactive Forest")
    col7, col8 = st.columns(2)
    with col7:
        n_estimators  = st.slider("Árboles máximos por cliente", 10, 500,
                                 value=current_config["model"].get("n_estimators", 100), step=10)
        alpha_pf      = st.slider("α diversidad Proactive Forest", 0.05, 0.5,
                                 value=current_config["model"].get("alpha", 0.1), step=0.05)
        split_options = ["entropy", "gini"]
        split_crit    = st.selectbox("Criterio de split", split_options,
                                    index=split_options.index(
                                        current_config["model"].get("split_criterion", "entropy")))
        feat_options = ["prob", "log", "all"]
        feat_sel      = st.selectbox("Selección de features", feat_options,
                                    index=feat_options.index(
                                        current_config["model"].get("feature_selection", "prob")),
                                    help="'prob' = proactivo (recomendado para PF)")
    with col8:
        use_cpf       = st.checkbox("Usar Progressive Forest (CPF)",
                                   value=current_config["model"].get("use_progressive_stopping", True))
        convergence   = current_config["model"].get("convergence", 0.002)
        episode_size  = current_config["model"].get("episode_size", 5)
        if use_cpf:
            convergence  = st.number_input("Umbral convergencia CPF", 0.0001, 0.01,
                                           value=convergence, format="%.4f")
            episode_size = st.number_input("Tamaño episodio CPF", 2, 20,
                                          value=episode_size)
        verbose_cpf = st.checkbox("Verbose CPF (debug)",
                                 value=current_config.get("verbose", False))

    st.divider()

    # ── Agregación ────────────────────────────────────────────────────────────
    st.subheader("🔀 Estrategia de Agregación")
    strategy_options = list(STRATEGY_LABELS.keys())
    current_strategy = current_config["aggregation"].get("strategy", "s1_simple_pool")
    strategy_key = st.selectbox("Estrategia", strategy_options,
                                index=strategy_options.index(current_strategy) if current_strategy in strategy_options else 0,
                                format_func=lambda k: STRATEGY_LABELS[k])

    f1_weight  = current_config["aggregation"].get("f1_weight", 0.5)
    pcd_weight = current_config["aggregation"].get("pcd_weight", 0.5)
    if strategy_key in ("s4_global_f1_pcd", "s7_perclient_f1_pcd"):
        col9, col10 = st.columns(2)
        with col9:
            f1_weight = st.slider("Peso F1 (α)", 0.0, 1.0, value=f1_weight, step=0.05)
        with col10:
            pcd_weight = round(1.0 - f1_weight, 4)
            st.metric("Peso PCD (β)", f"{pcd_weight:.2f}")

    st.divider()

    # ── Predicción ────────────────────────────────────────────────────────────
    st.subheader("🎯 Predicción Híbrida")
    col11, col12 = st.columns(2)
    with col11:
        local_w = st.slider("Peso votos locales", 0.0, 1.0,
                           value=current_config["prediction"].get("local_weight", 0.4), step=0.05)
    with col12:
        global_w = round(1.0 - local_w, 4)
        st.metric("Peso votos globales", f"{global_w:.2f}")

    st.divider()

    # ── Reproducibilidad ──────────────────────────────────────────────────────
    st.subheader("🎲 Reproducibilidad")
    seed = st.number_input("Semilla global", 0, 99999,
                          value=current_config.get("seed", 42))

    st.divider()

    # ── Guardar ───────────────────────────────────────────────────────────────
    if st.button("💾 Guardar configuración", type="primary"):
        import numpy as np
        np.random.seed(seed)

        cfg = {
            "dataset": {
                "type": dataset_type,
                "file_path": file_path if dataset_type in ["Iris", "Students Dropout", "CSV personalizado"] else "",
                "train_path": str(PROJECT_ROOT / "data" / "NSL-KDD_train.csv") if dataset_type == "NSL-KDD" else (file_path if dataset_type == "Students Dropout" else ""),
                "test_path": str(PROJECT_ROOT / "data" / "NSL-KDD_test.csv") if dataset_type == "NSL-KDD" else "",
                "target_column": target_column,
                "test_size": test_size,
                "scale": scale,
                "scaler_type": current_config["dataset"].get("scaler_type", "standard"),
            },
            "federation": {
                "n_clients":       n_clients,
                "distribution":    distribution,
                "dirichlet_alpha": dirichlet_alpha,
            },
            "model": {
                "n_estimators":          n_estimators,
                "alpha":                 alpha_pf,
                "split_criterion":       split_crit,
                "feature_selection":     feat_sel,
                "use_progressive_stopping": use_cpf,
                "convergence":           convergence,
                "episode_size":          episode_size,
            },
            "aggregation": {
                "strategy":    strategy_key,
                "f1_weight":   f1_weight,
                "pcd_weight":  pcd_weight,
                "convergence": convergence,
                "episode_size": episode_size,
            },
            "prediction": {
                "local_weight":  local_w,
                "global_weight": global_w,
            },
            "verbose": verbose_cpf,
            "seed":    seed,
        }

        # Cargar el dataset ahora para validar rutas
        try:
            ds = _load_dataset(cfg)
            cfg["_dataset_split"] = ds
            
            # Guardar configuración en archivo
            if save_config_to_file(cfg):
                st.success("💾 Configuración guardada en archivo y cargada en memoria.")
            
            st.session_state["fl_config"] = cfg
            st.success(f"✅ Configuración guardada. Dataset '{ds.dataset_name}' cargado: "
                       f"{ds.X_train.shape[0]} train / {ds.X_test.shape[0]} test, "
                       f"{len(ds.class_names)} clases.")
        except Exception as e:
            st.error(f"❌ Error al cargar el dataset: {e}")
            import traceback
            with st.expander("Traceback completo"):
                st.code(traceback.format_exc())

    # Mostrar información sobre configuración guardada
    if CONFIG_FILE.exists():
        st.info(f"📁 Configuración guardada en: `{CONFIG_FILE}`")
        if st.button("🔄 Recargar configuración guardada"):
            st.rerun()


def _load_dataset(cfg):
    """Instancia el adaptador correcto según la config."""
    d = cfg["dataset"]

    if d["type"] == "Iris":
        # Iris se carga desde sklearn, el path es opcional y se ignora
        adapter = IrisAdapter(
            data_path=None,
            train_test_split_ratio=1 - d.get("test_size", 0.2),
            scale=d.get("scale", True),
            scaler_type=d.get("scaler_type", "standard")
        )
    elif d["type"] == "NSL-KDD":
        adapter = NslKddAdapter(
            train_path=str(PROJECT_ROOT / "data" / "NSL-KDD_train.csv"),
            test_path=str(PROJECT_ROOT / "data" / "NSL-KDD_test.csv"),
            scale=d.get("scale", True),
            scaler_type=d.get("scaler_type", "standard")
        )
    elif d["type"] == "Students Dropout":
        categorical_cols = [
            "Marital status", "Application mode", "Application order", "Course",
            "Daytime/evening attendance", "Previous qualification", "Nacionality",
            "Mother's qualification", "Father's qualification", "Mother's occupation",
            "Father's occupation", "Displaced", "Educational special needs", "Debtor",
            "Tuition fees up to date", "Gender", "Scholarship holder", "International",
            "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
            "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
            "Curricular units 1st sem (without evaluations)", "Curricular units 2nd sem (credited)",
            "Curricular units 2nd sem (enrolled)", "Curricular units 2nd sem (evaluations)",
            "Curricular units 2nd sem (approved)", "Curricular units 2nd sem (without evaluations)"
        ]
        adapter = GenericCsvAdapter(
            name="students_dropout",
            train_path=d.get("file_path", "data/students_dropout.csv"),
            target_column="Target",
            test_size=d.get("test_size", 0.2),
            categorical_features=categorical_cols,
            scale=d.get("scale", True),
            scaler_type=d.get("scaler_type", "standard"),
            seed=cfg.get("seed", 42),
            sep=";"
        )
    else:
        raise ValueError(f"Tipo de dataset no soportado: {d['type']}")

    return adapter.load()
