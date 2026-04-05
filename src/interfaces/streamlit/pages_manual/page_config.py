"""Página 1 — Configuración completa del experimento."""
import streamlit as st
import json
from pathlib import Path

from src.infrastructure.dataset.iris_adapter import IrisAdapter
from src.infrastructure.dataset.nslkdd_adapter import NslKddAdapter
from src.infrastructure.dataset.csv_adapter import GenericCsvAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_FILE = CONFIG_DIR / "last_config.json"

# ── Categorical columns per dataset (centralized) ─────────────────────────────
CATEGORICAL_COLS = {
    "students_dropout": [
        "Marital status", "Application mode", "Application order", "Course",
        "Daytime/evening attendance", "Previous qualification", "Nacionality",
        "Mother's qualification", "Father's qualification", "Mother's occupation",
        "Father's occupation", "Displaced", "Educational special needs", "Debtor",
        "Tuition fees up to date", "Gender", "Scholarship holder", "International",
        "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (without evaluations)", "Curricular units 2nd sem (credited)",
        "Curricular units 2nd sem (enrolled)", "Curricular units 2nd sem (evaluations)",
        "Curricular units 2nd sem (approved)", "Curricular units 2nd sem (without evaluations)",
    ],
    "nursery": ["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health"],
    "car": ["buying", "maint", "doors", "persons", "lug_boot", "safety"],
}

DATASET_PRESETS = {
    "Iris": {
        "file_path": "data/iris.csv", "target_column": "class", "test_size": 0.2,
        "scale": True, "sep": ",", "categorical_key": None,
        "info": "150 muestras, 4 features, 3 clases (setosa, versicolor, virginica)",
    },
    "Letter": {
        "file_path": "data/letter.csv", "target_column": "class", "test_size": 0.2,
        "scale": True, "sep": ",", "categorical_key": None,
        "info": "20,000 muestras, 16 features, 26 clases (A-Z)",
    },
    "Optdigits": {
        "file_path": "data/optdigits.csv", "target_column": "class", "test_size": 0.2,
        "scale": True, "sep": ",", "categorical_key": None,
        "info": "5,620 muestras, 64 features (8x8 píxeles), 10 clases (0-9)",
    },
    "Spambase": {
        "file_path": "data/spambase.csv", "target_column": "class", "test_size": 0.2,
        "scale": True, "sep": ",", "categorical_key": None,
        "info": "4,601 muestras, 57 features, 2 clases (spam/ham)",
    },
    "Nursery": {
        "file_path": "data/nursery.csv", "target_column": "class", "test_size": 0.2,
        "scale": True, "sep": ",", "categorical_key": "nursery",
        "info": "12,960 muestras, 8 features categóricas, 5 clases",
    },
    "Sonar": {
        "file_path": "data/sonar.csv", "target_column": "Class", "test_size": 0.2,
        "scale": True, "sep": ",", "categorical_key": None,
        "info": "208 muestras, 60 features, 2 clases (Rock/Mine)",
    },
    "Vowel": {
        "file_path": "data/vowel.csv", "target_column": "Class", "test_size": 0.2,
        "scale": True, "sep": ",", "categorical_key": None,
        "info": "990 muestras, 10 features, 11 clases",
    },
    "Car": {
        "file_path": "data/car.csv", "target_column": "class", "test_size": 0.2,
        "scale": True, "sep": ",", "categorical_key": "car",
        "info": "1,728 muestras, 6 features categóricas, 4 clases (unacc, acc, good, vgood)",
    },
    "Students Dropout": {
        "file_path": "data/students_dropout.csv", "target_column": "Target", "test_size": 0.2,
        "scale": True, "sep": ";", "categorical_key": "students_dropout",
        "info": "~4K muestras, 36 features, 3 clases (Dropout, Graduate, Enrolled)",
    },
    "NSL-KDD": {
        "file_path": "", "target_column": "class", "test_size": 0.0,
        "scale": True, "sep": ",", "categorical_key": None,
        "info": "~148K train, ~22K test, 41 features, 5 clases",
    },
    "CSV personalizado": {
        "file_path": "", "target_column": "class", "test_size": 0.2,
        "scale": True, "sep": ",", "categorical_key": None,
        "info": "Carga tu propio archivo CSV",
    },
}

STRATEGY_LABELS = {
    "s1_simple_pool":        "S1 — Simple Pool (todos los árboles, sin ordenar)",
    "s2_global_accuracy":    "S2 — Global, orden por Accuracy + Progressive",
    "s3_global_f1":          "S3 — Global, orden por Macro-F1 + Progressive",
    "s4_global_f1_pcd":      "S4 — Global, orden por α·F1 + β·PCD + Progressive",
    "s5_perclient_accuracy": "S5 — Per-Client, orden por Accuracy + Progressive",
    "s6_perclient_f1":       "S6 — Per-Client, orden por Macro-F1 + Progressive",
    "s7_perclient_f1_pcd":   "S7 — Per-Client, orden por α·F1 + β·PCD + Progressive",
    "pw":                    "PW — Progressive Windows (ventanas + score dinámico F1+Diversidad)",
}


def save_config_to_file(config: dict):
    try:
        CONFIG_DIR.mkdir(exist_ok=True)
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            config_to_save = {k: v for k, v in config.items() if k != "_dataset_split"}
            json.dump(config_to_save, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Error al guardar configuración: {e}")
        return False


def load_config_from_file() -> dict:
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _load_dataset(cfg):
    """Instancia el adaptador correcto según la config."""
    d = cfg["dataset"]
    dtype = d["type"]

    if dtype == "Iris":
        adapter = IrisAdapter(
            data_path=None,
            train_test_split_ratio=1 - d.get("test_size", 0.2),
            scale=d.get("scale", True),
            scaler_type=d.get("scaler_type", "standard"),
        )
    elif dtype == "NSL-KDD":
        adapter = NslKddAdapter(
            train_path=str(PROJECT_ROOT / "data" / "NSL-KDD_train.csv"),
            test_path=str(PROJECT_ROOT / "data" / "NSL-KDD_test.csv"),
            scale=d.get("scale", True),
            scaler_type=d.get("scaler_type", "standard"),
        )
    else:
        preset = DATASET_PRESETS.get(dtype, {})
        cat_key = preset.get("categorical_key")
        categorical_cols = CATEGORICAL_COLS.get(cat_key, []) if cat_key else []

        adapter = GenericCsvAdapter(
            name=dtype.lower().replace(" ", "_"),
            train_path=str(PROJECT_ROOT / d.get("file_path", "")),
            target_column=d.get("target_column", "class"),
            test_size=d.get("test_size", 0.2),
            categorical_features=categorical_cols,
            scale=d.get("scale", True),
            scaler_type=d.get("scaler_type", "standard"),
            seed=cfg.get("seed", 42),
            sep=d.get("sep", ","),
        )
    return adapter.load()


def get_default_config():
    return {
        "dataset": {
            "type": "Iris", "file_path": "", "target_column": "class",
            "test_size": 0.2, "scale": True, "scaler_type": "standard",
        },
        "federation": {
            "n_clients": 3, "distribution": "iid", "dirichlet_alpha": 0.5,
        },
        "model": {
            "n_estimators": 100, "alpha": 0.1, "split_criterion": "entropy",
            "feature_selection": "prob", "use_progressive_stopping": True,
            "convergence": 0.002, "episode_size": 5,
        },
        "aggregation": {
            "strategy": "s6_perclient_f1", "f1_weight": 0.5, "pcd_weight": 0.5,
            "convergence": 0.002, "episode_size": 5,
            "window_size": 5, "max_rounds": 20, "alpha": 0.5,
        },
        "prediction": {"local_weight": 0.4, "global_weight": 0.6},
        "verbose": False, "seed": 42,
    }


def render():
    st.header("⚙️ Configuración del Experimento")

    saved_config = load_config_from_file()
    default_config = get_default_config()
    current_config = {**default_config, **saved_config}

    # ── Dataset ───────────────────────────────────────────────────────────────
    st.subheader("📂 Dataset")
    dataset_options = list(DATASET_PRESETS.keys())
    current_type = current_config["dataset"]["type"]
    idx = dataset_options.index(current_type) if current_type in dataset_options else 0
    dataset_type = st.selectbox("Tipo de dataset", dataset_options, index=idx)

    preset = DATASET_PRESETS[dataset_type]
    st.info(f"**{dataset_type}:** {preset['info']}")

    scale = st.checkbox("Escalar features", value=current_config["dataset"].get("scale", True))

    # Dataset-specific settings
    test_size = current_config["dataset"].get("test_size", 0.2)
    target_column = preset["target_column"]
    file_path = preset["file_path"]
    sep = preset["sep"]

    if dataset_type == "NSL-KDD":
        st.success("✓ Archivos separados Train/Test en data/")
        test_size = 0.0
    elif dataset_type == "CSV personalizado":
        col_a, col_b = st.columns(2)
        with col_a:
            file_path = st.text_input("Ruta del archivo CSV", value=current_config["dataset"].get("file_path", ""))
        with col_b:
            target_column = st.text_input("Columna objetivo", value=current_config["dataset"].get("target_column", "class"))
        test_size = st.slider("Tamaño del conjunto de test", 0.1, 0.5, value=test_size, step=0.05)
        sep = st.selectbox("Separador CSV", [",", ";", "\t"], index=[",", ";", "\t"].index(current_config["dataset"].get("sep", ",")))
    else:
        if dataset_type not in ("Iris", "NSL-KDD"):
            test_size = st.slider("Tamaño del conjunto de test", 0.1, 0.5, value=test_size, step=0.05)

    st.divider()

    # ── Federación ────────────────────────────────────────────────────────────
    st.subheader("🔗 Federación")
    col1, col2 = st.columns(2)
    with col1:
        n_clients = st.slider("Número de clientes", 2, 20,
                              value=current_config["federation"].get("n_clients", 3))
        dist_opts = ["iid", "noniid_dirichlet"]
        distribution = st.selectbox("Distribución de datos", dist_opts,
                                    index=dist_opts.index(current_config["federation"].get("distribution", "iid")))
    with col2:
        dirichlet_alpha = current_config["federation"].get("dirichlet_alpha", 0.5)
        if distribution == "noniid_dirichlet":
            dirichlet_alpha = st.slider("Parámetro Dirichlet α", 0.1, 5.0, value=dirichlet_alpha, step=0.1)

    st.divider()

    # ── Modelo ────────────────────────────────────────────────────────────────
    st.subheader("🌲 Modelo — Proactive Forest")
    col3, col4 = st.columns(2)
    with col3:
        n_estimators = st.slider("Árboles máximos por cliente", 10, 500,
                                 value=current_config["model"].get("n_estimators", 100), step=10)
        alpha_pf = st.slider("α diversidad Proactive Forest", 0.05, 0.5,
                             value=current_config["model"].get("alpha", 0.1), step=0.05)
        split_opts = ["entropy", "gini"]
        split_crit = st.selectbox("Criterio de split", split_opts,
                                  index=split_opts.index(current_config["model"].get("split_criterion", "entropy")))
    with col4:
        use_cpf = st.checkbox("Usar Progressive Forest (CPF)",
                              value=current_config["model"].get("use_progressive_stopping", True))
        convergence = current_config["model"].get("convergence", 0.002)
        episode_size = current_config["model"].get("episode_size", 5)
        if use_cpf:
            convergence = st.number_input("Umbral convergencia CPF", 0.0001, 0.01, value=convergence, format="%.4f")
            episode_size = st.number_input("Tamaño episodio CPF", 2, 20, value=episode_size)
        verbose_cpf = st.checkbox("Verbose CPF (debug)", value=current_config.get("verbose", False))

    st.divider()

    # ── Agregación ────────────────────────────────────────────────────────────
    st.subheader("🔀 Estrategia de Agregación")
    strategy_options = list(STRATEGY_LABELS.keys())
    current_strategy = current_config["aggregation"].get("strategy", "s6_perclient_f1")
    strategy_key = st.selectbox("Estrategia", strategy_options,
                                index=strategy_options.index(current_strategy) if current_strategy in strategy_options else 0,
                                format_func=lambda k: STRATEGY_LABELS[k])

    f1_weight = current_config["aggregation"].get("f1_weight", 0.5)
    pcd_weight = current_config["aggregation"].get("pcd_weight", 0.5)
    window_size = current_config["aggregation"].get("window_size", 5)
    max_rounds = current_config["aggregation"].get("max_rounds", 20)
    alpha_score = current_config["aggregation"].get("alpha", 0.5)

    if strategy_key in ("s4_global_f1_pcd", "s7_perclient_f1_pcd"):
        col5, col6 = st.columns(2)
        with col5:
            f1_weight = st.slider("Peso F1 (α)", 0.0, 1.0, value=f1_weight, step=0.05)
        with col6:
            pcd_weight = round(1.0 - f1_weight, 4)
            st.metric("Peso PCD (β)", f"{pcd_weight:.2f}")
    elif strategy_key == "pw":
        st.info("🔄 **Progressive Windows**: Entrenamiento por ventanas + selección secuencial con score dinámico")
        col5, col6, col7 = st.columns(3)
        with col5:
            window_size = st.number_input("Tamaño ventana (W)", 2, 20, value=window_size)
        with col6:
            max_rounds = st.number_input("Máximo rondas (R_MAX)", 5, 50, value=max_rounds)
        with col7:
            alpha_score = st.slider("α Score (F1 vs Diversidad)", 0.0, 1.0, value=alpha_score, step=0.05)

    st.divider()

    # ── Predicción ────────────────────────────────────────────────────────────
    st.subheader("🎯 Predicción Híbrida")
    col8, col9 = st.columns(2)
    with col8:
        local_w = st.slider("Peso votos locales", 0.0, 1.0,
                            value=current_config["prediction"].get("local_weight", 0.4), step=0.05)
    with col9:
        global_w = round(1.0 - local_w, 4)
        st.metric("Peso votos globales", f"{global_w:.2f}")

    st.divider()

    # ── Reproducibilidad ──────────────────────────────────────────────────────
    seed = st.number_input("Semilla global", 0, 99999, value=current_config.get("seed", 42))

    st.divider()

    # ── Guardar ───────────────────────────────────────────────────────────────
    if st.button("💾 Guardar configuración", type="primary"):
        import numpy as np
        np.random.seed(seed)

        cfg = {
            "dataset": {
                "type": dataset_type,
                "file_path": file_path,
                "train_path": str(PROJECT_ROOT / "data" / "NSL-KDD_train.csv") if dataset_type == "NSL-KDD" else "",
                "test_path": str(PROJECT_ROOT / "data" / "NSL-KDD_test.csv") if dataset_type == "NSL-KDD" else "",
                "target_column": target_column,
                "test_size": test_size,
                "scale": scale,
                "scaler_type": current_config["dataset"].get("scaler_type", "standard"),
                "sep": sep,
            },
            "federation": {
                "n_clients": n_clients,
                "distribution": distribution,
                "dirichlet_alpha": dirichlet_alpha,
            },
            "model": {
                "n_estimators": n_estimators,
                "alpha": alpha_pf,
                "split_criterion": split_crit,
                "use_progressive_stopping": use_cpf,
                "convergence": convergence,
                "episode_size": episode_size,
            },
            "aggregation": {
                "strategy": strategy_key,
                "f1_weight": f1_weight,
                "pcd_weight": pcd_weight,
                "convergence": convergence,
                "episode_size": episode_size,
                "window_size": window_size,
                "max_rounds": max_rounds,
                "alpha": alpha_score,
            },
            "prediction": {"local_weight": local_w, "global_weight": global_w},
            "verbose": verbose_cpf,
            "seed": seed,
        }

        try:
            ds = _load_dataset(cfg)
            cfg["_dataset_split"] = ds
            if save_config_to_file(cfg):
                st.success("💾 Configuración guardada en archivo y cargada en memoria.")
            st.session_state["fl_config"] = cfg
            st.success(f"✅ Dataset '{ds.dataset_name}' cargado: "
                       f"{ds.X_train.shape[0]} train / {ds.X_test.shape[0]} test, "
                       f"{len(ds.class_names)} clases.")
        except Exception as e:
            st.error(f"❌ Error al cargar el dataset: {e}")
            import traceback
            with st.expander("Traceback completo"):
                st.code(traceback.format_exc())

    if CONFIG_FILE.exists():
        st.info(f"📁 Configuración guardada en: `{CONFIG_FILE}`")
        if st.button("🔄 Recargar configuración guardada"):
            st.rerun()
