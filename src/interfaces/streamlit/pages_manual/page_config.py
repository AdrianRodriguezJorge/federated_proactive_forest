"""Página 1 — Configuración completa del experimento."""

import json
from pathlib import Path
from typing import Any, Dict, List
import streamlit as st

from src.infrastructure.dataset.dataset_factory import DatasetFactory
from src.interfaces.streamlit.components.constants import (
    DATASET_PRESETS,
    STRATEGY_LABELS,
    S9_VARIANT_LABELS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONFIG_DIR = PROJECT_ROOT / "configs"
CONFIG_FILE = CONFIG_DIR / "last_config.json"


def save_config_to_file(config: Dict[str, Any]) -> bool:
    """Saves entire experiment configuration mapping to a JSON file.

    Args:
        config (Dict[str, Any]): Target configuration mapping to write.

    Returns:
        bool: True if configuration was successfully saved, False otherwise.
    """
    try:
        CONFIG_DIR.mkdir(exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            config_to_save = {
                k: v for k, v in config.items() if k != "_dataset_split"
            }
            json.dump(config_to_save, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Error al guardar configuración: {e}")
        return False


def load_config_from_file() -> Dict[str, Any]:
    """Loads saved configuration from file system.

    Returns:
        Dict[str, Any]: Loaded configuration map.
    """
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        import logging

        logging.warning(f"Could not load config from {CONFIG_FILE}: {e}")
        return {}
    return {}


def _validate_dataset_split(ds: Any) -> List[str]:
    """Validate that the loaded dataset split is usable.

    Args:
        ds (Any): Target dataset split.

    Returns:
        List[str]: List of warning/error messages.
    """
    warnings = []
    if ds.X_train.shape[0] == 0:
        return ["❌ El conjunto de entrenamiento está vacío."]

    if len(ds.class_names) < 2:
        warnings.append(
            f"❌ Se necesitan al menos 2 clases, se encontraron "
            f"{len(ds.class_names)}."
        )
    if ds.X_train.shape[1] == 0:
        return ["❌ No se encontraron features en el dataset."]
    if ds.X_train.shape[0] < 10:
        warnings.append(
            f"⚠️ Muy pocas muestras de entrenamiento ({ds.X_train.shape[0]}). "
            f"El modelo puede no converger."
        )
    return warnings


@st.cache_data(ttl=3600, show_spinner="Cargando dataset...")
def _load_dataset_cached(cfg_hash: str, cfg_json: str) -> Any:
    """Cached wrapper around _load_dataset.

    Args:
        cfg_hash (str): Hash key of the configuration.
        cfg_json (str): Serialized JSON configuration.

    Returns:
        Any: Loaded dataset split object.
    """
    import json

    cfg = json.loads(cfg_json)
    return _load_dataset_uncached(cfg)


def _load_dataset_uncached(cfg: Dict[str, Any]) -> Any:
    """Instanciación del adaptador correcto centralizado."""
    return DatasetFactory.load_from_config(
        cfg["dataset"], project_root=PROJECT_ROOT
    )


def _load_dataset(cfg: Dict[str, Any]) -> Any:
    """Load dataset with caching, hashing configuration."""
    import json

    cfg_clean = {k: v for k, v in cfg.items() if k != "_dataset_split"}
    cfg_json = json.dumps(cfg_clean, sort_keys=True, default=str)
    cfg_hash = str(hash(cfg_json))
    return _load_dataset_cached(cfg_hash, cfg_json)


def get_default_config() -> Dict[str, Any]:
    """Provides standard default configuration dict.

    Returns:
        Dict[str, Any]: Default configuration dict.
    """
    return {
        "dataset": {
            "type": "Iris",
            "file_path": "",
            "target_column": "class",
            "test_size": 0.15,
            "scale": True,
            "scaler_type": "standard",
        },
        "federation": {
            "n_clients": 3,
            "distribution": "iid",
            "dirichlet_alpha": 0.5,
        },
        "model": {
            "n_estimators": 100,
            "alpha": 0.1,
            "split_criterion": "entropy",
            "feature_selection": "prob",
            "use_progressive_stopping": True,
            "local_convergence_threshold": 0.002,
            "local_episode_size": 5,
        },
        "aggregation": {
            "strategy": "s6_perclient_f1",
            "f1_weight": 0.5,
            "pcd_weight": 0.5,
            "global_convergence_threshold": 0.002,
            "global_episode_size": 5,
            "window_size": 5,
            "max_rounds": 20,
            "alpha": 0.5,
        },
        "prediction": {
            "local_weight": 0.4,
            "global_weight": 0.6,
            "use_weighted": True,
        },
        "verbose": False,
        "seed": 42,
    }


def render() -> None:
    """Renders page_config Streamlit layout with entire settings widgets."""
    st.header("⚙️ Configuración del Experimento")

    saved_config = load_config_from_file()
    default_config = get_default_config()
    current_config = {**default_config, **saved_config}

    # ── Initialization of variables ──────────────────────────────────────────
    strategy_key = current_config["aggregation"].get(
        "strategy", "s6_perclient_f1"
    )
    window_size = current_config["aggregation"].get("window_size", 5)
    max_rounds = current_config["aggregation"].get("max_rounds", 20)

    s9_variant = current_config.get("aggregation", {}).get(
        "variant", "S9_MEAN"
    )
    s9_local_roulette_weight = float(
        current_config.get("aggregation", {}).get("local_roulette_weight", 0.1)
    )
    s9_window_size = int(
        current_config.get("aggregation", {}).get("window_size", 5)
    )
    s9_max_rounds = int(
        current_config.get("aggregation", {}).get("max_rounds", 20)
    )

    convergence = current_config["model"].get(
        "local_convergence_threshold", 0.002
    )
    episode_size = current_config["model"].get("local_episode_size", 5)
    convergence_agg = current_config["aggregation"].get(
        "global_convergence_threshold",
        current_config["aggregation"].get("convergence", convergence),
    )
    episode_size_agg = current_config["aggregation"].get(
        "global_episode_size",
        current_config["aggregation"].get("episode_size", episode_size),
    )

    f1_weight = current_config["aggregation"].get("f1_weight", 0.5)
    pcd_weight = current_config["aggregation"].get("pcd_weight", 0.5)
    pw_local_weight = current_config.get("prediction", {}).get(
        "local_weight", 0.5
    )
    use_weighted = current_config.get("prediction", {}).get(
        "use_weighted", True
    )
    local_w = current_config.get("prediction", {}).get("local_weight", 0.4)

    n_estimators = current_config["model"].get("n_estimators", 100)
    alpha_pf = current_config["model"].get("alpha", 0.1)
    split_crit = current_config["model"].get("split_criterion", "entropy")
    feat_sel = current_config["model"].get("feature_selection", "prob")
    use_cpf = current_config["model"].get("use_progressive_stopping", True)
    verbose_cpf = current_config.get("verbose", False)

    t_max = current_config["aggregation"].get("t_max", n_estimators)
    seed = current_config.get("seed", 42)

    is_pw_s9 = strategy_key in ("pw", "s9_roulette")
    is_progressive = strategy_key in (
        "s2_global_accuracy",
        "s3_global_f1",
        "s4_global_f1_pcd",
        "s5_perclient_accuracy",
        "s6_perclient_f1",
        "s7_perclient_f1_pcd",
        "pw",
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    st.subheader("📂 Dataset")
    dataset_options = list(DATASET_PRESETS.keys())
    current_type = current_config["dataset"]["type"]
    idx = (
        dataset_options.index(current_type)
        if current_type in dataset_options
        else 0
    )
    dataset_type = st.selectbox("Tipo de dataset", dataset_options, index=idx)

    preset = DATASET_PRESETS[dataset_type]
    st.info(f"**{dataset_type}:** {preset['info']}")

    scale = st.checkbox(
        "Escalar features",
        value=current_config["dataset"].get("scale", True),
        help="Normalizar las features para mejorar la convergencia del modelo.",
    )
    if scale:
        scaler_opts = ["standard", "minmax"]
        scaler_type = st.selectbox(
            "Tipo de escalador",
            scaler_opts,
            index=scaler_opts.index(
                current_config["dataset"].get("scaler_type", "standard")
            ),
            help="StandardScaler o MinMaxScaler.",
        )
    else:
        scaler_type = "none"

    test_size = current_config["dataset"].get("test_size", 0.15)
    target_column = preset["target_column"]
    file_path = preset["file_path"]
    sep = preset["sep"]

    if dataset_type == "CSV personalizado":
        col_a, col_b = st.columns(2)
        with col_a:
            file_path = st.text_input(
                "Ruta del archivo CSV",
                value=current_config["dataset"].get("file_path", ""),
                help="Ruta absoluta o relativa al archivo CSV.",
            )
        with col_b:
            target_column = st.text_input(
                "Columna objetivo (variable a predecir)",
                value=current_config["dataset"].get("target_column", "class"),
                help="Nombre exacto de la columna que contiene las etiquetas.",
            )
        test_size = st.slider(
            "Tamaño del conjunto de test",
            0.1,
            0.5,
            value=test_size,
            step=0.05,
            help="Proporción de datos reservados para evaluación.",
        )
        sep = st.selectbox(
            "Separador CSV",
            [",", ";", "\t"],
            index=[",", ";", "\t"].index(
                current_config["dataset"].get("sep", ",")
            ),
            help="Carácter utilizado como separador en el archivo CSV.",
        )

        if file_path:
            is_abs = Path(file_path).is_absolute()
            csv_path = PROJECT_ROOT / file_path if not is_abs else Path(file_path)
            if not csv_path.exists():
                st.error(f"❌ Archivo no encontrado: `{csv_path}`")
            else:
                st.success(f"✓ Archivo encontrado: `{csv_path}`")
                try:
                    import pandas as pd

                    sample_df = pd.read_csv(csv_path, sep=sep, nrows=100)
                    all_cols = [
                        c for c in sample_df.columns if c != target_column
                    ]
                    detected_cat = []
                    for col in all_cols:
                        is_object = sample_df[col].dtype == "object"
                        is_low_cardinality = sample_df[col].nunique() < 10
                        if is_object or is_low_cardinality:
                            detected_cat.append(col)
                    if detected_cat:
                        st.info(
                            f"🔍 Columnas categóricas detectadas: "
                            f"{len(detected_cat)}"
                        )
                    else:
                        st.info("🔍 No se detectaron columnas categóricas.")
                except Exception as e:
                    import logging

                    logging.debug(f"Auto-detection of categorical failed: {e}")
    else:
        if dataset_type != "Iris":
            test_size = st.slider(
                "Tamaño del conjunto de test",
                0.1,
                0.5,
                value=test_size,
                step=0.05,
            )

    st.divider()

    # ── Federación ────────────────────────────────────────────────────────────
    st.subheader("🔗 Federación")
    col1, col2 = st.columns(2)
    with col1:
        n_clients = st.slider(
            "Número de clientes",
            2,
            20,
            value=current_config["federation"].get("n_clients", 3),
        )
        dist_opts = ["iid", "noniid_dirichlet"]
        distribution = st.selectbox(
            "Distribución de datos",
            dist_opts,
            index=dist_opts.index(
                current_config["federation"].get("distribution", "iid")
            ),
        )
    with col2:
        dirichlet_alpha = current_config["federation"].get(
            "dirichlet_alpha", 0.5
        )
        if distribution == "noniid_dirichlet":
            dirichlet_alpha = st.slider(
                "Parámetro Dirichlet α",
                0.1,
                5.0,
                value=dirichlet_alpha,
                step=0.1,
            )

    st.divider()

    # ── Modelo ────────────────────────────────────────────────────────────────
    st.subheader("🌲 Modelo — Proactive Forest")

    feat_opts = ["prob", "sqrt", "log2", "all"]

    col3, col4 = st.columns(2)
    with col3:
        if is_pw_s9:
            st.info(
                "💡 **n_estimators** se calculará automáticamente: "
                "`ventana * rondas`."
            )
            n_estimators = current_config["model"].get("n_estimators", 50)
        else:
            n_estimators = st.slider(
                "Árboles máximos por cliente",
                10,
                500,
                value=current_config["model"].get("n_estimators", 100),
                step=10,
            )

        alpha_pf = st.slider(
            "α diversidad Proactive Forest",
            0.05,
            0.5,
            value=alpha_pf,
            step=0.05,
        )
        split_opts = ["entropy", "gini"]
        split_crit = st.selectbox(
            "Criterio de split",
            split_opts,
            index=split_opts.index(
                current_config["model"].get("split_criterion", "entropy")
            ),
            help="Medida para evaluar la calidad del split en cada nodo.",
        )
        feat_sel = st.selectbox(
            "Selección de features (m)",
            feat_opts,
            index=feat_opts.index(
                current_config["model"].get("feature_selection", "prob")
            ),
            help="Número de features candidatas por nodo.",
        )
    with col4:
        use_cpf = st.checkbox(
            "Usar Progressive Forest (CPF)",
            value=use_cpf,
            help="Criterio de parada progresivo local del cliente.",
        )
        if use_cpf:
            convergence = st.number_input(
                "Umbral convergencia CPF (local)",
                0.0,
                1.0,
                value=convergence,
                format="%.4f",
                help="Delta local: Mejora mínima local requerida.",
            )
            episode_size = st.number_input(
                "Tamaño episodio CPF (local)",
                2,
                20,
                value=episode_size,
                help="Árboles entrenados antes de evaluar convergencia local.",
            )
        verbose_cpf = st.checkbox(
            "Verbose CPF (debug)", value=current_config.get("verbose", False)
        )

    st.divider()

    # ── Agregación ────────────────────────────────────────────────────────────
    st.subheader("🔀 Estrategia de Agregación")
    strategy_options = list(STRATEGY_LABELS.keys())
    strategy_key = st.selectbox(
        "Estrategia",
        strategy_options,
        index=(
            strategy_options.index(strategy_key)
            if strategy_key in strategy_options
            else 0
        ),
        format_func=lambda k: STRATEGY_LABELS[k],
        help="Algoritmo de agregación seleccionado.",
    )

    if not is_pw_s9:
        t_max = st.number_input(
            "T_MAX (máx. árboles en bosque global)",
            10,
            500,
            value=t_max,
            step=10,
            help="Número máximo de árboles en bosque global tras agregación.",
        )
    else:
        is_s9 = strategy_key == "s9_roulette"
        t_max_s9 = s9_window_size * s9_max_rounds
        t_max_pw = window_size * max_rounds
        t_max = t_max_s9 if is_s9 else t_max_pw
        st.caption(f"📏 T_MAX teórico: {t_max} árboles.")

    if is_progressive:
        st.caption(
            f"📌 **{STRATEGY_LABELS[strategy_key]}** — parada progresiva global"
        )
        col_conv, col_ep = st.columns(2)
        with col_conv:
            convergence_agg = st.number_input(
                "Umbral convergencia (global)",
                0.0,
                1.0,
                value=convergence_agg,
                format="%.4f",
                help="Delta global: Mejora mínima global requerida.",
            )
        with col_ep:
            episode_size_agg = st.number_input(
                "Tamaño episodio (global)",
                2,
                20,
                value=episode_size_agg,
                help="Árboles añadidos antes de evaluar convergencia global.",
            )

    if strategy_key in ("s4_global_f1_pcd", "s7_perclient_f1_pcd"):
        col5, col6 = st.columns(2)
        with col5:
            f1_weight = st.slider(
                "Peso F1 (α)",
                0.0,
                1.0,
                value=f1_weight,
                step=0.05,
                help="Peso de Macro-F1 en el score combinado.",
            )
        with col6:
            pcd_weight = round(1.0 - f1_weight, 4)
            st.metric("Peso PCD (β)", f"{pcd_weight:.2f}")

    if strategy_key == "pw":
        st.info("🔄 **Progressive Windows**: Agregación por ventanas.")
        col5, col6, col7 = st.columns(3)
        with col5:
            window_size = st.number_input(
                "Tamaño ventana (W)",
                2,
                20,
                value=window_size,
                help="Árboles entrenados por ronda por cliente.",
            )
        with col6:
            max_rounds = st.number_input(
                "Máximo rondas (R_MAX)",
                5,
                50,
                value=max_rounds,
                help="Rondas máximas de Round Robin.",
            )
        with col7:
            f1_weight = st.slider(
                "α Score (F1 vs Diversidad)",
                0.0,
                1.0,
                value=f1_weight,
                step=0.05,
                help="Peso F1 en el score dinámico.",
            )
            pcd_weight = round(1.0 - f1_weight, 4)
        use_weighted = st.checkbox(
            "Ponderar por origen (local/global)",
            value=use_weighted,
            help="Los árboles locales y globales tienen pesos distintos.",
            key="pw_use_weighted",
        )
        if use_weighted:
            pw_local_weight = st.slider(
                "Peso predicción local (PW)",
                0.0,
                1.0,
                value=pw_local_weight,
                step=0.05,
                help="Peso de árboles locales en inferencia PW.",
            )
        else:
            st.caption("⚖️ Cada árbol vota con el mismo peso.")

    if strategy_key == "s9_roulette":
        st.info("🎰 **Ruleta Global de Atributos**: Agregación de vectores.")
        col_s9a, col_s9b = st.columns(2)
        with col_s9a:
            s9_variant_options = list(S9_VARIANT_LABELS.keys())
            variant_idx = (
                s9_variant_options.index(s9_variant)
                if s9_variant in s9_variant_options
                else 0
            )
            s9_variant = st.selectbox(
                "Variante de agregacion",
                s9_variant_options,
                index=variant_idx,
                format_func=lambda k: S9_VARIANT_LABELS[k],
                help="Método de agregación de vectores.",
            )
            s9_window_size = st.number_input(
                "Arboles por ventana (W)",
                2,
                20,
                value=s9_window_size,
                help="Árboles locales antes de enviar vector de ruleta.",
            )
        with col_s9b:
            s9_local_roulette_weight = st.slider(
                "Local Roulette Weight (local_roulette_weight)",
                0.0,
                1.0,
                value=s9_local_roulette_weight,
                step=0.05,
                help="Peso asignado a la ruleta local del cliente.",
            )
            s9_max_rounds = st.number_input(
                "Rondas maximas",
                5,
                50,
                value=s9_max_rounds,
                help="Número máximo de rondas federadas.",
            )

    st.divider()

    # ── Predicción ────────────────────────────────────────────────────────────
    if strategy_key == "s9_roulette":
        st.subheader("🎯 Predicción")
        st.caption(
            "En S9, la inferencia es local usando el modelo de la ruleta."
        )
        local_w = 0.0
    elif strategy_key != "pw":
        st.subheader("🎯 Predicción Híbrida")
        use_weighted = st.checkbox(
            "Ponderar por origen (local/global)",
            value=use_weighted,
            help="Los árboles locales y globales tienen pesos distintos.",
            key="pred_use_weighted",
        )
        if use_weighted:
            col8, col9 = st.columns(2)
            with col8:
                local_w = st.slider(
                    "Peso votos locales",
                    0.0,
                    1.0,
                    value=local_w,
                    step=0.05,
                    help="Proporción de votos locales en inferencia híbrida.",
                )
            with col9:
                global_w = round(1.0 - local_w, 4)
                st.metric("Peso votos globales", f"{global_w:.2f}")
        else:
            local_w = 0.5
            st.caption("⚖️ Cada árbol vota con el mismo peso.")
    else:
        local_w = pw_local_weight

    st.divider()

    # ── Reproducibilidad ──────────────────────────────────────────────────────
    seed = st.number_input(
        "Semilla global", 0, 99999, value=current_config.get("seed", 42)
    )

    st.divider()

    # ── Guardar ───────────────────────────────────────────────────────────────
    if st.button("💾 Guardar configuración", type="primary"):
        import numpy as np

        np.random.seed(seed)

        if strategy_key == "s9_roulette":
            t_max_calc = s9_window_size * s9_max_rounds
            window_size_calc = s9_window_size
            max_rounds_calc = s9_max_rounds
        elif strategy_key == "pw":
            t_max_calc = window_size * max_rounds
            window_size_calc = window_size
            max_rounds_calc = max_rounds
        else:
            t_max_calc = t_max
            window_size_calc = 5
            max_rounds_calc = 20

        is_weight_strat = strategy_key in (
            "s4_global_f1_pcd",
            "s7_perclient_f1_pcd",
            "pw",
        )
        f1_w_calc = f1_weight if is_weight_strat else 0.5
        pcd_w_calc = pcd_weight if is_weight_strat else 0.5

        pred_local_w = pw_local_weight if strategy_key == "pw" else local_w

        cfg = {
            "dataset": {
                "type": dataset_type,
                "file_path": file_path,
                "train_path": "",
                "test_path": "",
                "target_column": target_column,
                "test_size": test_size,
                "scale": scale,
                "scaler_type": scaler_type,
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
                "feature_selection": feat_sel,
                "use_progressive_stopping": use_cpf,
                "local_convergence_threshold": convergence,
                "local_episode_size": episode_size,
            },
            "aggregation": {
                "strategy": strategy_key,
                "t_max": t_max_calc,
                "f1_weight": f1_w_calc,
                "pcd_weight": pcd_w_calc,
                "global_convergence_threshold": convergence_agg,
                "global_episode_size": episode_size_agg,
                "window_size": window_size_calc,
                "max_rounds": max_rounds_calc,
                "variant": s9_variant if strategy_key == "s9_roulette" else "",
                "local_roulette_weight": s9_local_roulette_weight if strategy_key == "s9_roulette" else 0.0,
            },
            "prediction": {
                "local_weight": pred_local_w,
                "global_weight": round(1.0 - pred_local_w, 4),
                "use_weighted": use_weighted,
            },
            "verbose": verbose_cpf,
            "seed": seed,
        }

        try:
            ds = _load_dataset(cfg)

            val_warnings = _validate_dataset_split(ds)
            has_errors = any(w.startswith("❌") for w in val_warnings)
            if has_errors:
                for w in val_warnings:
                    if w.startswith("❌"):
                        st.error(w)
                st.warning(
                    "⚠️ Corrige los parámetros del dataset e inténtalo."
                )
                return

            cfg["_dataset_split"] = ds
            if save_config_to_file(cfg):
                st.success("💾 Configuración guardada en archivo.")
            st.session_state["fl_config"] = cfg
            st.success(
                f"✅ Dataset '{ds.dataset_name}' cargado: "
                f"{ds.X_train.shape[0]} train / {ds.X_test.shape[0]} test, "
                f"{len(ds.class_names)} clases."
            )

            for w in val_warnings:
                if w.startswith("⚠️"):
                    st.warning(w)
        except Exception as e:
            st.error(f"❌ Error al cargar el dataset: {e}")
            import traceback

            with st.expander("Traceback completo"):
                st.code(traceback.format_exc())

    if CONFIG_FILE.exists():
        st.info(f"📁 Configuración guardada en: `{CONFIG_FILE}`")
        if st.button("🔄 Recargar configuración guardada"):
            st.rerun()
