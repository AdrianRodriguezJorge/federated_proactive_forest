"""Página 1 — Configuración completa del experimento."""
import streamlit as st


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

    # ── Dataset ───────────────────────────────────────────────────────────────
    st.subheader("📂 Dataset")
    col1, col2 = st.columns(2)
    with col1:
        dataset_type = st.selectbox("Tipo de dataset",
                                    ["NSL-KDD", "Iris", "CSV personalizado"])
    with col2:
        scale = st.checkbox("Escalar features", value=True)
        scaler_type = st.selectbox("Scaler", ["standard", "minmax"],
                                   disabled=not scale)

    # ──────────────────────────────────────────────────────────────────────────
    # CONFIGURACIÓN ESPECÍFICA POR DATASET
    # ──────────────────────────────────────────────────────────────────────────
    
    if dataset_type == "NSL-KDD":
        st.markdown("##### 🔧 Configuración NSL-KDD")
        col3, col4 = st.columns(2)
        with col3:
            train_path = st.text_input("📄 Ruta KDDTrain+.csv",
                                       value="data/NSL-KDD_train.csv",
                                       key="nslkdd_train")
        with col4:
            test_path = st.text_input("📄 Ruta KDDTest+.csv",
                                      value="data/NSL-KDD_test.csv",
                                      key="nslkdd_test")
        target_col = "class"
        cat_features = ["protocol_type", "service", "flag"]
        csv_name = "nslkdd"
        
    elif dataset_type == "Iris":
        st.markdown("##### 🌸 Configuración Iris")
        st.success("✓ Dataset Iris cargado automáticamente")
        st.markdown("""
        **📊 Características:**
        - Muestras: 150
        - Features: 4 (sepallength, sepalwidth, petallength, petalwidth)
        - Clases: 3 (setosa, versicolor, virginica)
        - Split: Automático 70-30 Train/Test
        """)
        train_path = "data/iris.csv"
        test_path = None
        target_col = "class"
        cat_features = []
        csv_name = "iris"
        
    else:  # CSV personalizado
        st.markdown("##### ⚙️ Configuración CSV Personalizado")
        col3, col4 = st.columns(2)
        with col3:
            train_path = st.text_input("📄 Ruta CSV entrenamiento",
                                       key="csv_train")
        with col4:
            test_path  = st.text_input("📄 Ruta CSV test (dejar vacío para split automático: 80-20)",
                                       key="csv_test")
        col5, col6 = st.columns(2)
        with col5:
            target_col   = st.text_input("🎯 Columna target", value="class", key="csv_target")
        with col6:
            cat_input = st.text_input("🏷️  Columnas categóricas (separadas por coma)", 
                                      key="csv_cat")
            cat_features = [c.strip() for c in cat_input.split(",") if c.strip()]
        csv_name     = st.text_input("📛 Nombre del dataset", value="custom", key="csv_name")

    st.divider()

    # ── Federación ────────────────────────────────────────────────────────────
    st.subheader("🔗 Federación")
    col5, col6 = st.columns(2)
    with col5:
        n_clients    = st.slider("Número de clientes", 2, 20, 5)
        distribution = st.selectbox("Distribución de datos",
                                    ["iid", "noniid_dirichlet"])
    with col6:
        dirichlet_alpha = 0.5
        if distribution == "noniid_dirichlet":
            dirichlet_alpha = st.slider("Parámetro Dirichlet α", 0.1, 5.0, 0.5, 0.1)
        val_split = st.slider("Fracción validación local (metadatos)", 0.1, 0.4, 0.2, 0.05)

    st.divider()

    # ── Modelo ────────────────────────────────────────────────────────────────
    st.subheader("🌲 Modelo — Proactive Forest")
    col7, col8 = st.columns(2)
    with col7:
        n_estimators  = st.slider("Árboles máximos por cliente", 10, 500, 100, 10)
        alpha_pf      = st.slider("α diversidad Proactive Forest", 0.05, 0.5, 0.1, 0.05)
        split_crit    = st.selectbox("Criterio de split", ["entropy", "gini"])
        feat_sel      = st.selectbox("Selección de features",
                                     ["prob", "log", "all"],
                                     help="'prob' = proactivo (recomendado para PF)")
    with col8:
        use_cpf       = st.checkbox("Usar Progressive Forest (CPF)", value=True)
        convergence   = 0.002
        episode_size  = 5
        if use_cpf:
            convergence  = st.number_input("Umbral convergencia CPF", 0.0001, 0.01,
                                           0.002, format="%.4f")
            episode_size = st.number_input("Tamaño episodio CPF", 2, 20, 5)
        verbose_cpf = st.checkbox("Verbose CPF (debug)", value=False)

    st.divider()

    # ── Agregación ────────────────────────────────────────────────────────────
    st.subheader("🔀 Estrategia de Agregación")
    strategy_key = st.selectbox("Estrategia", list(STRATEGY_LABELS.keys()),
                                format_func=lambda k: STRATEGY_LABELS[k])

    f1_weight  = 0.5
    pcd_weight = 0.5
    if strategy_key in ("s4_global_f1_pcd", "s7_perclient_f1_pcd"):
        col9, col10 = st.columns(2)
        with col9:
            f1_weight = st.slider("Peso F1 (α)", 0.0, 1.0, 0.5, 0.05)
        with col10:
            pcd_weight = round(1.0 - f1_weight, 4)
            st.metric("Peso PCD (β)", f"{pcd_weight:.2f}")

    st.divider()

    # ── Predicción ────────────────────────────────────────────────────────────
    st.subheader("🎯 Predicción Híbrida")
    col11, col12 = st.columns(2)
    with col11:
        local_w = st.slider("Peso votos locales", 0.0, 1.0, 0.4, 0.05)
    with col12:
        global_w = round(1.0 - local_w, 4)
        st.metric("Peso votos globales", f"{global_w:.2f}")

    st.divider()

    # ── Reproducibilidad ──────────────────────────────────────────────────────
    st.subheader("🎲 Reproducibilidad")
    seed = st.number_input("Semilla global", 0, 99999, 42)

    st.divider()

    # ── Guardar ───────────────────────────────────────────────────────────────
    if st.button("💾 Guardar configuración", type="primary"):
        import numpy as np
        np.random.seed(seed)

        cfg = {
            "dataset": {
                "type":     dataset_type,
                "name":     csv_name,
                "train_path": train_path,
                "test_path":  test_path,
                "target_column": target_col,
                "categorical_features": cat_features,
                "scale": scale,
                "scaler_type": scaler_type,
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
            "metadata": {
                "validation_split": val_split,
            },
            "verbose": verbose_cpf,
            "seed":    seed,
        }

        # Cargar el dataset ahora para validar rutas
        try:
            ds = _load_dataset(cfg)
            cfg["_dataset_split"] = ds
            st.session_state["fl_config"] = cfg
            st.success(f"✅ Configuración guardada. Dataset '{ds.dataset_name}' cargado: "
                       f"{ds.X_train.shape[0]} train / {ds.X_test.shape[0]} test, "
                       f"{len(ds.class_names)} clases.")
        except Exception as e:
            st.error(f"❌ Error al cargar el dataset: {e}")


def _load_dataset(cfg):
    """Instancia el adaptador correcto según la config."""
    d = cfg["dataset"]
    if d["type"] == "NSL-KDD":
        from src.infrastructure.dataset.nslkdd_adapter import NslKddAdapter
        adapter = NslKddAdapter(
            train_path=d["train_path"],
            test_path=d["test_path"],
            scale=d.get("scale", True),
            scaler_type=d.get("scaler_type", "standard"),
        )
    elif d["type"] == "Iris":
        from src.infrastructure.dataset.iris_adapter import IrisAdapter
        adapter = IrisAdapter(
            data_path=d["train_path"],  # Para Iris, train_path es la ruta al CSV
            scale=d.get("scale", True),
            scaler_type=d.get("scaler_type", "standard"),
            train_test_split_ratio=0.7,  # 70% train, 30% test
        )
    else:
        from src.infrastructure.dataset.csv_adapter import GenericCsvAdapter
        adapter = GenericCsvAdapter(
            name=d.get("name", "custom"),
            train_path=d["train_path"],
            test_path=d.get("test_path") or None,
            target_column=d["target_column"],
            categorical_features=d.get("categorical_features", []),
            scale=d.get("scale", True),
            scaler_type=d.get("scaler_type", "standard"),
        )
    return adapter.load()
