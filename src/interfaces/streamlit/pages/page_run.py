"""Página 2 — Ejecutar el experimento federado."""
import streamlit as st
import time
import json
from pathlib import Path

# Importar adaptadores de dataset
from src.infrastructure.dataset.iris_adapter import IrisAdapter
from src.infrastructure.dataset.nslkdd_adapter import NslKddAdapter
from src.infrastructure.dataset.csv_adapter import GenericCsvAdapter

# Configuración de persistencia (igual que en page_config.py)
CONFIG_DIR = Path("config")
CONFIG_FILE = CONFIG_DIR / "last_config.json"


def load_config_from_file() -> dict:
    """Carga la configuración desde un archivo JSON."""
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"No se pudo cargar la configuración guardada: {e}")
    return {}


def create_dataset_adapter(config: dict):
    """Crea el adaptador de dataset basado en la configuración."""
    dataset_config = config["dataset"]

    if dataset_config["type"] == "Iris":
        return IrisAdapter(
            data_path=dataset_config.get("file_path", "data/iris.csv"),
            train_test_split_ratio=1 - dataset_config.get("test_size", 0.2),
            scale=dataset_config.get("scale", True),
            scaler_type=dataset_config.get("scaler_type", "standard")
        )
    elif dataset_config["type"] == "NSL-KDD":
        return NslKddAdapter(
            train_path="data/NSL-KDD_train.csv",
            test_path="data/NSL-KDD_test.csv",
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


def _load_dataset(config: dict):
    """Carga el dataset usando el adaptador correspondiente."""
    adapter = create_dataset_adapter(config)
    return adapter.load()


def render():
    st.header("▶️ Ejecutar Experimento Federado")

    # Intentar obtener configuración de session_state
    cfg = st.session_state.get("fl_config")
    
    # Si no hay configuración en session_state, intentar cargar la última guardada
    if not cfg:
        st.info("🔄 Cargando última configuración guardada...")
        saved_cfg = load_config_from_file()
        if saved_cfg:
            try:
                # Cargar configuración guardada y crear dataset_split
                ds = _load_dataset(saved_cfg)
                saved_cfg["_dataset_split"] = ds
                st.session_state["fl_config"] = saved_cfg
                cfg = saved_cfg
                st.success("✅ Configuración cargada exitosamente")
            except Exception as e:
                st.error(f"❌ Error al cargar la configuración guardada: {e}")
                import traceback
                with st.expander("Traceback completo"):
                    st.code(traceback.format_exc())
                st.warning("⚠️ Crea una nueva configuración en la página '⚙️ Configuración'.")
                return
        else:
            st.warning("⚠️ No hay configuración guardada. Crea una configuración en la página '⚙️ Configuración'.")
            return

    # Resumen de la config activa
    with st.expander("📋 Configuración activa", expanded=False):
        clean = {k: v for k, v in cfg.items() if k != "_dataset_split"}
        st.json(clean)

    ds = cfg.get("_dataset_split")
    if ds:
        col1, col2, col3 = st.columns(3)
        col1.metric("Dataset", ds.dataset_name)
        col2.metric("Train / Test", f"{ds.X_train.shape[0]} / {ds.X_test.shape[0]}")
        col3.metric("Clases", len(ds.class_names))

    st.divider()

    if st.button("🚀 Ejecutar ronda federada", type="primary"):
        progress  = st.progress(0, text="Inicializando...")
        status    = st.empty()
        log_lines = []
        log_box   = st.expander("📜 Log de ejecución", expanded=True)

        def step_cb(msg: str, pct: int, detail: str = ""):
            progress.progress(pct / 100, text=f"{msg} ({pct}%)")
            status.info(f"**{msg}**  {detail}")
            ts = time.strftime("%H:%M:%S")
            log_lines.append(f"[{ts}] {pct:3d}% | {msg} {detail}")
            log_box.code("\n".join(log_lines[-30:]))

        try:
            from src.application.fl_orchestrator import FLEXOrchestrator
            orch    = FLEXOrchestrator(cfg, step_callback=step_cb)
            # Get dataset split from config
            ds = cfg.get('_dataset_split')
            orch.setup_federation(ds)
            results = orch.run_federated_round()

            st.session_state["fl_results"] = results
            progress.progress(1.0, text="✅ Completado")
            st.success("🎉 Ronda federada completada. Explora los resultados en Ranking y Métricas.")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy global",  f"{results.global_accuracy:.4f}")
            c2.metric("Macro-F1 global",  f"{results.global_macro_f1:.4f}")
            c3.metric("Árboles global",   results.n_trees_global)
            c4.metric("Estrategia",       results.strategy_id)

            # Tabla rápida de clientes
            st.subheader("Resumen por cliente")
            import pandas as pd
            rows = []
            for cid in results.client_ids:
                meta = results.client_metadata.get(cid)
                rep  = results.client_reports.get(cid)
                if meta and rep:
                    rows.append({
                        "Cliente":      cid,
                        "Árboles loc.": int(meta.n_trees) if meta.n_trees else 0,
                        "Acc local":    f"{meta.accuracy:.4f}",
                        "F1 local":     f"{meta.macro_f1:.4f}",
                        "PCD local":    f"{meta.pcd:.4f}",
                        "Sel. en global": int(len(results.selected_ids.get(cid, []))),
                        "Acc extendido": f"{rep.accuracy:.4f}",
                    })
            if rows:
                df_summary = pd.DataFrame(rows).set_index("Cliente")
                st.dataframe(df_summary, use_container_width=True)

        except Exception as exc:
            st.error(f"❌ Error durante la ejecución:\n\n```\n{exc}\n```")
            import traceback
            with st.expander("Traceback completo"):
                st.code(traceback.format_exc())
