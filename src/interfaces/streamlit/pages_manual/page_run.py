"""Página 2 — Ejecutar el experimento federado."""
import streamlit as st
import time
import json
import pandas as pd
import numpy as np
from src.interfaces.streamlit.pages_manual.page_config import (
    CONFIG_FILE, _load_dataset
)
from src.interfaces.streamlit.state.experiment_logger import save_experiment_log

MAX_LOG_LINES = 50


def load_config_from_file() -> dict:
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"No se pudo cargar la configuración guardada: {e}")
    return {}


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

    # Lazy load dataset if config exists but dataset object is missing
    if cfg and cfg.get("_dataset_split") is None:
        with st.spinner("📦 Cargando dataset para el experimento..."):
            try:
                from src.interfaces.streamlit.pages_manual.page_config import _load_dataset
                ds = _load_dataset(cfg)
                cfg["_dataset_split"] = ds
                st.session_state["fl_config"] = cfg
            except Exception as e:
                st.error(f"❌ Error al cargar el dataset: {e}")

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

    # ── Re-run confirmation ───────────────────────────────────────────────────
    has_results = st.session_state.get("fl_results") is not None
    
    col_run, col_clear = st.columns([1, 4])
    with col_run:
        run_button = st.button("🚀 Ejecutar", type="primary", use_container_width=True)
    with col_clear:
        if has_results:
            if st.button("🗑️ Limpiar Resultados", use_container_width=False):
                st.session_state.pop("fl_results", None)
                st.rerun()

    if run_button:
        progress  = st.progress(0, text="Inicializando...")
        status    = st.empty()
        log_placeholder = st.empty()  # Para el log acumulativo
        log_lines = []  # Usar lista para evitar problemas de closure

        def step_cb(msg: str, pct: int, detail: str = ""):
            progress.progress(pct / 100, text=f"{msg} ({pct}%)")
            status.info(f"**{msg}**  {detail}")
            ts = time.strftime("%H:%M:%S")
            log_lines.append(f"[{ts}] {pct:3d}% | {msg} {detail}")
            # Limitar líneas de log para evitar fuga de memoria
            if len(log_lines) > MAX_LOG_LINES:
                log_lines.pop(0)
            log_placeholder.code("\n".join(log_lines))

        try:
            strategy_key = cfg.get("aggregation", {}).get("strategy", "")
            is_s9 = strategy_key == "s9_roulette"

            if is_s9:
                from src.application.orchestrators.roulette_orchestrator import RouletteOrchestrator
                orch = RouletteOrchestrator(cfg, step_callback=step_cb)
                ds = cfg.get('_dataset_split')
                orch.setup_federation(ds, seed=cfg.get("seed", 42))
                results = orch.run_federated_round(n_bootstrap=100)
            else:
                from src.application.orchestrators import FLEXOrchestrator
                orch    = FLEXOrchestrator(cfg, step_callback=step_cb)
                ds = cfg.get('_dataset_split')
                orch.setup_federation(ds)
                results = orch.run_federated_round(n_bootstrap=100)

            st.session_state["fl_results"] = results
            progress.progress(1.0, text="Completado")
            st.success("Ronda federada completada. Explora los resultados en Ranking y Metricas.")

            # ── Save experiment log ────────────────────────────────────────
            log_path = save_experiment_log(results, cfg)
            if log_path:
                st.toast(f"📝 Log guardado: {log_path.name}", icon="✅")

            # ── Dashboard de Resultados (KPIs) ──────────────────────────────────
            st.markdown("### 📊 Resumen de la Ejecución")
            
            # Fila 1: Métricas de eficacia
            row1_c1, row1_c2 = st.columns(2)
            with row1_c1:
                st.metric("🎯 Accuracy (Referencia Ensamble)", f"{results.global_accuracy:.4f}")
            with row1_c2:
                st.metric("📈 Macro-F1 (Referencia Ensamble)", f"{results.global_macro_f1:.4f}")

            # Fila 2: Métricas técnicas y federación
            row2_c1, row2_c2, row2_c3 = st.columns([1, 2, 1])
            with row2_c1:
                comm_kb = getattr(results, 'total_communication_bytes', 0) / 1024
                st.metric("📡 Coste Comm.", f"{comm_kb:.2f} KB")
            with row2_c2:
                is_s9 = "S9" in results.strategy_id
                strat_label = f"S9 ({getattr(results, 'roulette_variant', 'N/A')})" if is_s9 else results.strategy_id
                st.metric("🧠 Estrategia", strat_label)
            with row2_c3:
                if is_s9:
                    st.metric("⚖️ Beta (Local)", f"{getattr(results, 'beta', 0.0):.2f}")
                else:
                    st.metric("🌲 Árboles Global", results.n_trees_global)

            # ── Dashboards & Details (Optional expanders to avoid redundancy) ────────
            if is_s9:
                with st.expander("🎰 Ver detalles de la Ruleta Global", expanded=False):
                    from src.interfaces.streamlit.components.s9_dashboard import render_s9_dashboard
                    render_s9_dashboard(results)
            else:
                with st.expander("📊 Ver evolución de convergencia", expanded=False):
                    from src.interfaces.streamlit.components.convergence_dashboard import render_convergence_dashboard
                    render_convergence_dashboard(results, cfg)

            # ── Comparison Table (Quick View) ────────────────────────────────
            with st.expander("👥 Resumen por cliente (Vista rápida)", expanded=False):
                from src.interfaces.streamlit.components.metrics_table import render_comparison_table
                render_comparison_table(results)
            
            st.info("💡 Para un análisis profundo, curvas de precisión y matrices de confusión, dirígete a la página **📊 Métricas**.")

        except Exception as exc:
            st.error(f"❌ Error durante la ejecución:\n\n```\n{exc}\n```")
            import traceback
            with st.expander("Traceback completo"):
                st.code(traceback.format_exc())
    
    # ── Persistent Results Display ──────────────────────────────────────────
    elif st.session_state.get("fl_results"):
        results = st.session_state["fl_results"]
        st.success("📊 Resultados de la última ejecución listos.")
        
        # ── Dashboard de Resultados (KPIs) ──────────────────────────────────
        st.markdown("### 📊 Resumen de la Ejecución")
        
        # Fila 1: Métricas de eficacia
        row1_c1, row1_c2 = st.columns(2)
        with row1_c1:
            st.metric("🎯 Accuracy (Referencia Ensamble)", f"{results.global_accuracy:.4f}")
        with row1_c2:
            st.metric("📈 Macro-F1 (Referencia Ensamble)", f"{results.global_macro_f1:.4f}")

        # Fila 2: Métricas técnicas y federación
        row2_c1, row2_c2, row2_c3 = st.columns([1, 2, 1])
        with row2_c1:
            comm_kb = getattr(results, 'total_communication_bytes', 0) / 1024
            st.metric("📡 Coste Comm.", f"{comm_kb:.2f} KB")
        with row2_c2:
            is_s9 = "S9" in results.strategy_id
            strat_label = f"S9 ({getattr(results, 'roulette_variant', 'N/A')})" if is_s9 else results.strategy_id
            st.metric("🧠 Estrategia", strat_label)
        with row2_c3:
            if is_s9:
                st.metric("⚖️ Beta (Local)", f"{getattr(results, 'beta', 0.0):.2f}")
            else:
                st.metric("🌲 Árboles Global", results.n_trees_global)

        # ── Optional Dashboards & Table ──────────────────────────────────
        if is_s9:
            with st.expander("🎰 Ver detalles de la Ruleta Global", expanded=False):
                from src.interfaces.streamlit.components.s9_dashboard import render_s9_dashboard
                render_s9_dashboard(results)
        else:
            with st.expander("📊 Ver evolución de convergencia", expanded=False):
                from src.interfaces.streamlit.components.convergence_dashboard import render_convergence_dashboard
                render_convergence_dashboard(results, cfg)

        with st.expander("👥 Resumen por cliente (Vista rápida)", expanded=False):
            from src.interfaces.streamlit.components.metrics_table import render_comparison_table
            render_comparison_table(results)

