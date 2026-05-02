"""Página 2 — Ejecutar el experimento federado."""
import streamlit as st
import time
import json
import pandas as pd
from src.interfaces.streamlit.pages_manual.page_config import (
    CONFIG_FILE, _load_dataset
)

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
            from src.application.orchestrators import FLEXOrchestrator
            orch    = FLEXOrchestrator(cfg, step_callback=step_cb)
            # Get dataset split from config
            ds = cfg.get('_dataset_split')
            orch.setup_federation(ds)
            results = orch.run_federated_round()

            st.session_state["fl_results"] = results
            progress.progress(1.0, text="✅ Completado")
            st.success(" Ronda federada completada. Explora los resultados en Ranking y Métricas.")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy global",  f"{results.global_accuracy:.4f}")
            c2.metric("Macro-F1 global",  f"{results.global_macro_f1:.4f}")
            c3.metric("Árboles global",   results.n_trees_global)
            c4.metric("Estrategia",       results.strategy_id)

            # Progressive Forest Convergence Dashboard (S2-S7, PW)
            if hasattr(results, 'round_logs') and results.round_logs:
                st.divider()
                st.subheader("📊 Dashboard de Convergencia Progresiva")
                
                # Metrics at the top
                rev_c1, rev_c2, rev_c3 = st.columns(3)
                
                is_converged = results.convergence_round is not None
                
                if is_converged:
                    rev_c1.metric("Episodio/Ronda de parada", results.convergence_round)
                    logs = results.round_logs
                    last_idx = results.convergence_round - 1
                    if 0 < last_idx < len(logs):
                        current_acc = logs[last_idx].get('accuracy', 0.0)
                        prev_acc = logs[last_idx - 1].get('accuracy', 0.0)
                        improvement = current_acc - prev_acc
                        st.success(f"🎯 **Parada Temprana**: El modelo convergió en el episodio {results.convergence_round}. La mejora fue de `{improvement:.5f}` (<= 0.002).")
                    else:
                        st.success(f"🎯 **Parada Temprana**: El modelo convergió en el episodio {results.convergence_round}.")
                else:
                    rev_c1.metric("Episodio/Ronda de parada", "No hubo")
                    logs = results.round_logs
                    if len(logs) >= 2:
                        current_acc = logs[-1].get('accuracy', 0.0)
                        prev_acc = logs[-2].get('accuracy', 0.0)
                        improvement = current_acc - prev_acc
                        conv_thr = cfg.get("aggregation", {}).get("convergence", 0.002)
                        st.warning(f"⚠️ **Límite alcanzado**: No hubo convergencia. Se seleccionaron todos los árboles. Mejora final: `{improvement:.5f}` (> {conv_thr}).")
                    else:
                        st.warning("⚠️ **Límite alcanzado**: No hubo convergencia. Se seleccionaron todos los árboles disponibles.")
                
                with st.expander("📈 Visualizar evolución de la agregación", expanded=True):
                    # Prepare data for plotting with flexible key mapping
                    plot_rows = []
                    for log in results.round_logs:
                        # Map keys flexibly to support different strategy implementations
                        r_idx = log.get('round') or log.get('episode') or 0
                        acc   = log.get('accuracy') or log.get('round_accuracy') or 0.0
                        trees = log.get('n_trees') or log.get('trees_after') or 0
                        f1    = log.get('macro_f1') or 0.0
                        
                        plot_rows.append({
                            'Ronda': r_idx,
                            'Accuracy (Val)': acc,
                            'Macro-F1 (Val)': f1,
                            'Árboles Totales': trees
                        })
                    
                    df_plot = pd.DataFrame(plot_rows)
                    
                    # Accuracy/F1 Chart
                    st.write("**Evolución de Desempeño Global**")
                    st.line_chart(df_plot.set_index('Ronda')[['Accuracy (Val)', 'Macro-F1 (Val)']])
                    
                    # Forest Size Chart
                    st.write("**Evolución del tamaño del bosque**")
                    st.bar_chart(df_plot.set_index('Ronda')['Árboles Totales'])
                    
                    # Detailed table
                    st.write("**Detalle de agregación por episodio**")
                    st.dataframe(df_plot.set_index('Ronda'), use_container_width=True)
            elif results.strategy_id == "PW":
                st.warning("No se encontraron logs detallados de rondas para esta ejecución.")

            # Tabla rápida de clientes
            st.subheader("Resumen por cliente")
            from src.domain.metrics.forest_evaluator import ForestEvaluator
            rows = []
            for cid in results.client_ids:
                meta = results.client_metadata.get(cid)
                if meta:
                    # Calcular accuracy híbrida
                    y_pred = results.client_hybrid_predictions.get(cid)
                    if y_pred is not None:
                        hybrid_report = ForestEvaluator.evaluate_from_predictions(
                            y_pred, results.y_test, results.class_names, 0, 0.0
                        )
                        acc_hibrido = f"{hybrid_report.accuracy:.4f}"
                    else:
                        acc_hibrido = "N/A"
                    
                    rows.append({
                        "Cliente":      cid,
                        "Árboles loc.": int(meta.n_trees),
                        "Acc local":    f"{meta.accuracy:.4f}",
                        "F1 local":     f"{meta.macro_f1:.4f}",
                        "PCD local":    f"{meta.pcd:.4f}",
                        "Sel. en global": int(len(results.selected_ids.get(str(cid), []))),
                        "Acc híbrido": acc_hibrido,
                    })
            if rows:
                df_summary = pd.DataFrame(rows).set_index("Cliente")
                st.dataframe(df_summary, use_container_width=True)

        except Exception as exc:
            st.error(f"❌ Error durante la ejecución:\n\n```\n{exc}\n```")
            import traceback
            with st.expander("Traceback completo"):
                st.code(traceback.format_exc())
    
    # ── Persistent Results Display ──────────────────────────────────────────
    elif st.session_state.get("fl_results"):
        results = st.session_state["fl_results"]
        st.success("📊 Mostrando resultados de la última ejecución.")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy global",  f"{results.global_accuracy:.4f}")
        c2.metric("Macro-F1 global",  f"{results.global_macro_f1:.4f}")
        c3.metric("Árboles global",   results.n_trees_global)
        c4.metric("Estrategia",       results.strategy_id)

        # Progressive Forest Convergence Dashboard (S2-S7, PW)
        if hasattr(results, 'round_logs') and results.round_logs:
            st.divider()
            st.subheader("📊 Dashboard de Convergencia Progresiva")
            
            logs = results.round_logs
            rev_c1, rev_c2, rev_c3 = st.columns(3)
            with rev_c1:
                st.metric("Episodios", len(logs))
            with rev_c2:
                conv_round = results.convergence_round
                st.metric("Convergencia", f"Episodio {conv_round}" if conv_round else "No alcanzó")
            with rev_c3:
                if len(logs) >= 2:
                    improvement = logs[-1].get('accuracy', 0.0) - logs[-2].get('accuracy', 0.0)
                    st.metric("Mejora final", f"{improvement:.5f}")
                else:
                    st.metric("Mejora final", "N/A")

            if results.convergence_round:
                st.success(f"🎯 El modelo convergió en el episodio {results.convergence_round}.")
            else:
                st.warning("⚠️ El modelo se detuvo por alcanzar el límite de árboles (sin convergencia).")

            with st.expander("📈 Visualizar evolución de la agregación", expanded=True):
                plot_rows = []
                for log in logs:
                    r_idx = log.get('round') or log.get('episode') or 0
                    acc   = log.get('accuracy') or log.get('round_accuracy') or 0.0
                    trees = log.get('n_trees') or log.get('trees_after') or 0
                    f1    = log.get('macro_f1') or 0.0
                    plot_rows.append({
                        'Ronda': r_idx,
                        'Accuracy (Val)': acc,
                        'Macro-F1 (Val)': f1,
                        'Árboles Totales': trees
                    })
                df_plot = pd.DataFrame(plot_rows)
                st.write("**Evolución de Desempeño Global**")
                st.line_chart(df_plot.set_index('Ronda')[['Accuracy (Val)', 'Macro-F1 (Val)']])
                st.write("**Evolución del tamaño del bosque**")
                st.bar_chart(df_plot.set_index('Ronda')['Árboles Totales'])

        # Tabla rápida de clientes
        st.subheader("Resumen por cliente")
        from src.domain.metrics.forest_evaluator import ForestEvaluator
        rows = []
        for cid in results.client_ids:
            meta = results.client_metadata.get(cid)
            if meta:
                y_pred = results.client_hybrid_predictions.get(cid)
                if y_pred is not None:
                    hybrid_report = ForestEvaluator.evaluate_from_predictions(
                        y_pred, results.y_test, results.class_names, 0, 0.0
                    )
                    acc_hibrido = f"{hybrid_report.accuracy:.4f}"
                else:
                    acc_hibrido = "N/A"
                
                rows.append({
                    "Cliente":      cid,
                    "Árboles loc.": int(meta.n_trees),
                    "Acc local":    f"{meta.accuracy:.4f}",
                    "F1 local":     f"{meta.macro_f1:.4f}",
                    "PCD local":    f"{meta.pcd:.4f}",
                    "Sel. en global": int(len(results.selected_ids.get(str(cid), []))),
                    "Acc híbrido": acc_hibrido,
                })
        if rows:
            df_summary = pd.DataFrame(rows).set_index("Cliente")
            st.dataframe(df_summary, use_container_width=True)

