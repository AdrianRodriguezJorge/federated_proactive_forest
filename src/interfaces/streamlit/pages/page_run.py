"""Página 2 — Ejecutar el experimento federado."""
import streamlit as st
import time


def render():
    st.header("▶️ Ejecutar Experimento Federado")

    cfg = st.session_state.get("fl_config")
    if not cfg:
        st.warning("⚠️ Primero guarda la configuración en la página '⚙️ Configuración'.")
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
