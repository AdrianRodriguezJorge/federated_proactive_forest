"""Página 4 — Métricas completas por cliente o modelo global."""
import streamlit as st
import pandas as pd


def render():
    st.header("📊 Métricas del Modelo")

    results = st.session_state.get("fl_results")
    if not results:
        st.warning("⚠️ Ejecuta primero el experimento en la página '▶️ Ejecutar'.")
        return

    is_s9 = results.strategy_id.startswith("S9")
    
    # ── Comparación de todos los clientes ───────────────────────────
    if is_s9:
        st.subheader("📈 Comparación de Desempeño entre Clientes")
    else:
        st.subheader("📈 Comparación de todos los Clientes vs Global")
    
    # Tabla comparativa (Global vs Clientes)
    from src.interfaces.streamlit.components.metrics_table import render_comparison_table
    render_comparison_table(results)
    
    st.divider()

    # ── Selector de modelo ────────────────────────────────────────────────────
    if is_s9:
        options = [f"👤 {cid}" for cid in results.client_ids]
        if not options:
            st.error("❌ No hay clientes disponibles para analizar.")
            return
        sel = st.selectbox("Seleccionar cliente a analizar", options)
        if sel is None:
            return
        cid = sel.replace("👤 ", "")
        report = results.client_reports.get(cid)
        if report is None and cid.isdigit():
            report = results.client_reports.get(int(cid))
        
        if report is None:
            st.error(f"No se encontraron reportes métricos para el cliente '{cid}'.")
            return
            
        title = f"Modelo Local — {cid}"
        meta = results.client_metadata.get(cid) or results.client_metadata.get(int(cid) if cid.isdigit() else cid)
    else:
        options = ["🌐 Modelo Global"] + [f"👤 {cid}" for cid in results.client_ids]
        sel = st.selectbox("Seleccionar modelo a analizar", options)

        if sel == "🌐 Modelo Global":
            report = results.global_report
            title  = "Modelo Global (Bosque Federado)"
            meta   = None
        else:
            cid    = sel.replace("👤 ", "")
            # For clients, use pre-calculated report
            report = results.client_reports.get(cid)
            if report is None and cid.isdigit():
                report = results.client_reports.get(int(cid))
            
            if report is None:
                st.error(f"No se encontraron reportes métricos para el cliente '{cid}'.")
                return
                
            title  = f"Modelo Local — {cid} (inferencia híbrida local+global)"
            meta   = results.client_metadata.get(cid) or results.client_metadata.get(int(cid) if cid.isdigit() else cid)

    st.subheader(f"📋 {title}")

    if meta:
        if isinstance(meta, dict):
            n_trees = meta.get('n_trees', 0)
            # S9 doesn't have "selected_local_tree_ids" as it builds iterative forests
            st.caption(f"Árboles locales en el bosque: {n_trees}")
        else:
            n_trees = getattr(meta, 'n_trees', 0)
            selected = len(getattr(meta, 'selected_local_tree_ids', []))
            st.caption(f"Árboles locales entrenados: {n_trees} | "
                       f"Seleccionados en global: {selected}")

    # ── KPIs ──────────────────────────────────────────────────────────────────
    if is_s9:
        st.markdown("#### Métricas del Cliente")
    else:
        st.markdown("#### Métricas Globales")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy",        f"{report.accuracy:.4f}")
    c2.metric("Macro-F1",        f"{report.macro_f1:.4f}")
    c3.metric("Macro Precision", f"{report.macro_precision:.4f}")
    c4.metric("Macro Recall",    f"{report.macro_recall:.4f}")
    c5.metric("PCD (diversidad)", f"{report.pcd:.4f}",
              help="Pair Classifier Disagreement — mayor = más diverso")
    c6.metric("Árboles en bosque", report.forest_size)

    st.divider()

    # ── Matriz de confusión ───────────────────────────────────────────────────
    st.markdown("#### Matriz de Confusión")
    cm = report.confusion_matrix
    class_names = report.class_names

    try:
        import plotly.express as px
        fig = px.imshow(cm, text_auto=True, aspect="auto",
                        x=class_names, y=class_names,
                        labels={"x": "Predicho", "y": "Real", "color": "Instancias"},
                        color_continuous_scale="Blues",
                        title="Matriz de Confusión")
        fig.update_layout(height=max(300, 60 * len(class_names)))
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        # Fallback sin plotly
        df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
        st.dataframe(df_cm.style.background_gradient(cmap="Blues"), use_container_width=True)

    st.divider()

    # ── Métricas por clase ────────────────────────────────────────────────────
    st.markdown("#### Métricas por Clase")
    per_df = pd.DataFrame({
        "Clase":     class_names,
        "Precision": [round(report.per_class_prec[c], 4)   for c in class_names],
        "Recall":    [round(report.per_class_recall[c], 4) for c in class_names],
        "F1-Score":  [round(report.per_class_f1[c], 4)     for c in class_names],
    })

    try:
        import plotly.express as px
        melted = per_df.melt(id_vars="Clase", var_name="Métrica", value_name="Valor")
        fig2 = px.bar(melted, x="Clase", y="Valor", color="Métrica", barmode="group",
                      title="Precision / Recall / F1 por clase",
                      color_discrete_map={
                          "Precision": "#2E75B6",
                          "Recall":    "#00708F",
                          "F1-Score":  "#375623",
                      })
        fig2.update_layout(height=400, yaxis_range=[0, 1])
        st.plotly_chart(fig2, use_container_width=True)
    except ImportError:
        pass

    st.dataframe(per_df.set_index("Clase"), use_container_width=True)
