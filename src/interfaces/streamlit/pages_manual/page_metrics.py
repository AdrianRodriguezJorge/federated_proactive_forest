"""Página 4 — Métricas completas por cliente o modelo global."""
import streamlit as st
import pandas as pd


def render():
    st.header("📊 Métricas del Modelo")

    results = st.session_state.get("fl_results")
    if not results:
        st.warning("⚠️ Ejecuta primero el experimento en la página '▶️ Ejecutar'.")
        return

    # ── Comparación de todos los clientes vs Global ───────────────────────────
    st.subheader("📈 Comparación de todos los Clientes vs Global")
    
    # Preparar datos para la tabla
    comparison_data = []
    
    # Modelo Global
    global_report = results.global_report
    comparison_data.append({
        "Modelo": "🌐 Global",
        "Accuracy": f"{global_report.accuracy:.4f}",
        "Macro-F1": f"{global_report.macro_f1:.4f}",
        "Macro Precision": f"{global_report.macro_precision:.4f}",
        "Macro Recall": f"{global_report.macro_recall:.4f}",
        "PCD": f"{global_report.pcd:.4f}",
        "Tamaño Bosque": global_report.forest_size
    })
    
    # Clientes (con inferencia híbrida)
    from src.domain.metrics.forest_evaluator import ForestEvaluator
    for cid in results.client_ids:
        y_pred = results.client_hybrid_predictions.get(cid)
        meta = results.client_metadata.get(cid)
        if y_pred is not None:
            forest_size = results.client_hybrid_forest_sizes.get(cid, 0)
            client_report = ForestEvaluator.evaluate_from_predictions(
                y_pred, results.y_test, results.class_names, forest_size, pcd=0.0
            )
            pcd_value = f"{meta.pcd:.4f}" if meta else "N/A"
            comparison_data.append({
                "Modelo": f"👤 {cid}",
                "Accuracy": f"{client_report.accuracy:.4f}",
                "Macro-F1": f"{client_report.macro_f1:.4f}",
                "Macro Precision": f"{client_report.macro_precision:.4f}",
                "Macro Recall": f"{client_report.macro_recall:.4f}",
                "PCD": pcd_value,
                "Tamaño Bosque": client_report.forest_size
            })
    
    # Mostrar tabla
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison.set_index("Modelo"), use_container_width=True)
    
    st.divider()

    # ── Selector de modelo ────────────────────────────────────────────────────
    options = ["🌐 Modelo Global"] + [f"👤 {cid}" for cid in results.client_ids]
    sel = st.selectbox("Seleccionar modelo a analizar", options)

    if sel == "🌐 Modelo Global":
        report = results.global_report
        title  = "Modelo Global (Bosque Federado)"
        meta   = None
    else:
        cid    = sel.replace("👤 ", "")
        # For clients, compute report from hybrid predictions
        from src.domain.metrics.forest_evaluator import ForestEvaluator
        y_pred = results.client_hybrid_predictions.get(cid)
        if y_pred is None:
            st.error("No se encontraron predicciones híbridas para este cliente.")
            return
        forest_size = results.client_hybrid_forest_sizes.get(cid, 0)
        report = ForestEvaluator.evaluate_from_predictions(
            y_pred, results.y_test, results.class_names, forest_size, pcd=0.0
        )
        title  = f"Modelo Local — {cid} (inferencia híbrida local+global)"
        meta   = results.client_metadata.get(cid)

    st.subheader(f"📋 {title}")

    if meta:
        st.caption(f"Árboles locales entrenados: {meta.n_trees} | "
                   f"Seleccionados en global: {len(meta.selected_local_tree_ids)}")

    # ── KPIs ──────────────────────────────────────────────────────────────────
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
