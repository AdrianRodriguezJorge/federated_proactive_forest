"""Reusable dashboard for progressive aggregation convergence results."""
import streamlit as st
import pandas as pd
from typing import Any

def render_convergence_dashboard(results: Any, config: dict):
    """Renders charts showing performance and forest size evolution across rounds."""
    if not hasattr(results, 'round_logs') or not results.round_logs:
        st.warning("No se encontraron logs detallados de rondas para esta ejecución.")
        return

    st.divider()
    st.subheader("📊 Dashboard de Convergencia Progresiva")
    
    rev_c1, rev_c2, rev_c3 = st.columns(3)
    
    logs = results.round_logs
    is_converged = results.convergence_round is not None
    
    with rev_c1:
        st.metric("Episodios totales", len(logs))
    with rev_c2:
        st.metric("Convergencia", f"Episodio {results.convergence_round}" if is_converged else "No alcanzó")
    with rev_c3:
        if len(logs) >= 2:
            improvement = logs[-1].get('accuracy', 0.0) - logs[-2].get('accuracy', 0.0)
            st.metric("Mejora final", f"{improvement:.5f}")
        else:
            st.metric("Mejora final", "N/A")
            
    if is_converged:
        st.success(f"🎯 **Parada Temprana**: El modelo convergió en el episodio {results.convergence_round}.")
    else:
        conv_thr = config.get("aggregation", {}).get("global_convergence_threshold", 
                    config.get("aggregation", {}).get("convergence", 0.002))
        st.warning(f"⚠️ **Límite alcanzado**: No se detectó convergencia significativa (Umbral: {conv_thr}).")
    
    st.write("---")
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
    
    st.write("**Detalle de agregación por episodio**")
    st.dataframe(df_plot.set_index('Ronda'), use_container_width=True)
