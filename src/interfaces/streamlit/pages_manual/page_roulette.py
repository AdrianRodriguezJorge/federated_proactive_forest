"""Página 5 — Evolución de la Ruleta Global (S9)."""
import streamlit as st
import pandas as pd
import numpy as np

def render():
    st.header("🎰 Evolución de la Ruleta Global (S9)")

    results = st.session_state.get("fl_results")
    if not results or not hasattr(results, 'roulette_history'):
        st.warning("⚠️ Esta página solo está disponible para experimentos S9.")
        return

    st.subheader("📈 Historial de Probabilidades de Atributos")
    
    with st.expander("📈 Evolución de Probabilidades de Atributos", expanded=True):
        from src.interfaces.streamlit.components.s9_dashboard import render_s9_dashboard
        render_s9_dashboard(results)

    st.divider()
    
    # Communication cost breakdown
    st.subheader("📡 Costo de Comunicación")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Bytes", f"{results.total_communication_bytes} B")
        st.metric("Total KB", f"{results.total_communication_bytes/1024:.2f} KB")
    
    with col2:
        st.metric("Variante", results.roulette_variant)
        st.metric("Beta (Balance)", f"{results.beta:.2f}")

    # Round breakdown table
    st.write("**Detalle de comunicación por ronda**")
    comm_data = []
    for i, (up, down) in enumerate(zip(results.upload_bytes_per_round, results.download_bytes_per_round)):
        comm_data.append({
            "Ronda": i + 1,
            "Upload (B)": up,
            "Download (B)": down,
            "Total (B)": up + down
        })
    st.dataframe(pd.DataFrame(comm_data).set_index("Ronda"), use_container_width=True)
