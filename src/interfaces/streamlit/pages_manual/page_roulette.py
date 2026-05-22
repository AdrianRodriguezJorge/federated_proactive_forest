"""Página 5 — Evolución de la Ruleta Global (S9)."""

import pandas as pd
import streamlit as st


def render() -> None:
    """Renders page_roulette Streamlit page for strategy S9."""
    st.header("🎰 Evolución de la Ruleta Global (S9)")

    results = st.session_state.get("fl_results")
    if not results or not hasattr(results, "roulette_history"):
        st.warning("⚠️ Esta página solo está para experimentos S9.")
        return

    st.subheader("📈 Historial de Probabilidades de Atributos")

    exp_title = "📈 Evolución de Probabilidades de Atributos"
    with st.expander(exp_title, expanded=True):
        from src.interfaces.streamlit.components.s9_dashboard import (
            render_s9_dashboard,
        )

        render_s9_dashboard(results)

    st.divider()

    # Communication cost breakdown
    st.subheader("📡 Costo de Comunicación")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Bytes", f"{results.total_communication_bytes} B")
        st.metric(
            "Total KB", f"{results.total_communication_bytes / 1024:.2f} KB"
        )

    with col2:
        st.metric("Variante", results.roulette_variant)
        st.metric("Peso Ruleta Local", f"{results.local_roulette_weight:.2f}")

    # Round breakdown table
    st.write("**Detalle de comunicación por ronda**")
    comm_data = []
    up_list = results.upload_bytes_per_round
    down_list = results.download_bytes_per_round
    for i, (up, down) in enumerate(zip(up_list, down_list)):
        comm_data.append(
            {
                "Ronda": i + 1,
                "Upload (B)": up,
                "Download (B)": down,
                "Total (B)": up + down,
            }
        )
    df_comm = pd.DataFrame(comm_data).set_index("Ronda")
    st.dataframe(df_comm, use_container_width=True)
