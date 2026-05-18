"""Reusable dashboard for S9 Global Attribute Roulette results."""

from typing import Any
import pandas as pd
import streamlit as st


def render_s9_dashboard(results: Any) -> None:
    """Renders S9 specific metrics, roulette evolution, and cost.

    Args:
        results (Any): FLResults from the federated round.
    """
    st.divider()
    st.subheader("🎰 Dashboard de Ruleta Global")

    rev_c1, rev_c2, rev_c3 = st.columns(3)
    rev_c1.metric("Rondas totales", results.num_rounds)

    conv_text = (
        f"Ronda {results.convergence_round}"
        if results.convergence_round
        else "Máx. alcanzado"
    )
    rev_c2.metric("Convergencia", conv_text)
    rev_c3.metric("Atributos", getattr(results, "n_features", "N/A"))

    st.write("---")
    # history is list of dicts: [{'round': 1, 'global_roulette': [...]}, ...]
    history = results.roulette_history
    if not history:
        st.info("No hay datos históricos de ruleta disponibles.")
        return

    rounds = [h["round"] for h in history]
    probs = [h["global_roulette"] for h in history]

    # Use real feature names if available
    n_features = len(probs[0])
    feature_names = getattr(results, "feature_names", None)
    if not feature_names or len(feature_names) != n_features:
        feature_names = [f"Feature {i}" for i in range(n_features)]

    df_hist = pd.DataFrame(probs, columns=feature_names, index=rounds)
    df_hist.index.name = "Ronda"

    st.write("**Historial de Ruleta Global**")
    st.line_chart(df_hist)

    # Heatmap
    st.write("**Mapa de Calor de Importancia**")
    try:
        import plotly.express as px

        fig = px.imshow(
            df_hist.T,
            labels=dict(x="Ronda", y="Atributo", color="Probabilidad"),
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.dataframe(
            df_hist.T.style.background_gradient(cmap="viridis"),
            use_container_width=True,
        )

    # Also show communication breakdown
    st.write("**Costo de Comunicación (Acumulado)**")
    comm_rows = []
    acc_bytes = 0

    # Ensure results has the lists
    fallback_val = results.total_communication_bytes / results.num_rounds
    fallback_list = [fallback_val] * results.num_rounds
    up_list = getattr(results, "upload_bytes_per_round", fallback_list)
    down_list = getattr(
        results, "download_bytes_per_round", [0] * results.num_rounds
    )

    for i, (up, down) in enumerate(zip(up_list, down_list)):
        acc_bytes += up + down
        comm_rows.append({"Ronda": i + 1, "KB Acumulado": acc_bytes / 1024})
    st.area_chart(pd.DataFrame(comm_rows).set_index("Ronda"))
