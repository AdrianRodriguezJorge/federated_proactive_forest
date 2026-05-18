"""Reusable dashboard for progressive aggregation convergence results."""

from typing import Any, Dict
import pandas as pd
import streamlit as st


def render_convergence_dashboard(results: Any, config: Dict[str, Any]) -> None:
    """Renders charts showing performance and forest size evolution.

    Args:
        results (Any): FLResults from the federated round.
        config (Dict[str, Any]): Entire experiment configuration dict.
    """
    if not hasattr(results, "round_logs") or not results.round_logs:
        st.warning(
            "No se encontraron logs detallados de rondas para esta "
            "ejecución."
        )
        return

    st.divider()
    st.subheader("📊 Dashboard de Convergencia Progresiva")

    rev_c1, rev_c2, rev_c3 = st.columns(3)

    logs = results.round_logs
    is_converged = results.convergence_round is not None

    with rev_c1:
        st.metric("Episodios totales", len(logs))
    with rev_c2:
        conv_text = (
            f"Episodio {results.convergence_round}"
            if is_converged
            else "No alcanzó"
        )
        st.metric("Convergencia", conv_text)
    with rev_c3:
        if len(logs) >= 2:
            improvement = logs[-1].get("accuracy", 0.0) - logs[-2].get(
                "accuracy", 0.0
            )
            st.metric("Mejora final", f"{improvement:.5f}")
        else:
            st.metric("Mejora final", "N/A")

    if is_converged:
        st.success(
            f"🎯 **Parada Temprana**: El modelo convergió en el "
            f"episodio {results.convergence_round}."
        )
    else:
        agg_cfg = config.get("aggregation", {})
        conv_thr = agg_cfg.get(
            "global_convergence_threshold", agg_cfg.get("convergence", 0.002)
        )
        st.warning(
            f"⚠️ **Límite alcanzado**: No se detectó convergencia "
            f"significativa (Umbral: {conv_thr})."
        )

    st.write("---")
    plot_rows = []
    for log in logs:
        r_idx = log.get("round") or log.get("episode") or 0
        acc = log.get("accuracy") or log.get("round_accuracy") or 0.0
        trees = log.get("n_trees") or log.get("trees_after") or 0
        f1 = log.get("macro_f1") or 0.0

        plot_rows.append(
            {
                "Ronda": r_idx,
                "Accuracy (Val)": acc,
                "Macro-F1 (Val)": f1,
                "Árboles Totales": trees,
            }
        )

    df_plot = pd.DataFrame(plot_rows)

    st.write("**Evolución de Desempeño Global**")
    st.line_chart(
        df_plot.set_index("Ronda")[["Accuracy (Val)", "Macro-F1 (Val)"]]
    )

    st.write("**Evolución del tamaño del bosque**")
    st.bar_chart(df_plot.set_index("Ronda")["Árboles Totales"])

    st.write("**Detalle de agregación por episodio**")
    st.dataframe(df_plot.set_index("Ronda"), use_container_width=True)
