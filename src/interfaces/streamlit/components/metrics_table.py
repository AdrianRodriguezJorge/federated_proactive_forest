"""Reusable component for the Global vs Clients metrics comparison table."""

from typing import Any
import pandas as pd
import streamlit as st


def render_comparison_table(results: Any) -> None:
    """Renders a dataframe comparing global model vs each client.

    Args:
        results (Any): FLResults from the federated round.
    """
    comparison_data = []

    # Global Model (Skip for S8 strategy as requested)
    is_s8 = results.strategy_id.startswith("S8")
    if not is_s8:
        global_report = results.global_report
        comparison_data.append(
            {
                "Modelo": "🌐 Global",
                "Accuracy": f"{global_report.accuracy:.4f}",
                "Macro-F1": f"{global_report.macro_f1:.4f}",
                "Macro Precision": f"{global_report.macro_precision:.4f}",
                "Macro Recall": f"{global_report.macro_recall:.4f}",
                "PCD": f"{global_report.pcd:.4f}",
                "Tamaño Bosque": global_report.forest_size,
            }
        )

    # Clients
    for cid in results.client_ids:
        report = results.client_reports.get(
            cid
        ) or results.client_reports.get(str(cid))
        if report:
            comparison_data.append(
                {
                    "Modelo": f"👤 {cid}",
                    "Accuracy": f"{report.accuracy:.4f}",
                    "Macro-F1": f"{report.macro_f1:.4f}",
                    "Macro Precision": f"{report.macro_precision:.4f}",
                    "Macro Recall": f"{report.macro_recall:.4f}",
                    "PCD": f"{report.pcd:.4f}",
                    "Tamaño Bosque": report.forest_size,
                }
            )

    if not comparison_data:
        st.warning(
            "⚠️ No se encontraron métricas para mostrar en la tabla "
            "comparativa."
        )
        return

    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison.set_index("Modelo"), use_container_width=True)
