"""Metrics panel component for Streamlit."""
import streamlit as st
import pandas as pd
from typing import Dict, Any
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def render_metrics_panel(report: Any, title: str) -> None:
    """
    Render the metrics panel component.

    Args:
        report: Forest report object
        title: Panel title
    """
    st.subheader(title)

    # KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{report.accuracy:.4f}")
    col2.metric("Macro F1", f"{report.macro_f1:.4f}")
    col3.metric("Macro Precision", f"{report.macro_precision:.4f}")
    col4.metric("Macro Recall", f"{report.macro_recall:.4f}")

    # Additional metrics
    col5, col6, col7 = st.columns(3)
    col5.metric("PCD", f"{report.pcd:.4f}")
    col6.metric("Forest Size", report.forest_size)

    # Confusion matrix
    if hasattr(report, 'confusion_matrix') and report.confusion_matrix is not None:
        st.subheader("Confusion Matrix")
        if PLOTLY_AVAILABLE:
            fig = go.Figure(data=go.Heatmap(
                z=report.confusion_matrix,
                x=list(range(report.confusion_matrix.shape[1])),
                y=list(range(report.confusion_matrix.shape[0])),
                colorscale='Blues'
            ))
            fig.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="Actual"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(report.confusion_matrix, use_container_width=True)
            st.caption("Install plotly for interactive heatmap: pip install plotly")

    # Per-class metrics if available
    if hasattr(report, 'per_class_f1') and report.per_class_f1:
        st.subheader("Per-Class F1 Scores")
        classes = list(report.per_class_f1.keys())
        f1_scores = list(report.per_class_f1.values())

        if PLOTLY_AVAILABLE:
            fig = go.Figure(data=[
                go.Bar(x=classes, y=f1_scores, marker_color='lightblue')
            ])
            fig.update_layout(
                title="F1 Score by Class",
                xaxis_title="Class",
                yaxis_title="F1 Score"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            per_class_df = pd.DataFrame({
                "Class": classes,
                "F1-Score": f1_scores
            })
            st.bar_chart(per_class_df.set_index("Class"))