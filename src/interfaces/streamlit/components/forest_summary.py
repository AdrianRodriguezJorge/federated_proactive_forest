"""Forest summary component."""
import streamlit as st
from typing import Any


def render_forest_summary(forest: Any, metadata: Any = None) -> None:
    """
    Render forest summary information.

    Args:
        forest: Forest model
        metadata: Optional metadata
    """
    st.subheader("Forest Summary")

    if hasattr(forest, 'get_trees'):
        n_trees = len(forest.get_trees())
        st.metric("Number of Trees", n_trees)

    if metadata:
        if hasattr(metadata, 'n_trees'):
            st.metric("Local Trees", metadata.n_trees)
        if hasattr(metadata, 'selected_local_tree_ids'):
            st.metric("Selected Global Trees", len(metadata.selected_local_tree_ids))
        if hasattr(metadata, 'accuracy'):
            st.metric("Local Accuracy", f"{metadata.accuracy:.4f}")
        if hasattr(metadata, 'macro_f1'):
            st.metric("Local Macro F1", f"{metadata.macro_f1:.4f}")
        if hasattr(metadata, 'pcd'):
            st.metric("Local PCD", f"{metadata.pcd:.4f}")