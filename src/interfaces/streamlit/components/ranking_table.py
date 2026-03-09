"""Ranking table component for Streamlit."""
import streamlit as st
import pandas as pd
from typing import List, Dict, Any


def render_ranking_table(tree_entries: List[Any], selected_ids: Dict[str, List[int]],
                        client_colors: Dict[str, str]) -> None:
    """
    Render the ranking table component.

    Args:
        tree_entries: List of tree entries
        selected_ids: Selected tree IDs per client
        client_colors: Color mapping for clients
    """
    if not tree_entries:
        st.warning("No tree entries to display")
        return

    # Convert to dataframe
    data = []
    for entry in tree_entries:
        is_selected = entry.tree_local_id in selected_ids.get(entry.client_id, [])
        data.append({
            'Client': entry.client_id,
            'Tree ID': f"{entry.client_id}_{entry.tree_local_id}",
            'Accuracy': entry.accuracy,
            'F1': entry.macro_f1,
            'PCD': entry.pcd,
            'Score': entry.score,
            'Selected': '✅' if is_selected else '❌'
        })

    df = pd.DataFrame(data)

    # Apply styling
    def color_selected(val):
        if val == '✅':
            return 'background-color: #d4edda'
        return ''

    styled_df = df.style.applymap(color_selected, subset=['Selected'])

    # Add client colors
    def color_client(client):
        return f'background-color: {client_colors.get(client, "#ffffff")}'

    styled_df = styled_df.applymap(color_client, subset=['Client'])

    st.dataframe(styled_df, use_container_width=True)

    # Summary stats
    total = len(tree_entries)
    selected = sum(1 for entry in tree_entries
                  if entry.tree_local_id in selected_ids.get(entry.client_id, []))

    st.caption(f"Total trees: {total} | Selected: {selected} | Discarded: {total - selected}")