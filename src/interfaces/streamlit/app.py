"""
Entrypoint de la interfaz Streamlit.
Ejecutar: streamlit run src/interfaces/streamlit/app.py
"""
import sys
import os
# Asegurar que el root del proyecto está en el path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st

st.set_page_config(
    page_title="Federated Proactive Forest",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Navegación manual con selectbox en sidebar ────────────────────────────────
pages = {
    "⚙️  Configuración":  "config",
    "▶️  Ejecutar":        "run",
    "🏆 Ranking de Árboles": "ranking",
    "📊 Métricas":         "metrics",
}

st.sidebar.title("🌲 Federated Proactive Forest")
st.sidebar.markdown("---")
page = st.sidebar.selectbox("Navegación", list(pages.keys()))

# Estado global
if "fl_config"  not in st.session_state: st.session_state["fl_config"]  = None
if "fl_results" not in st.session_state: st.session_state["fl_results"] = None

# ── Cargar página seleccionada ────────────────────────────────────────────────
key = pages[page]

if key == "config":
    from src.interfaces.streamlit.pages import page_config
    page_config.render()
elif key == "run":
    from src.interfaces.streamlit.pages import page_run
    page_run.render()
elif key == "ranking":
    from src.interfaces.streamlit.pages import page_ranking
    page_ranking.render()
elif key == "metrics":
    from src.interfaces.streamlit.pages import page_metrics
    page_metrics.render()
