"""
Entrypoint de la interfaz Streamlit.
Ejecutar: streamlit run src/interfaces/streamlit/app.py
"""
import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
from src.interfaces.streamlit.state.session_state import SessionState

st.set_page_config(
    page_title="Federated Proactive Forest",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialize session state ─────────────────────────────────────────────────
SessionState.initialize_defaults()

# ── Navigation ────────────────────────────────────────────────────────────────
st.sidebar.title("🌲 Federated Proactive Forest")
st.sidebar.markdown("---")

# Get current strategy to customize sidebar
fl_config = st.session_state.get("fl_config") or {}
strategy = fl_config.get("aggregation", {}) or {}
strategy_id = strategy.get("strategy", "")
is_s9 = (strategy_id == "s9_roulette")

pages = [
    "⚙️  Configuración",
    "▶️  Ejecutar",
]

if is_s9:
    pages.append("🎰 Evolución de Ruleta")
else:
    pages.append("🏆 Ranking de Árboles")

pages.append("📊 Métricas")

page = st.sidebar.radio("Navegación", pages)

# Track current page
SessionState.set("current_page", page)

# ── Load selected page ───────────────────────────────────────────────────────
if page == "⚙️  Configuración":
    from src.interfaces.streamlit.pages_manual import page_config
    page_config.render()
elif page == "▶️  Ejecutar":
    from src.interfaces.streamlit.pages_manual import page_run
    page_run.render()
elif page == "🎰 Evolución de Ruleta":
    from src.interfaces.streamlit.pages_manual import page_roulette
    page_roulette.render()
elif page == "🏆 Ranking de Árboles":
    from src.interfaces.streamlit.pages_manual import page_ranking
    page_ranking.render()
elif page == "📊 Métricas":
    from src.interfaces.streamlit.pages_manual import page_metrics
    page_metrics.render()
