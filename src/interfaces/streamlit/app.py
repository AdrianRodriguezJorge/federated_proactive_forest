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

# ── Custom CSS for Theme Consistency ─────────────────────────────────────────
st.markdown("""
    <style>
    /* MIMETIZAR MENSAJES CON EL FONDO DE LA PÁGINA (SAGE THEME) */
    [data-testid="stAlert"], .stAlert {
        background: #F1F3E0 !important;
        background-color: #F1F3E0 !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Asegurar que los contenedores internos también sean del color del fondo */
    [data-testid="stAlert"] > div, .stAlert > div {
        background: #F1F3E0 !important;
        background-color: #F1F3E0 !important;
    }

    /* Asegurar que todo el texto sea del color oscuro restaurado */
    [data-testid="stAlert"] div, [data-testid="stAlert"] p, [data-testid="stAlert"] span {
        color: #2C2C2C !important;
        font-weight: 500 !important;
    }

    /* COLOR DEL ICONO Y TEXTO DE ALERTAS (Sincronizado con el tema) */
    [data-testid="stAlert"] svg {
        fill: #2C2C2C !important;
    }

    /* Mantener alertas minimalistas (sin fondo azul, texto oscuro) */
    div[data-testid="stNotificationContentSuccess"], 
    div[data-testid="stNotificationContentInfo"],
    .stAlert {
        background-color: #F1F3E0 !important;
        background: #F1F3E0 !important;
        border: none !important;
        box-shadow: none !important;
        border-radius: 0.5rem !important;
        overflow: hidden !important;
    }

    [data-testid="stAlert"] div, [data-testid="stAlert"] p, [data-testid="stAlert"] span {
        color: #2C2C2C !important;
        font-weight: 500 !important;
    }

    /* BOTONES EN NEGRITA */
    button p {
        font-weight: bold !important;
    }
    </style>
""", unsafe_allow_html=True)

# Auto-load last config from file if session state is empty
if st.session_state.get("fl_config") is None:
    from src.interfaces.streamlit.pages_manual.page_config import load_config_from_file, _load_dataset
    saved_cfg = load_config_from_file()
    if saved_cfg:
        try:
            # We don't load the full dataset here to keep app startup fast, 
            # but we load the metadata so navigation is correct.
            # page_run or page_config will handle the full dataset load if needed.
            st.session_state["fl_config"] = saved_cfg
        except Exception:
            pass

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

# ── Scroll to Top on Page Change ─────────────────────────────────────────────
if "last_page" not in st.session_state:
    st.session_state.last_page = page

if st.session_state.last_page != page:
    st.session_state.last_page = page
    # Inject JS to scroll the main container to the top
    st.components.v1.html(
        """
        <script>
            window.parent.document.querySelector('section.main').scrollTo(0, 0);
        </script>
        """,
        height=0
    )

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
