"""Centralized session state management for Streamlit app."""
import streamlit as st
from typing import Any


class SessionState:
    """Manages Streamlit session state in a centralized way."""

    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """Get value from session state."""
        return st.session_state.get(key, default)

    @staticmethod
    def set(key: str, value: Any) -> None:
        """Set value in session state."""
        st.session_state[key] = value

    @staticmethod
    def initialize_defaults() -> None:
        """Initialize default values in session state."""
        defaults = {
            "fl_config": None,
            "fl_results": None,
            "current_page": "config"
        }

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @staticmethod
    def clear_results() -> None:
        """Clear experiment results."""
        st.session_state["fl_results"] = None

    @staticmethod
    def is_configured() -> bool:
        """Check if configuration is set."""
        return st.session_state.get("fl_config") is not None

    @staticmethod
    def is_executed() -> bool:
        """Check if experiment has been executed."""
        return st.session_state.get("fl_results") is not None