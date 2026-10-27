"""
Streamlit web UI for AI Drone Search & Rescue.

Run from the project root (this directory):
    streamlit run streamlit_app.py

Does not import Pygame or the desktop `ui/*` modules.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from web import session
from web.components_coverage import render_sidebar as render_coverage_sidebar
from web.components_search import render_sidebar as render_search_sidebar
from web.map_view import render_map_figure
from web.results_page import render as render_results


st.set_page_config(
    page_title="AI Drone SAR — Web",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

session.ensure_initialized()

st.title("AI Drone Search & Rescue")
st.caption("Intelligent Search Under Uncertainty — browser simulation (V1)")

section = st.radio(
    "Section",
    options=["Simulation", "Benchmark results"],
    horizontal=True,
    key="app_section",
)

if section == "Simulation":
    st.session_state.mode = st.radio(
        "Mode",
        options=["Coverage", "Search"],
        horizontal=True,
        key="mode_radio",
    )

    session.advance_simulation_if_running()

    sidebar = st.sidebar
    sidebar.header("Controls")
    if st.session_state.mode == "Coverage":
        render_coverage_sidebar()
    else:
        render_search_sidebar()

    fig = render_map_figure(
        mode=st.session_state.mode,
        coverage=session.coverage_mission(),
        search=session.search_mission(),
        show_heatmap=st.session_state.show_heatmap,
        debug_target=st.session_state.debug_target,
    )
    st.pyplot(fig, clear_figure=True)

    if session.mission_still_running_for_autoplay():
        time.sleep(max(0.02, 0.12 / max(1, int(st.session_state.speed))))
        st.rerun()
else:
    st.sidebar.header("Benchmark results")
    st.sidebar.caption("Read-only view of saved CSV and methodology report.")
    render_results()
