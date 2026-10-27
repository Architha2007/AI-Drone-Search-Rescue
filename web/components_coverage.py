"""Streamlit sidebar controls for coverage mode."""

from __future__ import annotations

import streamlit as st

from config import PRESET_MAPS
from simulation.mission import MissionState
from web import session


def render_sidebar() -> None:
    mission = session.coverage_mission()

    st.subheader("Coverage controls")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Start / Pause", key="cov_start_pause", use_container_width=True):
            session.toggle_start_pause()
    with c2:
        if st.button("Step", key="cov_step", use_container_width=True):
            session.step_once()
    with c3:
        if st.button("Reset", key="cov_reset", use_container_width=True):
            session.reset_mission()

    if st.button("New random map", key="cov_new_map", use_container_width=True):
        session.new_random_map()

    algo = st.selectbox(
        "Pathfinder",
        options=["astar", "bfs"],
        index=0 if st.session_state.coverage_algorithm == "astar" else 1,
        format_func=lambda x: "A*" if x == "astar" else "BFS",
        key="cov_algo_select",
    )
    if algo != st.session_state.coverage_algorithm:
        st.session_state.coverage_algorithm = algo
        session.set_coverage_algorithm(algo)

    preset = st.selectbox(
        "Scenario preset",
        options=["(Random map)"] + list(PRESET_MAPS.keys()),
        key="cov_preset_select",
    )
    if preset != "(Random map)":
        if st.button(f"Load preset: {preset}", key="cov_load_preset", use_container_width=True):
            session.load_preset(preset)

    st.session_state.speed = st.select_slider(
        "Simulation speed (steps per refresh)",
        options=[1, 5, 10, 20],
        value=st.session_state.speed,
        key="cov_speed",
    )

    st.divider()
    st.subheader("Mission status")
    state_label = mission.state.name
    st.metric("State", state_label)
    st.metric("Coverage", f"{mission.coverage_percentage:.1f}%")
    st.progress(min(1.0, mission.coverage_percentage / 100.0))
    st.metric("Survivors discovered", mission.survivors_discovered)
    st.metric("Survivors reached", mission.survivors_reached)
    st.metric("Pathfinder ops", mission.pathfinding_operations)
    st.caption(mission.status_message)

    if mission.state == MissionState.COMPLETED:
        st.success("Coverage mission complete.")
