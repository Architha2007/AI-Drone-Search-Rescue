"""Streamlit sidebar controls for probabilistic search mode."""

from __future__ import annotations

import streamlit as st

from config import PRESET_MAPS
from simulation.search_mission import SearchMissionState
from web import session
from web.policy_factory import SEARCH_POLICY_NAMES


def render_sidebar() -> None:
    sm = session.search_mission()

    st.subheader("Search controls")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Start / Pause", key="srch_start_pause", use_container_width=True):
            session.toggle_start_pause()
    with c2:
        if st.button("Step", key="srch_step", use_container_width=True):
            session.step_once()
    with c3:
        if st.button("Reset", key="srch_reset", use_container_width=True):
            session.reset_mission()

    if st.button("New random map", key="srch_new_map", use_container_width=True):
        session.new_random_map()

    policy = st.selectbox(
        "Search policy",
        options=SEARCH_POLICY_NAMES,
        index=SEARCH_POLICY_NAMES.index(st.session_state.search_policy),
        key="srch_policy_select",
    )
    if policy != st.session_state.search_policy:
        session.set_search_policy(policy)

    preset = st.selectbox(
        "Scenario preset",
        options=["(Random map)"] + list(PRESET_MAPS.keys()),
        key="srch_preset_select",
    )
    if preset != "(Random map)":
        if st.button(f"Load preset: {preset}", key="srch_load_preset", use_container_width=True):
            session.load_preset(preset)

    st.session_state.show_heatmap = st.toggle(
        "Belief heatmap",
        value=st.session_state.show_heatmap,
        key="srch_heatmap",
    )
    st.session_state.debug_target = st.toggle(
        "Debug: show hidden target",
        value=st.session_state.debug_target,
        key="srch_debug",
    )

    st.session_state.speed = st.select_slider(
        "Simulation speed (steps per refresh)",
        options=[1, 5, 10, 20],
        value=st.session_state.speed,
        key="srch_speed",
    )

    st.divider()
    st.subheader("Mission status")
    st.metric("State", sm.state.name)
    st.metric("Policy", sm.policy.name)
    st.metric("Battery", f"{sm.battery_percentage:.1f}%")
    st.progress(max(0.0, min(1.0, sm.battery_remaining)))
    st.metric("Coverage", f"{sm.coverage_percentage:.1f}%")
    st.metric("Steps", sm.step_count)
    st.metric("Distance", sm.distance_traveled)

    if sm.termination_reason:
        st.info(sm.termination_reason)
    if sm.state == SearchMissionState.TARGET_FOUND:
        st.success("Target detected.")
    elif sm.state == SearchMissionState.BATTERY_DEPLETED:
        st.warning("Battery depleted.")
    elif sm.state == SearchMissionState.COMPLETED:
        st.warning("Search ended without confirmed detection.")

    st.divider()
    st.subheader("Mission log")
    log_text = "\n".join(sm.log[-40:]) if sm.log else "(no events yet)"
    st.text_area("Events", value=log_text, height=220, disabled=True)
