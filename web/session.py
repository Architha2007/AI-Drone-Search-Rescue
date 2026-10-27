"""Streamlit session state: mission objects and simulation transport controls."""

from __future__ import annotations

import random
from typing import Optional

import streamlit as st

from config import DEFAULT_RANDOM_SEED, PRESET_MAPS
from simulation.environment import DisasterEnvironment
from simulation.mission import MissionController, MissionState
from simulation.search_mission import SearchMission, SearchMissionState
from web.policy_factory import SEARCH_POLICY_NAMES, make_search_policy


def ensure_initialized() -> None:
    """Create mission controllers once per browser session."""
    if st.session_state.get("web_initialized"):
        return

    seed = DEFAULT_RANDOM_SEED
    st.session_state.seed = seed
    st.session_state.mode = "Coverage"
    st.session_state.auto_run = False
    st.session_state.speed = 5
    st.session_state.show_heatmap = True
    st.session_state.debug_target = False
    st.session_state.search_policy = SEARCH_POLICY_NAMES[0]
    st.session_state.coverage_algorithm = "astar"

    st.session_state.coverage_mission = MissionController(
        env=DisasterEnvironment(seed=seed),
        algorithm_name="astar",
    )
    search_env = DisasterEnvironment(seed=seed)
    st.session_state.search_mission = SearchMission(
        env=search_env,
        policy=make_search_policy(st.session_state.search_policy, seed=seed),
        seed=seed,
    )
    st.session_state.web_initialized = True


def coverage_mission() -> MissionController:
    ensure_initialized()
    return st.session_state.coverage_mission


def search_mission() -> SearchMission:
    ensure_initialized()
    return st.session_state.search_mission


def is_auto_run() -> bool:
    return bool(st.session_state.get("auto_run"))


def set_auto_run(value: bool) -> None:
    st.session_state.auto_run = value


def toggle_start_pause() -> None:
    if st.session_state.mode == "Coverage":
        mission = coverage_mission()
        if mission.state == MissionState.RUNNING:
            mission.pause()
            set_auto_run(False)
        else:
            mission.start()
            set_auto_run(True)
    else:
        sm = search_mission()
        if sm.state == SearchMissionState.RUNNING:
            sm.pause()
            set_auto_run(False)
        elif sm.state in (
            SearchMissionState.IDLE,
            SearchMissionState.PAUSED,
        ):
            sm.start()
            set_auto_run(True)


def step_once() -> None:
    if st.session_state.mode == "Coverage":
        mission = coverage_mission()
        if mission.state == MissionState.RUNNING:
            mission.pause()
            set_auto_run(False)
        mission.step()
    else:
        sm = search_mission()
        if sm.state == SearchMissionState.RUNNING:
            sm.pause()
            set_auto_run(False)
        if sm.state not in (
            SearchMissionState.TARGET_FOUND,
            SearchMissionState.BATTERY_DEPLETED,
            SearchMissionState.COMPLETED,
        ):
            sm.step()


def reset_mission() -> None:
    set_auto_run(False)
    if st.session_state.mode == "Coverage":
        coverage_mission().reset()
    else:
        search_mission().reset()


def new_random_map() -> None:
    set_auto_run(False)
    new_seed = random.randint(1, 99999)
    st.session_state.seed = new_seed
    if st.session_state.mode == "Coverage":
        coverage_mission().new_map(seed=new_seed)
    else:
        sm = search_mission()
        sm.env.generate_random_map(seed=new_seed)
        sm.reset(seed=new_seed)


def load_preset(scenario_name: str) -> None:
    set_auto_run(False)
    layout = PRESET_MAPS.get(scenario_name)
    if layout is None:
        return
    if st.session_state.mode == "Coverage":
        coverage_mission().load_fixed_map(layout=layout, map_name=scenario_name)
    else:
        sm = search_mission()
        sm.env.load_fixed_map(layout=layout)
        sm.reset()


def set_coverage_algorithm(name: str) -> None:
    set_auto_run(False)
    coverage_mission().set_algorithm(name.lower())


def set_search_policy(name: str) -> None:
    if name not in SEARCH_POLICY_NAMES:
        return
    set_auto_run(False)
    st.session_state.search_policy = name
    sm = search_mission()
    sm.policy = make_search_policy(name, seed=st.session_state.seed)
    sm.reset()


def advance_simulation_if_running() -> None:
    """Advance simulation by `speed` steps when autoplay is enabled."""
    if not is_auto_run():
        return

    speed = max(1, int(st.session_state.get("speed", 5)))
    if st.session_state.mode == "Coverage":
        mission = coverage_mission()
        if mission.state != MissionState.RUNNING:
            set_auto_run(False)
            return
        for _ in range(speed):
            mission.step()
            if mission.state == MissionState.COMPLETED:
                set_auto_run(False)
                break
    else:
        sm = search_mission()
        if sm.state != SearchMissionState.RUNNING:
            set_auto_run(False)
            return
        for _ in range(speed):
            sm.step()
            if sm.state != SearchMissionState.RUNNING:
                set_auto_run(False)
                break


def mission_still_running_for_autoplay() -> bool:
    if not is_auto_run():
        return False
    if st.session_state.mode == "Coverage":
        return coverage_mission().state == MissionState.RUNNING
    return search_mission().state == SearchMissionState.RUNNING
