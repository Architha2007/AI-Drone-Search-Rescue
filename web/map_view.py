"""Matplotlib grid renderer for Streamlit (no Pygame)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from config import (
    COLOR_BASE,
    COLOR_DRONE,
    COLOR_EXPLORED,
    COLOR_GRID_LINE,
    COLOR_OBSTACLE_BUILDING,
    COLOR_OBSTACLE_DEBRIS,
    COLOR_PLANNED_PATH,
    COLOR_PROB_HIGH,
    COLOR_PROB_LOW,
    COLOR_PROB_MED,
    COLOR_PROB_ZERO,
    COLOR_SURVIVOR_DISCOVERED,
    COLOR_SURVIVOR_GROUND_TRUTH,
    COLOR_SURVIVOR_RESCUED,
    COLOR_TRAIL,
    COLOR_UNEXPLORED,
    HEATMAP_ALPHA,
    SENSOR_RANGE,
)
from simulation.environment import CellType, DisasterEnvironment
from simulation.mission import MissionController
from simulation.search_mission import SearchMission


def _lerp_color(
    c1: Tuple[int, int, int],
    c2: Tuple[int, int, int],
    t: float,
) -> Tuple[int, int, int]:
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def _prob_to_color(prob: float, max_prob: float) -> Tuple[int, int, int]:
    if max_prob <= 0.0:
        return COLOR_PROB_ZERO
    t = min(1.0, prob / max_prob)
    if t < 0.5:
        return _lerp_color(COLOR_PROB_LOW, COLOR_PROB_MED, t * 2.0)
    return _lerp_color(COLOR_PROB_MED, COLOR_PROB_HIGH, (t - 0.5) * 2.0)


def _to_unit_rgb(color: Tuple[int, int, int]) -> Tuple[float, float, float]:
    return (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)


def _blend(
    base: Tuple[int, int, int],
    overlay: Tuple[int, int, int],
    alpha: float,
) -> Tuple[int, int, int]:
    a = max(0.0, min(1.0, alpha))
    return (
        int(base[0] * (1 - a) + overlay[0] * a),
        int(base[1] * (1 - a) + overlay[1] * a),
        int(base[2] * (1 - a) + overlay[2] * a),
    )


def _base_cell_color(env: DisasterEnvironment, x: int, y: int) -> Tuple[int, int, int]:
    pos = (x, y)
    if pos in env.obstacles:
        if env.obstacles[pos] == CellType.BUILDING:
            return COLOR_OBSTACLE_BUILDING
        return COLOR_OBSTACLE_DEBRIS
    if pos in env.explored_cells:
        return COLOR_EXPLORED
    return COLOR_UNEXPLORED


def _build_rgb_grid(
    env: DisasterEnvironment,
    *,
    search_mission: Optional[SearchMission] = None,
    show_heatmap: bool = False,
    debug_target: bool = False,
) -> np.ndarray:
    rows, cols = env.rows, env.cols
    grid = np.zeros((rows, cols, 3), dtype=float)

    beliefs = {}
    max_prob = 1.0
    if search_mission is not None and show_heatmap:
        beliefs = search_mission.belief_map.get_all_beliefs()
        if beliefs:
            max_prob = max(beliefs.values()) or 1.0

    for y in range(rows):
        for x in range(cols):
            rgb = _base_cell_color(env, x, y)
            if search_mission is not None and show_heatmap and env.is_traversable(x, y):
                prob = beliefs.get((x, y), 0.0)
                if prob > 1e-8:
                    heat = _prob_to_color(prob, max_prob)
                    alpha = (HEATMAP_ALPHA / 255.0) * min(1.0, prob / max_prob)
                    rgb = _blend(rgb, heat, alpha)
            grid[y, x] = _to_unit_rgb(rgb)

    if search_mission is not None and debug_target:
        target = search_mission.belief_map.debug_hidden_target
        if target is not None:
            tx, ty = target
            if env.is_valid(tx, ty):
                grid[ty, tx] = _to_unit_rgb(COLOR_SURVIVOR_GROUND_TRUTH)

    bx, by = env.base_pos
    if env.is_valid(bx, by):
        grid[by, bx] = _to_unit_rgb(COLOR_BASE)

    return grid


def _plot_path(ax, path: List[Tuple[int, int]], color: Tuple[int, int, int], width: float, alpha: float = 1.0) -> None:
    if len(path) < 2:
        return
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    ax.plot(xs, ys, color=_to_unit_rgb(color), linewidth=width, alpha=alpha, zorder=4)


def render_map_figure(
    *,
    mode: str,
    coverage: Optional[MissionController] = None,
    search: Optional[SearchMission] = None,
    show_heatmap: bool = False,
    debug_target: bool = False,
) -> Figure:
    """Build a matplotlib figure for the current simulation state."""
    if mode == "Search" and search is not None:
        env = search.env
        sm = search
        mission = None
    else:
        env = coverage.env if coverage is not None else DisasterEnvironment()
        sm = None
        mission = coverage

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#12161e")
    ax.set_facecolor("#12161e")

    rgb = _build_rgb_grid(
        env,
        search_mission=sm,
        show_heatmap=show_heatmap and sm is not None,
        debug_target=debug_target and sm is not None,
    )
    ax.imshow(rgb, origin="upper", interpolation="nearest", zorder=1)

    rows, cols = env.rows, env.cols
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.grid(True, color=_to_unit_rgb(COLOR_GRID_LINE), linewidth=0.4, zorder=2)
    ax.tick_params(colors="#a0aec0", labelsize=7)

    if sm is not None:
        trail = sm.drone.path_history
        plan = [sm.drone.position] + list(sm.planned_path)
        _plot_path(ax, trail, COLOR_TRAIL, 2.0, alpha=0.55)
        _plot_path(ax, plan, COLOR_PLANNED_PATH, 2.5)
        dx, dy = sm.drone.position
    elif mission is not None:
        trail = mission.drone.path_history
        plan = [mission.drone.position] + list(mission.planned_path)
        _plot_path(ax, trail, COLOR_TRAIL, 2.0, alpha=0.55)
        _plot_path(ax, plan, COLOR_PLANNED_PATH, 2.5)
        dx, dy = mission.drone.position

        detector = mission.detector
        for s_pos in detector.discovered_survivors:
            if s_pos in detector.reached_survivors:
                continue
            sx, sy = s_pos
            ax.scatter(
                [sx],
                [sy],
                marker="D",
                s=80,
                color=[_to_unit_rgb(COLOR_SURVIVOR_DISCOVERED)],
                zorder=5,
            )
        for s_pos in detector.reached_survivors:
            sx, sy = s_pos
            ax.scatter(
                [sx],
                [sy],
                marker="P",
                s=100,
                color=[_to_unit_rgb(COLOR_SURVIVOR_RESCUED)],
                zorder=5,
            )
    else:
        dx, dy = env.base_pos

    fov_r = SENSOR_RANGE + 0.45
    circle = plt.Circle((dx, dy), fov_r, fill=False, color=_to_unit_rgb(COLOR_DRONE), linewidth=1.2, alpha=0.7, zorder=6)
    ax.add_patch(circle)
    ax.scatter(
        [dx],
        [dy],
        s=140,
        color=[_to_unit_rgb(COLOR_DRONE)],
        edgecolors="white",
        linewidths=1.2,
        zorder=7,
    )

    title = "Search Mode — Probabilistic SAR" if sm is not None else "Coverage Mode — Area Sweep"
    ax.set_title(title, color="white", fontsize=12, pad=10)
    ax.set_xlabel("Grid X", color="#a0aec0", fontsize=9)
    ax.set_ylabel("Grid Y", color="#a0aec0", fontsize=9)

    fig.tight_layout()
    return fig
