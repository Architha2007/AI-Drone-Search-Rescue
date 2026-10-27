"""
SearchMission — controller for the uncertain-target probabilistic search mode.

This is a **separate, independent** mission mode from the existing coverage-based
MissionController.  It answers the research question:
    "Where should the drone search next when the target's location is
     unknown and battery power is limited?"

Key responsibilities:
    - Manage battery consumption (terrain-dependent costs).
    - Drive belief updates after each sensor observation.
    - Delegate destination selection to pluggable SearchPolicy objects.
    - Maintain a mission log of destinations, observations, belief updates.
    - Terminate the mission cleanly when:
        (a) the target is detected with high confidence, or
        (b) battery drops to/below the safety reserve.
"""
from __future__ import annotations

from enum import Enum, auto
from typing import List, Optional, Tuple

from algorithms.astar import AStarPlanner
from config import (
    BATTERY_MOVE_COST_DEBRIS,
    BATTERY_MOVE_COST_EMPTY,
    BATTERY_SAFETY_RESERVE,
    INITIAL_BATTERY,
    MISSION_LOG_MAX_ENTRIES,
    SENSOR_RANGE,
)
from simulation.belief_map import BeliefMap
from simulation.drone import VirtualDrone
from simulation.environment import CellType, DisasterEnvironment


class SearchMissionState(Enum):
    """Execution state of the uncertain-target search mission."""
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    TARGET_FOUND = auto()       # Successful detection
    BATTERY_DEPLETED = auto()   # Mission ended due to insufficient battery
    COMPLETED = auto()          # Area exhausted without detection


class SearchMission:
    """
    Controller for uncertain-target probabilistic search with battery constraints.

    Usage pattern:
        mission = SearchMission(env, policy, seed=42)
        mission.start()
        while mission.state == SearchMissionState.RUNNING:
            mission.step()
    """

    def __init__(
        self,
        env: DisasterEnvironment,
        policy: "SearchPolicy",   # noqa: F821 — forward ref
        seed: Optional[int] = None,
        initial_battery: float = INITIAL_BATTERY,
        move_cost_empty: float = BATTERY_MOVE_COST_EMPTY,
        move_cost_debris: float = BATTERY_MOVE_COST_DEBRIS,
        safety_reserve: float = BATTERY_SAFETY_RESERVE,
    ) -> None:
        self.env = env
        self.policy = policy
        self.seed = seed

        # Battery parameters
        self.initial_battery = initial_battery
        self.move_cost_empty = move_cost_empty
        self.move_cost_debris = move_cost_debris
        self.safety_reserve = safety_reserve

        # Core components
        self.drone = VirtualDrone(
            start_pos=env.base_pos,
            sensor_range=SENSOR_RANGE,
        )
        self.belief_map = BeliefMap(env, seed=seed)

        # Pathfinder (A* for low-level routing)
        self._pathfinder = AStarPlanner()

        # Battery
        self.battery_remaining: float = initial_battery
        self.battery_consumed: float = 0.0

        # Mission state
        self.state: SearchMissionState = SearchMissionState.IDLE
        self.step_count: int = 0
        self.planned_path: List[Tuple[int, int]] = []
        self.current_destination: Optional[Tuple[int, int]] = None
        self.termination_reason: str = ""

        # Metrics
        self.cells_visited: int = 0
        self.distance_traveled: int = 0
        self.detection_confirmed: bool = False

        # Mission log
        self.log: List[str] = []

        # Elapsed time
        self.elapsed_time_sec: float = 0.0

        # Initial scan
        self._initial_log_entry()

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start or resume the search mission."""
        if self.state != SearchMissionState.TARGET_FOUND:
            self.state = SearchMissionState.RUNNING
            self._log(f"[START] Search mission started. Policy: {self.policy.name}")

    def pause(self) -> None:
        """Pause the search mission."""
        if self.state == SearchMissionState.RUNNING:
            self.state = SearchMissionState.PAUSED
            self._log("[PAUSE] Mission paused by user.")

    def resume(self) -> None:
        """Resume from paused state."""
        if self.state == SearchMissionState.PAUSED:
            self.state = SearchMissionState.RUNNING
            self._log("[RESUME] Mission resumed.")

    def update_clock(self, delta_time_sec: float) -> None:
        """Accumulate mission elapsed time."""
        if self.state == SearchMissionState.RUNNING:
            self.elapsed_time_sec += delta_time_sec

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    def step(self) -> None:
        """
        Execute one discrete simulation step:
        1. Validate / continue planned path or select new destination.
        2. Advance drone one cell, deduct battery.
        3. Simulate sensor observation, update Bayesian beliefs.
        4. Check termination conditions.
        """
        if self.state != SearchMissionState.RUNNING:
            return

        self.step_count += 1

        # Check whether we can afford to move at all
        if not self._can_afford_any_move():
            self._terminate(SearchMissionState.BATTERY_DEPLETED, "Battery at safety reserve. Mission terminated.")
            return

        # Re-plan if path is empty or obstructed
        if not self.planned_path or not self.env.is_traversable(*self.planned_path[0]):
            self._select_next_destination()
            if not self.planned_path:
                self._terminate(SearchMissionState.COMPLETED, "No reachable unvisited cells remain.")
                return

        # Advance drone by one step
        next_pos = self.planned_path.pop(0)
        move_cost = self._terrain_move_cost(next_pos)

        # Safety check: refuse to move if it would drain below reserve
        if self.battery_remaining - move_cost < self.safety_reserve:
            self._terminate(SearchMissionState.BATTERY_DEPLETED,
                            f"Battery would fall below {self.safety_reserve:.0%} reserve. Halting.")
            return

        moved = self.drone.move_to(next_pos, self.env)
        if not moved:
            self.planned_path = []  # Path blocked; replan next step
            return

        # Consume battery
        self.battery_remaining -= move_cost
        self.battery_consumed += move_cost
        self.distance_traveled += 1
        self.cells_visited = len(self.env.explored_cells)

        # Simulate sensor observation and update beliefs
        detection, is_true_positive = self.belief_map.simulate_sensor_reading(self.drone.position)
        self.belief_map.update_observation(
            sensor_center=self.drone.position,
            detection_occurred=detection,
            step=self.step_count,
        )
        self.belief_map.mark_cell_searched(self.drone.position)

        # Log observation
        obs_str = "DETECTED" if detection else "no-detect"
        batt_pct = self.battery_remaining * 100
        self._log(
            f"[Step {self.step_count:4d}] pos={self.drone.position} "
            f"obs={obs_str} batt={batt_pct:.1f}%"
        )

        # Check target confirmed detection
        if self.belief_map.target_detected and not self.detection_confirmed:
            self.detection_confirmed = True
            self._log(
                f"[DETECTION] Target confirmed at step {self.step_count}! "
                f"Battery remaining: {batt_pct:.1f}%"
            )
            self._terminate(SearchMissionState.TARGET_FOUND,
                            f"Target detected at step {self.step_count}.")
            return

        # Check destination reached → replan
        if not self.planned_path or self.drone.position == self.current_destination:
            self.planned_path = []

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, new_env: Optional[DisasterEnvironment] = None, seed: Optional[int] = None) -> None:
        """Reset mission to initial state on the same or a new environment."""
        if new_env is not None:
            self.env = new_env
        if seed is not None:
            self.seed = seed

        self.drone = VirtualDrone(start_pos=self.env.base_pos, sensor_range=SENSOR_RANGE)
        self.belief_map = BeliefMap(self.env, seed=self.seed)
        self.env.explored_cells.clear()
        self.env.mark_explored_area(*self.env.base_pos, radius=SENSOR_RANGE)

        self.battery_remaining = self.initial_battery
        self.battery_consumed = 0.0
        self.state = SearchMissionState.IDLE
        self.step_count = 0
        self.planned_path = []
        self.current_destination = None
        self.termination_reason = ""
        self.cells_visited = 0
        self.distance_traveled = 0
        self.detection_confirmed = False
        self.elapsed_time_sec = 0.0
        self.log = []
        self._log("[RESET] Mission reset.")

    # ------------------------------------------------------------------
    # Metrics properties
    # ------------------------------------------------------------------

    @property
    def battery_percentage(self) -> float:
        """Battery remaining as a percentage 0.0–100.0."""
        return self.battery_remaining * 100.0

    @property
    def coverage_percentage(self) -> float:
        """Fraction of reachable cells visited."""
        reach = len(self.env.reachable_cells)
        if reach == 0:
            return 0.0
        explored = len([c for c in self.env.explored_cells if c in self.env.reachable_cells])
        return min(100.0, (explored / reach) * 100.0)

    @property
    def is_active(self) -> bool:
        return self.state == SearchMissionState.RUNNING

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _can_afford_any_move(self) -> bool:
        """True if battery remaining exceeds safety reserve by at least one step cost."""
        min_cost = min(self.move_cost_empty, self.move_cost_debris)
        return (self.battery_remaining - min_cost) >= self.safety_reserve

    def _terrain_move_cost(self, pos: Tuple[int, int]) -> float:
        """Return the battery cost to enter cell pos based on surrounding terrain."""
        x, y = pos
        # Check neighbours for debris
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                nbr = (x + dx, y + dy)
                if nbr in self.env.obstacles and self.env.obstacles[nbr] == CellType.DEBRIS:
                    return self.move_cost_debris
        return self.move_cost_empty

    def _select_next_destination(self) -> None:
        """Ask the search policy for the next destination and plan a path to it."""
        destination = self.policy.select_next(
            current_pos=self.drone.position,
            belief_map=self.belief_map,
            env=self.env,
            battery_remaining=self.battery_remaining,
            safety_reserve=self.safety_reserve,
        )

        if destination is None:
            return

        self.current_destination = destination
        path, _, _ = self._pathfinder.find_path(self.drone.position, destination, self.env)
        if path and len(path) > 1:
            self.planned_path = path[1:]  # Trim start position
            self._log(
                f"[PLAN] Policy={self.policy.name} -> dest={destination} "
                f"path_len={len(self.planned_path)}"
            )

    def _terminate(self, final_state: SearchMissionState, reason: str) -> None:
        self.state = final_state
        self.termination_reason = reason
        self._log(f"[END] {reason}")

    def _log(self, message: str) -> None:
        """Append a message to the mission log (capped at max entries)."""
        self.log.append(message)
        if len(self.log) > MISSION_LOG_MAX_ENTRIES:
            self.log = self.log[-MISSION_LOG_MAX_ENTRIES:]

    def _initial_log_entry(self) -> None:
        """Log initial state information."""
        self._log(
            f"[INIT] Env={self.env.rows}x{self.env.cols} "
            f"reachable={len(self.env.reachable_cells)} "
            f"battery={self.battery_percentage:.0f}%"
        )
        self._log(
            f"[INIT] detection_prob={self.belief_map.detection_probability:.2f} "
            f"false_pos={self.belief_map.false_positive_rate:.2f}"
        )


# ---------------------------------------------------------------------------
# Search Policy base class
# ---------------------------------------------------------------------------

class SearchPolicy:
    """
    Abstract base class for uncertain-target search destination-selection policies.

    Subclasses implement `select_next()` which returns the next cell to navigate to.
    Low-level pathfinding (e.g., finding the actual route) is handled externally by
    the SearchMission using AStarPlanner.

    This separation keeps destination-selection algorithms pure: they reason about
    WHICH cell to visit next, not HOW to physically navigate there.
    """

    name: str = "base"

    def select_next(
        self,
        current_pos: Tuple[int, int],
        belief_map: BeliefMap,
        env: DisasterEnvironment,
        battery_remaining: float,
        safety_reserve: float,
    ) -> Optional[Tuple[int, int]]:
        """
        Select the next destination cell to navigate toward.

        Parameters
        ----------
        current_pos     : Current drone grid position.
        belief_map      : Current probability distribution (read-only).
        env             : Disaster environment (obstacles, reachable cells, etc.).
        battery_remaining : Battery fraction remaining (0.0–1.0).
        safety_reserve  : Minimum battery fraction before mission must stop.

        Returns
        -------
        A grid coordinate (x, y) to navigate to, or None if no candidates remain.
        """
        raise NotImplementedError
