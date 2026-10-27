"""Mission controller coordinating autonomous drone navigation, search, and rescue."""
from enum import Enum, auto
import time
from typing import List, Optional, Set, Tuple

from algorithms.astar import AStarPlanner
from algorithms.bfs import BFSPlanner
from algorithms.coverage import CoveragePlanner
from algorithms.target_selection import SearchStrategy, TargetSelector
from config import BASE_COORDINATES, SENSOR_RANGE
from detection.simulated_detector import SimulatedDetector
from simulation.drone import VirtualDrone
from simulation.environment import DisasterEnvironment


class MissionState(Enum):
    """Execution state of the search and rescue mission."""
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()


class MissionController:
    """
    Central controller managing simulation execution, dynamic replanning,
    sensor scanning, and performance metrics tracking.
    """

    def __init__(
        self,
        env: Optional[DisasterEnvironment] = None,
        algorithm_name: str = "astar",
    ):
        self.env = env if env is not None else DisasterEnvironment()
        self.drone = VirtualDrone(start_pos=self.env.base_pos, sensor_range=SENSOR_RANGE)
        self.detector = SimulatedDetector(sensor_range=SENSOR_RANGE)

        # Pathfinding planners
        self.astar_planner = AStarPlanner()
        self.bfs_planner = BFSPlanner()
        self.active_algorithm_name: str = algorithm_name.lower()

        # Coverage and target selection
        self.coverage_planner = CoveragePlanner(self.env)
        self.target_selector = TargetSelector(strategy=SearchStrategy.BALANCED)
        self.sweep_sequence: List[Tuple[int, int]] = self.coverage_planner.generate_boustrophedon_pattern()

        # Mission state
        self.state: MissionState = MissionState.IDLE
        self.current_target: Optional[Tuple[int, int]] = None
        self.planned_path: List[Tuple[int, int]] = []
        self.status_message: str = "System ready. Press Start to initiate autonomous mission."

        # Metrics
        self.pathfinding_operations: int = 0
        self.replanning_events: int = 0
        self.elapsed_time_sec: float = 0.0
        self._last_tick_time: Optional[float] = None

        # Initial scan at base
        self._perform_scan()

    def _get_active_planner(self):
        """Return the currently selected pathfinder."""
        if self.active_algorithm_name == "bfs":
            return self.bfs_planner
        return self.astar_planner

    def set_algorithm(self, name: str) -> None:
        """Switch pathfinding algorithm between 'astar' and 'bfs'."""
        name_clean = name.lower()
        if name_clean in ("astar", "bfs"):
            self.active_algorithm_name = name_clean
            self.status_message = f"Pathfinder set to {name_clean.upper()}."
            # Trigger replan with the new algorithm if currently navigating
            if self.state == MissionState.RUNNING and self.current_target is not None:
                self.replan(force=True)

    def _perform_scan(self) -> None:
        """Scan field of view around current drone location for survivors."""
        newly_found = self.detector.scan(self.drone.position, self.env)
        if newly_found:
            first_found = newly_found[0]
            self.status_message = f"Survivor detected at grid {first_found}!"

        self.detector.check_reached(self.drone.position)

    def plan_path_to(self, target: Tuple[int, int]) -> bool:
        """
        Calculate path from current drone position to target using active planner.
        Returns True if a path was found, False if unreachable.
        """
        self.pathfinding_operations += 1
        planner = self._get_active_planner()
        path, _, _ = planner.find_path(self.drone.position, target, self.env)

        if not path:
            self.target_selector.mark_unreachable(target)
            return False

        # Path includes start position at index 0, trim it for next waypoints
        self.planned_path = path[1:] if len(path) > 1 else []
        self.current_target = target
        return True

    def replan(self, force: bool = False) -> bool:
        """
        Evaluate mission priorities and compute a new route.
        Called when target reached, target unreachable, or obstacle detected along path.
        """
        self.replanning_events += 1

        # Select next destination according to priorities
        target, reason = self.target_selector.select_next_target(
            current_pos=self.drone.position,
            base_pos=self.env.base_pos,
            env=self.env,
            discovered_survivors=self.detector.discovered_survivors,
            reached_survivors=self.detector.reached_survivors,
            sweep_sequence=self.sweep_sequence,
        )

        if target is None:
            # Mission finished
            self.state = MissionState.COMPLETED
            self.planned_path = []
            self.current_target = None
            self.status_message = "Mission complete! Area searched, survivors addressed, secured at base."
            return True

        self.status_message = reason
        success = self.plan_path_to(target)

        # If primary candidate is unreachable, iterate to find reachable alternative
        attempts = 0
        while not success and attempts < 15:
            attempts += 1
            alt_target, alt_reason = self.target_selector.select_next_target(
                current_pos=self.drone.position,
                base_pos=self.env.base_pos,
                env=self.env,
                discovered_survivors=self.detector.discovered_survivors,
                reached_survivors=self.detector.reached_survivors,
                sweep_sequence=self.sweep_sequence,
            )
            if alt_target is None or alt_target == target:
                break
            target = alt_target
            self.status_message = f"Replanning: {alt_reason}"
            success = self.plan_path_to(target)

        if not success:
            # Check if all reachable cells are already explored
            unexplored = [p for p in self.env.reachable_cells if p not in self.env.explored_cells]
            if not unexplored:
                if self.drone.position == self.env.base_pos:
                    self.state = MissionState.COMPLETED
                    self.status_message = "All reachable cells explored. Mission completed at base."
                else:
                    self.status_message = "All reachable sectors explored. Returning to base."
                    self.plan_path_to(self.env.base_pos)
            else:
                self.status_message = "Warning: Current targets unreachable due to debris blocking passage."

        return success

    def step(self) -> None:
        """
        Execute one discrete simulation step:
        1. Validate planned path against dynamic obstacles.
        2. Advance drone 1 cell along path.
        3. Scan environment and update exploration.
        4. Replan if destination reached or path obstructed.
        """
        if self.state not in (MissionState.RUNNING, MissionState.IDLE):
            return

        # Path obstruction check (Dynamic Obstacle handling)
        if self.planned_path:
            next_cell = self.planned_path[0]
            if not self.env.is_traversable(*next_cell):
                self.status_message = f"Obstacle detected at {next_cell}! Dynamic replanning engaged."
                self.planned_path = []
                self.replan(force=True)
                return

        # If no path planned, compute one
        if not self.planned_path:
            self.replan()
            if not self.planned_path:
                return

        # Advance drone by one grid cell
        next_pos = self.planned_path.pop(0)
        moved = self.drone.move_to(next_pos, self.env)

        if moved:
            self._perform_scan()
            # If reached current waypoint/target, replan for next objective
            if self.drone.position == self.current_target or not self.planned_path:
                self.replan()
        else:
            # Failed to move (obstacle or out of bounds)
            self.status_message = f"Movement to {next_pos} blocked. Recalculating path..."
            self.planned_path = []
            self.replan(force=True)

    def update_clock(self, delta_time_sec: float) -> None:
        """Accumulate mission elapsed time when running."""
        if self.state == MissionState.RUNNING:
            self.elapsed_time_sec += delta_time_sec

    def start(self) -> None:
        """Start or resume mission execution."""
        if self.state != MissionState.COMPLETED:
            self.state = MissionState.RUNNING
            self.status_message = f"Mission active. Searching via {self.active_algorithm_name.upper()}..."

    def pause(self) -> None:
        """Pause mission execution."""
        if self.state == MissionState.RUNNING:
            self.state = MissionState.PAUSED
            self.status_message = "Simulation paused. Press Resume to continue."

    def resume(self) -> None:
        """Resume paused simulation."""
        self.start()

    def reset(self) -> None:
        """Reset mission to initial starting condition on current map."""
        self.state = MissionState.IDLE
        self.drone.reset(self.env.base_pos)
        self.detector.reset()
        self.target_selector.clear_unreachable()
        self.env.explored_cells.clear()
        self.env.mark_explored_area(*self.env.base_pos, radius=SENSOR_RANGE)

        self.current_target = None
        self.planned_path = []
        self.pathfinding_operations = 0
        self.replanning_events = 0
        self.elapsed_time_sec = 0.0

        self.sweep_sequence = self.coverage_planner.generate_boustrophedon_pattern()
        self._perform_scan()
        self.status_message = "Mission reset. System ready at base."

    def new_map(self, seed: Optional[int] = None) -> None:
        """Generate a fresh disaster environment map."""
        self.env.generate_random_map(seed=seed)
        self.reset()
        self.status_message = f"New disaster map generated (Seed: {self.env.seed})."

    def load_fixed_map(self, layout: Optional[List[str]] = None, map_name: str = "Benchmark") -> None:
        """Load benchmark or preset disaster scenario map."""
        self.env.load_fixed_map(layout=layout)
        self.reset()
        self.status_message = f"Scenario loaded: {map_name}."

    # ---------------------------------------------------------
    # Performance & Status Metrics
    # ---------------------------------------------------------
    @property
    def total_traversable(self) -> int:
        return self.env.total_traversable_cells

    @property
    def total_reachable(self) -> int:
        return self.env.total_reachable_cells

    @property
    def explored_traversable(self) -> int:
        """Count of traversable cells that have been explored."""
        return len([c for c in self.env.explored_cells if self.env.is_traversable(*c)])

    @property
    def coverage_percentage(self) -> float:
        """Coverage percentage based on reachable traversable cells."""
        reach = self.total_reachable
        if reach == 0:
            return 0.0
        explored_reach = len([c for c in self.env.explored_cells if c in self.env.reachable_cells])
        return min(100.0, (explored_reach / reach) * 100.0)

    @property
    def survivors_discovered(self) -> int:
        return self.detector.total_discovered

    @property
    def survivors_reached(self) -> int:
        return self.detector.total_reached

    @property
    def ground_truth_survivors_count(self) -> int:
        """Ground truth count for post-mission evaluation only."""
        return len(self.env.ground_truth_survivors)

    @property
    def ground_truth_remaining_undiscovered(self) -> int:
        """Evaluation ground truth count remaining undiscovered."""
        return len(self.env.ground_truth_survivors - self.detector.discovered_survivors)
