"""Virtual autonomous drone model for search and rescue operations."""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Set, Tuple

from config import BASE_COORDINATES, SENSOR_RANGE

if TYPE_CHECKING:
    from simulation.environment import DisasterEnvironment


class VirtualDrone:
    """Autonomous search and rescue drone model."""

    def __init__(
        self,
        start_pos: Tuple[int, int] = BASE_COORDINATES,
        sensor_range: int = SENSOR_RANGE,
    ):
        self.base_pos: Tuple[int, int] = start_pos
        self.position: Tuple[int, int] = start_pos
        self.sensor_range: int = sensor_range

        # Movement tracking
        self.path_history: List[Tuple[int, int]] = [start_pos]
        self.distance_traveled: int = 0
        self.orientation_angle: float = 0.0  # For UI animation/rotation

    def move_to(self, target_pos: Tuple[int, int], env: DisasterEnvironment) -> bool:
        """
        Attempt to move the drone to target_pos.
        Enforces safety constraints:
        1. Target must be within bounds.
        2. Target must NOT be an obstacle.
        3. Movement must be strictly 4-directional step (adjacent).
        Returns True if successful, False otherwise.
        """
        tx, ty = target_pos
        cx, cy = self.position

        # Boundary check
        if not env.is_valid(tx, ty):
            return False

        # Obstacle check
        if not env.is_traversable(tx, ty):
            return False

        # Adjacency check (Manhattan distance == 1)
        if abs(tx - cx) + abs(ty - cy) != 1:
            return False

        # Calculate heading angle for rendering (degrees)
        dx = tx - cx
        dy = ty - cy
        if dx == 1:
            self.orientation_angle = 90.0
        elif dx == -1:
            self.orientation_angle = 270.0
        elif dy == 1:
            self.orientation_angle = 180.0
        elif dy == -1:
            self.orientation_angle = 0.0

        # Execute move
        self.position = target_pos
        self.distance_traveled += 1
        self.path_history.append(target_pos)

        # Mark explored in environment
        env.mark_explored_area(tx, ty, radius=self.sensor_range)
        return True

    def get_observable_cells(self, env: DisasterEnvironment) -> Set[Tuple[int, int]]:
        """Return the set of valid cells currently within sensor range."""
        cx, cy = self.position
        observable: Set[Tuple[int, int]] = set()
        for dy in range(-self.sensor_range, self.sensor_range + 1):
            for dx in range(-self.sensor_range, self.sensor_range + 1):
                nx, ny = cx + dx, cy + dy
                if env.is_valid(nx, ny):
                    observable.add((nx, ny))
        return observable

    def reset(self, new_base: Optional[Tuple[int, int]] = None) -> None:
        """Reset drone state back to the base station."""
        if new_base is not None:
            self.base_pos = new_base
        self.position = self.base_pos
        self.distance_traveled = 0
        self.path_history = [self.base_pos]
        self.orientation_angle = 0.0
