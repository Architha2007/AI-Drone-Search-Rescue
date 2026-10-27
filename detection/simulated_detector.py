"""Simulated survivor detector based on drone sensor field of view."""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Set, Tuple

from config import SENSOR_RANGE

if TYPE_CHECKING:
    from simulation.environment import DisasterEnvironment


class SimulatedDetector:
    """
    Simulates onboard drone sensors (optical, infrared, thermal).
    Scans the area within sensor range and discovers survivors hidden in the environment.
    Strictly keeps undiscovered survivor locations hidden from mission logic and UI display.
    """

    def __init__(self, sensor_range: int = SENSOR_RANGE):
        self.sensor_range: int = sensor_range
        # Discovered survivors (revealed to UI and mission planner)
        self.discovered_survivors: Set[Tuple[int, int]] = set()
        # Survivors where drone has physically visited their exact cell
        self.reached_survivors: Set[Tuple[int, int]] = set()

    def scan(
        self,
        drone_pos: Tuple[int, int],
        env: DisasterEnvironment,
    ) -> List[Tuple[int, int]]:
        """
        Scan all cells within sensor radius around drone_pos.
        Returns a list of newly discovered survivors detected in this scan.
        """
        dx_range = range(-self.sensor_range, self.sensor_range + 1)
        dy_range = range(-self.sensor_range, self.sensor_range + 1)
        newly_found: List[Tuple[int, int]] = []

        for dy in dy_range:
            for dx in dx_range:
                cell = (drone_pos[0] + dx, drone_pos[1] + dy)
                if env.is_valid(*cell) and cell in env.ground_truth_survivors:
                    if cell not in self.discovered_survivors:
                        self.discovered_survivors.add(cell)
                        newly_found.append(cell)

        # Check if drone reached any discovered survivor's cell
        if drone_pos in self.discovered_survivors:
            self.reached_survivors.add(drone_pos)

        return newly_found

    def check_reached(self, drone_pos: Tuple[int, int]) -> bool:
        """Mark survivor as physically reached if drone is at survivor location."""
        if drone_pos in self.discovered_survivors:
            self.reached_survivors.add(drone_pos)
            return True
        return False

    @property
    def total_discovered(self) -> int:
        """Publicly exposed count of discovered survivors."""
        return len(self.discovered_survivors)

    @property
    def total_reached(self) -> int:
        """Publicly exposed count of reached survivors."""
        return len(self.reached_survivors)

    def reset(self) -> None:
        """Reset detection history for new mission or map reset."""
        self.discovered_survivors.clear()
        self.reached_survivors.clear()
