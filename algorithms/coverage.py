"""Coverage path planning algorithms for systematic disaster area search."""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from simulation.environment import DisasterEnvironment


class CoveragePlanner:
    """
    Coverage Path Planner using Boustrophedon (Lawnmower) Sweeping and Dynamic Relocation.

    Theoretical Complexity:
    - Initial Sweep Generation: O(W * H) where W and H are the grid dimensions.
    - Next Target Selection: O(|V|) where V is the set of unvisited reachable cells.
    - Overall Complete Coverage: Visits all N reachable traversable cells in bounded steps.
    """

    def __init__(self, env: DisasterEnvironment):
        self.env = env

    def generate_boustrophedon_pattern(self) -> List[Tuple[int, int]]:
        """
        Generate a row-by-row boustrophedon (lawnmower) sweeping order.
        Even rows traverse left-to-right (x: 0 -> cols-1).
        Odd rows traverse right-to-left (x: cols-1 -> 0).
        Filters out non-traversable or unreachable cells.
        """
        pattern: List[Tuple[int, int]] = []
        for y in range(self.env.rows):
            x_range = range(self.env.cols) if (y % 2 == 0) else range(self.env.cols - 1, -1, -1)
            for x in x_range:
                pos = (x, y)
                if self.env.is_traversable(*pos) and pos in self.env.reachable_cells:
                    pattern.append(pos)
        return pattern

    def get_next_sweep_target(
        self,
        current_pos: Tuple[int, int],
        explored_cells: Set[Tuple[int, int]],
        sweep_sequence: List[Tuple[int, int]],
    ) -> Optional[Tuple[int, int]]:
        """
        Get the next unvisited cell according to the boustrophedon sweep order.
        If the immediate next sweep cell is already explored, advances forward.
        """
        for target in sweep_sequence:
            if target not in explored_cells and self.env.is_traversable(*target):
                return target
        return None

    def get_closest_unexplored_target(
        self,
        current_pos: Tuple[int, int],
        explored_cells: Set[Tuple[int, int]],
    ) -> Optional[Tuple[int, int]]:
        """
        Find the nearest unvisited reachable traversable cell to current_pos using Manhattan distance.
        Used as a fallback or optimization when the sweep pattern creates long transits.
        """
        unvisited = [
            pos for pos in self.env.reachable_cells
            if pos not in explored_cells and self.env.is_traversable(*pos)
        ]
        if not unvisited:
            return None

        cx, cy = current_pos
        # Sort by Manhattan distance to current position
        unvisited.sort(key=lambda p: abs(p[0] - cx) + abs(p[1] - cy))
        return unvisited[0]
