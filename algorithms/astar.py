"""A* (A-Star) Pathfinding Algorithm using Manhattan Distance heuristic."""
from __future__ import annotations

import heapq
import time
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from simulation.environment import DisasterEnvironment


def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """
    Compute Manhattan distance heuristic between points a and b.
    h(a, b) = |x1 - x2| + |y1 - y2|
    Admissible and consistent on 4-directional grids without diagonal movement.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class AStarPlanner:
    """
    A* Shortest Path Planner for 2D Grid Environments.

    Theoretical Complexity:
    - Time Complexity: O(|E| + |V| log |V|) in the worst case where V is the set of
      traversable grid cells (at most W * H) and E is the set of edges (at most 4 * |V|).
      In practice, the consistent Manhattan heuristic focuses search toward the target,
      visiting significantly fewer nodes than uninformed algorithms.
    - Space Complexity: O(|V|) to maintain open set, closed set, and came_from mapping.
    """

    def __init__(self, heuristic_fn: Callable[[Tuple[int, int], Tuple[int, int]], int] = manhattan_distance):
        self.heuristic_fn = heuristic_fn

    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        env: DisasterEnvironment,
    ) -> Tuple[List[Tuple[int, int]], int, float]:
        """
        Find the shortest path from start to goal avoiding obstacles.

        Returns:
            Tuple of:
            - path: List of (x, y) coordinates from start to goal (inclusive). Empty if unreachable.
            - nodes_explored: Count of unique nodes popped from the priority queue.
            - elapsed_time_ms: Execution duration in milliseconds.
        """
        start_time = time.perf_counter()

        # Edge cases: validation
        if not env.is_valid(*start) or not env.is_valid(*goal):
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return [], 0, elapsed

        # Goal or start inside obstacle
        if not env.is_traversable(*start) or not env.is_traversable(*goal):
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return [], 0, elapsed

        # Start is goal
        if start == goal:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return [start], 1, elapsed

        # Priority Queue: (f_score, tie_breaker, current_node)
        tie_counter = 0
        open_set: List[Tuple[float, int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (self.heuristic_fn(start, goal), tie_counter, start))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        f_score: Dict[Tuple[int, int], float] = {start: float(self.heuristic_fn(start, goal))}
        visited_closed: Set[Tuple[int, int]] = set()

        nodes_explored = 0

        while open_set:
            current_f, _, current = heapq.heappop(open_set)

            if current in visited_closed:
                continue

            visited_closed.add(current)
            nodes_explored += 1

            # Goal reached: reconstruct path
            if current == goal:
                path = []
                curr = current
                while curr in came_from:
                    path.append(curr)
                    curr = came_from[curr]
                path.append(start)
                path.reverse()
                elapsed = (time.perf_counter() - start_time) * 1000.0
                return path, nodes_explored, elapsed

            for neighbor in env.get_neighbors(*current):
                if neighbor in visited_closed:
                    continue

                tentative_g = g_score[current] + 1.0  # Unit step cost on grid

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = self.heuristic_fn(neighbor, goal)
                    f = tentative_g + h
                    f_score[neighbor] = f
                    tie_counter += 1
                    heapq.heappush(open_set, (f, tie_counter, neighbor))

        # Goal is unreachable
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return [], nodes_explored, elapsed


# Convenience functional interface
def astar_search(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    env: DisasterEnvironment,
) -> Tuple[List[Tuple[int, int]], int, float]:
    """Helper wrapper for AStarPlanner."""
    planner = AStarPlanner()
    return planner.find_path(start, goal, env)
