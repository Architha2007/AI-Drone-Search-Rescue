"""Breadth-First Search (BFS) pathfinding baseline algorithm."""
from __future__ import annotations

from collections import deque
import time
from typing import TYPE_CHECKING, Dict, List, Set, Tuple

if TYPE_CHECKING:
    from simulation.environment import DisasterEnvironment


class BFSPlanner:
    """
    Breadth-First Search (BFS) Shortest Path Planner for unweighted grid graphs.

    Theoretical Complexity:
    - Time Complexity: O(|V| + |E|) where V is the number of traversable cells and
      E is the number of 4-connected edges (E <= 4|V|). Expands outward uniformly
      in all directions without heuristic guidance.
    - Space Complexity: O(|V|) to maintain queue, visited set, and predecessor mappings.
    """

    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        env: DisasterEnvironment,
    ) -> Tuple[List[Tuple[int, int]], int, float]:
        """
        Find an unweighted shortest path from start to goal using BFS.

        Returns:
            Tuple of:
            - path: List of (x, y) coordinates from start to goal (inclusive). Empty if unreachable.
            - nodes_explored: Count of unique nodes dequeued.
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

        queue: deque[Tuple[int, int]] = deque([start])
        visited: Set[Tuple[int, int]] = {start}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        nodes_explored = 0

        while queue:
            current = queue.popleft()
            nodes_explored += 1

            if current == goal:
                # Reconstruct path
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
                if neighbor not in visited:
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    queue.append(neighbor)

        # Unreachable
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return [], nodes_explored, elapsed


# Convenience functional interface
def bfs_search(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    env: DisasterEnvironment,
) -> Tuple[List[Tuple[int, int]], int, float]:
    """Helper wrapper for BFSPlanner."""
    planner = BFSPlanner()
    return planner.find_path(start, goal, env)
