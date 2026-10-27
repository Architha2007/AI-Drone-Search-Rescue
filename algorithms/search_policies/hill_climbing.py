"""
Hill Climbing policy for uncertain-target drone search.

Principle
---------
Hill Climbing is a local search algorithm that iteratively improves a candidate
solution by examining neighbours.  In the context of drone search, a "solution"
is a candidate destination cell, and the "neighbourhood" is the set of cells
within a defined spatial radius.

Algorithm
---------
1. Begin with a random starting candidate drawn from the top-probability cells.
2. Evaluate the objective function at the current candidate.
3. Examine all cells within HILL_CLIMB_NEIGHBOR_RADIUS of the current candidate.
4. If any neighbour has a higher score: move to that neighbour (uphill step).
5. If no neighbour is better: we've reached a LOCAL OPTIMUM.

Handling local optima (documented)
------------------------------------
Pure hill climbing is vulnerable to getting stuck at local optima in the belief
landscape.  This implementation handles this with RANDOM RESTARTS:
    - When a local optimum is reached, start from a new random candidate from
      the unexplored high-probability cells.
    - Repeat up to HILL_CLIMB_MAX_RESTARTS times.
    - Track the globally best candidate found across all restarts.

This converts simple hill climbing into "iterated local search" — a practical
and well-known technique for escaping local optima while retaining the simplicity
of the gradient-following approach.

Objective function (documented)
--------------------------------
    score(c) = belief_prob(c) - lambda * manhattan_distance(current_pos, c)

The linear travel penalty ensures that the drone balances maximizing detection
probability with minimizing battery cost (distance).

Battery awareness
-----------------
Candidates requiring more than usable_battery × BATTERY_MOVE_COST_EMPTY to
reach from the drone's current position are excluded from consideration.
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING, List, Optional, Set, Tuple

from algorithms.astar import manhattan_distance
from config import (
    BATTERY_MOVE_COST_EMPTY,
    HILL_CLIMB_MAX_RESTARTS,
    HILL_CLIMB_NEIGHBOR_RADIUS,
)
from simulation.search_mission import SearchPolicy

if TYPE_CHECKING:
    from simulation.belief_map import BeliefMap
    from simulation.environment import DisasterEnvironment


class HillClimbingPolicy(SearchPolicy):
    """
    Hill Climbing with random restarts for uncertain-target destination selection.

    The search iteratively improves a candidate destination by examining a
    spatial neighbourhood and moving to the best-scoring neighbour.  Random
    restarts help escape local optima in the belief probability landscape.
    """

    name: str = "Hill Climbing"

    def __init__(
        self,
        neighbor_radius: int = HILL_CLIMB_NEIGHBOR_RADIUS,
        max_restarts: int = HILL_CLIMB_MAX_RESTARTS,
        travel_penalty: float = 0.02,
        seed: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        neighbor_radius : Spatial radius defining the hill climbing neighbourhood.
        max_restarts    : Number of random restarts when a local optimum is reached.
        travel_penalty  : Weight on Manhattan distance in the objective function.
        seed            : RNG seed for reproducible restarts.
        """
        self.neighbor_radius = neighbor_radius
        self.max_restarts = max_restarts
        self.travel_penalty = travel_penalty
        self._rng = random.Random(seed)

    def select_next(
        self,
        current_pos: Tuple[int, int],
        belief_map: "BeliefMap",
        env: "DisasterEnvironment",
        battery_remaining: float,
        safety_reserve: float,
    ) -> Optional[Tuple[int, int]]:
        """
        Run hill climbing with random restarts; return globally best destination found.
        """
        usable_battery = battery_remaining - safety_reserve
        beliefs = belief_map.get_all_beliefs()
        candidates = self._get_candidates(beliefs, env, current_pos, usable_battery)

        if not candidates:
            return None

        global_best_cell: Optional[Tuple[int, int]] = None
        global_best_score: float = float("-inf")

        for restart in range(self.max_restarts):
            # Start from a randomly chosen high-probability cell
            if not candidates:
                break
            start_cell = self._rng.choice(candidates[:max(1, len(candidates) // 3)])

            local_best = start_cell
            local_best_score = self._score(start_cell, beliefs, current_pos)

            # Hill climb from start_cell
            improved = True
            while improved:
                improved = False
                neighbours = self._get_neighbourhood(local_best, env, beliefs, current_pos, usable_battery)
                for nbr in neighbours:
                    nbr_score = self._score(nbr, beliefs, current_pos)
                    if nbr_score > local_best_score:
                        local_best_score = nbr_score
                        local_best = nbr
                        improved = True

            # Update global best
            if local_best_score > global_best_score:
                global_best_score = local_best_score
                global_best_cell = local_best

        return global_best_cell

    def _score(
        self,
        cell: Tuple[int, int],
        beliefs: dict,
        current_pos: Tuple[int, int],
    ) -> float:
        """Objective: P(target at cell) - lambda * travel_cost(cell)."""
        prob = beliefs.get(cell, 0.0)
        dist = manhattan_distance(current_pos, cell)
        return prob - self.travel_penalty * dist

    def _get_candidates(
        self,
        beliefs: dict,
        env: "DisasterEnvironment",
        current_pos: Tuple[int, int],
        usable_battery: float,
    ) -> List[Tuple[int, int]]:
        """Return reachable, affordable cells sorted by belief probability (desc)."""
        result = []
        for cell, prob in beliefs.items():
            if prob < 1e-8:
                continue
            if not env.is_traversable(*cell) or cell not in env.reachable_cells:
                continue
            if cell == current_pos:
                continue
            dist = manhattan_distance(current_pos, cell)
            if dist * BATTERY_MOVE_COST_EMPTY > usable_battery:
                continue
            result.append((prob, cell))
        result.sort(reverse=True)
        return [c for _, c in result]

    def _get_neighbourhood(
        self,
        cell: Tuple[int, int],
        env: "DisasterEnvironment",
        beliefs: dict,
        current_pos: Tuple[int, int],
        usable_battery: float,
    ) -> List[Tuple[int, int]]:
        """Return all traversable reachable cells within neighbour_radius of cell."""
        cx, cy = cell
        neighbours = []
        r = self.neighbor_radius
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx == 0 and dy == 0:
                    continue
                nbr = (cx + dx, cy + dy)
                if not env.is_valid(*nbr):
                    continue
                if not env.is_traversable(*nbr):
                    continue
                if nbr not in env.reachable_cells:
                    continue
                if nbr == current_pos:
                    continue
                dist = manhattan_distance(current_pos, nbr)
                if dist * BATTERY_MOVE_COST_EMPTY > usable_battery:
                    continue
                neighbours.append(nbr)
        return neighbours
