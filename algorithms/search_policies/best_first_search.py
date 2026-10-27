"""
Best-First Search policy for uncertain-target drone search.

Principle
---------
Best-First Search (BFS in the decision-making sense, not graph BFS) is a greedy
strategy: at each destination selection step it evaluates ALL candidate cells and
picks the one with the highest heuristic score.

Heuristic (documented)
-----------------------
    score(c) = belief_probability(c) / (travel_cost(c) + 1)

where:
    belief_probability(c) = P(target at c) from the Bayesian belief map.
    travel_cost(c)        = Manhattan distance from current drone position to c.
                            Adding 1 prevents division-by-zero when the drone
                            is already at c.

Rationale:
    - We want to search cells that are both HIGH probability (likely to contain
      the target) AND LOW travel cost (cheap to reach with limited battery).
    - Dividing probability by distance implements a value-efficiency trade-off:
      a slightly lower-probability cell nearby can outrank a higher-probability
      cell that is far away.

This is the baseline greedy policy.  It does not plan ahead (no lookahead),
so it can make locally optimal decisions that are globally suboptimal (e.g.,
zig-zagging across the map).

Battery awareness
-----------------
Cells that would require more battery than is safely available are excluded.
Travel cost is estimated as a lower bound (Manhattan distance × min_step_cost).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

from algorithms.astar import manhattan_distance
from simulation.search_mission import SearchPolicy

if TYPE_CHECKING:
    from simulation.belief_map import BeliefMap
    from simulation.environment import DisasterEnvironment


class BestFirstSearchPolicy(SearchPolicy):
    """
    Greedy Best-First Search: selects the candidate cell maximising
    belief_probability / (manhattan_distance + 1).
    """

    name: str = "Best-First"

    def __init__(self, min_belief_threshold: float = 1e-6) -> None:
        """
        Parameters
        ----------
        min_belief_threshold :
            Cells with probability below this value are skipped to avoid
            wasting battery on effectively-zero candidates.
        """
        self.min_belief_threshold = min_belief_threshold

    def select_next(
        self,
        current_pos: Tuple[int, int],
        belief_map: "BeliefMap",
        env: "DisasterEnvironment",
        battery_remaining: float,
        safety_reserve: float,
    ) -> Optional[Tuple[int, int]]:
        """
        Select the cell maximising score = P(target) / (dist + 1).
        Excludes cells that are unreachable or would drain battery below reserve.
        """
        from config import BATTERY_MOVE_COST_EMPTY

        usable_battery = battery_remaining - safety_reserve
        best_cell = None
        best_score = -1.0

        for cell, prob in belief_map.get_all_beliefs().items():
            if prob < self.min_belief_threshold:
                continue
            if not env.is_traversable(*cell):
                continue
            if cell not in env.reachable_cells:
                continue
            if cell == current_pos:
                continue

            dist = manhattan_distance(current_pos, cell)
            # Battery feasibility lower bound: at least 'dist' steps
            if dist * BATTERY_MOVE_COST_EMPTY > usable_battery:
                continue

            score = prob / (dist + 1)
            if score > best_score:
                best_score = score
                best_cell = cell

        return best_cell
