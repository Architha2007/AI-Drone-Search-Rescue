"""
A* with Belief-Driven Destination Selection.

Role of A* in Uncertain-Target Search (documented)
----------------------------------------------------
A* is fundamentally a **pathfinding** algorithm.  Given a fixed start and
goal, it finds the shortest (or least-cost) path through a known graph using
an admissible heuristic.  A* does NOT know:
    - Where the hidden target is.
    - What the probability distribution over target locations is.
    - Which cell to search next.

Therefore, A* **alone** cannot function as a complete uncertain-target search
policy.  To integrate A* into this framework, we need TWO components:
    1. A **destination selector** that uses the belief map to choose WHICH
       cell to navigate to next.
    2. A* as the **pathfinder** that computes HOW to physically reach that cell.

This module implements the destination-selection component using the TRUE A*
path cost rather than a Manhattan-distance approximation.

Why A* Belief differs from Best-First Search
--------------------------------------------
BestFirstSearchPolicy scores destinations as:
    score(c) = P(target at c) / (manhattan_distance(c) + 1)

This uses Manhattan distance as a cheap proxy for travel cost.  On an open
grid this is exact, but near obstacles the actual path is longer — Manhattan
distance underestimates the true cost.

AStarBeliefPolicy scores destinations as:
    score(c) = P(target at c) / (actual_astar_path_length(c) + 1)

where actual_astar_path_length(c) is computed by running A* from the drone's
current position to c, respecting all obstacle constraints.

Consequences:
- In open terrain the two policies choose identically (path length == Manhattan
  distance on obstacle-free grids).
- Near walls, debris clusters, or corridors, Best-First may select a cell that
  appears close by Manhattan measure but requires a long detour.  A* Belief
  correctly penalises these cells and may choose a cell that is farther as-the-
  crow-flies but shorter by actual navigable path.
- The policy is more computationally expensive per decision step (O(N log N)
  per candidate instead of O(1)) — this is reflected in the benchmark timing.

Computational optimisation
--------------------------
Running full A* to every candidate is O(K * |V| log|V|) per step.  We limit
the candidate set to the top-M cells by belief probability (default M=40),
since cells with negligible probability are not worth the routing cost.
We also cache A* path lengths within a single call to avoid redundant searches
when the same source position is used across candidates.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from algorithms.astar import AStarPlanner, manhattan_distance
from config import BATTERY_MOVE_COST_EMPTY
from simulation.search_mission import SearchPolicy

if TYPE_CHECKING:
    from simulation.belief_map import BeliefMap
    from simulation.environment import DisasterEnvironment


class AStarBeliefPolicy(SearchPolicy):
    """
    A*-integrated belief-driven destination selection.

    Uses the TRUE A* path length (accounting for obstacles) as the travel cost
    denominator, making it genuinely distinct from BestFirstSearchPolicy which
    uses the cheaper Manhattan distance approximation.

    Decision score: P(target at c) / (astar_path_length(current -> c) + 1)
    """

    name: str = "A* Belief"

    def __init__(
        self,
        min_belief_threshold: float = 1e-6,
        max_candidates: int = 40,
    ) -> None:
        """
        Parameters
        ----------
        min_belief_threshold :
            Cells below this probability are skipped without running A*.
        max_candidates :
            Only run A* for the top-N cells by belief probability.
            This bounds computation cost while preserving quality.
        """
        self.min_belief_threshold = min_belief_threshold
        self.max_candidates = max_candidates
        self._planner = AStarPlanner()

    def select_next(
        self,
        current_pos: Tuple[int, int],
        belief_map: "BeliefMap",
        env: "DisasterEnvironment",
        battery_remaining: float,
        safety_reserve: float,
    ) -> Optional[Tuple[int, int]]:
        """
        Select the next destination using true A* path length as the cost estimate.

        Steps:
        1. Filter candidates by minimum belief threshold and traversability.
        2. Sort candidates by belief probability (descending) — take top-M.
        3. Run A* from current_pos to each candidate to get exact path length.
        4. Score each candidate: P(target at c) / (path_length + 1).
        5. Return the highest-scoring reachable candidate.
        """
        usable_battery = battery_remaining - safety_reserve

        # --- Step 1 & 2: collect and pre-filter candidates ---
        candidates: List[Tuple[float, Tuple[int, int]]] = []
        for cell, prob in belief_map.get_all_beliefs().items():
            if prob < self.min_belief_threshold:
                continue
            if not env.is_traversable(*cell):
                continue
            if cell not in env.reachable_cells:
                continue
            if cell == current_pos:
                continue

            # Quick Manhattan lower-bound battery check before running A*
            lb_dist = manhattan_distance(current_pos, cell)
            if lb_dist * BATTERY_MOVE_COST_EMPTY > usable_battery:
                continue

            candidates.append((prob, cell))

        if not candidates:
            return None

        # Sort by probability descending, take top-M to bound A* calls
        candidates.sort(key=lambda t: t[0], reverse=True)
        candidates = candidates[: self.max_candidates]

        # --- Step 3 & 4: score using actual A* path length ---
        best_cell: Optional[Tuple[int, int]] = None
        best_score: float = -1.0

        for prob, cell in candidates:
            path, _nodes, _ms = self._planner.find_path(current_pos, cell, env)

            if not path or len(path) < 2:
                continue  # Unreachable by A*, or already there (handled above)

            # Actual navigable path length in grid steps (path includes start cell)
            path_length = len(path) - 1

            # Battery check with actual path cost (not lower bound)
            if path_length * BATTERY_MOVE_COST_EMPTY > usable_battery:
                continue

            # Core scoring: P(target at c) / (actual_path_length + 1)
            # This is strictly superior to prob / (manhattan + 1) near obstacles.
            score = prob / (path_length + 1)

            if score > best_score:
                best_score = score
                best_cell = cell

        return best_cell
