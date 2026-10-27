"""
Beam Search policy for uncertain-target drone search.

Principle
---------
Beam Search is a heuristic search strategy that retains only the best k partial
plans (the "beam") at each planning depth level.  It combines the structure of
BFS-style tree expansion with bounded memory.

Unlike pure greedy (Best-First), Beam Search looks ahead BEAM_SEARCH_DEPTH steps
and evaluates multi-step candidate sequences before committing to a decision.

Algorithm
---------
1. Start with the current drone position as the initial state.
2. For each depth level d = 1 … BEAM_SEARCH_DEPTH:
     a. Expand each beam candidate: generate up to k neighbouring cells
        weighted by belief probability (sample or rank).
     b. Score each expanded sequence using the objective function.
     c. Retain only the top BEAM_WIDTH sequences (prune the rest).
3. Select the sequence with the highest cumulative score.
4. Return the FIRST cell of the best sequence as the next destination.
   (The rest of the sequence is discarded — only one step is executed.)

Objective function (documented)
--------------------------------
    score(seq) = sum_over_seq[ belief_prob(c) * decay^depth ] - lambda * total_dist

where:
    decay        : A discount factor (0 < decay <= 1) that down-weights
                   far-future cells to reflect uncertainty.
    total_dist   : Estimated Manhattan travel length of the sequence.
    lambda       : Trade-off parameter between detection benefit and travel cost.

This multi-step lookahead allows Beam Search to avoid local optima that trap
the pure greedy Best-First policy (e.g., bee-lining to one cluster while
ignoring a high-value region only slightly farther away).

Battery awareness
-----------------
Sequences that would exhaust battery before completion are pruned during expansion.
"""
from __future__ import annotations

import heapq
from typing import TYPE_CHECKING, List, Optional, Tuple

from algorithms.astar import manhattan_distance
from config import BATTERY_MOVE_COST_EMPTY, BEAM_SEARCH_DEPTH, BEAM_WIDTH
from simulation.search_mission import SearchPolicy

if TYPE_CHECKING:
    from simulation.belief_map import BeliefMap
    from simulation.environment import DisasterEnvironment


class BeamSearchPolicy(SearchPolicy):
    """
    Beam Search: retain top-k partial search plans across a planning horizon.

    Parameters
    ----------
    beam_width     : Number of candidate sequences to retain (k).
    search_depth   : Planning horizon (lookahead steps).
    decay          : Future-reward discount factor per step.
    travel_penalty : Weight applied to estimated travel distance in the score.
    """

    name: str = "Beam Search"

    def __init__(
        self,
        beam_width: int = BEAM_WIDTH,
        search_depth: int = BEAM_SEARCH_DEPTH,
        decay: float = 0.85,
        travel_penalty: float = 0.01,
    ) -> None:
        self.beam_width = beam_width
        self.search_depth = search_depth
        self.decay = decay
        self.travel_penalty = travel_penalty

    def select_next(
        self,
        current_pos: Tuple[int, int],
        belief_map: "BeliefMap",
        env: "DisasterEnvironment",
        battery_remaining: float,
        safety_reserve: float,
    ) -> Optional[Tuple[int, int]]:
        """
        Run beam search and return the first cell of the best candidate sequence.
        """
        usable_battery = battery_remaining - safety_reserve
        beliefs = belief_map.get_all_beliefs()

        # Beam: list of (neg_score, sequence, cumulative_dist)
        # We use negated score because heapq is a min-heap.
        # sequence: list of cells visited in order (not including current_pos)
        initial_beam: List[Tuple[float, List[Tuple[int, int]], float]] = [
            (0.0, [], 0.0)
        ]

        beam = initial_beam

        for depth in range(self.search_depth):
            candidates: List[Tuple[float, List[Tuple[int, int]], float]] = []

            for neg_score, seq, cum_dist in beam:
                tip = seq[-1] if seq else current_pos

                # Generate candidate next cells: top cells by belief near tip
                next_cells = self._get_candidate_cells(tip, beliefs, env, current_pos, seq)

                for cell in next_cells:
                    step_dist = manhattan_distance(tip, cell)
                    new_dist = cum_dist + step_dist

                    # Battery feasibility
                    if new_dist * BATTERY_MOVE_COST_EMPTY > usable_battery:
                        continue

                    disc_factor = (self.decay ** (depth + 1))
                    cell_prob = beliefs.get(cell, 0.0)
                    new_score = (-neg_score) + cell_prob * disc_factor - self.travel_penalty * step_dist
                    new_seq = seq + [cell]
                    candidates.append((-new_score, new_seq, new_dist))

            if not candidates:
                break

            # Retain top-k candidates
            candidates.sort(key=lambda t: t[0])  # smallest neg_score = best
            beam = candidates[:self.beam_width]

        if not beam:
            return self._fallback(current_pos, beliefs, env, usable_battery)

        # Best beam sequence is beam[0]; return its first step
        _, best_seq, _ = beam[0]
        if not best_seq:
            return self._fallback(current_pos, beliefs, env, usable_battery)

        return best_seq[0]

    def _get_candidate_cells(
        self,
        tip: Tuple[int, int],
        beliefs: dict,
        env: "DisasterEnvironment",
        origin: Tuple[int, int],
        already_visited: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """
        Return top candidate cells near `tip` by belief probability.
        We look at cells within a neighbourhood sorted by probability,
        ignoring cells already in the current sequence.
        """
        visited_set = set(already_visited)
        # Gather reachable traversable cells with non-trivial belief
        scored = [
            (prob, cell)
            for cell, prob in beliefs.items()
            if prob > 1e-8
            and cell not in visited_set
            and cell != tip
            and env.is_traversable(*cell)
            and cell in env.reachable_cells
        ]
        # Sort by probability desc, take top (beam_width * 2) to limit branching
        scored.sort(reverse=True)
        return [c for _, c in scored[:self.beam_width * 2]]

    def _fallback(
        self,
        current_pos: Tuple[int, int],
        beliefs: dict,
        env: "DisasterEnvironment",
        usable_battery: float,
    ) -> Optional[Tuple[int, int]]:
        """Fallback: return highest-probability reachable cell."""
        best = None
        best_prob = -1.0
        for cell, prob in beliefs.items():
            if not env.is_traversable(*cell) or cell not in env.reachable_cells:
                continue
            if cell == current_pos:
                continue
            dist = manhattan_distance(current_pos, cell)
            if dist * BATTERY_MOVE_COST_EMPTY > usable_battery:
                continue
            if prob > best_prob:
                best_prob = prob
                best = cell
        return best
