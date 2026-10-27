"""Intelligent target selection with prioritized survivor rescue and coverage fallback."""
from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from simulation.environment import DisasterEnvironment


class SearchStrategy(Enum):
    """Target selection strategy."""
    BALANCED = auto()        # Sweeps area, immediately reroutes to rescue discovered survivors
    PURE_COVERAGE = auto()   # Focuses strictly on map coverage first, rescues at the end
    CLOSEST_FIRST = auto()   # Greedy closest unvisited cell exploration


class TargetSelector:
    """
    Selects the next navigation target based on mission priorities:
    1. Unrescued discovered survivors.
    2. Systematic area coverage.
    3. Return to base when complete.
    """

    def __init__(self, strategy: SearchStrategy = SearchStrategy.BALANCED):
        self.strategy: SearchStrategy = strategy
        self.unreachable_targets: Set[Tuple[int, int]] = set()

    def select_next_target(
        self,
        current_pos: Tuple[int, int],
        base_pos: Tuple[int, int],
        env: DisasterEnvironment,
        discovered_survivors: Set[Tuple[int, int]],
        reached_survivors: Set[Tuple[int, int]],
        sweep_sequence: List[Tuple[int, int]],
    ) -> Tuple[Optional[Tuple[int, int]], str]:
        """
        Determine the next target coordinate and description.

        Returns:
            Tuple of (target_coordinate, reason_string).
            target_coordinate is None if mission is finished at base.
        """
        cx, cy = current_pos

        # 1. Prioritize unreached discovered survivors in BALANCED mode
        if self.strategy == SearchStrategy.BALANCED:
            pending_rescue = [
                s for s in discovered_survivors
                if s not in reached_survivors and s not in self.unreachable_targets and env.is_traversable(*s)
            ]
            if pending_rescue:
                # Select closest survivor to current position
                pending_rescue.sort(key=lambda s: abs(s[0] - cx) + abs(s[1] - cy))
                target = pending_rescue[0]
                return target, f"Investigating/Rescuing discovered survivor at {target}"

        # 2. Select next unvisited cell from coverage sweep
        unexplored_reachable = [
            p for p in env.reachable_cells
            if p not in env.explored_cells
            and p not in self.unreachable_targets
            and env.is_traversable(*p)
        ]

        if unexplored_reachable:
            if self.strategy == SearchStrategy.CLOSEST_FIRST:
                unexplored_reachable.sort(key=lambda p: abs(p[0] - cx) + abs(p[1] - cy))
                target = unexplored_reachable[0]
                return target, f"Exploring nearest cell at {target}"

            # Standard Boustrophedon sweep preference
            for sweep_cell in sweep_sequence:
                if (
                    sweep_cell not in env.explored_cells
                    and sweep_cell not in self.unreachable_targets
                    and env.is_traversable(*sweep_cell)
                ):
                    return sweep_cell, f"Executing coverage sweep towards {sweep_cell}"

            # Fallback to nearest reachable if sweep cells were skipped
            unexplored_reachable.sort(key=lambda p: abs(p[0] - cx) + abs(p[1] - cy))
            target = unexplored_reachable[0]
            return target, f"Exploring unvisited sector at {target}"

        # 3. If in PURE_COVERAGE mode, now attend to any remaining discovered survivors
        pending_rescue = [
            s for s in discovered_survivors
            if s not in reached_survivors and s not in self.unreachable_targets and env.is_traversable(*s)
        ]
        if pending_rescue:
            pending_rescue.sort(key=lambda s: abs(s[0] - cx) + abs(s[1] - cy))
            target = pending_rescue[0]
            return target, f"Rescuing survivor at {target} after coverage sweep"

        # 4. All coverage complete and all survivors reached -> Return to base
        if current_pos != base_pos:
            return base_pos, f"Coverage complete. Returning to base station at {base_pos}"

        # 5. Already at base with everything explored
        return None, "Mission fully accomplished. Drone secured at base."

    def mark_unreachable(self, target: Tuple[int, int]) -> None:
        """Mark a target as unreachable to prevent infinite replanning loops."""
        self.unreachable_targets.add(target)

    def clear_unreachable(self) -> None:
        """Clear unreachable cache (e.g. after environment modifications)."""
        self.unreachable_targets.clear()
