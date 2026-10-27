"""
Probabilistic Belief Map for uncertain-target search.

Research question: Where should the drone search next when the target's
location is unknown and battery power is limited?

This module maintains a probability distribution P(target at cell c) over
all valid, traversable search cells.  The distribution is updated via Bayes'
Rule each time the drone makes a sensor observation (detection or no-detection).

Key invariants:
    - Probabilities are always non-negative.
    - The distribution sums to 1.0 after every update.
    - A failed observation (no detection) decreases the observed cell's probability
      proportionally to the detection probability — it does NOT set the cell to 0.
    - The hidden target location is stored here and is NEVER exposed to any search
      policy or UI element unless debug_mode is explicitly enabled.
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from config import (
    DETECTION_PROBABILITY,
    FALSE_POSITIVE_RATE,
    SENSOR_RANGE,
    TERRAIN_PRIOR_EMPTY,
    TERRAIN_PRIOR_NEAR_BUILDING,
    TERRAIN_PRIOR_NEAR_DEBRIS,
)
from simulation.environment import CellType

if TYPE_CHECKING:
    from simulation.environment import DisasterEnvironment


class BeliefMap:
    """
    Maintains a probability distribution over traversable search cells.

    Bayesian update rule
    --------------------
    Let p_c = P(target at cell c).
    When the drone observes a DETECTION at cell d (within sensor range r):
        unnorm_c = p_c * P(detect | target=c, obs=positive)
        where P(detect | target=c, obs=positive) = detection_prob  if c == d
                                                  = false_pos_rate  otherwise

    When the drone observes NO DETECTION at cell d (within sensor range r):
        unnorm_c = p_c * P(no-detect | target=c, obs=negative)
        where P(no-detect | target=c, obs=negative) = 1 - detection_prob  if c in sensor_cells
                                                      = 1.0                otherwise

    After computing unnormalized values, we renormalize so probabilities sum to 1.
    This ensures failed observations reduce probability but never drive it to zero.
    """

    def __init__(
        self,
        env: "DisasterEnvironment",
        seed: Optional[int] = None,
        detection_probability: float = DETECTION_PROBABILITY,
        false_positive_rate: float = FALSE_POSITIVE_RATE,
        sensor_range: int = SENSOR_RANGE,
    ) -> None:
        self.env = env
        self.detection_probability = detection_probability
        self.false_positive_rate = false_positive_rate
        self.sensor_range = sensor_range

        self._rng = random.Random(seed)

        # Probability distribution: cell -> probability
        self._beliefs: Dict[Tuple[int, int], float] = {}

        # Hidden ground-truth target — NEVER exposed outside this class
        # except through the debug_mode accessor.
        self._hidden_target: Optional[Tuple[int, int]] = None

        # Track whether target has been detected
        self.target_detected: bool = False
        self.detection_step: Optional[int] = None

        # Initialize
        self._place_hidden_target()
        self._initialize_priors()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _place_hidden_target(self) -> None:
        """Randomly place a hidden target on a reachable traversable cell."""
        candidates = [
            c for c in self.env.reachable_cells
            if self.env.is_traversable(*c) and c != self.env.base_pos
        ]
        if not candidates:
            self._hidden_target = None
            return
        self._hidden_target = self._rng.choice(candidates)

    def _initialize_priors(self) -> None:
        """
        Build the prior probability distribution using terrain-based likelihoods.

        The unnormalized prior for a cell is weighted by:
        - TERRAIN_PRIOR_NEAR_DEBRIS  if any adjacent cell is CellType.DEBRIS
        - TERRAIN_PRIOR_NEAR_BUILDING if any adjacent cell is CellType.BUILDING
        - TERRAIN_PRIOR_EMPTY        otherwise

        Only valid traversable reachable cells get non-zero priors.
        """
        raw: Dict[Tuple[int, int], float] = {}
        obstacles = self.env.obstacles

        for cell in self.env.reachable_cells:
            if not self.env.is_traversable(*cell):
                continue

            x, y = cell
            weight = TERRAIN_PRIOR_EMPTY

            # Examine 8-connected neighbours for nearby terrain features
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    nbr = (x + dx, y + dy)
                    if nbr in obstacles:
                        if obstacles[nbr] == CellType.DEBRIS:
                            weight = max(weight, TERRAIN_PRIOR_NEAR_DEBRIS)
                        elif obstacles[nbr] == CellType.BUILDING:
                            weight = max(weight, TERRAIN_PRIOR_NEAR_BUILDING)

            raw[cell] = weight

        self._beliefs = self._normalize(raw)

    # ------------------------------------------------------------------
    # Core probability interface
    # ------------------------------------------------------------------

    def get_probability(self, x: int, y: int) -> float:
        """Return P(target at (x, y)).  Returns 0.0 for invalid / obstacle cells."""
        return self._beliefs.get((x, y), 0.0)

    def get_all_beliefs(self) -> Dict[Tuple[int, int], float]:
        """Return a shallow copy of the current belief distribution."""
        return dict(self._beliefs)

    def get_top_k_cells(self, k: int) -> List[Tuple[Tuple[int, int], float]]:
        """Return the top-k cells by probability as [(cell, prob), ...]."""
        return sorted(self._beliefs.items(), key=lambda kv: kv[1], reverse=True)[:k]

    def max_belief_cell(self) -> Optional[Tuple[int, int]]:
        """Return the cell with the highest current belief probability."""
        if not self._beliefs:
            return None
        return max(self._beliefs, key=lambda c: self._beliefs[c])

    # ------------------------------------------------------------------
    # Bayesian update
    # ------------------------------------------------------------------

    def update_observation(
        self,
        sensor_center: Tuple[int, int],
        detection_occurred: bool,
        step: int = 0,
    ) -> None:
        """
        Update beliefs using Bayes' rule based on a sensor observation.

        Parameters
        ----------
        sensor_center :
            The grid cell where the drone is currently located (observation origin).
        detection_occurred :
            True  → sensor fired (may be true positive or false positive).
            False → sensor did NOT fire (may be true negative or missed detection).
        step :
            Current simulation step (recorded if target detected).
        """
        # Determine sensor footprint (cells within sensor_range)
        scx, scy = sensor_center
        sensor_cells = set()
        for dy in range(-self.sensor_range, self.sensor_range + 1):
            for dx in range(-self.sensor_range, self.sensor_range + 1):
                c = (scx + dx, scy + dy)
                if self.env.is_valid(*c) and self.env.is_traversable(*c):
                    sensor_cells.add(c)

        updated: Dict[Tuple[int, int], float] = {}

        for cell, prior in self._beliefs.items():
            if prior <= 0.0:
                continue

            if detection_occurred:
                # P(obs=detection | target=cell)
                if cell in sensor_cells:
                    # Could be a true positive
                    likelihood = self.detection_probability
                else:
                    # Must be a false positive
                    likelihood = self.false_positive_rate
            else:
                # P(obs=no-detection | target=cell)
                if cell in sensor_cells:
                    # Could be a missed detection
                    likelihood = 1.0 - self.detection_probability
                else:
                    # Target outside sensor range → observation is uninformative
                    likelihood = 1.0

            updated[cell] = prior * likelihood

        self._beliefs = self._normalize(updated)

        # Check if actual target is in sensor range this step
        if (
            self._hidden_target is not None
            and self._hidden_target in sensor_cells
            and detection_occurred
            and not self.target_detected
        ):
            self.target_detected = True
            self.detection_step = step

    def mark_cell_searched(self, cell: Tuple[int, int]) -> None:
        """
        Aggressively reduce belief for a physically visited cell.
        We do not set to zero; we apply detection-probability-based decay.
        """
        if cell not in self._beliefs:
            return
        raw = dict(self._beliefs)
        # Strong reduction: apply missed-detection likelihood squared
        raw[cell] *= (1.0 - self.detection_probability) ** 2
        self._beliefs = self._normalize(raw)

    # ------------------------------------------------------------------
    # Simulate noisy sensor observation
    # ------------------------------------------------------------------

    def simulate_sensor_reading(
        self,
        sensor_center: Tuple[int, int],
    ) -> Tuple[bool, bool]:
        """
        Simulate whether the sensor fires at sensor_center, given the hidden target.

        Returns
        -------
        (detection_occurred, is_true_positive)
            detection_occurred : bool — whether the sensor reported a detection.
            is_true_positive   : bool — whether the target was actually in range
                                        (only accurate in post-analysis / debug).
        """
        scx, scy = sensor_center
        sensor_cells = set()
        for dy in range(-self.sensor_range, self.sensor_range + 1):
            for dx in range(-self.sensor_range, self.sensor_range + 1):
                c = (scx + dx, scy + dy)
                if self.env.is_valid(*c) and self.env.is_traversable(*c):
                    sensor_cells.add(c)

        target_in_range = self._hidden_target in sensor_cells

        if target_in_range:
            # True positive with probability detection_probability
            detection_occurred = self._rng.random() < self.detection_probability
            return detection_occurred, True
        else:
            # False alarm with probability false_positive_rate
            detection_occurred = self._rng.random() < self.false_positive_rate
            return detection_occurred, False

    # ------------------------------------------------------------------
    # Debug / evaluation interface (hidden target)
    # ------------------------------------------------------------------

    @property
    def debug_hidden_target(self) -> Optional[Tuple[int, int]]:
        """
        Reveal the actual hidden target location.

        **WARNING**: This property must NEVER be called by any search policy or
        used in belief-update logic — that would violate the research assumption.
        Intended only for:
          - Debug overlay rendering (when debug_mode is toggled on)
          - Post-mission evaluation (success / failure classification)
          - Unit testing
        """
        return self._hidden_target

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(raw: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
        """
        Normalize a dictionary of non-negative values to sum to 1.0.

        Handles edge cases:
        - If all values are zero (impossible), fall back to uniform distribution.
        - Negative values are clamped to zero before normalization.
        """
        total = sum(max(0.0, v) for v in raw.values())
        if total <= 0.0:
            # Impossible distribution — fall back to uniform
            n = len(raw)
            if n == 0:
                return {}
            uniform_p = 1.0 / n
            return {k: uniform_p for k in raw}
        return {k: max(0.0, v) / total for k, v in raw.items()}

    def reset(
        self,
        env: Optional["DisasterEnvironment"] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Re-initialize belief map for a new mission (same or different environment)."""
        if env is not None:
            self.env = env
        if seed is not None:
            self._rng = random.Random(seed)
        self.target_detected = False
        self.detection_step = None
        self._place_hidden_target()
        self._initialize_priors()

    @property
    def valid_cell_count(self) -> int:
        """Number of cells tracked in the belief distribution."""
        return len(self._beliefs)
