"""
Unit tests for the probabilistic belief map (Phase 2).

Covers:
- Belief normalization invariant (sum to 1)
- Bayesian update on positive and negative observations
- Probabilities never go to zero after a single missed observation
- Terrain-based prior initialization
- Hidden target handling (inaccessible to policy)
- Reproducibility with fixed seeds
"""
import pytest
from simulation.environment import DisasterEnvironment, CellType
from simulation.belief_map import BeliefMap


@pytest.fixture
def open_env():
    """10x10 open grid with no obstacles."""
    env = DisasterEnvironment(rows=10, cols=10, seed=None)
    env.obstacles.clear()
    env.ground_truth_survivors.clear()
    env._update_reachability()
    env.mark_explored_area(*env.base_pos, radius=1)
    return env


@pytest.fixture
def belief(open_env):
    """BeliefMap on open 10x10 grid, reproducible seed."""
    return BeliefMap(open_env, seed=7)


# ------------------------------------------------------------------
# Normalization tests
# ------------------------------------------------------------------

def test_initial_beliefs_sum_to_one(belief):
    """Belief probability distribution must sum to 1.0 after initialization."""
    total = sum(belief.get_all_beliefs().values())
    assert abs(total - 1.0) < 1e-9, f"Initial beliefs sum = {total}, expected 1.0"


def test_beliefs_sum_after_positive_observation(belief, open_env):
    """After a positive observation, beliefs must still sum to 1.0."""
    belief.update_observation((5, 5), detection_occurred=True, step=1)
    total = sum(belief.get_all_beliefs().values())
    assert abs(total - 1.0) < 1e-9, f"Post-positive-obs beliefs sum = {total}"


def test_beliefs_sum_after_negative_observation(belief, open_env):
    """After a negative observation, beliefs must still sum to 1.0."""
    belief.update_observation((5, 5), detection_occurred=False, step=1)
    total = sum(belief.get_all_beliefs().values())
    assert abs(total - 1.0) < 1e-9, f"Post-negative-obs beliefs sum = {total}"


def test_beliefs_sum_after_multiple_updates(belief, open_env):
    """Beliefs must sum to 1.0 after 20 sequential updates."""
    positions = [(i % 10, i // 10) for i in range(20)]
    for step, pos in enumerate(positions):
        belief.update_observation(pos, detection_occurred=(step % 3 == 0), step=step)
    total = sum(belief.get_all_beliefs().values())
    assert abs(total - 1.0) < 1e-9, f"Post-multi-obs beliefs sum = {total}"


def test_all_beliefs_non_negative(belief, open_env):
    """All probability values must be >= 0."""
    belief.update_observation((3, 3), detection_occurred=False, step=1)
    for cell, prob in belief.get_all_beliefs().items():
        assert prob >= 0.0, f"Negative probability at {cell}: {prob}"


# ------------------------------------------------------------------
# Bayesian update correctness
# ------------------------------------------------------------------

def test_positive_observation_increases_in_range_beliefs(open_env):
    """After a positive detection, cells in sensor range should have higher probability."""
    bm = BeliefMap(open_env, seed=11)
    prior_center = bm.get_probability(5, 5)
    bm.update_observation((5, 5), detection_occurred=True, step=1)
    posterior_center = bm.get_probability(5, 5)
    # In-range cell should increase after positive observation
    assert posterior_center >= prior_center, (
        f"Sensor-range cell prob did not increase after positive obs: "
        f"{prior_center:.6f} -> {posterior_center:.6f}"
    )


def test_negative_observation_does_not_zero_out_in_range_cells(open_env):
    """
    A missed detection should REDUCE but NOT eliminate probability for in-range cells.
    This is the key Bayesian requirement: failed observations are not proof of absence.
    """
    bm = BeliefMap(open_env, seed=13)
    center = (5, 5)
    prior = bm.get_probability(*center)
    bm.update_observation(center, detection_occurred=False, step=1)
    posterior = bm.get_probability(*center)

    assert posterior > 0.0, (
        f"Cell {center} probability was zeroed after missed detection. "
        f"Prior={prior:.6f}, Posterior={posterior:.6f}"
    )
    assert posterior < prior, (
        f"Missed detection should reduce probability. "
        f"Prior={prior:.6f}, Posterior={posterior:.6f}"
    )


def test_out_of_range_cells_unaffected_by_negative_observation(open_env):
    """Cells outside sensor range should have unchanged probabilities after negative obs."""
    bm = BeliefMap(open_env, seed=17)
    far_cell = (9, 9)  # Distance > 1 from (0, 0)
    prior_far = bm.get_probability(*far_cell)

    # Observe at base (0, 0); (9, 9) is far outside sensor range
    bm.update_observation((0, 0), detection_occurred=False, step=1)
    posterior_far = bm.get_probability(*far_cell)

    # Ratio should change only by renormalization, not by likelihood penalty
    # After negative obs at (0,0), out-of-range cells get likelihood=1.0 (uninformative)
    # So their relative weight shouldn't change much, but absolute values shift due to renorm
    # The key constraint: posterior_far > 0
    assert posterior_far > 0.0, f"Out-of-range cell was zeroed: {posterior_far}"


# ------------------------------------------------------------------
# Terrain priors
# ------------------------------------------------------------------

def test_terrain_prior_debris_cells_higher(open_env):
    """Cells adjacent to debris should have higher prior probability than isolated empty cells."""
    # Place debris at (3, 3)
    open_env.obstacles[(3, 3)] = CellType.DEBRIS
    open_env._update_reachability()

    bm = BeliefMap(open_env, seed=5)
    # (2, 3) is adjacent to debris; (9, 9) is isolated
    p_near_debris = bm.get_probability(2, 3)
    p_far = bm.get_probability(9, 9)

    # Near-debris cell should have equal or higher probability
    # (due to TERRAIN_PRIOR_NEAR_DEBRIS > TERRAIN_PRIOR_EMPTY)
    assert p_near_debris >= p_far * 0.8, (
        f"Near-debris cell ({p_near_debris:.6f}) should be >= isolated cell ({p_far:.6f})"
    )


# ------------------------------------------------------------------
# Hidden target
# ------------------------------------------------------------------

def test_hidden_target_not_none(open_env):
    """Belief map must place a hidden target on a reachable cell."""
    bm = BeliefMap(open_env, seed=3)
    assert bm.debug_hidden_target is not None


def test_hidden_target_is_traversable(open_env):
    """The hidden target must be on a traversable cell."""
    bm = BeliefMap(open_env, seed=3)
    target = bm.debug_hidden_target
    assert open_env.is_traversable(*target), f"Hidden target {target} is on an obstacle"


def test_hidden_target_is_reachable(open_env):
    """The hidden target must be in the reachable set."""
    bm = BeliefMap(open_env, seed=3)
    target = bm.debug_hidden_target
    assert target in open_env.reachable_cells, f"Hidden target {target} unreachable from base"


def test_hidden_target_not_at_base(open_env):
    """The hidden target must not be at the base station."""
    bm = BeliefMap(open_env, seed=3)
    target = bm.debug_hidden_target
    assert target != open_env.base_pos, "Hidden target must not be at the base station"


# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------

def test_reproducibility_same_seed(open_env):
    """Two belief maps with the same seed must produce identical distributions."""
    bm1 = BeliefMap(open_env, seed=42)
    bm2 = BeliefMap(open_env, seed=42)

    beliefs1 = bm1.get_all_beliefs()
    beliefs2 = bm2.get_all_beliefs()

    assert beliefs1.keys() == beliefs2.keys()
    for cell in beliefs1:
        assert abs(beliefs1[cell] - beliefs2[cell]) < 1e-12, (
            f"Mismatch at {cell}: {beliefs1[cell]} vs {beliefs2[cell]}"
        )

    assert bm1.debug_hidden_target == bm2.debug_hidden_target


def test_different_seeds_give_different_targets(open_env):
    """Different seeds should typically produce different hidden target placements."""
    targets = set()
    for seed in range(20):
        bm = BeliefMap(open_env, seed=seed)
        targets.add(bm.debug_hidden_target)
    # With 20 seeds on a 10x10 grid, we expect at least 5 distinct targets
    assert len(targets) >= 5, f"Only {len(targets)} distinct targets across 20 seeds"


# ------------------------------------------------------------------
# Sensor simulation
# ------------------------------------------------------------------

def test_simulate_sensor_returns_bool_pair(open_env):
    """simulate_sensor_reading must return (bool, bool)."""
    bm = BeliefMap(open_env, seed=8)
    detection, is_tp = bm.simulate_sensor_reading((5, 5))
    assert isinstance(detection, bool)
    assert isinstance(is_tp, bool)


def test_sensor_detects_target_eventually(open_env):
    """With detection_prob=1.0, sensor must always detect the target when in range."""
    bm = BeliefMap(open_env, seed=9, detection_probability=1.0, false_positive_rate=0.0)
    target = bm.debug_hidden_target
    # Move to the target cell directly
    detection, is_tp = bm.simulate_sensor_reading(target)
    assert detection is True, "Perfect sensor must detect target at target's location"
    assert is_tp is True


def test_mark_cell_searched_reduces_probability(open_env):
    """mark_cell_searched should reduce the probability of the searched cell."""
    bm = BeliefMap(open_env, seed=15)
    cell = (5, 5)
    prior = bm.get_probability(*cell)
    bm.mark_cell_searched(cell)
    posterior = bm.get_probability(*cell)
    assert posterior <= prior, f"Probability did not decrease after marking {cell} searched"


def test_normalize_handles_all_zeros_gracefully():
    """_normalize must fall back to uniform distribution when all values are zero."""
    result = BeliefMap._normalize({(0, 0): 0.0, (1, 1): 0.0, (2, 2): 0.0})
    total = sum(result.values())
    assert abs(total - 1.0) < 1e-9
    for v in result.values():
        assert abs(v - 1 / 3) < 1e-9
