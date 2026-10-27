"""Unit tests for Disaster Environment and Virtual Drone movement safety."""
import pytest
from simulation.drone import VirtualDrone
from simulation.environment import CellType, DisasterEnvironment


@pytest.fixture
def env():
    """Create test environment."""
    return DisasterEnvironment(rows=10, cols=10, seed=42)


def test_drone_never_enters_obstacle(env):
    """Test 6: The drone model strictly refuses to enter obstacle cells."""
    drone = VirtualDrone(start_pos=(0, 0))

    # Place an obstacle directly next to the drone at (1, 0)
    env.obstacles[(1, 0)] = CellType.BUILDING

    # Attempt to move into the obstacle
    moved = drone.move_to((1, 0), env)

    assert moved is False
    assert drone.position == (0, 0)
    assert (1, 0) not in drone.path_history[1:]


def test_drone_stays_within_map_boundaries(env):
    """Test 7: The drone model strictly refuses to move outside the grid boundaries."""
    drone = VirtualDrone(start_pos=(0, 0))

    # Attempt to move negative x
    assert drone.move_to((-1, 0), env) is False
    assert drone.position == (0, 0)

    # Attempt to move negative y
    assert drone.move_to((0, -1), env) is False
    assert drone.position == (0, 0)

    # Move to top-right corner and test upper boundaries
    drone.position = (9, 9)
    assert drone.move_to((10, 9), env) is False
    assert drone.move_to((9, 10), env) is False
    assert drone.position == (9, 9)


def test_drone_disallows_diagonal_teleportation(env):
    """Test that movement is strictly 4-directional unit steps."""
    drone = VirtualDrone(start_pos=(0, 0))
    # Diagonal jump
    assert drone.move_to((1, 1), env) is False
    assert drone.position == (0, 0)


def test_environment_dynamic_obstacle_toggle(env):
    """Test dynamic obstacle toggle and reachability recomputation."""
    target_pos = (5, 5)
    assert env.is_traversable(*target_pos)

    # Add obstacle
    added = env.toggle_obstacle(*target_pos, CellType.DEBRIS)
    assert added is True
    assert not env.is_traversable(*target_pos)
    assert env.obstacles[target_pos] == CellType.DEBRIS

    # Remove obstacle
    removed = env.toggle_obstacle(*target_pos)
    assert removed is False
    assert env.is_traversable(*target_pos)
    assert target_pos not in env.obstacles
