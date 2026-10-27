"""Unit tests for A* Pathfinding Planner."""
import pytest
from algorithms.astar import AStarPlanner, manhattan_distance
from simulation.environment import CellType, DisasterEnvironment


@pytest.fixture
def empty_env():
    """Create a 10x10 completely open grid."""
    env = DisasterEnvironment(rows=10, cols=10, seed=None)
    env.obstacles.clear()
    env.ground_truth_survivors.clear()
    env._update_reachability()
    return env


def test_astar_simple_unobstructed_path(empty_env):
    """Test 1: A* finds the optimal shortest path on a simple unobstructed grid."""
    planner = AStarPlanner()
    start = (0, 0)
    goal = (3, 3)

    path, nodes_explored, elapsed = planner.find_path(start, goal, empty_env)

    assert len(path) > 0
    assert path[0] == start
    assert path[-1] == goal

    # In 4-connected grid, shortest distance is Manhattan distance: |3-0| + |3-0| = 6 steps.
    # Path includes start node, so path length is 7.
    assert len(path) == manhattan_distance(start, goal) + 1

    # Verify each consecutive step is adjacent (distance == 1)
    for i in range(len(path) - 1):
        assert manhattan_distance(path[i], path[i + 1]) == 1


def test_astar_avoids_obstacles(empty_env):
    """Test 3a: A* navigates around wall obstacles without entering them."""
    planner = AStarPlanner()
    start = (0, 2)
    goal = (4, 2)

    # Place a vertical wall at x=2 with an opening only at (2, 0)
    for y in range(1, 10):
        empty_env.obstacles[(2, y)] = CellType.BUILDING
    empty_env._update_reachability()

    path, nodes_explored, elapsed = planner.find_path(start, goal, empty_env)

    assert len(path) > 0
    assert path[0] == start
    assert path[-1] == goal

    # Verify path never touches any obstacle cell
    for cell in path:
        assert cell not in empty_env.obstacles

    # Must pass through the gap at (2, 0)
    assert (2, 0) in path


def test_astar_unreachable_target(empty_env):
    """Test 4a: A* returns empty path when the target is completely enclosed by obstacles."""
    planner = AStarPlanner()
    start = (0, 0)
    goal = (5, 5)

    # Completely box in (5, 5)
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        empty_env.obstacles[(5 + dx, 5 + dy)] = CellType.BUILDING
    empty_env._update_reachability()

    path, nodes_explored, elapsed = planner.find_path(start, goal, empty_env)

    assert path == []
    assert nodes_explored > 0


def test_astar_start_equals_goal(empty_env):
    """Test 5a: A* returns a single-element path [start] when start == goal."""
    planner = AStarPlanner()
    start = (3, 4)
    goal = (3, 4)

    path, nodes_explored, elapsed = planner.find_path(start, goal, empty_env)

    assert path == [start]


def test_astar_invalid_or_obstacle_endpoints(empty_env):
    """Test invalid coordinates or endpoints located inside obstacles."""
    planner = AStarPlanner()

    # Out of bounds
    path, _, _ = planner.find_path((-1, 0), (3, 3), empty_env)
    assert path == []

    # Goal inside obstacle
    empty_env.obstacles[(4, 4)] = CellType.BUILDING
    path, _, _ = planner.find_path((0, 0), (4, 4), empty_env)
    assert path == []
