"""Unit tests for BFS Pathfinding Planner."""
import pytest
from algorithms.bfs import BFSPlanner
from algorithms.astar import manhattan_distance
from simulation.environment import CellType, DisasterEnvironment


@pytest.fixture
def empty_env():
    """Create a 10x10 completely open grid."""
    env = DisasterEnvironment(rows=10, cols=10, seed=None)
    env.obstacles.clear()
    env.ground_truth_survivors.clear()
    env._update_reachability()
    return env


def test_bfs_simple_unweighted_shortest_path(empty_env):
    """Test 2: BFS finds the shortest path on an unweighted grid."""
    planner = BFSPlanner()
    start = (0, 0)
    goal = (3, 3)

    path, nodes_explored, elapsed = planner.find_path(start, goal, empty_env)

    assert len(path) > 0
    assert path[0] == start
    assert path[-1] == goal

    # On unweighted 4-directional grid, shortest path has length = manhattan_distance + 1
    assert len(path) == manhattan_distance(start, goal) + 1

    # Verify each consecutive step is adjacent
    for i in range(len(path) - 1):
        assert manhattan_distance(path[i], path[i + 1]) == 1


def test_bfs_avoids_obstacles(empty_env):
    """Test 3b: BFS navigates around obstacles without entering them."""
    planner = BFSPlanner()
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

    # Path must avoid all obstacles
    for cell in path:
        assert cell not in empty_env.obstacles

    # Must pass through gap
    assert (2, 0) in path


def test_bfs_unreachable_target(empty_env):
    """Test 4b: BFS returns empty path when the target is unreachable."""
    planner = BFSPlanner()
    start = (0, 0)
    goal = (5, 5)

    # Completely box in (5, 5)
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        empty_env.obstacles[(5 + dx, 5 + dy)] = CellType.BUILDING
    empty_env._update_reachability()

    path, nodes_explored, elapsed = planner.find_path(start, goal, empty_env)

    assert path == []
    assert nodes_explored > 0


def test_bfs_start_equals_goal(empty_env):
    """Test 5b: BFS returns [start] when start == goal."""
    planner = BFSPlanner()
    start = (2, 3)
    goal = (2, 3)

    path, nodes_explored, elapsed = planner.find_path(start, goal, empty_env)

    assert path == [start]
    assert nodes_explored == 0 or nodes_explored == 1
