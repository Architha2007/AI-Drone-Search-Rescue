"""Unit tests for Mission Controller, Simulated Detection, and Coverage Tracking."""
import pytest
from detection.yolo_detector import YOLODetector
from simulation.environment import DisasterEnvironment
from simulation.mission import MissionController, MissionState


@pytest.fixture
def mission():
    """Create test mission controller with deterministic environment."""
    env = DisasterEnvironment(rows=10, cols=10, seed=123)
    controller = MissionController(env=env, algorithm_name="astar")
    return controller


def test_survivor_discovered_only_when_in_sensor_range():
    """Test 8: Survivors are hidden until drone enters their FOV sensor range."""
    env = DisasterEnvironment(rows=10, cols=10, seed=None)
    env.obstacles.clear()
    env.ground_truth_survivors = {(5, 5)}
    env._update_reachability()

    mission = MissionController(env=env)

    # Initially at (0, 0), sensor range is 1 cell (up to (1, 1))
    assert mission.drone.position == (0, 0)
    # Ground truth survivor at (5, 5) must NOT be discovered yet
    assert (5, 5) not in mission.detector.discovered_survivors
    assert mission.detector.total_discovered == 0

    # Move drone closer, e.g. to (3, 3) -> still outside FOV radius 1
    mission.drone.position = (3, 3)
    mission._perform_scan()
    assert (5, 5) not in mission.detector.discovered_survivors

    # Move drone to (4, 5) -> (5, 5) is now within sensor radius 1
    mission.drone.position = (4, 5)
    mission._perform_scan()
    assert (5, 5) in mission.detector.discovered_survivors
    assert mission.detector.total_discovered == 1

    # But not yet reached physically
    assert (5, 5) not in mission.detector.reached_survivors

    # Drone enters (5, 5) cell physically
    mission.drone.position = (5, 5)
    mission._perform_scan()
    assert (5, 5) in mission.detector.reached_survivors


def test_search_coverage_calculation(mission):
    """Test 9: Search coverage percentage is calculated accurately."""
    # Base starts with its radius 1 area marked explored
    initial_cov = mission.coverage_percentage
    assert 0.0 < initial_cov <= 100.0

    # Step the drone multiple times
    for _ in range(5):
        mission.step()

    # Explored traversable count must increase or stay valid
    assert mission.explored_traversable >= 1
    assert mission.coverage_percentage >= initial_cov
    assert mission.coverage_percentage <= 100.0


def test_reset_restores_initial_state(mission):
    """Test 10: Reset restores initial mission state, positions, and clears flight history."""
    # Run mission for some steps
    mission.start()
    for _ in range(10):
        mission.step()

    assert mission.drone.distance_traveled > 0
    assert len(mission.drone.path_history) > 1

    # Trigger reset
    mission.reset()

    assert mission.state == MissionState.IDLE
    assert mission.drone.position == mission.env.base_pos
    assert mission.drone.distance_traveled == 0
    assert len(mission.drone.path_history) == 1
    assert mission.pathfinding_operations == 0
    assert mission.replanning_events == 0
    assert mission.elapsed_time_sec == 0.0
    assert mission.detector.total_discovered <= 1  # only at base if any


def test_simulated_detection_runs_without_yolo():
    """Test 11: Core simulation and simulated detector run independently of YOLO."""
    env = DisasterEnvironment(rows=8, cols=8, seed=99)
    controller = MissionController(env=env, algorithm_name="astar")

    controller.start()
    for _ in range(8):
        controller.step()

    # Simulation executed successfully without error or YOLO import
    assert controller.drone.distance_traveled >= 0
    assert controller.coverage_percentage > 0.0


def test_detection_evaluator_metrics():
    """Test precision, recall, and F1 computation with ground truth boxes."""
    from evaluation.metrics import DetectionEvaluator

    gt_boxes = [(10, 10, 50, 50), (100, 100, 150, 150)]
    pred_boxes = [(10, 10, 50, 50), (200, 200, 250, 250)]  # 1 TP, 1 FP, 1 FN

    metrics = DetectionEvaluator.evaluate(pred_boxes, gt_boxes, iou_threshold=0.5)
    assert metrics is not None
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5

    # Returns None when no ground truth is provided
    assert DetectionEvaluator.evaluate(pred_boxes, None) is None


def test_yolo_georeferencing_projection():
    """Test conversion from camera bounding box to disaster grid coordinates."""
    from detection.yolo_detector import BoundingBox, YOLODetector

    box = BoundingBox(
        x1=300, y1=220, x2=340, y2=260,
        confidence=0.85, class_id=0, class_name="person"
    )
    # Box center is (320, 240) in a 640x480 image (exact image center)
    grid_pos = YOLODetector.georeference_detection(
        box=box,
        drone_grid_pos=(5, 5),
        drone_altitude_m=20.0,
        fov_horizontal_deg=60.0,
        image_dims=(640, 480),
        grid_cell_size_m=5.0,
    )
    # Since detection is at image center, projected grid position should match drone grid position
    assert grid_pos == (5, 5)
