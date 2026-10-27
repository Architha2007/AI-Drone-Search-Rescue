"""Detection package supporting both simulated FOV detection and optional YOLO computer vision."""
from detection.simulated_detector import SimulatedDetector
from detection.yolo_detector import BoundingBox, YOLODetector

__all__ = [
    "SimulatedDetector",
    "YOLODetector",
    "BoundingBox",
]
