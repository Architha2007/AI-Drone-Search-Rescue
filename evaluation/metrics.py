"""Performance metrics tracking, pathfinding benchmarks, and detection evaluation."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PathfindingMetric:
    """Benchmark result for a single pathfinding query."""
    algorithm: str
    start: Tuple[int, int]
    goal: Tuple[int, int]
    path_found: bool
    path_length: int
    nodes_explored: int
    execution_time_ms: float


@dataclass
class MissionEvaluationRecord:
    """Summary record for a completed simulation run."""
    algorithm_name: str
    grid_dimensions: Tuple[int, int]
    total_traversable: int
    total_reachable: int
    explored_traversable: int
    coverage_percentage: float
    ground_truth_survivors: int
    discovered_survivors: int
    remaining_undiscovered: int
    survivors_rescued: int
    total_distance_steps: int
    pathfinding_operations: int
    replanning_events: int
    elapsed_time_sec: float
    completed: bool


class DetectionEvaluator:
    """
    Evaluates computer vision detections against labeled ground-truth bounding boxes.
    Does NOT invent mock accuracy figures. Only computes metrics when ground truth is provided.
    """

    @staticmethod
    def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        """
        Compute Intersection over Union (IoU) between two boxes (x1, y1, x2, y2).
        """
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])

        inter_area = max(0, xb - xa) * max(0, yb - ya)
        box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        union_area = float(box_a_area + box_b_area - inter_area)
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    @classmethod
    def evaluate(
        cls,
        predicted_boxes: List[Tuple[int, int, int, int]],
        ground_truth_boxes: Optional[List[Tuple[int, int, int, int]]],
        iou_threshold: float = 0.5,
    ) -> Optional[Dict[str, float]]:
        """
        Calculate precision, recall, and F1-score.
        Returns None if ground_truth_boxes is None or empty.
        """
        if ground_truth_boxes is None:
            return None

        if len(ground_truth_boxes) == 0:
            if len(predicted_boxes) == 0:
                return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": len(predicted_boxes), "fn": 0}

        matched_gt = set()
        tp = 0
        fp = 0

        for p_box in predicted_boxes:
            best_iou = 0.0
            best_gt_idx = -1
            for idx, gt_box in enumerate(ground_truth_boxes):
                if idx in matched_gt:
                    continue
                iou = cls.compute_iou(p_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou >= iou_threshold and best_gt_idx != -1:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1

        fn = len(ground_truth_boxes) - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
