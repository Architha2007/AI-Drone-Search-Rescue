"""Evaluation package for benchmarking pathfinding algorithms and detection performance."""
from evaluation.compare_algorithms import (
    generate_comparison_plots,
    print_benchmark_summary,
    run_pathfinding_benchmark,
)
from evaluation.metrics import (
    DetectionEvaluator,
    MissionEvaluationRecord,
    PathfindingMetric,
)

__all__ = [
    "run_pathfinding_benchmark",
    "generate_comparison_plots",
    "print_benchmark_summary",
    "PathfindingMetric",
    "MissionEvaluationRecord",
    "DetectionEvaluator",
]
