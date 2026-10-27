"""Benchmarking and performance comparison between A* and BFS pathfinding algorithms."""
import os
import random
import time
from typing import Dict, List, Tuple

from algorithms.astar import AStarPlanner
from algorithms.bfs import BFSPlanner
from config import FIXED_BENCHMARK_10x10, GRID_COLS, GRID_ROWS
from evaluation.metrics import PathfindingMetric
from simulation.environment import DisasterEnvironment


def run_pathfinding_benchmark(
    num_trials: int = 30,
    seed: int = 42,
) -> Tuple[List[PathfindingMetric], List[PathfindingMetric]]:
    """
    Benchmark A* and BFS on identical disaster maps and identical (start, goal) queries.
    Returns:
        (astar_results, bfs_results)
    """
    random.seed(seed)
    astar_planner = AStarPlanner()
    bfs_planner = BFSPlanner()

    astar_metrics: List[PathfindingMetric] = []
    bfs_metrics: List[PathfindingMetric] = []

    # 1. Benchmark on the Fixed Benchmark Map
    fixed_env = DisasterEnvironment(seed=None)
    fixed_env.load_fixed_map(FIXED_BENCHMARK_10x10)

    # Test fixed map query
    start = fixed_env.base_pos
    goals = list(fixed_env.ground_truth_survivors)
    if goals:
        g = goals[0]
        # A*
        p_a, n_a, t_a = astar_planner.find_path(start, g, fixed_env)
        astar_metrics.append(PathfindingMetric("A*", start, g, bool(p_a), len(p_a), n_a, t_a))
        # BFS
        p_b, n_b, t_b = bfs_planner.find_path(start, g, fixed_env)
        bfs_metrics.append(PathfindingMetric("BFS", start, g, bool(p_b), len(p_b), n_b, t_b))

    # 2. Benchmark on randomized maps across multiple seeds
    for trial_idx in range(num_trials):
        trial_seed = seed + trial_idx + 100
        env = DisasterEnvironment(rows=GRID_ROWS, cols=GRID_COLS, seed=trial_seed)

        reachable_list = list(env.reachable_cells)
        if len(reachable_list) < 2:
            continue

        # Sample start and goal pairs
        sample_start = env.base_pos
        sample_goal = random.choice([c for c in reachable_list if c != sample_start])

        # Run A*
        p_a, n_a, t_a = astar_planner.find_path(sample_start, sample_goal, env)
        astar_metrics.append(
            PathfindingMetric("A*", sample_start, sample_goal, bool(p_a), len(p_a), n_a, t_a)
        )

        # Run BFS
        p_b, n_b, t_b = bfs_planner.find_path(sample_start, sample_goal, env)
        bfs_metrics.append(
            PathfindingMetric("BFS", sample_start, sample_goal, bool(p_b), len(p_b), n_b, t_b)
        )

    return astar_metrics, bfs_metrics


def generate_comparison_plots(
    astar_metrics: List[PathfindingMetric],
    bfs_metrics: List[PathfindingMetric],
    output_path: str = "results/algorithm_comparison.png",
) -> str:
    """
    Generate comparative Matplotlib visualizations for Explored Nodes, Execution Time, and Path Length.
    Saves figure to output_path.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return "Matplotlib/NumPy not installed. Cannot generate comparison plot."

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Filter successful runs
    valid_pairs = [
        (a, b) for a, b in zip(astar_metrics, bfs_metrics)
        if a.path_found and b.path_found
    ]

    if not valid_pairs:
        return "No valid pathfinding pairs to plot."

    astar_nodes = [a.nodes_explored for a, b in valid_pairs]
    bfs_nodes = [b.nodes_explored for a, b in valid_pairs]

    astar_times = [a.execution_time_ms for a, b in valid_pairs]
    bfs_times = [b.execution_time_ms for a, b in valid_pairs]

    astar_lens = [a.path_length for a, b in valid_pairs]
    bfs_lens = [b.path_length for a, b in valid_pairs]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Pathfinding Algorithm Performance Benchmark: A* vs BFS ({len(valid_pairs)} Trials)",
        fontsize=14,
        fontweight="bold",
    )

    # Subplot 1: Explored Nodes Comparison (Bar / Box)
    labels = ["A* Search", "BFS (Baseline)"]
    avg_nodes = [np.mean(astar_nodes), np.mean(bfs_nodes)]
    colors = ["#00bcd4", "#ff9800"]

    bars1 = axes[0].bar(labels, avg_nodes, color=colors, width=0.5, edgecolor="black")
    axes[0].set_title("Average Explored Nodes (Search Space)", fontweight="bold")
    axes[0].set_ylabel("Nodes Explored")
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + (max(avg_nodes) * 0.02),
            f"{yval:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Subplot 2: Execution Time (ms)
    avg_times = [np.mean(astar_times), np.mean(bfs_times)]
    bars2 = axes[1].bar(labels, avg_times, color=colors, width=0.5, edgecolor="black")
    axes[1].set_title("Average Execution Time (ms)", fontweight="bold")
    axes[1].set_ylabel("Time (ms)")
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + (max(avg_times) * 0.02),
            f"{yval:.3f} ms",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Subplot 3: Path Length Optimality Scatter
    axes[2].scatter(astar_lens, bfs_lens, color="#4caf50", alpha=0.7, edgecolors="black", s=50)
    max_len = max(max(astar_lens), max(bfs_lens), 10)
    axes[2].plot([0, max_len], [0, max_len], "r--", label="Optimal Parity (y = x)")
    axes[2].set_title("Path Length Optimality Check", fontweight="bold")
    axes[2].set_xlabel("A* Path Length (steps)")
    axes[2].set_ylabel("BFS Path Length (steps)")
    axes[2].legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return f"Comparison plots successfully saved to {output_path}"


def print_benchmark_summary(
    astar_metrics: List[PathfindingMetric],
    bfs_metrics: List[PathfindingMetric],
) -> None:
    """Print clean summary table to console."""
    valid_pairs = [
        (a, b) for a, b in zip(astar_metrics, bfs_metrics)
        if a.path_found and b.path_found
    ]

    print("\n" + "=" * 68)
    print("      ALGORITHM PERFORMANCE BENCHMARK: A* vs BFS COMPARISON      ")
    print("=" * 68)
    print(f"Total Trials Executed:       {len(astar_metrics)}")
    print(f"Successful Paths Found:      {len(valid_pairs)}")

    if valid_pairs:
        a_nodes = [a.nodes_explored for a, _ in valid_pairs]
        b_nodes = [b.nodes_explored for _, b in valid_pairs]
        a_times = [a.execution_time_ms for a, _ in valid_pairs]
        b_times = [b.execution_time_ms for _, b in valid_pairs]

        avg_a_nodes = sum(a_nodes) / len(a_nodes)
        avg_b_nodes = sum(b_nodes) / len(b_nodes)
        node_reduction = ((avg_b_nodes - avg_a_nodes) / avg_b_nodes) * 100.0 if avg_b_nodes > 0 else 0

        avg_a_time = sum(a_times) / len(a_times)
        avg_b_time = sum(b_times) / len(b_times)

        print("-" * 68)
        print(f"{'Metric':<28} | {'A* (Manhattan)':<16} | {'BFS (Baseline)':<16}")
        print("-" * 68)
        print(f"{'Avg Explored Nodes':<28} | {avg_a_nodes:<16.2f} | {avg_b_nodes:<16.2f}")
        print(f"{'Search Space Reduction':<28} | {f'{node_reduction:.1f}% fewer nodes':<16} | {'Baseline':<16}")
        print(f"{'Avg Query Time (ms)':<28} | {f'{avg_a_time:.4f} ms':<16} | {f'{avg_b_time:.4f} ms':<16}")
        print("=" * 68 + "\n")


if __name__ == "__main__":
    print("Starting A* vs BFS benchmark suite...")
    a_res, b_res = run_pathfinding_benchmark(num_trials=40, seed=42)
    print_benchmark_summary(a_res, b_res)
    plot_msg = generate_comparison_plots(a_res, b_res)
    print(plot_msg)
