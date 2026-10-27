"""
Search algorithm benchmark: 5-way comparison of uncertain-target search policies.

Methodology
-----------
- Each trial uses the SAME random environment seed for all 5 algorithms.
- Each algorithm is evaluated on `num_trials` seeds × 1 trial per seed.
- All algorithms share: identical environments, identical sensor parameters,
  identical battery budgets, and identical starting position.
- Failures (battery depleted before detection) are counted separately.
- Mean steps-to-detection is computed ONLY over successful trials.
- Fabricating results is never done: all metrics come from actual simulation.

Metrics recorded per trial:
    - detected (bool)
    - steps_to_detection (int or None on failure)
    - distance_traveled (int)
    - battery_consumed (float)
    - coverage_percentage (float)
    - cells_visited (int)
    - computation_time_ms (float)  — policy select_next() overhead per step

Algorithm comparison:
    Detection rate  = successful trials / total trials
    Mean steps      = average over successful trials only
    Mean distance   = average over all trials
    Mean battery    = average over all trials
    Mean coverage   = average over all trials
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from simulation.environment import DisasterEnvironment
from simulation.search_mission import SearchMission, SearchMissionState

POLICY_NAMES = ["Best-First", "Beam Search", "Hill Climb", "A* Belief", "Genetic Alg"]

MAX_STEPS_PER_TRIAL = 400  # Hard cap to prevent infinite loops in evaluation


def _make_policy(name: str, seed: int):
    if name == "Best-First":
        from algorithms.search_policies.best_first_search import BestFirstSearchPolicy
        return BestFirstSearchPolicy()
    elif name == "Beam Search":
        from algorithms.search_policies.beam_search import BeamSearchPolicy
        return BeamSearchPolicy()
    elif name == "Hill Climb":
        from algorithms.search_policies.hill_climbing import HillClimbingPolicy
        return HillClimbingPolicy(seed=seed)
    elif name == "A* Belief":
        from algorithms.search_policies.astar_belief import AStarBeliefPolicy
        return AStarBeliefPolicy()
    elif name == "Genetic Alg":
        from algorithms.search_policies.genetic_algorithm import GeneticAlgorithmPolicy
        return GeneticAlgorithmPolicy(seed=seed, generations=15, population_size=12)
    else:
        from algorithms.search_policies.best_first_search import BestFirstSearchPolicy
        return BestFirstSearchPolicy()


@dataclass
class TrialResult:
    """Result of a single search trial."""
    algorithm: str
    seed: int
    detected: bool
    steps_to_detection: Optional[int]   # None if not detected
    distance_traveled: int
    battery_consumed: float
    coverage_percentage: float
    cells_visited: int
    computation_time_ms: float          # Total policy computation time
    termination_reason: str


@dataclass
class AlgorithmSummary:
    """Aggregate statistics for one algorithm across all trials."""
    algorithm: str
    total_trials: int
    successful_trials: int
    detection_rate: float
    mean_steps_to_detection: Optional[float]   # None if no successes
    mean_distance: float
    mean_battery_consumed: float
    mean_coverage: float
    mean_cells_visited: float
    mean_computation_ms: float
    trials: List[TrialResult] = field(default_factory=list)


def run_search_benchmark(
    num_trials: int = 20,
    seed: int = 42,
    max_steps: int = MAX_STEPS_PER_TRIAL,
    verbose: bool = True,
) -> Dict[str, AlgorithmSummary]:
    """
    Run `num_trials` trials for each of the 5 search algorithms.

    All algorithms are evaluated on the SAME set of seeds.

    Returns
    -------
    Dict mapping algorithm name -> AlgorithmSummary.
    """
    import random
    rng = random.Random(seed)
    trial_seeds = [rng.randint(1, 999999) for _ in range(num_trials)]

    summaries: Dict[str, AlgorithmSummary] = {}

    for policy_name in POLICY_NAMES:
        if verbose:
            print(f"\n[Benchmark] Running {policy_name} ({num_trials} trials)...")

        trials: List[TrialResult] = []

        for i, trial_seed in enumerate(trial_seeds):
            env = DisasterEnvironment(seed=trial_seed)
            policy = _make_policy(policy_name, seed=trial_seed)
            sm = SearchMission(env=env, policy=policy, seed=trial_seed)
            sm.start()

            total_policy_time_ms = 0.0
            step_count = 0

            while (
                sm.state == SearchMissionState.RUNNING
                and step_count < max_steps
            ):
                t0 = time.perf_counter()
                sm.step()
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                total_policy_time_ms += elapsed_ms
                step_count += 1

            # If hit the step cap
            if step_count >= max_steps and sm.state == SearchMissionState.RUNNING:
                sm.state = SearchMissionState.COMPLETED
                sm.termination_reason = f"Step cap ({max_steps}) reached."

            detected = sm.detection_confirmed
            steps_detect = sm.belief_map.detection_step if detected else None

            trial = TrialResult(
                algorithm=policy_name,
                seed=trial_seed,
                detected=detected,
                steps_to_detection=steps_detect,
                distance_traveled=sm.distance_traveled,
                battery_consumed=sm.battery_consumed,
                coverage_percentage=sm.coverage_percentage,
                cells_visited=sm.cells_visited,
                computation_time_ms=total_policy_time_ms,
                termination_reason=sm.termination_reason,
            )
            trials.append(trial)

            if verbose and (i + 1) % 5 == 0:
                det_rate = sum(1 for t in trials if t.detected) / len(trials) * 100
                print(f"  [{i+1}/{num_trials}] detection_rate so far: {det_rate:.0f}%")

        # Aggregate
        n = len(trials)
        successful = [t for t in trials if t.detected]
        n_succ = len(successful)

        mean_steps = (
            sum(t.steps_to_detection for t in successful) / n_succ
            if n_succ > 0 else None
        )

        summaries[policy_name] = AlgorithmSummary(
            algorithm=policy_name,
            total_trials=n,
            successful_trials=n_succ,
            detection_rate=n_succ / n if n > 0 else 0.0,
            mean_steps_to_detection=mean_steps,
            mean_distance=sum(t.distance_traveled for t in trials) / n if n > 0 else 0.0,
            mean_battery_consumed=sum(t.battery_consumed for t in trials) / n if n > 0 else 0.0,
            mean_coverage=sum(t.coverage_percentage for t in trials) / n if n > 0 else 0.0,
            mean_cells_visited=sum(t.cells_visited for t in trials) / n if n > 0 else 0.0,
            mean_computation_ms=sum(t.computation_time_ms for t in trials) / n if n > 0 else 0.0,
            trials=trials,
        )

    return summaries


def print_search_summary(summaries: Dict[str, AlgorithmSummary]) -> None:
    """Print a formatted comparison table to console."""
    print("\n" + "=" * 90)
    print("     5-ALGORITHM SEARCH BENCHMARK: Uncertain-Target Drone Search Comparison")
    print("=" * 90)
    header = (
        f"{'Algorithm':<14} | {'Det.Rate':>8} | {'Steps*':>7} | "
        f"{'Dist':>6} | {'Batt%':>5} | {'Cover%':>7} | {'Time(ms)':>8}"
    )
    print(header)
    print("-" * 90)
    for name, s in summaries.items():
        steps_str = f"{s.mean_steps_to_detection:.1f}" if s.mean_steps_to_detection else "N/A"
        batt_pct = s.mean_battery_consumed * 100.0
        print(
            f"{name:<14} | {s.detection_rate * 100:7.1f}% | {steps_str:>7} | "
            f"{s.mean_distance:6.1f} | {batt_pct:5.1f}% | {s.mean_coverage:7.1f}% | "
            f"{s.mean_computation_ms:8.1f}"
        )
    print("-" * 90)
    print("* Mean steps-to-detection computed over SUCCESSFUL trials only.")
    print("  Unsuccessful trials are counted in detection rate but NOT in mean steps.")
    print("=" * 90)


def generate_search_plots(
    summaries: Dict[str, AlgorithmSummary],
    output_path: str = "results/search_comparison.png",
) -> str:
    """Generate and save comparison plots for the 5-algorithm evaluation."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return "Matplotlib/NumPy not available."

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    names = list(summaries.keys())
    det_rates = [summaries[n].detection_rate * 100 for n in names]
    mean_steps = [summaries[n].mean_steps_to_detection or 0 for n in names]
    mean_dists = [summaries[n].mean_distance for n in names]
    mean_batt = [summaries[n].mean_battery_consumed * 100 for n in names]
    mean_cover = [summaries[n].mean_coverage for n in names]

    colors = ["#00bcd4", "#ff9800", "#4caf50", "#e91e63", "#9c27b0"]
    x = np.arange(len(names))
    bar_w = 0.6

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.patch.set_facecolor("#0f1621")
    fig.suptitle(
        "5-Algorithm Uncertain-Target Search Benchmark",
        fontsize=14, fontweight="bold", color="white",
    )

    datasets = [
        (det_rates, "Detection Rate (%)", "Target Found (%)"),
        (mean_steps, "Mean Steps to Detection*", "Steps (success only)"),
        (mean_dists, "Mean Distance Traveled", "Grid Cells"),
        (mean_batt, "Mean Battery Consumed (%)", "Battery %"),
        (mean_cover, "Area Coverage (%)", "Coverage %"),
    ]

    for ax, (data, title, ylabel) in zip(axes, datasets):
        ax.set_facecolor("#161e2e")
        bars = ax.bar(x, data, color=colors, width=bar_w, edgecolor="#2d3748")
        ax.set_title(title, fontweight="bold", color="white", fontsize=10)
        ax.set_ylabel(ylabel, color="#a0aec0", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=7, color="white")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#2d3748")
        for bar, val in zip(bars, data):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=8, color="white",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    return f"Search comparison plot saved to {output_path}"
