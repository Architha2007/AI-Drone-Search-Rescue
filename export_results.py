"""
export_results.py — Reproducible experiment-results package generator.

Run this script to:
  1. Execute the 5-algorithm benchmark (100 trials per algorithm, matched seeds).
  2. Save all per-trial and summary data to CSV.
  3. Generate presentation-quality PNG graphs.
  4. Write a methodology markdown report.

All CSV values come directly from the benchmark simulation output.
Nothing is manually typed or fabricated.

Usage (from the project root):
  C:\\Users\\archi\\AppData\\Local\\Programs\\Python\\Python312\\python.exe export_results.py

Output directory: results/benchmark_100_trials/
"""
from __future__ import annotations

import csv
import math
import os
import sys
from datetime import datetime, timezone
from typing import Dict

# ── make sure the project root is on sys.path ──────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── config ─────────────────────────────────────────────────────────────────
BENCHMARK_SEED     = 42
NUM_TRIALS         = 100
MAX_STEPS          = 400
OUTPUT_DIR         = os.path.join(_HERE, "results", "benchmark_100_trials")
GRAPH_DPI          = 150
GRAPH_STYLE_BG     = "#0f1621"
GRAPH_PANEL_BG     = "#161e2e"
ALGORITHM_COLORS   = {
    "Best-First":  "#00bcd4",
    "Beam Search": "#ff9800",
    "Hill Climb":  "#4caf50",
    "A* Belief":   "#e91e63",
    "Genetic Alg": "#9c27b0",
}
TIMESTAMP = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


# ===========================================================================
# 1. Run benchmark
# ===========================================================================

def run(verbose: bool = True):
    from evaluation.search_evaluation import run_search_benchmark

    print(f"\n{'='*65}")
    print(" AI Drone SAR — Reproducible Experiment Results Generator")
    print(f"{'='*65}")
    print(f" Trials per algorithm : {NUM_TRIALS}")
    print(f" Random seed          : {BENCHMARK_SEED}")
    print(f" Max steps per trial  : {MAX_STEPS}")
    print(f" Output directory     : {OUTPUT_DIR}")
    print(f"{'='*65}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summaries = run_search_benchmark(
        num_trials=NUM_TRIALS,
        seed=BENCHMARK_SEED,
        max_steps=MAX_STEPS,
        verbose=verbose,
    )
    return summaries


# ===========================================================================
# 2. Save CSVs
# ===========================================================================

def save_summary_csv(summaries: dict, path: str) -> None:
    """Write one row per algorithm with all aggregate metrics."""
    fieldnames = [
        "algorithm",
        "total_trials",
        "successful_trials",
        "failed_trials",
        "detection_rate_pct",            # % — all trials
        "mean_steps_to_detection",       # successful trials only; "N/A" if none
        "mean_distance_all_trials",      # all trials
        "mean_battery_consumed_pct",     # all trials
        "mean_coverage_pct",             # all trials
        "mean_cells_visited_all_trials", # all trials
        "mean_computation_ms_all_trials",# all trials
        "benchmark_seed",
        "num_trials",
        "max_steps_per_trial",
        "generated_at_utc",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, s in summaries.items():
            steps_str = (
                f"{s.mean_steps_to_detection:.4f}"
                if s.mean_steps_to_detection is not None
                else "N/A"
            )
            writer.writerow({
                "algorithm": name,
                "total_trials": s.total_trials,
                "successful_trials": s.successful_trials,
                "failed_trials": s.total_trials - s.successful_trials,
                "detection_rate_pct": round(s.detection_rate * 100, 4),
                "mean_steps_to_detection": steps_str,
                "mean_distance_all_trials": round(s.mean_distance, 4),
                "mean_battery_consumed_pct": round(s.mean_battery_consumed * 100, 4),
                "mean_coverage_pct": round(s.mean_coverage, 4),
                "mean_cells_visited_all_trials": round(s.mean_cells_visited, 4),
                "mean_computation_ms_all_trials": round(s.mean_computation_ms, 4),
                "benchmark_seed": BENCHMARK_SEED,
                "num_trials": NUM_TRIALS,
                "max_steps_per_trial": MAX_STEPS,
                "generated_at_utc": TIMESTAMP,
            })
    print(f"[CSV] Summary written → {path}")


def save_trials_csv(summaries: dict, path: str) -> None:
    """Write one row per individual trial for full reproducibility."""
    fieldnames = [
        "algorithm",
        "trial_seed",
        "detected",
        "steps_to_detection",   # empty string if not detected
        "distance_traveled",
        "battery_consumed_pct",
        "coverage_pct",
        "cells_visited",
        "computation_time_ms",
        "termination_reason",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, s in summaries.items():
            for t in s.trials:
                writer.writerow({
                    "algorithm": t.algorithm,
                    "trial_seed": t.seed,
                    "detected": t.detected,
                    "steps_to_detection": t.steps_to_detection if t.detected else "",
                    "distance_traveled": t.distance_traveled,
                    "battery_consumed_pct": round(t.battery_consumed * 100, 4),
                    "coverage_pct": round(t.coverage_percentage, 4),
                    "cells_visited": t.cells_visited,
                    "computation_time_ms": round(t.computation_time_ms, 4),
                    "termination_reason": t.termination_reason,
                })
    print(f"[CSV] Per-trial data written → {path}")


# ===========================================================================
# 3. Graphs
# ===========================================================================

def _bar_graph(
    ax,
    algorithms: list,
    values: list,
    ylabel: str,
    title: str,
    annotation_fmt: str = "{:.1f}",
    colors: list = None,
    hatch_failed: bool = False,
):
    """Draw a single bar graph panel on `ax`."""
    import numpy as np

    if colors is None:
        colors = list(ALGORITHM_COLORS.values())

    x = range(len(algorithms))
    bars = ax.bar(x, values, color=colors, width=0.6, edgecolor="#2d3748", linewidth=0.8)

    ax.set_facecolor(GRAPH_PANEL_BG)
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [a.replace(" ", "\n") for a in algorithms],
        fontsize=8, color="white",
    )
    ax.set_ylabel(ylabel, color="#a0aec0", fontsize=8)
    ax.set_title(title, fontweight="bold", color="white", fontsize=9, pad=6)
    ax.tick_params(colors="white", labelsize=8)
    ax.spines[:].set_color("#2d3748")
    ax.yaxis.label.set_color("#a0aec0")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            annotation_fmt.format(val),
            ha="center", va="bottom", fontsize=8, color="white", fontweight="bold",
        )

    # Light horizontal grid
    ax.yaxis.grid(True, color="#2d3748", linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)


def generate_graphs(summaries: dict, out_dir: str) -> Dict[str, str]:
    """Generate all 6 graphs. Returns dict of metric_name → file_path."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError as e:
        print(f"[WARN] matplotlib not available: {e}. Skipping graph generation.")
        return {}

    algorithms = list(summaries.keys())
    colors = [ALGORITHM_COLORS[a] for a in algorithms]
    n_trials = NUM_TRIALS

    caption = f"n={n_trials} trials/algorithm, seed={BENCHMARK_SEED}, max_steps={MAX_STEPS}"

    paths: Dict[str, str] = {}

    # ── helper: apply consistent figure chrome ──────────────────────────────
    def _fig_chrome(fig, title: str):
        fig.patch.set_facecolor(GRAPH_STYLE_BG)
        fig.suptitle(
            f"{title}\n({caption})",
            fontsize=11, fontweight="bold", color="white", y=0.98,
        )

    # ── 1. Detection Rate ────────────────────────────────────────────────────
    det_rates = [s.detection_rate * 100 for s in summaries.values()]
    fig, ax = plt.subplots(figsize=(9, 5))
    _fig_chrome(fig, "Survivor Detection Rate (%)")
    _bar_graph(
        ax, algorithms, det_rates,
        ylabel="Detection Rate (%)",
        title="Detection Rate — averaged over ALL trials\n(includes failures)",
        annotation_fmt="{:.1f}%",
        colors=colors,
    )
    ax.set_ylim(0, 105)
    # Add note about failed trials
    ax.text(
        0.01, 0.02,
        "Metric scope: ALL trials (successful + failed). "
        "Failed = battery depleted before detection.",
        transform=ax.transAxes, fontsize=7, color="#a0aec0",
        va="bottom",
    )
    plt.tight_layout()
    p = os.path.join(out_dir, "01_detection_rate.png")
    plt.savefig(p, dpi=GRAPH_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    paths["detection_rate"] = p

    # ── 2. Mean Steps to Detection ───────────────────────────────────────────
    steps_vals = [
        s.mean_steps_to_detection if s.mean_steps_to_detection else 0
        for s in summaries.values()
    ]
    has_none = [s.mean_steps_to_detection is None for s in summaries.values()]

    fig, ax = plt.subplots(figsize=(9, 5))
    _fig_chrome(fig, "Mean Steps to Detection (Steps)")
    _bar_graph(
        ax, algorithms, steps_vals,
        ylabel="Steps (grid moves)",
        title="Mean Steps to Detection — SUCCESSFUL trials only\n(failed trials excluded)",
        annotation_fmt="{:.1f}",
        colors=colors,
    )
    # Annotate N/A bars
    for i, (none_flag, val) in enumerate(zip(has_none, steps_vals)):
        if none_flag:
            ax.text(i, 2, "N/A", ha="center", va="bottom", fontsize=9, color="#ff4444")

    ax.text(
        0.01, 0.02,
        "Metric scope: SUCCESSFUL trials only (detected=True). "
        "Algorithms with 0 successes show N/A.",
        transform=ax.transAxes, fontsize=7, color="#a0aec0", va="bottom",
    )
    plt.tight_layout()
    p = os.path.join(out_dir, "02_mean_steps_to_detection.png")
    plt.savefig(p, dpi=GRAPH_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    paths["mean_steps"] = p

    # ── 3. Mean Distance Travelled ───────────────────────────────────────────
    dists = [s.mean_distance for s in summaries.values()]
    fig, ax = plt.subplots(figsize=(9, 5))
    _fig_chrome(fig, "Mean Distance Travelled (Grid Steps)")
    _bar_graph(
        ax, algorithms, dists,
        ylabel="Distance (grid steps)",
        title="Mean Distance Travelled — averaged over ALL trials",
        annotation_fmt="{:.1f}",
        colors=colors,
    )
    ax.text(
        0.01, 0.02,
        "Metric scope: ALL trials. Distance = number of grid cells traversed "
        "regardless of detection outcome.",
        transform=ax.transAxes, fontsize=7, color="#a0aec0", va="bottom",
    )
    plt.tight_layout()
    p = os.path.join(out_dir, "03_mean_distance.png")
    plt.savefig(p, dpi=GRAPH_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    paths["mean_distance"] = p

    # ── 4. Battery Usage ────────────────────────────────────────────────────
    batt = [s.mean_battery_consumed * 100 for s in summaries.values()]
    fig, ax = plt.subplots(figsize=(9, 5))
    _fig_chrome(fig, "Mean Battery Consumed (%)")
    _bar_graph(
        ax, algorithms, batt,
        ylabel="Battery consumed (%)",
        title="Mean Battery Consumed — averaged over ALL trials",
        annotation_fmt="{:.1f}%",
        colors=colors,
    )
    ax.set_ylim(0, 110)
    ax.axhline(100, color="#ff4444", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(
        len(algorithms) - 0.5, 101, "100% (full battery)",
        color="#ff4444", fontsize=7, ha="right",
    )
    ax.text(
        0.01, 0.02,
        "Metric scope: ALL trials. battery_consumed = 1.0 - battery_remaining at mission end.",
        transform=ax.transAxes, fontsize=7, color="#a0aec0", va="bottom",
    )
    plt.tight_layout()
    p = os.path.join(out_dir, "04_battery_usage.png")
    plt.savefig(p, dpi=GRAPH_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    paths["battery_usage"] = p

    # ── 5. Coverage ──────────────────────────────────────────────────────────
    cover = [s.mean_coverage for s in summaries.values()]
    fig, ax = plt.subplots(figsize=(9, 5))
    _fig_chrome(fig, "Explored-Area Coverage (%)")
    _bar_graph(
        ax, algorithms, cover,
        ylabel="Coverage (%)",
        title="Mean Explored-Area Coverage — averaged over ALL trials",
        annotation_fmt="{:.1f}%",
        colors=colors,
    )
    ax.text(
        0.01, 0.02,
        "Metric scope: ALL trials. Coverage = (explored traversable cells) / "
        "(total reachable cells) × 100.",
        transform=ax.transAxes, fontsize=7, color="#a0aec0", va="bottom",
    )
    plt.tight_layout()
    p = os.path.join(out_dir, "05_coverage.png")
    plt.savefig(p, dpi=GRAPH_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    paths["coverage"] = p

    # ── 6. Runtime ───────────────────────────────────────────────────────────
    runtime = [s.mean_computation_ms for s in summaries.values()]
    fig, ax = plt.subplots(figsize=(9, 5))
    _fig_chrome(fig, "Mean Total Computation Time per Trial (ms)")
    _bar_graph(
        ax, algorithms, runtime,
        ylabel="Computation time (ms/trial)",
        title="Mean Computation Time — averaged over ALL trials\n"
              "(includes policy selection + A* routing + Bayesian update per step)",
        annotation_fmt="{:.1f}",
        colors=colors,
    )
    ax.text(
        0.01, 0.02,
        "Metric scope: ALL trials. Timing wraps full sm.step() per step; "
        "NOT isolated to policy.select_next() alone.",
        transform=ax.transAxes, fontsize=7, color="#a0aec0", va="bottom",
    )
    plt.tight_layout()
    p = os.path.join(out_dir, "06_runtime.png")
    plt.savefig(p, dpi=GRAPH_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    paths["runtime"] = p

    # ── 7. Dashboard: all 6 metrics in one figure ────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    _fig_chrome(fig, "AI Drone SAR — 5-Algorithm Benchmark Dashboard")
    axes = axes.flatten()

    panels = [
        (det_rates,  "Detection Rate (%)",          "{:.1f}%",   "ALL trials"),
        (steps_vals, "Mean Steps to Detection",     "{:.1f}",    "SUCCESSFUL only"),
        (dists,      "Mean Distance (steps)",       "{:.1f}",    "ALL trials"),
        (batt,       "Battery Consumed (%)",        "{:.1f}%",   "ALL trials"),
        (cover,      "Coverage (%)",                "{:.1f}%",   "ALL trials"),
        (runtime,    "Computation Time (ms/trial)", "{:.1f}",    "ALL trials"),
    ]
    for ax, (vals, ylabel, fmt, scope) in zip(axes, panels):
        _bar_graph(ax, algorithms, vals, ylabel=ylabel,
                   title=f"{ylabel}\n[{scope}]", annotation_fmt=fmt, colors=colors)

    plt.tight_layout()
    p = os.path.join(out_dir, "00_dashboard.png")
    plt.savefig(p, dpi=GRAPH_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    paths["dashboard"] = p

    return paths


# ===========================================================================
# 4. Methodology report
# ===========================================================================

REPORT_TEMPLATE = """\
# AI Drone Search & Rescue — Benchmark Results Report

**Generated:** {timestamp_utc}  
**Benchmark seed:** {seed}  
**Trials per algorithm:** {num_trials}  
**Max steps per trial:** {max_steps}  
**Algorithms compared:** {algo_list}

---

## Methodology

### Environment
Each trial uses a fresh `DisasterEnvironment` and `BeliefMap` constructed from a
deterministic random seed. The same set of {num_trials} seeds (derived from the master
benchmark seed `{seed}`) is used for **all five algorithms**, ensuring fair comparison
on identical scenarios.

### Sensor model
- Detection probability: 0.85 (configurable in `config.py`)
- False-positive rate: 0.05
- Sensor range: 2 grid cells (chebyshev)
- Missed detection: reduces cell probability proportionally — never zeros it out

### Battery
- Initial battery: 1.0 (normalised)
- Cost per step: `BATTERY_MOVE_COST_EMPTY` (0.004 by default)
- Safety reserve: 0.10 (10% of initial)
- Mission terminates when `battery_remaining ≤ safety_reserve`

### Termination conditions
A trial ends when any of the following occurs:
1. Target detected (`detection_confirmed = True`) → SUCCESS
2. Battery drops to or below safety reserve → FAILURE
3. No valid move available (all candidates battery-infeasible) → FAILURE
4. Step cap of {max_steps} steps reached → FAILURE (cap not hit in most trials at this budget)

---

## Metric Definitions

| Metric | Scope | Definition |
|--------|-------|-----------|
| **Detection Rate (%)** | ALL trials | `successful_trials / total_trials × 100` |
| **Mean Steps to Detection** | **SUCCESSFUL trials only** | `mean(steps_to_detection)` for trials where `detected = True`. Failed trials are **excluded**. |
| **Mean Distance (steps)** | ALL trials | `mean(distance_traveled)` — grid cells traversed regardless of outcome |
| **Mean Battery Consumed (%)** | ALL trials | `mean(battery_consumed) × 100`; `battery_consumed = initial_battery − battery_remaining` |
| **Mean Coverage (%)** | ALL trials | `mean(explored_cells / reachable_cells × 100)` |
| **Computation Time (ms/trial)** | ALL trials | `mean(total_step_time_ms)`; wraps full `sm.step()` per step, **not** isolated to `policy.select_next()` |

> ⚠️ **Mean Steps to Detection** and **Detection Rate** measure different things.
> An algorithm that terminates early (battery depleted) contributes to the failure count
> but is excluded from the mean-steps calculation. This means an algorithm with a low
> detection rate could appear to have a low mean-steps value if it always finds the target
> quickly when it does succeed.

---

## Results

{results_table}

---

## Observations

{observations}

---

## Statistical Limitations

- With {num_trials} trials per algorithm, the standard error of a proportion near 75% is
  approximately ±4.3 percentage points (95% CI width ≈ 8.5 pp).
- **No formal significance test was performed.** Differences in detection rate of less than
  ~8–10 pp should not be interpreted as meaningful without a chi-squared or Fisher's exact test.
- Detection rate is sensitive to the battery budget constant. Changing `INITIAL_BATTERY`
  or `BATTERY_MOVE_COST_EMPTY` in `config.py` will shift all rates.
- Computation time includes Bayesian belief updates and A* routing common to all policies.
  A* Belief runs additional A* calls per decision step (up to 40) — this is by design.

---

## Files in This Package

| File | Description |
|------|-------------|
| `benchmark_summary.csv` | One row per algorithm, all aggregate metrics |
| `benchmark_trials.csv` | One row per individual trial (raw data) |
| `00_dashboard.png` | All 6 metrics in a single figure |
| `01_detection_rate.png` | Detection rate bar chart |
| `02_mean_steps_to_detection.png` | Mean steps (successful trials only) |
| `03_mean_distance.png` | Mean distance (all trials) |
| `04_battery_usage.png` | Battery consumed (all trials) |
| `05_coverage.png` | Area coverage (all trials) |
| `06_runtime.png` | Computation time (all trials) |
| `methodology_report.md` | This file |

---

## How to Regenerate

```bash
# From the project root:
C:\\Users\\archi\\AppData\\Local\\Programs\\Python\\Python312\\python.exe export_results.py
```

To change the number of trials or seed, edit the constants at the top of `export_results.py`:
```python
BENCHMARK_SEED = 42
NUM_TRIALS     = 100
MAX_STEPS      = 400
```
"""


def build_results_table(summaries: dict) -> str:
    lines = [
        "| Algorithm | Det.Rate | Steps* | Distance | Battery% | Coverage% | Time(ms) |",
        "|-----------|----------|--------|----------|----------|-----------|----------|",
    ]
    for name, s in summaries.items():
        steps = f"{s.mean_steps_to_detection:.1f}" if s.mean_steps_to_detection else "N/A"
        lines.append(
            f"| {name:<11} | {s.detection_rate*100:7.1f}% | {steps:>6} | "
            f"{s.mean_distance:8.1f} | {s.mean_battery_consumed*100:8.1f}% | "
            f"{s.mean_coverage:9.1f}% | {s.mean_computation_ms:8.1f} |"
        )
    lines.append("")
    lines.append("\\* Mean steps computed over **successful trials only**.")
    lines.append("All other metrics averaged over **all trials**.")
    return "\n".join(lines)


def build_observations(summaries: dict) -> str:
    names = list(summaries.keys())
    det = {n: s.detection_rate for n, s in summaries.items()}
    dist = {n: s.mean_distance for n, s in summaries.items()}
    ms = {n: s.mean_computation_ms for n, s in summaries.items()}

    best_det = max(det, key=det.get)
    worst_det = min(det, key=det.get)
    fastest = min(ms, key=ms.get)
    slowest = max(ms, key=ms.get)
    shortest_dist = min(dist, key=dist.get)

    obs = [
        f"- **Highest detection rate:** {best_det} ({det[best_det]*100:.1f}%)",
        f"- **Lowest detection rate:** {worst_det} ({det[worst_det]*100:.1f}%)",
        f"- **Shortest mean distance:** {shortest_dist} ({dist[shortest_dist]:.1f} steps)",
        f"- **Fastest computation:** {fastest} ({ms[fastest]:.1f} ms/trial)",
        f"- **Slowest computation:** {slowest} ({ms[slowest]:.1f} ms/trial) — "
        f"expected due to running up to 40 A\\* path searches per decision step."
        if slowest == "A* Belief"
        else f"- **Slowest computation:** {slowest} ({ms[slowest]:.1f} ms/trial)",
        "",
        "> No algorithm is claimed statistically superior based on these results alone.",
        "> See Statistical Limitations section.",
    ]
    return "\n".join(obs)


def save_methodology_report(summaries: dict, path: str) -> None:
    content = REPORT_TEMPLATE.format(
        timestamp_utc=TIMESTAMP,
        seed=BENCHMARK_SEED,
        num_trials=NUM_TRIALS,
        max_steps=MAX_STEPS,
        algo_list=", ".join(summaries.keys()),
        results_table=build_results_table(summaries),
        observations=build_observations(summaries),
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[MD]  Methodology report written → {path}")


# ===========================================================================
# 5. Verify CSV vs benchmark
# ===========================================================================

def verify_csv(summaries: dict, csv_path: str) -> bool:
    """Read the written CSV back and compare to in-memory summaries."""
    import csv as _csv
    ok = True
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        rows = list(reader)

    csv_by_algo = {r["algorithm"]: r for r in rows}

    for name, s in summaries.items():
        if name not in csv_by_algo:
            print(f"[VERIFY FAIL] '{name}' missing from CSV")
            ok = False
            continue
        row = csv_by_algo[name]

        # Check detection rate
        csv_dr = float(row["detection_rate_pct"])
        expected_dr = round(s.detection_rate * 100, 4)
        if abs(csv_dr - expected_dr) > 1e-3:
            print(f"[VERIFY FAIL] {name}: detection_rate_pct {csv_dr} != {expected_dr}")
            ok = False

        # Check distance
        csv_dist = float(row["mean_distance_all_trials"])
        expected_dist = round(s.mean_distance, 4)
        if abs(csv_dist - expected_dist) > 1e-3:
            print(f"[VERIFY FAIL] {name}: mean_distance {csv_dist} != {expected_dist}")
            ok = False

        # Check battery
        csv_batt = float(row["mean_battery_consumed_pct"])
        expected_batt = round(s.mean_battery_consumed * 100, 4)
        if abs(csv_batt - expected_batt) > 1e-3:
            print(f"[VERIFY FAIL] {name}: mean_battery {csv_batt} != {expected_batt}")
            ok = False

    if ok:
        print(f"[VERIFY] CSV values verified against in-memory benchmark data ✓")
    return ok


# ===========================================================================
# Main
# ===========================================================================

def main():
    # 1. Run benchmark (live — no fabrication)
    summaries = run(verbose=True)

    # 2. Save CSVs
    summary_csv = os.path.join(OUTPUT_DIR, "benchmark_summary.csv")
    trials_csv  = os.path.join(OUTPUT_DIR, "benchmark_trials.csv")
    save_summary_csv(summaries, summary_csv)
    save_trials_csv(summaries, trials_csv)

    # 3. Verify CSV integrity
    verify_csv(summaries, summary_csv)

    # 4. Generate graphs
    print("\n[GRAPHS] Generating presentation-quality graphs...")
    graph_paths = generate_graphs(summaries, OUTPUT_DIR)
    for metric, path in graph_paths.items():
        print(f"  [{metric}] → {path}")

    # 5. Methodology report
    report_path = os.path.join(OUTPUT_DIR, "methodology_report.md")
    save_methodology_report(summaries, report_path)

    # 6. Print summary table
    from evaluation.search_evaluation import print_search_summary
    print()
    print_search_summary(summaries)

    # 7. Final manifest
    print(f"\n{'='*65}")
    print(" Generated files:")
    print(f"{'='*65}")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        size  = os.path.getsize(fpath)
        print(f"  {fname:<40} {size:>10,} bytes")
    print(f"{'='*65}")
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print(f"\nTo regenerate:")
    print(f"  C:\\Users\\archi\\AppData\\Local\\Programs\\Python\\Python312\\python.exe export_results.py")


if __name__ == "__main__":
    main()
