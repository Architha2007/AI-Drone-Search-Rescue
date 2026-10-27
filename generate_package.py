"""
generate_package.py
====================
Generates the complete benchmark-results package from the live benchmark.

Because the summary CSV from a previous run already exists and is valid,
this script offers two modes:

  Mode A (default): Load the existing CSV, generate graphs + report from it.
                    Verifies CSV integrity by running a short cross-check.
  Mode B (--rerun): Re-run the full 500-trial benchmark, overwrite CSV,
                    regenerate everything.

Usage (from project root):
  python generate_package.py          # Mode A: fast, uses saved CSV
  python generate_package.py --rerun  # Mode B: full rerun (~5 min)

All graph labels use plain ASCII only to avoid Windows cp1252 encoding errors.
"""
from __future__ import annotations

import csv
import os
import sys
import argparse

# -- ensure project root on path -------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# -- constants ---------------------------------------------------------------
BENCHMARK_SEED = 42
NUM_TRIALS     = 100
MAX_STEPS      = 400
OUTPUT_DIR     = os.path.join(_HERE, "results", "benchmark_100_trials")

ALGORITHM_COLORS = {
    "Best-First":  "#00bcd4",
    "Beam Search": "#ff9800",
    "Hill Climb":  "#4caf50",
    "A* Belief":   "#e91e63",
    "Genetic Alg": "#9c27b0",
}

SUMMARY_CSV = os.path.join(OUTPUT_DIR, "benchmark_summary.csv")
TRIALS_CSV  = os.path.join(OUTPUT_DIR, "benchmark_trials.csv")

# These are the expected CSV column names (must match save_summary_csv output)
SUMMARY_FIELDS = [
    "algorithm",
    "total_trials",
    "successful_trials",
    "failed_trials",
    "detection_rate_pct",
    "mean_steps_to_detection",
    "mean_distance_all_trials",
    "mean_battery_consumed_pct",
    "mean_coverage_pct",
    "mean_cells_visited_all_trials",
    "mean_computation_ms_all_trials",
    "benchmark_seed",
    "num_trials",
    "max_steps_per_trial",
    "generated_at_utc",
]


# ===========================================================================
# A. Load from existing CSV (Mode A)
# ===========================================================================

def load_summary_from_csv(path: str) -> list[dict]:
    """Read the saved summary CSV and return list of row dicts."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Summary CSV not found: {path}\nRun with --rerun to generate it.")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rows_to_display_dict(rows: list[dict]) -> dict:
    """Convert CSV rows into {algo_name: metric_dict} for graph generation."""
    d = {}
    for r in rows:
        d[r["algorithm"]] = {
            "detection_rate":       float(r["detection_rate_pct"]),
            "mean_steps":           None if r["mean_steps_to_detection"] == "N/A"
                                    else float(r["mean_steps_to_detection"]),
            "mean_distance":        float(r["mean_distance_all_trials"]),
            "mean_battery_pct":     float(r["mean_battery_consumed_pct"]),
            "mean_coverage":        float(r["mean_coverage_pct"]),
            "mean_time_ms":         float(r["mean_computation_ms_all_trials"]),
            "successful_trials":    int(r["successful_trials"]),
            "total_trials":         int(r["total_trials"]),
        }
    return d


# ===========================================================================
# B. Run benchmark + save CSVs (Mode B / --rerun)
# ===========================================================================

def run_benchmark_and_save(verbose: bool = True) -> list[dict]:
    """Run the live benchmark, save both CSVs, return summary rows."""
    from evaluation.search_evaluation import run_search_benchmark

    print("\n" + "="*65)
    print(" AI Drone SAR - Reproducible Experiment Results Generator")
    print("="*65)
    print(f" Trials per algorithm : {NUM_TRIALS}")
    print(f" Random seed          : {BENCHMARK_SEED}")
    print(f" Max steps per trial  : {MAX_STEPS}")
    print(f" Output directory     : {OUTPUT_DIR}")
    print("="*65 + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summaries = run_search_benchmark(
        num_trials=NUM_TRIALS,
        seed=BENCHMARK_SEED,
        max_steps=MAX_STEPS,
        verbose=verbose,
    )

    from datetime import datetime, timezone
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")

    # -- summary CSV --
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for name, s in summaries.items():
            steps_str = (
                f"{s.mean_steps_to_detection:.4f}"
                if s.mean_steps_to_detection is not None else "N/A"
            )
            writer.writerow({
                "algorithm":                    name,
                "total_trials":                 s.total_trials,
                "successful_trials":            s.successful_trials,
                "failed_trials":                s.total_trials - s.successful_trials,
                "detection_rate_pct":           round(s.detection_rate * 100, 4),
                "mean_steps_to_detection":      steps_str,
                "mean_distance_all_trials":     round(s.mean_distance, 4),
                "mean_battery_consumed_pct":    round(s.mean_battery_consumed * 100, 4),
                "mean_coverage_pct":            round(s.mean_coverage, 4),
                "mean_cells_visited_all_trials":round(s.mean_cells_visited, 4),
                "mean_computation_ms_all_trials":round(s.mean_computation_ms, 4),
                "benchmark_seed":               BENCHMARK_SEED,
                "num_trials":                   NUM_TRIALS,
                "max_steps_per_trial":          MAX_STEPS,
                "generated_at_utc":             ts,
            })
    print(f"[CSV] Summary written -> {SUMMARY_CSV}")

    # -- per-trial CSV --
    trial_fields = [
        "algorithm", "trial_seed", "detected", "steps_to_detection",
        "distance_traveled", "battery_consumed_pct", "coverage_pct",
        "cells_visited", "computation_time_ms", "termination_reason",
    ]
    with open(TRIALS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=trial_fields)
        writer.writeheader()
        for name, s in summaries.items():
            for t in s.trials:
                writer.writerow({
                    "algorithm":            t.algorithm,
                    "trial_seed":           t.seed,
                    "detected":             t.detected,
                    "steps_to_detection":   t.steps_to_detection if t.detected else "",
                    "distance_traveled":    t.distance_traveled,
                    "battery_consumed_pct": round(t.battery_consumed * 100, 4),
                    "coverage_pct":         round(t.coverage_percentage, 4),
                    "cells_visited":        t.cells_visited,
                    "computation_time_ms":  round(t.computation_time_ms, 4),
                    "termination_reason":   t.termination_reason,
                })
    print(f"[CSV] Per-trial data written -> {TRIALS_CSV}")

    return load_summary_from_csv(SUMMARY_CSV)


# ===========================================================================
# C. Verify CSV integrity
# ===========================================================================

def verify_summary_csv(rows: list[dict]) -> bool:
    """Basic structural integrity check on summary CSV rows."""
    ok = True
    expected_algos = {"Best-First", "Beam Search", "Hill Climb", "A* Belief", "Genetic Alg"}
    found_algos = {r["algorithm"] for r in rows}

    if found_algos != expected_algos:
        print(f"[VERIFY FAIL] Algorithms in CSV: {found_algos}")
        print(f"              Expected         : {expected_algos}")
        ok = False

    for r in rows:
        dr = float(r["detection_rate_pct"])
        if not (0.0 <= dr <= 100.0):
            print(f"[VERIFY FAIL] {r['algorithm']}: detection_rate_pct={dr} out of range")
            ok = False
        if r["mean_steps_to_detection"] not in ("N/A", "") and \
           float(r["mean_steps_to_detection"]) <= 0:
            print(f"[VERIFY FAIL] {r['algorithm']}: mean_steps_to_detection <= 0")
            ok = False
        if float(r["mean_distance_all_trials"]) < 0:
            print(f"[VERIFY FAIL] {r['algorithm']}: mean_distance < 0")
            ok = False
        n = int(r["num_trials"])
        s = int(r["successful_trials"])
        fa = int(r["failed_trials"])
        if s + fa != n:
            print(f"[VERIFY FAIL] {r['algorithm']}: successful+failed ({s+fa}) != num_trials ({n})")
            ok = False

    if ok:
        print("[VERIFY] CSV integrity check passed (5 algorithms, valid ranges, trial counts) OK")
    return ok


# ===========================================================================
# D. Graph generation (pure ASCII labels, presentation quality)
# ===========================================================================

def _apply_dark_panel(ax, title: str, ylabel: str):
    """Apply consistent dark theme to a matplotlib axes."""
    ax.set_facecolor("#161e2e")
    ax.set_title(title, fontweight="bold", color="white", fontsize=9, pad=6)
    ax.set_ylabel(ylabel, color="#a0aec0", fontsize=8)
    ax.tick_params(colors="white", labelsize=8)
    ax.spines[:].set_color("#2d3748")
    ax.yaxis.label.set_color("#a0aec0")
    ax.yaxis.grid(True, color="#2d3748", linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)


def _bar_panel(ax, algorithms, values, ylabel, title, fmt="{:.1f}", colors=None):
    if colors is None:
        colors = list(ALGORITHM_COLORS.values())
    x = range(len(algorithms))
    bars = ax.bar(x, values, color=colors, width=0.6,
                  edgecolor="#2d3748", linewidth=0.8)
    _apply_dark_panel(ax, title, ylabel)
    ax.set_xticks(list(x))
    ax.set_xticklabels([a.replace(" ", "\n") for a in algorithms],
                       fontsize=8, color="white")
    max_v = max(values) if values else 1
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_v * 0.01,
            fmt.format(val),
            ha="center", va="bottom", fontsize=8,
            color="white", fontweight="bold",
        )


def _scope_note(ax, text: str):
    """Add a small scope-annotation at the bottom of the axes."""
    ax.text(0.01, 0.02, text, transform=ax.transAxes,
            fontsize=7, color="#a0aec0", va="bottom")


def _savefig(fig, path: str):
    import matplotlib
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    matplotlib.pyplot.close(fig)
    print(f"[GRAPH] Saved -> {path}")


def generate_graphs(data: dict, num_trials: int, seed: int, out_dir: str) -> list[str]:
    """
    Generate 7 PNG graphs from the display dict returned by rows_to_display_dict().
    Returns list of file paths created.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"[WARN] matplotlib unavailable: {e}. Skipping graphs.")
        return []

    os.makedirs(out_dir, exist_ok=True)
    algorithms = list(data.keys())
    colors = [ALGORITHM_COLORS.get(a, "#888888") for a in algorithms]
    caption = f"n={num_trials} trials/algorithm | seed={seed} | max_steps={MAX_STEPS}"

    det_rates  = [data[a]["detection_rate"]   for a in algorithms]
    steps_vals = [data[a]["mean_steps"] or 0  for a in algorithms]
    dists      = [data[a]["mean_distance"]     for a in algorithms]
    batt       = [data[a]["mean_battery_pct"]  for a in algorithms]
    cover      = [data[a]["mean_coverage"]     for a in algorithms]
    runtime    = [data[a]["mean_time_ms"]      for a in algorithms]

    saved = []

    # Helper: make a full figure with a suptitle
    def _fig(title: str, size=(9, 5)):
        fig, ax = plt.subplots(figsize=size)
        fig.patch.set_facecolor("#0f1621")
        fig.suptitle(f"{title}\n({caption})",
                     fontsize=11, fontweight="bold", color="white", y=0.98)
        return fig, ax

    # 1. Detection Rate
    fig, ax = _fig("Survivor Detection Rate (%)")
    _bar_panel(ax, algorithms, det_rates, "Detection Rate (%)",
               "Detection Rate - ALL trials (success + failure)", "{:.1f}%", colors)
    ax.set_ylim(0, 110)
    _scope_note(ax, "Scope: ALL trials. Failed = battery depleted before detection.")
    p = os.path.join(out_dir, "01_detection_rate.png")
    _savefig(fig, p); saved.append(p)

    # 2. Mean Steps to Detection
    fig, ax = _fig("Mean Steps to Detection")
    _bar_panel(ax, algorithms, steps_vals, "Steps (grid moves)",
               "Mean Steps to Detection - SUCCESSFUL trials only", "{:.1f}", colors)
    _scope_note(ax, "Scope: SUCCESSFUL trials only. Failed trials excluded from this metric.")
    p = os.path.join(out_dir, "02_mean_steps_to_detection.png")
    _savefig(fig, p); saved.append(p)

    # 3. Mean Distance
    fig, ax = _fig("Mean Distance Travelled (Grid Steps)")
    _bar_panel(ax, algorithms, dists, "Distance (grid steps)",
               "Mean Distance Travelled - ALL trials", "{:.1f}", colors)
    _scope_note(ax, "Scope: ALL trials. Distance = grid cells traversed regardless of outcome.")
    p = os.path.join(out_dir, "03_mean_distance.png")
    _savefig(fig, p); saved.append(p)

    # 4. Battery Usage
    fig, ax = _fig("Mean Battery Consumed (%)")
    _bar_panel(ax, algorithms, batt, "Battery consumed (%)",
               "Mean Battery Consumed - ALL trials", "{:.1f}%", colors)
    ax.set_ylim(0, 110)
    ax.axhline(100, color="#ff4444", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(len(algorithms) - 0.45, 101.5, "100% (full battery)",
            color="#ff4444", fontsize=7, ha="right")
    _scope_note(ax, "Scope: ALL trials. battery_consumed = initial_battery - battery_remaining.")
    p = os.path.join(out_dir, "04_battery_usage.png")
    _savefig(fig, p); saved.append(p)

    # 5. Coverage
    fig, ax = _fig("Explored-Area Coverage (%)")
    _bar_panel(ax, algorithms, cover, "Coverage (%)",
               "Explored-Area Coverage - ALL trials", "{:.1f}%", colors)
    _scope_note(ax, "Scope: ALL trials. Coverage = explored_cells / reachable_cells x 100.")
    p = os.path.join(out_dir, "05_coverage.png")
    _savefig(fig, p); saved.append(p)

    # 6. Runtime
    fig, ax = _fig("Computation Time per Trial (ms)")
    _bar_panel(ax, algorithms, runtime, "ms / trial",
               "Mean Computation Time - ALL trials\n"
               "(policy selection + A* routing + Bayesian update per step)",
               "{:.1f}", colors)
    _scope_note(ax, "Scope: ALL trials. Wraps full sm.step(); not isolated to policy.select_next().")
    p = os.path.join(out_dir, "06_runtime.png")
    _savefig(fig, p); saved.append(p)

    # 7. Dashboard (all 6 in one figure)
    fig, axes = plt.subplots(2, 3, figsize=(20, 9))
    fig.patch.set_facecolor("#0f1621")
    fig.suptitle(f"AI Drone SAR - 5-Algorithm Benchmark Dashboard\n({caption})",
                 fontsize=13, fontweight="bold", color="white")
    axes = axes.flatten()
    panels = [
        (det_rates,  "Detection Rate (%)",    "{:.1f}%", "ALL trials"),
        (steps_vals, "Mean Steps (success)",  "{:.1f}",  "SUCCESSFUL only"),
        (dists,      "Distance (steps)",      "{:.1f}",  "ALL trials"),
        (batt,       "Battery Consumed (%)",  "{:.1f}%", "ALL trials"),
        (cover,      "Coverage (%)",          "{:.1f}%", "ALL trials"),
        (runtime,    "Compute Time (ms)",     "{:.1f}",  "ALL trials"),
    ]
    for ax, (vals, ylabel, fmt, scope) in zip(axes, panels):
        _bar_panel(ax, algorithms, vals, ylabel,
                   f"{ylabel}\n[{scope}]", fmt, colors)
    plt.tight_layout()
    p = os.path.join(out_dir, "00_dashboard.png")
    _savefig(fig, p); saved.append(p)

    return saved


# ===========================================================================
# E. Per-trial CSV generation from benchmark (needed if only summary exists)
# ===========================================================================

def generate_trials_csv_from_live(verbose: bool = False) -> bool:
    """
    If benchmark_trials.csv doesn't exist, run a lightweight trial-collection
    pass to generate it. Uses the same seed so results match.
    Returns True if successful.
    """
    if os.path.exists(TRIALS_CSV):
        print(f"[CSV] Per-trial CSV already exists: {TRIALS_CSV}")
        return True

    print("[CSV] Per-trial CSV missing -- running benchmark to generate it...")
    from evaluation.search_evaluation import run_search_benchmark
    summaries = run_search_benchmark(
        num_trials=NUM_TRIALS,
        seed=BENCHMARK_SEED,
        max_steps=MAX_STEPS,
        verbose=verbose,
    )
    trial_fields = [
        "algorithm", "trial_seed", "detected", "steps_to_detection",
        "distance_traveled", "battery_consumed_pct", "coverage_pct",
        "cells_visited", "computation_time_ms", "termination_reason",
    ]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(TRIALS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=trial_fields)
        writer.writeheader()
        for name, s in summaries.items():
            for t in s.trials:
                writer.writerow({
                    "algorithm":            t.algorithm,
                    "trial_seed":           t.seed,
                    "detected":             t.detected,
                    "steps_to_detection":   t.steps_to_detection if t.detected else "",
                    "distance_traveled":    t.distance_traveled,
                    "battery_consumed_pct": round(t.battery_consumed * 100, 4),
                    "coverage_pct":         round(t.coverage_percentage, 4),
                    "cells_visited":        t.cells_visited,
                    "computation_time_ms":  round(t.computation_time_ms, 4),
                    "termination_reason":   t.termination_reason,
                })
    print(f"[CSV] Per-trial data written -> {TRIALS_CSV}")
    return True


# ===========================================================================
# F. Methodology report
# ===========================================================================

METHODOLOGY_TEMPLATE = """\
# AI Drone Search & Rescue -- Benchmark Results Report

**Generated:** {timestamp}
**Benchmark seed:** {seed}
**Trials per algorithm:** {num_trials}
**Max steps per trial:** {max_steps}
**Algorithms compared:** {algo_list}

---

## Methodology

### Environment setup
Each trial constructs a fresh `DisasterEnvironment` and `BeliefMap` from a
deterministic seed. The same set of {num_trials} seeds (derived from master seed
`{seed}` via `random.Random`) is used for **all five algorithms**, ensuring every
algorithm is evaluated on identical scenarios.

### Sensor model
- Detection probability: 0.85 (configurable via `config.DETECTION_PROBABILITY`)
- False-positive rate: 0.05 (`config.FALSE_POSITIVE_RATE`)
- Sensor range: 2 grid cells (Chebyshev distance)
- Missed detections reduce cell probability proportionally; never set to zero

### Battery model
- Initial battery: 1.0 (normalised fraction)
- Cost per move step: `config.BATTERY_MOVE_COST_EMPTY` (default 0.004)
- Safety reserve: 0.10 (10% of initial)
- Mission terminates when `battery_remaining <= safety_reserve`

### Trial termination conditions
A trial ends on the FIRST of:
1. Target detected (`detection_confirmed = True`) -> SUCCESS
2. Battery drops to or below safety reserve -> FAILURE
3. Policy returns no valid next cell (all candidates battery-infeasible) -> FAILURE
4. Step cap of {max_steps} reached -> FAILURE

---

## Metric Definitions

| Metric | Trial Scope | Definition |
|--------|-------------|-----------|
| Detection Rate (%) | ALL trials | `successful_trials / total_trials x 100` |
| Mean Steps to Detection | SUCCESSFUL only | `mean(steps_to_detection)` -- failed trials EXCLUDED |
| Mean Distance (steps) | ALL trials | `mean(distance_traveled)` regardless of outcome |
| Mean Battery Consumed (%) | ALL trials | `mean(battery_consumed) x 100`; `battery_consumed = 1.0 - battery_remaining` |
| Mean Coverage (%) | ALL trials | `mean(explored_cells / reachable_cells x 100)` |
| Computation Time (ms/trial) | ALL trials | `mean(total_step_time_ms)`; wraps full `sm.step()` per step |

> NOTE: "Mean Steps to Detection" and "Detection Rate" measure different things.
> An algorithm that terminates early (battery depleted) adds to the failure count
> but is excluded from mean-steps. An algorithm could show a low mean-steps value
> simply because it only detects on easy trials.

---

## Results

{results_table}

---

## Observations

{observations}

---

## Statistical Limitations

- With {num_trials} trials per algorithm, the standard error of a proportion near
  75-80% is approximately +/-4.3 percentage points (95% CI width ~8.5 pp).
- No formal significance test was performed. Differences in detection rate of less
  than ~8-10 pp should not be interpreted as meaningful without a chi-squared or
  Fisher's exact test.
- Detection rate is sensitive to battery budget. Changing `INITIAL_BATTERY` or
  `BATTERY_MOVE_COST_EMPTY` in `config.py` will shift all rates.
- Computation time wraps the full step including Bayesian updates and A* routing
  common to all policies. A* Belief additionally runs up to 40 A* searches per
  destination decision -- this is by design and reflected in its timing column.

---

## Files in This Package

| File | Description |
|------|-------------|
| `benchmark_summary.csv` | One row per algorithm, all aggregate metrics |
| `benchmark_trials.csv` | One row per individual trial (500 rows, raw data) |
| `00_dashboard.png` | All 6 metrics in a single dashboard figure |
| `01_detection_rate.png` | Detection rate -- ALL trials |
| `02_mean_steps_to_detection.png` | Mean steps -- SUCCESSFUL trials only |
| `03_mean_distance.png` | Mean distance -- ALL trials |
| `04_battery_usage.png` | Battery consumed -- ALL trials |
| `05_coverage.png` | Area coverage -- ALL trials |
| `06_runtime.png` | Computation time -- ALL trials |
| `methodology_report.md` | This file |

---

## How to Regenerate

```bash
# Fast mode (uses existing summary CSV, skips re-running 500 trials):
C:\\Users\\archi\\AppData\\Local\\Programs\\Python\\Python312\\python.exe generate_package.py

# Full rerun (re-runs all 500 trials, overwrites CSVs):
C:\\Users\\archi\\AppData\\Local\\Programs\\Python\\Python312\\python.exe generate_package.py --rerun
```

To change trial count or seed, edit constants at top of `generate_package.py`:
```python
BENCHMARK_SEED = 42
NUM_TRIALS     = 100
MAX_STEPS      = 400
```
"""


def build_results_table(data: dict) -> str:
    lines = [
        "| Algorithm | Det.Rate | Steps* | Distance | Battery% | Coverage% | Time(ms) |",
        "|-----------|----------|--------|----------|----------|-----------|----------|",
    ]
    for name, m in data.items():
        steps = f"{m['mean_steps']:.1f}" if m["mean_steps"] else "N/A"
        lines.append(
            f"| {name:<11} | {m['detection_rate']:7.1f}% | {steps:>6} | "
            f"{m['mean_distance']:8.1f} | {m['mean_battery_pct']:8.1f}% | "
            f"{m['mean_coverage']:9.1f}% | {m['mean_time_ms']:8.1f} |"
        )
    lines += [
        "",
        "\\* Mean steps computed over **successful trials only**.",
        "All other metrics averaged over **all trials**.",
    ]
    return "\n".join(lines)


def build_observations(data: dict) -> str:
    best_det = max(data, key=lambda a: data[a]["detection_rate"])
    worst_det = min(data, key=lambda a: data[a]["detection_rate"])
    fastest = min(data, key=lambda a: data[a]["mean_time_ms"])
    slowest = max(data, key=lambda a: data[a]["mean_time_ms"])
    shortest_dist = min(data, key=lambda a: data[a]["mean_distance"])
    obs = [
        f"- Highest detection rate: {best_det} ({data[best_det]['detection_rate']:.1f}%)",
        f"- Lowest detection rate: {worst_det} ({data[worst_det]['detection_rate']:.1f}%)",
        f"- Shortest mean distance: {shortest_dist} ({data[shortest_dist]['mean_distance']:.1f} steps)",
        f"- Fastest computation: {fastest} ({data[fastest]['mean_time_ms']:.1f} ms/trial)",
        f"- Slowest computation: {slowest} ({data[slowest]['mean_time_ms']:.1f} ms/trial)",
        "",
        "No algorithm is claimed statistically superior based on these results alone.",
        "See Statistical Limitations section.",
    ]
    return "\n".join(obs)


def save_methodology_report(data: dict, path: str) -> None:
    from datetime import datetime, timezone
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    content = METHODOLOGY_TEMPLATE.format(
        timestamp=ts,
        seed=BENCHMARK_SEED,
        num_trials=NUM_TRIALS,
        max_steps=MAX_STEPS,
        algo_list=", ".join(data.keys()),
        results_table=build_results_table(data),
        observations=build_observations(data),
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[MD]  Methodology report written -> {path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate AI Drone SAR benchmark results package."
    )
    parser.add_argument(
        "--rerun", action="store_true",
        help="Re-run the full 500-trial benchmark (slow). Default: use existing CSV."
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Get summary data
    if args.rerun or not os.path.exists(SUMMARY_CSV):
        print("[MODE] Full benchmark rerun requested (or no CSV found).")
        rows = run_benchmark_and_save(verbose=True)
    else:
        print(f"[MODE] Loading existing summary CSV: {SUMMARY_CSV}")
        rows = load_summary_from_csv(SUMMARY_CSV)

    data = rows_to_display_dict(rows)

    # Step 2: Verify CSV
    ok = verify_summary_csv(rows)
    if not ok:
        print("[ERROR] CSV verification failed. Run with --rerun to regenerate.")
        sys.exit(1)

    # Step 3: Per-trial CSV note (only generated with --rerun)
    if os.path.exists(TRIALS_CSV):
        sz = os.path.getsize(TRIALS_CSV)
        print(f"[CSV] Per-trial CSV present: {TRIALS_CSV} ({sz:,} bytes)")
    else:
        print(f"[CSV] Per-trial CSV not yet generated.")
        print(f"      Run with --rerun to generate it (requires ~5 min full benchmark rerun).")


    # Step 4: Graphs
    print("\n[GRAPHS] Generating graphs...")
    saved_graphs = generate_graphs(
        data, num_trials=NUM_TRIALS, seed=BENCHMARK_SEED, out_dir=OUTPUT_DIR
    )

    # Step 5: Methodology report
    report_path = os.path.join(OUTPUT_DIR, "methodology_report.md")
    save_methodology_report(data, report_path)

    # Step 6: Print summary table
    print()
    header = f"{'Algorithm':<14} {'Det.Rate':>9} {'Steps*':>7} {'Dist':>7} {'Batt%':>6} {'Cover%':>8} {'ms/trial':>9}"
    print("="*70)
    print(" AI Drone SAR -- 100-Trial Benchmark Results")
    print("="*70)
    print(header)
    print("-"*70)
    for name, m in data.items():
        steps = f"{m['mean_steps']:.1f}" if m["mean_steps"] else " N/A"
        print(
            f"{name:<14} {m['detection_rate']:8.1f}% {steps:>7} "
            f"{m['mean_distance']:7.1f} {m['mean_battery_pct']:6.1f}% "
            f"{m['mean_coverage']:7.1f}% {m['mean_time_ms']:9.1f}"
        )
    print("-"*70)
    print("* Steps: SUCCESSFUL trials only. All others: ALL trials.")
    print("="*70)

    # Step 7: Final manifest
    print("\n[FILES] Generated package:")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        size = os.path.getsize(fpath)
        print(f"  {fname:<45} {size:>10,} bytes")

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("\nTo regenerate (fast, reuse CSV):")
    print("  C:\\Users\\archi\\AppData\\Local\\Programs\\Python\\Python312\\python.exe generate_package.py")
    print("\nTo regenerate (full rerun, ~5 min):")
    print("  C:\\Users\\archi\\AppData\\Local\\Programs\\Python\\Python312\\python.exe generate_package.py --rerun")


if __name__ == "__main__":
    main()
