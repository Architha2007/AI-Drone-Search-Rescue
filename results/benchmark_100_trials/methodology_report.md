# AI Drone Search & Rescue -- Benchmark Results Report

**Generated:** 2026-10-27T06:00:58Z
**Benchmark seed:** 42
**Trials per algorithm:** 100
**Max steps per trial:** 400
**Algorithms compared:** Best-First, Beam Search, Hill Climb, A* Belief, Genetic Alg

---

## Methodology

### Environment setup
Each trial constructs a fresh `DisasterEnvironment` and `BeliefMap` from a
deterministic seed. The same set of 100 seeds (derived from master seed
`42` via `random.Random`) is used for **all five algorithms**, ensuring every
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
4. Step cap of 400 reached -> FAILURE

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

| Algorithm | Det.Rate | Steps* | Distance | Battery% | Coverage% | Time(ms) |
|-----------|----------|--------|----------|----------|-----------|----------|
| Best-First  |    75.0% |   68.6 |     93.6 |     49.7% |      43.8% |     41.4 |
| Beam Search |    73.0% |   74.5 |     99.8 |     52.9% |      47.0% |    119.8 |
| Hill Climb  |    70.0% |   64.8 |     98.3 |     49.6% |      42.1% |    211.9 |
| A* Belief   |    80.0% |   74.4 |     92.5 |     50.1% |      47.5% |    231.4 |
| Genetic Alg |    81.0% |   75.3 |     93.3 |     49.4% |      46.0% |     69.5 |

\* Mean steps computed over **successful trials only**.
All other metrics averaged over **all trials**.

---

## Observations

- Highest detection rate: Genetic Alg (81.0%)
- Lowest detection rate: Hill Climb (70.0%)
- Shortest mean distance: A* Belief (92.5 steps)
- Fastest computation: Best-First (41.4 ms/trial)
- Slowest computation: A* Belief (231.4 ms/trial)

No algorithm is claimed statistically superior based on these results alone.
See Statistical Limitations section.

---

## Statistical Limitations

- With 100 trials per algorithm, the standard error of a proportion near
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
C:\Users\archi\AppData\Local\Programs\Python\Python312\python.exe generate_package.py

# Full rerun (re-runs all 500 trials, overwrites CSVs):
C:\Users\archi\AppData\Local\Programs\Python\Python312\python.exe generate_package.py --rerun
```

To change trial count or seed, edit constants at top of `generate_package.py`:
```python
BENCHMARK_SEED = 42
NUM_TRIALS     = 100
MAX_STEPS      = 400
```
