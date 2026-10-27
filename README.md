# AI Drone Search and Rescue Simulation

An autonomous, software-based 2D disaster response simulation built with Python, Pygame, NumPy, OpenCV, and Artificial Intelligence algorithms.

> **Phase 2 — Intelligent Search Under Uncertainty** is now implemented.  
> The simulator answers the research question: *"Where should the drone search next when the target's location is unknown and battery power is limited?"*  
> See §1-A for the new probabilistic search mode.

---

## 1. Project Overview and Objectives

In disaster-affected environments (earthquakes, structural collapses, floods), rapid reconnaissance and survivor localization are critical for emergency rescue teams. This project simulates an autonomous aerial search and rescue (SAR) drone deployed over an urban disaster zone.

### Core Objectives
1. **Autonomous Exploration & Complete Coverage:** Plan systematic sweeping routes across disaster-affected terrain to explore all reachable sectors.
2. **Obstacle Avoidance & Navigation:** Safely navigate around collapsed buildings, rubble, and dynamic debris using heuristic ($A^*$) and baseline (BFS) pathfinding algorithms.
3. **Survivor Localization:** Detect hidden survivors via simulated multi-spectral Field of View (FOV) sensors or optional YOLO computer vision.
4. **Dynamic Replanning:** Reactively calculate new optimal trajectories when encountering newly collapsed debris or discovering survivors needing rescue.
5. **Empirical Benchmarking:** Quantitatively compare pathfinding algorithms in terms of path optimality, nodes explored (search space), and runtime latency.

> [!NOTE]
> **Simulation Disclaimer:** This software is an academic computer simulation and does not interface with or control physical drone hardware. Real-world deployment would require flight controllers (e.g. PX4/ArduPilot), RTK-GPS / visual-inertial odometry, radio telemetry, collision avoidance sonars/LiDAR, and civil aviation safety validation.

---

## 1-A. Phase 2: Intelligent Search Under Uncertainty

### Research Question
> **Where should the drone search next when the target's location is unknown and battery power is limited?**

Phase 2 adds a complete probabilistic search system layered on top of the existing coverage simulation. The two modes are independently accessible at runtime (press **M** to switch).

### Key New Features

| Feature | Description |
|---------|-------------|
| **Bayesian Belief Map** | Maintains `P(target at cell c)` over all traversable cells. Updated via Bayes' Rule on every sensor observation. |
| **Terrain-Based Priors** | Cells near debris and buildings receive higher prior probabilities (survivors are more likely to be trapped near structures). |
| **Noisy Sensor Model** | Simulates stochastic detection with configurable `detection_probability` (default 0.85) and `false_positive_rate` (default 0.05). A missed detection reduces probability but never zeros it out. |
| **Battery System** | Drone has a finite energy budget. Debris-adjacent moves cost more. Mission terminates if battery drops below a safety reserve. |
| **Probability Heatmap** | Semi-transparent overlay (press **H**) shows real-time belief distribution in blue→amber→red gradient. |
| **5 Search Policies** | Best-First, Beam Search, Hill Climbing, A* Belief, Genetic Algorithm — all with genuine algorithmic differences. |
| **Debug Mode** | Press **D** to reveal the hidden target location (never used by any policy; for verification only). |
| **Mission Log** | Scrolling sidebar log showing every observation, plan, and detection event. |
| **5-Algorithm Benchmark** | Automated repeated-trial comparison with detection rate, mean steps, distance, battery, and coverage metrics. |

### 5 Search Policy Algorithms

#### 1. Best-First Search (`algorithms/search_policies/best_first_search.py`)
Greedy destination selection using: `score(c) = P(target at c) / (manhattan_distance(c) + 1)`
Balances detection probability against travel cost. No lookahead — locally optimal.

#### 2. Beam Search (`algorithms/search_policies/beam_search.py`)
Retains the top-k partial route plans (`BEAM_WIDTH=5`) across `BEAM_SEARCH_DEPTH=4` planning steps. Uses discounted cumulative belief scores. Multi-step lookahead avoids local optima that trap greedy search.

#### 3. Hill Climbing with Random Restarts (`algorithms/search_policies/hill_climbing.py`)
Iterative local search: starts at a high-probability cell, moves to the best neighbour within radius `r`. Uses `HILL_CLIMB_MAX_RESTARTS=5` random restarts to escape local optima in the belief landscape.

#### 4. A\* Belief (`algorithms/search_policies/astar_belief.py`)
A\* functions as the **pathfinder** (routes between cells), not the destination selector. This policy documents the distinction clearly: A\* routes, a separate belief-probability heuristic decides *where* to navigate. Score: `P(target at c) / (A* cost estimate + 1)`.

#### 5. Genetic Algorithm (`algorithms/search_policies/genetic_algorithm.py`)
Evolves a population of candidate search routes over `GA_GENERATIONS=30` generations using tournament selection, order-based crossover (OX), and random-replacement mutation. Fitness balances discounted detection probability against travel cost and battery feasibility.

### Running Search Mode

```bash
# Interactive: press M to switch to Search mode
python main.py

# Start directly in Search mode
python main.py --search-mode

# Run 5-algorithm search benchmark (no display required)
python main.py --search-benchmark

# Run original A* vs BFS pathfinding benchmark
python main.py --benchmark

# Run all tests (83 tests)
pytest tests/ -v

# Generate benchmark graphs from saved CSV (fast, ~10 s)
python generate_package.py

# Full benchmark rerun: 100 trials x 5 algorithms (~5 min)
python generate_package.py --rerun
```

> **Note**: Use `C:\Users\archi\AppData\Local\Programs\Python\Python312\python.exe` if `python` defaults to a different version.

> **YOLO detector**: The simulated sensor detector works out of the box. Real YOLOv8 person
> detection requires `pip install ultralytics` (not included in `requirements.txt`). Press **V**
> to toggle vision mode; without ultralytics installed the simulator falls back to the simulated detector automatically.

### Keyboard Controls (Full)

| Key | Action |
|-----|--------|
| **M** | Toggle Coverage ↔ Search mode |
| **SPACE** | Start / Pause mission |
| **S** | Single step |
| **R** | Reset mission |
| **N** | New random environment |
| **A** | Cycle search policy (Search) / Toggle A\*/BFS (Coverage) |
| **H** | Toggle probability heatmap (Search mode) |
| **D** | Toggle debug target reveal (Search mode) |
| **T** | Cycle preset scenarios (Coverage) |
| **V** | Toggle YOLO vision camera |
| **1–4** | Set simulation speed 1×/5×/10×/20× |
| **ESC / Q** | Quit |

### Bayesian Update Mathematics

Let `p_c = P(target at cell c)` before observation.

**Positive detection** at sensor centre `s`:
```
p_c' ∝ p_c × P(detect | target=c) where:
    P(detect | c ∈ sensor_range) = detection_probability   (true positive)
    P(detect | c ∉ sensor_range) = false_positive_rate      (false alarm)
```

**No detection** at sensor centre `s`:
```
p_c' ∝ p_c × P(no-detect | target=c) where:
    P(no-detect | c ∈ sensor_range) = 1 − detection_probability  (missed detection)
    P(no-detect | c ∉ sensor_range) = 1.0                         (uninformative)
```

After each update: `normalize(p)` so `Σ p_c = 1.0`. Probabilities never reach zero.

---

## 2. System Architecture

The project is structured according to modular software engineering principles, separating environment simulation, pathfinding algorithms, survivor detection, user interface, and empirical evaluation:

```
AI_Drone_Search_Rescue/
├── config.py                 # Central simulation parameters, grid sizing, and UI palette
├── main.py                   # Main entry point and game loop coordinator
├── requirements.txt          # Python dependency specifications
├── README.md                 # Project architecture, documentation, and user guide
├── simulation/
│   ├── __init__.py
│   ├── environment.py        # 2D disaster grid, obstacle generation, reachability
│   ├── drone.py              # Virtual drone kinematic model, safety bounds, history
│   └── mission.py            # Central mission controller coordinating search & rescue
├── algorithms/
│   ├── __init__.py
│   ├── astar.py              # A* search planner with Manhattan distance heuristic
│   ├── bfs.py                # Breadth-First Search baseline shortest path planner
│   ├── coverage.py           # Boustrophedon sweeping coverage planner
│   ├── target_selection.py   # Prioritized objective selector (rescue vs. coverage)
│   └── search_policies/      # ── Phase 2: 5 uncertain-target search algorithms ──
│       ├── __init__.py
│       ├── best_first_search.py  # Greedy: belief_prob / travel_cost
│       ├── beam_search.py        # Multi-step lookahead, top-k partial plans
│       ├── hill_climbing.py      # Local search with random restarts
│       ├── astar_belief.py       # A* routing + belief-driven destination
│       └── genetic_algorithm.py  # Evolutionary route optimisation
├── simulation/
│   ├── __init__.py
│   ├── environment.py        # 2D disaster grid, obstacle generation, reachability
│   ├── drone.py              # Virtual drone kinematic model, safety bounds, history
│   ├── mission.py            # Coverage mission controller (A*/BFS)
│   ├── belief_map.py         # ★ Bayesian probability distribution over target locations
│   └── search_mission.py     # ★ Uncertain-target search controller + battery tracking
├── detection/
│   ├── __init__.py
│   ├── simulated_detector.py # Sensor FOV survivor detector (ground truth decoupled)
│   └── yolo_detector.py      # Ultralytics YOLOv8 detector with camera projection
├── ui/
│   ├── __init__.py
│   ├── renderer.py           # Pygame 2D grid renderer + heatmap overlay (Phase 2)
│   ├── controls.py           # Interactive button and toggle UI widgets
│   └── dashboard.py          # Dual-mode dashboard: coverage + search (Phase 2)
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py            # Precision, recall, F1, and mission metrics
│   ├── compare_algorithms.py # A* vs BFS pathfinding benchmark
│   └── search_evaluation.py  # ★ 5-algorithm search benchmark with fair comparisons
├── tests/
│   ├── __init__.py
│   ├── test_astar.py         # Unit tests for A* pathfinding
│   ├── test_bfs.py           # Unit tests for BFS baseline
│   ├── test_environment.py   # Unit tests for obstacles, bounds, and drone movement
│   ├── test_mission.py       # Unit tests for coverage, reset, and detection modes
│   ├── test_belief.py        # ★ Unit tests for BeliefMap (19 tests)
│   └── test_search.py        # ★ Unit tests for SearchMission + 5 policies (35 tests)
├── images/                   # Directory for aerial images and camera feeds
└── results/                  # Generated benchmark graphs and evaluation artifacts
```

---

## 3. Algorithm Details & Theoretical Complexity

### A. A* Search (`algorithms/astar.py`)
- **Purpose:** Identifies the shortest collision-free path between the drone's position and target coordinates, guided toward the goal by an admissible heuristic.
- **Heuristic:** Manhattan distance on a 4-connected grid without diagonal movement:
  $$h(n) = |x_n - x_{\text{goal}}| + |y_n - y_{\text{goal}}|$$
  Because unit step cost is 1 and diagonal movement is disallowed, Manhattan distance is both **admissible** ($h(n) \le h^*(n)$) and **consistent** ($h(n) \le c(n, n') + h(n')$), guaranteeing that $A^*$ finds the optimal shortest path.
- **Time Complexity:** $\mathcal{O}(|E| + |V| \log |V|)$ worst-case on grid graphs (where $|V| \le W \times H$ and $|E| \le 4|V|$). In practice, heuristic pruning reduces explored nodes by approximately 50% compared to uninformed search.
- **Space Complexity:** $\mathcal{O}(|V|)$ to maintain open priority queue, visited closed set, and predecessor mappings.

#### Pseudocode:
```text
function AStar(start, goal, environment):
    open_set = MinPriorityQueue()
    open_set.insert(start, priority=h(start, goal))
    g_score[start] = 0
    f_score[start] = h(start, goal)
    came_from = map()

    while open_set is not empty:
        current = open_set.extract_min()
        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in environment.get_neighbors(current):
            tentative_g = g_score[current] + 1
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + h(neighbor, goal)
                open_set.insert_or_update(neighbor, f_score[neighbor])

    return Failure (Unreachable)
```

---

### B. Breadth-First Search (`algorithms/bfs.py`)
- **Purpose:** Serves as the unweighted graph-theoretic baseline for shortest-path comparison.
- **Mechanism:** Explores nodes layer-by-layer uniformly in all directions using a FIFO queue.
- **Time Complexity:** $\mathcal{O}(|V| + |E|) = \mathcal{O}(W \times H)$.
- **Space Complexity:** $\mathcal{O}(|V|)$.

#### Pseudocode:
```text
function BFS(start, goal, environment):
    queue = Queue()
    queue.push(start)
    visited = {start}
    came_from = map()

    while queue is not empty:
        current = queue.pop()
        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in environment.get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.push(neighbor)

    return Failure (Unreachable)
```

---

### C. Coverage Path Planning (`algorithms/coverage.py`)
- **Strategy:** Boustrophedon (lawnmower) row-by-row sweeping.
- **Mechanism:**
  1. Even rows are traversed left-to-right; odd rows right-to-left.
  2. Obstacle cells and unreachable pockets are filtered out during precomputation.
  3. When an obstacle interrupts the sweep, the planner invokes $A^*$ (or BFS) to route around the blockage to the next unvisited cell.
  4. If an unvisited pocket cannot be reached, the algorithm falls back to the nearest reachable unvisited cell until 100% of reachable traversable cells have been cleared.

---

### D. Prioritized Target Selection (`algorithms/target_selection.py`)
- **Hierarchy:**
  1. **Discovered Survivors Pending Rescue (Highest Priority):** Reroutes immediately to reach survivor locations.
  2. **Systematic Area Coverage:** Targets the next unvisited cell along the sweeping pattern.
  3. **Return to Base:** Once 100% reachable coverage is achieved and all survivors are reached, automatically returns the drone to base $(0, 0)$.

---

## 4. Survivor Detection Modes

### 1. Simulated FOV Detection (`detection/simulated_detector.py`)
- Survivors have ground-truth coordinates hidden in the environment.
- The drone's simulated multi-spectral FOV sensor has a configurable radius (default: $r = 1$, representing a $3 \times 3$ grid footprint).
- Survivors are strictly discovered only when entering the sensor halo.
- Public telemetry counters only display discovered survivors, never leaking undiscovered ground truth.

### 2. Real AI Computer Vision (`detection/yolo_detector.py`)
- Utilizes Ultralytics YOLOv8 and OpenCV to detect human survivors in aerial photographs.
- Filters strictly for the COCO `person` class (Class ID 0).
- Fully decoupled: if YOLO or OpenCV is not installed, or model weights are missing, the core simulation runs smoothly without errors or warnings.
- **No Automatic Downloads:** Model weights are not downloaded silently. Users can place `yolov8n.pt` in the project directory or pass an explicit download flag.

#### Georeferencing / Camera Projection
A 2D bounding box $(u, v)$ in image pixels does not directly give GPS or grid coordinates. The module implements a flat-earth pinhole camera projection:

$$\Delta x = \left( u - \frac{W}{2} \right) \cdot \frac{2 Z \tan\left(\frac{\text{FOV}_h}{2}\right)}{W}$$

$$\Delta y = \left( v - \frac{H}{2} \right) \cdot \frac{2 Z \tan\left(\frac{\text{FOV}_h}{2}\right)}{W}$$

$$X_{\text{world}} = X_{\text{drone}} + \Delta x, \quad Y_{\text{world}} = Y_{\text{drone}} + \Delta y$$

$$\text{Grid Cell} = \left( \left\lfloor \frac{X_{\text{world}}}{\text{cell\_size}} \right\rceil, \left\lfloor \frac{Y_{\text{world}}}{\text{cell\_size}} \right\rceil \right)$$

---

## 5. Installation & Setup (Windows)

### Prerequisites
- Python 3.11 or Python 3.12
- PowerShell or Windows Command Prompt

### Step 1: Open PowerShell and navigate to the project directory
```powershell
cd C:\Users\mejaw\.gemini\antigravity\scratch\AI_Drone_Search_Rescue
```

### Step 2: (Optional) Create and activate a Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Core Dependencies
```powershell
python -m pip install -r requirements.txt
```

### Step 4: (Optional) Install YOLO for Real AI Detection
```powershell
python -m pip install ultralytics
```

---

## 6. How to Run the Simulation

### Launch Interactive Graphical Simulation
```powershell
python main.py
```

### Run with Specific Disaster Scenario, Algorithm, or Random Seed
```powershell
# Launch specific preset disaster scenario
python main.py --scenario "Urban Earthquake"
python main.py --scenario "Flooded District"
python main.py --scenario "Industrial Factory"
python main.py --scenario "Rubble Labyrinth"
python main.py --scenario "Benchmark 10x10"

# Launch with specific algorithm and custom procedural seed
python main.py --algorithm astar --seed 12345
python main.py --algorithm bfs --seed 42
```

### Run Automated Performance Benchmark & Generate Graphs
```powershell
python main.py --benchmark
```
*Generates a 3-subplot comparison plot saved to `results/algorithm_comparison.png`.*

### Run Standalone Image Detection
```powershell
python main.py --detect-image images/sample_disaster_aerial.jpg
```

---

## 7. Controls & User Interface Guide

### Keyboard Shortcuts
| Key | Action |
| :--- | :--- |
| **SPACE** | Start / Pause simulation |
| **S** | Single-step drone navigation (useful for debugging) |
| **R** | Reset mission to base on current map |
| **N** | Generate a new randomized disaster map |
| **T** | Load deterministic fixed benchmark map |
| **A** | Toggle active algorithm between **A\*** and **BFS** |
| **V** | Toggle camera feed overlay (Simulated vs YOLO) |
| **1, 2, 3, 4** | Change simulation speed (1x, 5x, 10x, 20x) |
| **ESC / Q** | Exit application |

### Mouse Controls
- **Left-Click on Grid:** Click any cell on the disaster map to dynamically place or remove rubble debris. The drone will instantly engage **dynamic replanning** if its planned path is obstructed.
- **Left-Click on Sidebar:** Click any button on the interactive dashboard panel.

---

## 8. Map Legend

| Symbol / Color | Meaning |
| :--- | :--- |
| **Green Pad with 'H'** | Base Station / Drone Launch Pad |
| **Neon Cyan Drone** | Autonomous Quadcopter with animated rotors |
| **Cyan Translucent Halo** | Sensor Field of View (FOV) |
| **Slate Gray Block** | Collapsed / Intact Building (Impassable) |
| **Red-Brick Rubble** | Debris Hazard (Impassable) |
| **Dark Blue Fog** | Unexplored / Unsearched Sector |
| **Light Slate Gray** | Explored / Cleared Sector |
| **Pulsing Orange Diamond** | Discovered Survivor (Awaiting Rescue) |
| **Bright Green Cross** | Successfully Reached / Rescued Survivor |
| **Bright Yellow Line** | Current Planned Path |
| **Light Blue Line** | Drone Flight History Trail |

---

## 9. Running Automated Tests

Run the full Pytest test suite:
```powershell
python -m pytest tests -v
```

The test suite validates:
1. Shortest path optimality on unobstructed grids.
2. BFS unweighted baseline correctness.
3. Strict obstacle avoidance.
4. Graceful handling of unreachable targets.
5. Handling start equal to goal.
6. Safe kinematic boundaries: drone never enters obstacles.
7. Drone stays strictly within grid boundaries.
8. Survivor discovery only occurs within sensor range.
9. Coverage percentage calculation accuracy.
10. Complete state reset functionality.
11. Decoupled offline operation without YOLO.
12. Precision, Recall, and F1-score evaluation metrics.
13. Camera-to-world georeferencing projection math.

---

## 10. Performance Benchmarks ($A^*$ vs BFS)

Sample results obtained over 41 randomized disaster grid trials:

| Metric | $A^*$ (Manhattan Heuristic) | BFS (Baseline) | Performance Difference |
| :--- | :--- | :--- | :--- |
| **Average Explored Nodes** | **82.41** | 166.54 | **50.5% fewer nodes explored** |
| **Average Query Latency** | **0.1652 ms** | 0.2061 ms | **19.8% faster execution** |
| **Path Optimality** | 100% Shortest Path | 100% Shortest Path | Equivalent (both optimal) |

The generated comparison visualization is saved in [results/algorithm_comparison.png](file:///C:/Users/mejaw/.gemini/antigravity/scratch/AI_Drone_Search_Rescue/results/algorithm_comparison.png).

---

## 11. Known Limitations & Future Work

1. **2D Planar Assumption:** Current navigation operates in 2D grid space. Future work can extend the state space to 3D voxel grids ($x, y, z$) incorporating flight altitude adjustments to clear variable-height debris.
2. **Deterministic Kinematics:** Drone motion currently models discrete grid transitions. Future iterations can add continuous physical dynamics (acceleration, wind drift, inertia, battery depletion models).
3. **Simulated Fog/Smoke Interference:** Sensor FOV could incorporate probabilistic detection models (e.g., thermal sensor attenuation under heavy smoke or fire).
