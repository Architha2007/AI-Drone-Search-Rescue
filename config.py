"""Configuration constants for AI Drone Search and Rescue simulation."""
from dataclasses import dataclass
from typing import Tuple
import os

# ---------------------------------------------------------
# Grid & Map Configuration
# ---------------------------------------------------------
GRID_ROWS: int = 20
GRID_COLS: int = 20
CELL_SIZE: int = 34  # Pixels per grid cell (20 * 34 = 680 px)

# ---------------------------------------------------------
# Window & Display Dimensions
# ---------------------------------------------------------
MAP_WIDTH: int = GRID_COLS * CELL_SIZE  # 680 px
SIDEBAR_WIDTH: int = 380                # 380 px
WINDOW_WIDTH: int = MAP_WIDTH + SIDEBAR_WIDTH  # 1060 px
WINDOW_HEIGHT: int = GRID_ROWS * CELL_SIZE     # 680 px

FPS: int = 60
DEFAULT_SPEED_MULTIPLIER: int = 5  # Drone steps per second default

# ---------------------------------------------------------
# Simulation Settings
# ---------------------------------------------------------
DEFAULT_RANDOM_SEED: int = 42
DEFAULT_OBSTACLE_DENSITY: float = 0.20  # 20% obstacles
DEFAULT_SURVIVOR_COUNT: int = 6
SENSOR_RANGE: int = 1  # Radius in grid cells: 1 means 3x3 surrounding window

BASE_COORDINATES: Tuple[int, int] = (0, 0)

# ---------------------------------------------------------
# Color Palette (Dark Theme / High Contrast Modern UI)
# ---------------------------------------------------------
COLOR_BG: Tuple[int, int, int] = (18, 22, 30)             # Deep charcoal navy
COLOR_GRID_LINE: Tuple[int, int, int] = (32, 38, 50)      # Subtle grid line
COLOR_UNEXPLORED: Tuple[int, int, int] = (25, 30, 42)     # Fog of war / unvisited
COLOR_EXPLORED: Tuple[int, int, int] = (45, 54, 72)       # Visited / cleared
COLOR_BASE: Tuple[int, int, int] = (46, 204, 113)         # Emerald base pad
COLOR_OBSTACLE_BUILDING: Tuple[int, int, int] = (80, 88, 108)  # Slate building block
COLOR_OBSTACLE_DEBRIS: Tuple[int, int, int] = (192, 57, 43)    # Broken debris red-brick
COLOR_DRONE: Tuple[int, int, int] = (0, 212, 255)         # Neon cyan
COLOR_DRONE_GLOW: Tuple[int, int, int] = (0, 212, 255)    # Translucent halo
COLOR_PLANNED_PATH: Tuple[int, int, int] = (241, 196, 15) # Warm gold path
COLOR_TRAIL: Tuple[int, int, int] = (52, 152, 219)        # Flight path trail
COLOR_SURVIVOR_DISCOVERED: Tuple[int, int, int] = (230, 126, 34)  # Vibrant amber/orange
COLOR_SURVIVOR_RESCUED: Tuple[int, int, int] = (46, 204, 113)     # Confirmed rescued green
COLOR_SURVIVOR_GROUND_TRUTH: Tuple[int, int, int] = (255, 105, 180) # Pink (ground truth only)

# UI Elements
COLOR_PANEL_BG: Tuple[int, int, int] = (26, 32, 44)       # Sidebar panel
COLOR_PANEL_BORDER: Tuple[int, int, int] = (45, 55, 72)
COLOR_TEXT_PRIMARY: Tuple[int, int, int] = (240, 243, 246)
COLOR_TEXT_MUTED: Tuple[int, int, int] = (160, 174, 192)
COLOR_TEXT_ACCENT: Tuple[int, int, int] = (0, 212, 255)
COLOR_BUTTON_BG: Tuple[int, int, int] = (45, 55, 72)
COLOR_BUTTON_HOVER: Tuple[int, int, int] = (74, 85, 104)
COLOR_BUTTON_ACTIVE: Tuple[int, int, int] = (0, 168, 204)
COLOR_SUCCESS: Tuple[int, int, int] = (46, 204, 113)
COLOR_WARNING: Tuple[int, int, int] = (241, 196, 15)
COLOR_DANGER: Tuple[int, int, int] = (231, 76, 60)

# ---------------------------------------------------------
# Battery / Energy System
# ---------------------------------------------------------
INITIAL_BATTERY: float = 1.0          # Full charge = 1.0
BATTERY_MOVE_COST_EMPTY: float = 0.004  # Cost per step on open terrain
BATTERY_MOVE_COST_DEBRIS: float = 0.006  # Higher cost navigating near debris
BATTERY_SAFETY_RESERVE: float = 0.10   # Reserve 10% — mission halts below this

# Battery meter colors
COLOR_BATTERY_HIGH: Tuple[int, int, int] = (46, 204, 113)   # Green
COLOR_BATTERY_MED: Tuple[int, int, int] = (241, 196, 15)    # Yellow
COLOR_BATTERY_LOW: Tuple[int, int, int] = (231, 76, 60)     # Red

# ---------------------------------------------------------
# Probabilistic Search (Uncertain-Target) Configuration
# ---------------------------------------------------------
DETECTION_PROBABILITY: float = 0.85   # P(detect | target in range and sensor fires)
FALSE_POSITIVE_RATE: float = 0.05     # P(false alarm | no target)

# Terrain prior likelihood weights (unnormalized — will be normalized)
TERRAIN_PRIOR_EMPTY: float = 1.0
TERRAIN_PRIOR_NEAR_DEBRIS: float = 1.8   # Survivors likely trapped near debris
TERRAIN_PRIOR_NEAR_BUILDING: float = 1.4  # Likely sheltering near buildings

# Heatmap overlay colors (low-prob -> high-prob)
COLOR_PROB_ZERO: Tuple[int, int, int] = (18, 22, 30)       # Same as background
COLOR_PROB_LOW: Tuple[int, int, int] = (20, 60, 80)        # Dark teal
COLOR_PROB_MED: Tuple[int, int, int] = (180, 100, 0)       # Amber
COLOR_PROB_HIGH: Tuple[int, int, int] = (255, 60, 0)       # Hot red-orange
HEATMAP_ALPHA: int = 160                                    # Overlay transparency

# ---------------------------------------------------------
# Search Policy Algorithm Configuration
# ---------------------------------------------------------
BEAM_WIDTH: int = 5                  # Retain top-k candidate destinations
BEAM_SEARCH_DEPTH: int = 4          # Planning horizon for beam expansion
GA_POPULATION_SIZE: int = 20        # Genetic algorithm population
GA_GENERATIONS: int = 30            # Number of GA generations
GA_MUTATION_RATE: float = 0.2       # Probability of route mutation
GA_ELITE_SIZE: int = 4              # Top individuals carried over
HILL_CLIMB_MAX_RESTARTS: int = 5    # Hill climbing random restarts
HILL_CLIMB_NEIGHBOR_RADIUS: int = 3 # Neighbourhood size for hill climbing

# ---------------------------------------------------------
# Mission Log
# ---------------------------------------------------------
MISSION_LOG_MAX_ENTRIES: int = 200  # Maximum entries in the searchable mission log
MISSION_LOG_VISIBLE: int = 8        # Lines shown in the UI log panel

# ---------------------------------------------------------
# Test / Benchmark Fixed Map Layouts
# ---------------------------------------------------------
# 1. Deterministic 10x10 benchmark map for reproducible algorithm comparisons
FIXED_BENCHMARK_10x10 = [
    "....#.....",
    ".##.#.###.",
    "..#.#...#.",
    "#.#.###.#.",
    "#.#...#.#.",
    "#.###.#.#.",
    "#...#.#.#.",
    "###.###.#.",
    "S.......G.",
    "..........",
]

# 2. Urban Earthquake Collapse Zone (20x20)
MAP_URBAN_EARTHQUAKE = [
    "S...................",
    ".####.####..####.##.",
    ".#..#.#..#..#..#.#..",
    ".#..#.####..####.##.",
    "....D..........D....",
    ".####.####..####.##.",
    ".#..G.#..#..#..G.#..",
    ".####.####..####.##.",
    "....D..........D....",
    "....................",
    ".#####.#####..#####.",
    ".#...#.#...#..#...#.",
    ".#...#.#.G.#..#...#.",
    ".#####.#####..#####.",
    "....D..........D....",
    ".####.####..####.##.",
    ".#..#.#..#..#..#.#..",
    ".#G.#.####..#.G#.##.",
    "....D..........D....",
    "..............G.....",
]

# 3. Flooded District & Canal Breach (20x20)
MAP_FLOODED_DISTRICT = [
    "S.........##........",
    "..........##........",
    ".###...G..##...###..",
    ".###......##...###..",
    "..........##........",
    ".....###........G...",
    ".....###............",
    "################.###",
    "....................",
    "...G.......D........",
    "...........D........",
    "###.################",
    "..........##........",
    "..........##...###..",
    "..###.....##...###..",
    "..###.G...##........",
    "....................",
    "################.###",
    "........G...........",
    "....................",
]

# 4. Industrial Factory & Hazardous Storage (20x20)
MAP_INDUSTRIAL_FACTORY = [
    "S...................",
    ".######...######....",
    ".#....#...#....#....",
    ".#..G.#...#.G..#....",
    ".#....#...#....#....",
    ".##.###...###.##....",
    "....................",
    "....DDDD....DDDD....",
    "....................",
    ".########...########",
    ".#......#...#......#",
    ".#.####.#...#.####.#",
    ".#.#..#.#...#.#..#.#",
    ".#.#G.#.#...#.#.G#.#",
    ".#.####.#...#.####.#",
    ".#......#...#......#",
    ".##.#####...#####.##",
    "....................",
    ".........G..........",
    "....................",
]

# 5. Rubble Labyrinth & Structural Maze (20x20)
MAP_RUBBLE_LABYRINTH = [
    "S...................",
    ".#################..",
    ".................#..",
    "..##############.#..",
    "..#............#.#..",
    "..#.##########.#.#..",
    "..#.#........#.#.#..",
    "..#.#.######.#.#.#..",
    "..#.#.#....#.#.#.#..",
    "..#.#.#.G..#.#.#.#..",
    "..#.#.#....#.#.#.#..",
    "..#.#.######.#.#.#..",
    "..#.#........#.#.#..",
    "..#.##########.#.#..",
    "..#............#.#..",
    "..##############.#..",
    ".................#..",
    ".#################..",
    ".......G............",
    "....G.........G.....",
]

PRESET_MAPS = {
    "Urban Earthquake": MAP_URBAN_EARTHQUAKE,
    "Flooded District": MAP_FLOODED_DISTRICT,
    "Industrial Factory": MAP_INDUSTRIAL_FACTORY,
    "Rubble Labyrinth": MAP_RUBBLE_LABYRINTH,
    "Benchmark 10x10": FIXED_BENCHMARK_10x10,
}

