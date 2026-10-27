"""Disaster environment grid representation, obstacle generation, and ground truth."""
from enum import Enum, auto
import random
from typing import Dict, List, Optional, Set, Tuple

from config import (
    BASE_COORDINATES,
    DEFAULT_OBSTACLE_DENSITY,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SURVIVOR_COUNT,
    FIXED_BENCHMARK_10x10,
    GRID_COLS,
    GRID_ROWS,
)


class CellType(Enum):
    """Types of terrain and structures on the disaster grid."""
    EMPTY = auto()
    BUILDING = auto()
    DEBRIS = auto()


class DisasterEnvironment:
    """Represents the 2D disaster environment grid."""

    def __init__(
        self,
        rows: int = GRID_ROWS,
        cols: int = GRID_COLS,
        base_pos: Tuple[int, int] = BASE_COORDINATES,
        seed: Optional[int] = DEFAULT_RANDOM_SEED,
    ):
        self.rows: int = rows
        self.cols: int = cols
        self.base_pos: Tuple[int, int] = base_pos
        self.seed: Optional[int] = seed

        # Obstacles mapping: (x, y) -> CellType
        self.obstacles: Dict[Tuple[int, int], CellType] = {}

        # Ground-truth survivor locations (hidden from drone until detected)
        self.ground_truth_survivors: Set[Tuple[int, int]] = set()

        # Exploration history
        self.explored_cells: Set[Tuple[int, int]] = set()

        # Precalculated reachable traversable cells
        self.reachable_cells: Set[Tuple[int, int]] = set()

        # Generate initial map
        if seed is not None:
            self.generate_random_map(seed=seed)
        else:
            self.generate_random_map(seed=random.randint(1, 999999))

    def is_valid(self, x: int, y: int) -> bool:
        """Check if coordinates are within the grid bounds."""
        return 0 <= x < self.cols and 0 <= y < self.rows

    def is_traversable(self, x: int, y: int) -> bool:
        """Check if a cell is inside bounds and contains no obstacle."""
        if not self.is_valid(x, y):
            return False
        return (x, y) not in self.obstacles

    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get 4-directional traversable adjacent cells (North, East, South, West)."""
        candidates = [
            (x, y - 1),  # Up
            (x + 1, y),  # Right
            (x, y + 1),  # Down
            (x - 1, y),  # Left
        ]
        return [(nx, ny) for nx, ny in candidates if self.is_traversable(nx, ny)]

    def mark_explored(self, x: int, y: int) -> None:
        """Mark a cell as observed/explored by the drone."""
        if self.is_valid(x, y):
            self.explored_cells.add((x, y))

    def mark_explored_area(self, center_x: int, center_y: int, radius: int = 1) -> None:
        """Mark the area within sensor radius around (center_x, center_y) as explored."""
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = center_x + dx, center_y + dy
                if self.is_valid(nx, ny):
                    self.explored_cells.add((nx, ny))

    def _update_reachability(self) -> None:
        """Compute the set of all traversable cells reachable from the base via BFS."""
        self.reachable_cells.clear()
        if not self.is_traversable(*self.base_pos):
            return

        queue = [self.base_pos]
        visited = {self.base_pos}

        while queue:
            curr = queue.pop(0)
            self.reachable_cells.add(curr)
            for neighbor in self.get_neighbors(*curr):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    def generate_random_map(
        self,
        seed: Optional[int] = None,
        obstacle_density: float = DEFAULT_OBSTACLE_DENSITY,
        num_survivors: int = DEFAULT_SURVIVOR_COUNT,
    ) -> None:
        """
        Generate a randomized disaster map with buildings, debris, and survivors.
        Ensures base is clear and all survivors are placed on reachable cells.
        """
        if seed is not None:
            self.seed = seed
            random.seed(seed)

        self.obstacles.clear()
        self.ground_truth_survivors.clear()
        self.explored_cells.clear()

        # Reserve base area (base cell and its immediate neighbors)
        reserved_cells = {self.base_pos}
        bx, by = self.base_pos
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if self.is_valid(bx + dx, by + dy):
                reserved_cells.add((bx + dx, by + dy))

        total_cells = self.rows * self.cols
        target_obstacles = int(total_cells * obstacle_density)

        # Place clustered buildings and scattered debris
        placed_obstacles = 0

        # 1. Place some small building clusters (2x2 or 2x1)
        num_clusters = max(2, target_obstacles // 8)
        for _ in range(num_clusters):
            cx = random.randint(1, self.cols - 2)
            cy = random.randint(1, self.rows - 2)
            c_type = CellType.BUILDING if random.random() < 0.6 else CellType.DEBRIS
            w = random.randint(1, 2)
            h = random.randint(1, 2)
            for dy in range(h):
                for dx in range(w):
                    pos = (cx + dx, cy + dy)
                    if self.is_valid(*pos) and pos not in reserved_cells:
                        self.obstacles[pos] = c_type
                        placed_obstacles += 1

        # 2. Place remaining individual obstacles/debris
        while placed_obstacles < target_obstacles:
            rx = random.randint(0, self.cols - 1)
            ry = random.randint(0, self.rows - 1)
            pos = (rx, ry)
            if pos not in reserved_cells and pos not in self.obstacles:
                c_type = CellType.BUILDING if random.random() < 0.5 else CellType.DEBRIS
                self.obstacles[pos] = c_type
                placed_obstacles += 1

        # Calculate reachable cells from base
        self._update_reachability()

        # If base is trapped or coverage too low, prune obstacles around base
        if len(self.reachable_cells) < (total_cells * 0.4):
            # Clean up excessive obstacles
            keys_to_remove = [pos for pos in self.obstacles if abs(pos[0] - bx) + abs(pos[1] - by) <= 4]
            for pos in keys_to_remove:
                del self.obstacles[pos]
            self._update_reachability()

        # Place survivors strictly on reachable traversable cells (excluding base)
        available_cells = [c for c in self.reachable_cells if c != self.base_pos]
        survivors_to_place = min(num_survivors, len(available_cells))

        if survivors_to_place > 0:
            chosen = random.sample(available_cells, survivors_to_place)
            self.ground_truth_survivors = set(chosen)

        # Base is immediately explored
        self.mark_explored_area(*self.base_pos, radius=1)

    def load_fixed_map(self, layout: Optional[List[str]] = None) -> None:
        """
        Load a deterministic fixed layout for repeatable experiments.
        '.' = empty, '#' = building, 'D' = debris, 'S' = base/start, 'G' = survivor
        """
        if layout is None:
            layout = FIXED_BENCHMARK_10x10

        self.rows = len(layout)
        self.cols = len(layout[0])
        self.obstacles.clear()
        self.ground_truth_survivors.clear()
        self.explored_cells.clear()

        for y, row in enumerate(layout):
            for x, char in enumerate(row):
                pos = (x, y)
                if char == '#':
                    self.obstacles[pos] = CellType.BUILDING
                elif char == 'D':
                    self.obstacles[pos] = CellType.DEBRIS
                elif char == 'S':
                    self.base_pos = pos
                elif char == 'G':
                    self.ground_truth_survivors.add(pos)

        self._update_reachability()
        self.mark_explored_area(*self.base_pos, radius=1)

    def toggle_obstacle(self, x: int, y: int, obstacle_type: CellType = CellType.DEBRIS) -> bool:
        """
        Toggle an obstacle at (x, y) dynamically (for dynamic replanning testing).
        Returns True if obstacle was added, False if removed.
        """
        if not self.is_valid(x, y) or (x, y) == self.base_pos:
            return False

        if (x, y) in self.obstacles:
            del self.obstacles[(x, y)]
            self._update_reachability()
            return False
        else:
            self.obstacles[(x, y)] = obstacle_type
            # If a survivor was there, remove it
            self.ground_truth_survivors.discard((x, y))
            self._update_reachability()
            return True

    @property
    def total_traversable_cells(self) -> int:
        """Total number of in-bounds cells that are not obstacles."""
        return (self.rows * self.cols) - len(self.obstacles)

    @property
    def total_reachable_cells(self) -> int:
        """Total reachable cells from base."""
        return len(self.reachable_cells)
