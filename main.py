"""Main application entry point for AI Drone Search and Rescue simulation.

Supports two modes accessible from the UI (press M to switch):
  1. Coverage Mode  — existing A*/BFS multi-survivor area sweep.
  2. Search Mode    — probabilistic uncertain-target search with Bayesian
                      belief updates, battery constraints, and 5 algorithms.
"""
import argparse
import os
import sys
import time
from typing import Optional
# pygame is only imported when a display is needed (not in --headless / benchmark modes).
# This allows running pytest and benchmarks without a display server.
try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False

from config import (
    COLOR_BG,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SPEED_MULTIPLIER,
    FPS,
    MAP_WIDTH,
    PRESET_MAPS,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from detection.yolo_detector import YOLODetector
from evaluation.compare_algorithms import generate_comparison_plots, print_benchmark_summary, run_pathfinding_benchmark
from simulation.environment import DisasterEnvironment
from simulation.mission import MissionController, MissionState
from simulation.search_mission import SearchMission, SearchMissionState
from ui.dashboard import Dashboard, _make_search_policy
from ui.renderer import MapRenderer


class AIDroneSearchRescueApp:
    """Main application manager coordinating Pygame window, events, and simulation loop."""

    def __init__(
        self,
        seed: Optional[int] = DEFAULT_RANDOM_SEED,
        algorithm: str = "astar",
        scenario: Optional[str] = None,
        headless: bool = False,
        start_in_search_mode: bool = False,
    ):
        self.headless = headless
        self.seed = seed
        self.algorithm = algorithm

        # ----- Coverage mode (existing) -----
        self.env = DisasterEnvironment(seed=self.seed)
        self.mission = MissionController(env=self.env, algorithm_name=self.algorithm)

        if scenario:
            for name, layout in PRESET_MAPS.items():
                if scenario.lower() in name.lower():
                    self.mission.load_fixed_map(layout=layout, map_name=name)
                    break

        # ----- Search mode (new) -----
        self._search_env = DisasterEnvironment(seed=self.seed)
        policy = _make_search_policy("Best-First", seed=self.seed)
        self.search_mission = SearchMission(
            env=self._search_env,
            policy=policy,
            seed=self.seed,
        )

        # Simulation speed
        self.steps_per_second: float = float(DEFAULT_SPEED_MULTIPLIER)
        self.step_interval: float = 1.0 / self.steps_per_second
        self.last_step_time: float = time.time()

        # Computer Vision
        self.yolo_detector = YOLODetector()
        self.show_yolo_feed: bool = False

        # Mode
        self.search_mode_active: bool = start_in_search_mode

        if self.headless:
            self.screen = None
            self.renderer = None
            self.dashboard = None
            return

        # ----- Pygame init -----
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption(
            "AI Drone Search & Rescue — Intelligent Search Under Uncertainty"
        )

        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont("Segoe UI, Arial, Helvetica", 17, bold=True)
        self.font_main = pygame.font.SysFont("Segoe UI, Arial, Helvetica", 13, bold=True)
        self.font_small = pygame.font.SysFont("Segoe UI, Arial, Helvetica", 11)

        self.renderer = MapRenderer(self.mission)
        self.renderer.yolo_detector = self.yolo_detector
        self.renderer.search_mission = self.search_mission if self.search_mode_active else None

        self.dashboard = Dashboard(
            mission=self.mission,
            on_speed_change=self.set_speed,
            on_toggle_vision=self.toggle_vision_mode,
            on_toggle_search_mode=self._toggle_search_mode,
            on_toggle_heatmap=self._toggle_heatmap,
            on_toggle_debug=self._toggle_debug,
        )
        self.dashboard.search_mission = self.search_mission
        self.dashboard.search_mode_active = self.search_mode_active

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def set_speed(self, multiplier: int) -> None:
        self.steps_per_second = max(1.0, float(multiplier))
        self.step_interval = 1.0 / self.steps_per_second

    def toggle_vision_mode(self) -> None:
        self.show_yolo_feed = not self.show_yolo_feed
        if self.show_yolo_feed:
            sample_dir = os.path.join(os.path.dirname(__file__), "images")
            os.makedirs(sample_dir, exist_ok=True)
            sample_path = os.path.join(sample_dir, "test_survivor.jpg")
            if not os.path.exists(sample_path):
                self._create_mock_test_image(sample_path)
            annotated, detections, msg = self.yolo_detector.detect_person(sample_path)
            self.mission.status_message = f"Vision Mode: {msg}"
            self.renderer.yolo_detections = detections
            if annotated is not None:
                import cv2
                rgb_img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                h, w, _ = rgb_img.shape
                self.renderer.yolo_annotated_surface = pygame.image.frombuffer(
                    rgb_img.tobytes(), (w, h), "RGB"
                )
        else:
            self.renderer.yolo_annotated_surface = None
            self.mission.status_message = "Vision Mode: Simulated FOV sensor active."

    def _toggle_search_mode(self) -> None:
        """Switch between coverage and search modes."""
        self.search_mode_active = not self.search_mode_active
        if self.renderer:
            self.renderer.search_mission = self.search_mission if self.search_mode_active else None

    def _toggle_heatmap(self) -> None:
        """Toggle probability heatmap overlay."""
        if self.renderer:
            self.renderer.show_heatmap = self.dashboard.show_heatmap

    def _toggle_debug(self) -> None:
        """Toggle debug mode (reveal hidden target)."""
        if self.renderer:
            self.renderer.debug_show_target = self.dashboard.debug_mode

    def _create_mock_test_image(self, target_path: str) -> None:
        try:
            import numpy as np
            import cv2
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            img[:] = (35, 42, 54)
            cv2.circle(img, (320, 210), 24, (200, 200, 200), -1)
            cv2.ellipse(img, (320, 290), (40, 60), 0, 0, 360, (180, 150, 120), -1)
            cv2.putText(img, "AERIAL SEARCH CAMERA FEED", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 212, 255), 2)
            cv2.imwrite(target_path, img)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if mx < MAP_WIDTH:
                    grid_cell = self.renderer.screen_to_grid(mx, my)
                    if grid_cell is not None:
                        gx, gy = grid_cell
                        if not self.search_mode_active:
                            added = self.env.toggle_obstacle(gx, gy)
                            state_str = "added" if added else "removed"
                            self.mission.status_message = f"Dynamic debris {state_str} at ({gx}, {gy})."
                            if self.mission.state == MissionState.RUNNING:
                                self.mission.replan(force=True)
                else:
                    self.dashboard.handle_event(event)

            elif event.type == pygame.MOUSEMOTION:
                if event.pos[0] >= MAP_WIDTH:
                    self.dashboard.handle_event(event)

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                elif event.key == pygame.K_m:
                    self.dashboard._toggle_search_mode()
                elif event.key == pygame.K_SPACE:
                    if self.search_mode_active:
                        self.dashboard._srch_toggle_start_pause()
                    else:
                        self.dashboard._toggle_start_pause()
                elif event.key == pygame.K_s:
                    if self.search_mode_active:
                        self.dashboard._srch_step()
                    else:
                        self.dashboard._step_mission()
                elif event.key == pygame.K_r:
                    if self.search_mode_active:
                        self.dashboard._srch_reset()
                    else:
                        self.mission.reset()
                elif event.key == pygame.K_n:
                    if self.search_mode_active:
                        self.dashboard._srch_new_env()
                    else:
                        self.mission.new_map()
                elif event.key == pygame.K_t:
                    self.dashboard._cycle_preset_map()
                elif event.key == pygame.K_a:
                    if self.search_mode_active:
                        self.dashboard._cycle_search_policy()
                    else:
                        self.dashboard._toggle_algorithm()
                elif event.key == pygame.K_h:
                    if self.search_mode_active:
                        self.dashboard._toggle_heatmap()
                elif event.key == pygame.K_d:
                    if self.search_mode_active:
                        self.dashboard._toggle_debug()
                elif event.key == pygame.K_v:
                    self.dashboard._toggle_vision_mode()
                elif event.key == pygame.K_1:
                    self.dashboard._set_speed(1)
                elif event.key == pygame.K_2:
                    self.dashboard._set_speed(5)
                elif event.key == pygame.K_3:
                    self.dashboard._set_speed(10)
                elif event.key == pygame.K_4:
                    self.dashboard._set_speed(20)

        return True

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        running = True
        last_frame_time = time.time()

        while running:
            current_time = time.time()
            dt = current_time - last_frame_time
            last_frame_time = current_time

            running = self.handle_events()
            if not running:
                break

            # Update clocks
            self.mission.update_clock(dt)
            if self.search_mode_active:
                self.search_mission.update_clock(dt)

            # Advance simulation
            if (current_time - self.last_step_time) >= self.step_interval:
                if self.search_mode_active:
                    if self.search_mission.state == SearchMissionState.RUNNING:
                        self.search_mission.step()
                else:
                    if self.mission.state == MissionState.RUNNING:
                        self.mission.step()
                self.last_step_time = current_time

            # Render
            self.screen.fill(COLOR_BG)
            self.renderer.render(
                self.screen,
                self.font_main,
                self.font_small,
                show_yolo=self.show_yolo_feed and not self.search_mode_active,
            )
            self.dashboard.draw(
                self.screen,
                self.font_title,
                self.font_main,
                self.font_small,
            )

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autonomous AI Drone Search and Rescue Simulation."
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="astar",
        choices=["astar", "bfs"],
        help="Coverage-mode pathfinding algorithm ('astar' or 'bfs').",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED,
                        help="Random seed for repeatable map generation.")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run A* vs BFS pathfinding benchmark and exit.")
    parser.add_argument("--search-benchmark", action="store_true",
                        help="Run 5-algorithm search benchmark and exit.")
    parser.add_argument("--scenario", type=str, default=None,
                        help=f"Preset scenario. Options: {', '.join(PRESET_MAPS.keys())}")
    parser.add_argument("--detect-image", type=str, default=None,
                        help="Path to image for YOLO person detection test.")
    parser.add_argument("--headless", action="store_true",
                        help="Run without display (for CI and automated tests).")
    parser.add_argument("--search-mode", action="store_true",
                        help="Start directly in probabilistic search mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.benchmark:
        print("\n[AI Drone SAR] Running pathfinding benchmark...")
        a_metrics, b_metrics = run_pathfinding_benchmark(num_trials=40, seed=args.seed)
        print_benchmark_summary(a_metrics, b_metrics)
        plot_msg = generate_comparison_plots(a_metrics, b_metrics)
        print(f"[AI Drone SAR] {plot_msg}")
        return

    if args.search_benchmark:
        from evaluation.search_evaluation import run_search_benchmark, print_search_summary
        print("\n[AI Drone SAR] Running 5-algorithm search benchmark...")
        results = run_search_benchmark(num_trials=20, seed=args.seed)
        print_search_summary(results)
        return

    if args.detect_image:
        print(f"\n[AI Drone SAR] Testing YOLO detection on: {args.detect_image}")
        detector = YOLODetector()
        annotated, detections, msg = detector.detect_person(args.detect_image)
        print(f"[AI Drone SAR] Status: {msg}")
        for i, d in enumerate(detections):
            print(f"  Person #{i+1}: Conf={d.confidence:.2f}, Box=({d.x1},{d.y1},{d.x2},{d.y2})")
        return

    print("\n" + "=" * 65)
    print("       AI DRONE SEARCH & RESCUE — INTELLIGENT SEARCH MODE      ")
    print("=" * 65)
    print(f"Seed: {args.seed}   Coverage Algorithm: {args.algorithm.upper()}")
    print("\nControls:")
    print("  M           : Toggle Coverage ↔ Search mode")
    print("  SPACE       : Start / Pause mission")
    print("  S           : Single step")
    print("  R           : Reset mission")
    print("  N           : New random environment")
    print("  A           : Cycle algorithm (Search) / Toggle A*/BFS (Coverage)")
    print("  H           : Toggle probability heatmap (Search mode)")
    print("  D           : Toggle debug target reveal (Search mode)")
    print("  T           : Cycle preset scenarios (Coverage)")
    print("  V           : Toggle YOLO vision camera")
    print("  1-4         : Speed 1x / 5x / 10x / 20x")
    print("  ESC / Q     : Quit")
    print("=" * 65 + "\n")

    app = AIDroneSearchRescueApp(
        seed=args.seed,
        algorithm=args.algorithm,
        scenario=args.scenario,
        headless=args.headless,
        start_in_search_mode=args.search_mode,
    )
    app.run()


if __name__ == "__main__":
    main()
