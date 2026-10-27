"""Dashboard sidebar for mission metrics, controls, and environment legend.

Supports two modes:
  - Coverage Mode: existing A*/BFS multi-survivor area sweep (unchanged).
  - Search Mode:   probabilistic uncertain-target search with 5 algorithms,
                   battery meter, heatmap toggle, and mission log.
"""
import math
from typing import Callable, List, Optional, Tuple
import pygame

from config import (
    COLOR_BASE,
    COLOR_BATTERY_HIGH,
    COLOR_BATTERY_LOW,
    COLOR_BATTERY_MED,
    COLOR_DANGER,
    COLOR_DRONE,
    COLOR_EXPLORED,
    COLOR_OBSTACLE_BUILDING,
    COLOR_OBSTACLE_DEBRIS,
    COLOR_PANEL_BG,
    COLOR_PANEL_BORDER,
    COLOR_PLANNED_PATH,
    COLOR_PROB_HIGH,
    COLOR_PROB_LOW,
    COLOR_SUCCESS,
    COLOR_SURVIVOR_DISCOVERED,
    COLOR_SURVIVOR_RESCUED,
    COLOR_TEXT_ACCENT,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_PRIMARY,
    COLOR_TRAIL,
    COLOR_UNEXPLORED,
    COLOR_WARNING,
    MAP_WIDTH,
    MISSION_LOG_VISIBLE,
    PRESET_MAPS,
    SIDEBAR_WIDTH,
    WINDOW_HEIGHT,
)
from simulation.mission import MissionController, MissionState
from simulation.search_mission import SearchMission, SearchMissionState
from ui.controls import Button


# Search policy display names and import keys
SEARCH_POLICY_NAMES = [
    "Best-First",
    "Beam Search",
    "Hill Climb",
    "A* Belief",
    "Genetic Alg",
]


def _make_search_policy(name: str, seed: int = 42):
    """Factory: create a SearchPolicy instance by display name."""
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
        return GeneticAlgorithmPolicy(seed=seed)
    else:
        from algorithms.search_policies.best_first_search import BestFirstSearchPolicy
        return BestFirstSearchPolicy()


class Dashboard:
    """Renders real-time telemetry, mission controls, and interactive buttons.

    Dual-mode: coverage (existing) and probabilistic search (new).
    """

    def __init__(
        self,
        mission: MissionController,
        on_speed_change: Callable[[int], None],
        on_toggle_vision: Callable[[], None],
        on_toggle_search_mode: Optional[Callable[[], None]] = None,
        on_toggle_heatmap: Optional[Callable[[], None]] = None,
        on_toggle_debug: Optional[Callable[[], None]] = None,
    ):
        self.mission = mission
        self.on_speed_change = on_speed_change
        self.on_toggle_vision = on_toggle_vision
        self.on_toggle_search_mode = on_toggle_search_mode or (lambda: None)
        self.on_toggle_heatmap = on_toggle_heatmap or (lambda: None)
        self.on_toggle_debug = on_toggle_debug or (lambda: None)

        # Search mission reference (set by main app)
        self.search_mission: Optional[SearchMission] = None
        self.search_mode_active: bool = False
        self.show_heatmap: bool = False
        self.debug_mode: bool = False

        self.panel_rect = pygame.Rect(MAP_WIDTH, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT)
        self.buttons: List[Button] = []
        self.coverage_buttons: List[Button] = []  # Coverage-mode only
        self.search_buttons: List[Button] = []    # Search-mode only
        self.shared_buttons: List[Button] = []    # Always shown
        self.speed_buttons: List[Button] = []

        self.current_speed: int = 5
        self.vision_mode: str = "SIMULATED"
        self.preset_map_names: List[str] = list(PRESET_MAPS.keys())
        self.current_preset_idx: int = 0

        # Search algorithm selector
        self.search_policy_idx: int = 0

        # Log scroll offset
        self.log_scroll: int = 0

        self._setup_buttons()

    # ------------------------------------------------------------------
    # Button setup
    # ------------------------------------------------------------------

    # Vertical layout constants (pixels from top of window)
    # These are the pixel anchors for each section so nothing overlaps.
    #
    # 0      header title
    # 12     "AI DRONE SAR SYSTEM"
    # 32     subtitle
    # 50     mode toggle button  (h=24)
    # 80     state badge + clock (h=22)
    # 108    telemetry card      (coverage h=148 / search: batt 102..126, card 132..242)
    # 262    buttons start       (START/STEP row)
    # 300    RESET/NEW/SCENARIO row
    # 338    ALGO/VISION (or POLICY/DEBUG) row
    # 374    speed buttons row
    # 406    status box / mission log
    # 460    legend
    #
    # Total: 460 + ~220 = 680 = WINDOW_HEIGHT  (fits exactly)

    BTN_Y_START   = 262   # First button row
    BTN_Y_RESET   = 300
    BTN_Y_ALGO    = 338
    BTN_Y_SPEED   = 374
    STATUS_Y      = 406
    LEGEND_COV_Y  = 460
    LOG_Y         = 406
    LEGEND_SRCH_Y = 562   # log_h=148 -> 406+148+8=562

    def _setup_buttons(self) -> None:
        bx = MAP_WIDTH + 15
        bw_full = SIDEBAR_WIDTH - 30
        bw_half = (bw_full - 10) // 2
        bw_third = (bw_full - 16) // 3
        bw_quarter = (bw_full - 18) // 4

        # Mode toggle sits just below the header (always visible)
        self.btn_mode = Button(
            rect=(bx, 50, bw_full, 24),
            text="MODE: COVERAGE  [M to switch]",
            callback=self._toggle_search_mode,
            bg_color=(40, 60, 90),
        )
        self.shared_buttons.append(self.btn_mode)

        # --- COVERAGE MODE BUTTONS --- (placed below telemetry card)
        y_cov = self.BTN_Y_START

        self.btn_start_pause = Button(
            rect=(bx, y_cov, bw_half, 32),
            text="START",
            callback=self._toggle_start_pause,
            bg_color=(39, 174, 96),
        )
        self.btn_step = Button(
            rect=(bx + bw_half + 10, y_cov, bw_half, 32),
            text="STEP",
            callback=self._step_mission,
        )
        self.coverage_buttons.extend([self.btn_start_pause, self.btn_step])

        self.btn_reset = Button(
            rect=(bx, self.BTN_Y_RESET, bw_third, 30),
            text="RESET",
            callback=self.mission.reset,
        )
        self.btn_new_map = Button(
            rect=(bx + bw_third + 8, self.BTN_Y_RESET, bw_third, 30),
            text="NEW MAP",
            callback=lambda: self.mission.new_map(),
        )
        self.btn_fixed_map = Button(
            rect=(bx + (bw_third * 2) + 16, self.BTN_Y_RESET, bw_third, 30),
            text="SCENARIO",
            callback=self._cycle_preset_map,
        )
        self.coverage_buttons.extend([self.btn_reset, self.btn_new_map, self.btn_fixed_map])

        self.algo_button = Button(
            rect=(bx, self.BTN_Y_ALGO, bw_half, 30),
            text="ALGO: A*",
            callback=self._toggle_algorithm,
        )
        self.vision_button = Button(
            rect=(bx + bw_half + 10, self.BTN_Y_ALGO, bw_half, 30),
            text="DETECTOR: SIM",
            callback=self._toggle_vision_mode,
        )
        self.coverage_buttons.extend([self.algo_button, self.vision_button])

        speeds = [1, 5, 10, 20]
        for i, spd in enumerate(speeds):
            btn = Button(
                rect=(bx + i * (bw_quarter + 6), self.BTN_Y_SPEED, bw_quarter, 26),
                text=f"{spd}x",
                callback=lambda s=spd: self._set_speed(s),
            )
            if spd == self.current_speed:
                btn.is_active = True
            self.speed_buttons.append(btn)
            self.coverage_buttons.append(btn)

        # --- SEARCH MODE BUTTONS --- (same vertical anchors)
        self.btn_srch_start = Button(
            rect=(bx, self.BTN_Y_START, bw_half, 32),
            text="START",
            callback=self._srch_toggle_start_pause,
            bg_color=(39, 174, 96),
        )
        self.btn_srch_step = Button(
            rect=(bx + bw_half + 10, self.BTN_Y_START, bw_half, 32),
            text="STEP",
            callback=self._srch_step,
        )
        self.search_buttons.extend([self.btn_srch_start, self.btn_srch_step])

        self.btn_srch_reset = Button(
            rect=(bx, self.BTN_Y_RESET, bw_third, 30),
            text="RESET",
            callback=self._srch_reset,
        )
        self.btn_srch_new_map = Button(
            rect=(bx + bw_third + 8, self.BTN_Y_RESET, bw_third, 30),
            text="NEW ENV",
            callback=self._srch_new_env,
        )
        self.btn_srch_heatmap = Button(
            rect=(bx + (bw_third * 2) + 16, self.BTN_Y_RESET, bw_third, 30),
            text="HEATMAP",
            callback=self._toggle_heatmap,
            is_toggle=True,
        )
        self.search_buttons.extend([self.btn_srch_reset, self.btn_srch_new_map, self.btn_srch_heatmap])

        self.btn_srch_algo = Button(
            rect=(bx, self.BTN_Y_ALGO, bw_half, 30),
            text=f"POLICY: {SEARCH_POLICY_NAMES[0]}",
            callback=self._cycle_search_policy,
        )
        self.btn_srch_debug = Button(
            rect=(bx + bw_half + 10, self.BTN_Y_ALGO, bw_half, 30),
            text="DEBUG: OFF",
            callback=self._toggle_debug,
            is_toggle=True,
        )
        self.search_buttons.extend([self.btn_srch_algo, self.btn_srch_debug])

        for i, spd in enumerate(speeds):
            btn = Button(
                rect=(bx + i * (bw_quarter + 6), self.BTN_Y_SPEED, bw_quarter, 26),
                text=f"{spd}x",
                callback=lambda s=spd: self._set_speed(s),
            )
            if spd == self.current_speed:
                btn.is_active = True
            self.search_buttons.append(btn)

        # Collect all
        self.buttons = self.shared_buttons + self.coverage_buttons + self.search_buttons

    # ------------------------------------------------------------------
    # Coverage mode callbacks
    # ------------------------------------------------------------------

    def _toggle_start_pause(self) -> None:
        if self.mission.state == MissionState.RUNNING:
            self.mission.pause()
            self.btn_start_pause.text = "RESUME"
            self.btn_start_pause.bg_color = (230, 126, 34)
        else:
            self.mission.start()
            self.btn_start_pause.text = "PAUSE"
            self.btn_start_pause.bg_color = (231, 76, 60)

    def _step_mission(self) -> None:
        if self.mission.state == MissionState.RUNNING:
            self.mission.pause()
            self.btn_start_pause.text = "RESUME"
            self.btn_start_pause.bg_color = (230, 126, 34)
        self.mission.step()

    def _cycle_preset_map(self) -> None:
        name = self.preset_map_names[self.current_preset_idx]
        layout = PRESET_MAPS[name]
        self.mission.load_fixed_map(layout=layout, map_name=name)
        self.current_preset_idx = (self.current_preset_idx + 1) % len(self.preset_map_names)

    def _toggle_algorithm(self) -> None:
        if self.mission.active_algorithm_name == "astar":
            self.mission.set_algorithm("bfs")
            self.algo_button.text = "ALGO: BFS"
        else:
            self.mission.set_algorithm("astar")
            self.algo_button.text = "ALGO: A*"

    def _toggle_vision_mode(self) -> None:
        if self.vision_mode == "SIMULATED":
            self.vision_mode = "YOLO"
            self.vision_button.text = "DETECTOR: YOLO"
        else:
            self.vision_mode = "SIMULATED"
            self.vision_button.text = "DETECTOR: SIM"
        self.on_toggle_vision()

    def _set_speed(self, speed: int) -> None:
        self.current_speed = speed
        for btn in self.speed_buttons:
            btn.is_active = (btn.text == f"{speed}x")
        self.on_speed_change(speed)

    # ------------------------------------------------------------------
    # Search mode callbacks
    # ------------------------------------------------------------------

    def _srch_toggle_start_pause(self) -> None:
        sm = self.search_mission
        if sm is None:
            return
        if sm.state == SearchMissionState.RUNNING:
            sm.pause()
            self.btn_srch_start.text = "RESUME"
            self.btn_srch_start.bg_color = (230, 126, 34)
        elif sm.state in (SearchMissionState.IDLE, SearchMissionState.PAUSED):
            sm.start()
            self.btn_srch_start.text = "PAUSE"
            self.btn_srch_start.bg_color = (231, 76, 60)

    def _srch_step(self) -> None:
        sm = self.search_mission
        if sm is None:
            return
        if sm.state == SearchMissionState.RUNNING:
            sm.pause()
            self.btn_srch_start.text = "RESUME"
            self.btn_srch_start.bg_color = (230, 126, 34)
        if sm.state not in (SearchMissionState.TARGET_FOUND,
                             SearchMissionState.BATTERY_DEPLETED,
                             SearchMissionState.COMPLETED):
            sm.step()

    def _srch_reset(self) -> None:
        sm = self.search_mission
        if sm is None:
            return
        sm.reset()
        self.btn_srch_start.text = "START"
        self.btn_srch_start.bg_color = (39, 174, 96)
        self.log_scroll = 0

    def _srch_new_env(self) -> None:
        """Generate a new environment and reinitialize search mission."""
        if self.search_mission is None:
            return
        import random
        new_seed = random.randint(1, 99999)
        self.search_mission.env.generate_random_map(seed=new_seed)
        self.search_mission.reset()
        self.btn_srch_start.text = "START"
        self.btn_srch_start.bg_color = (39, 174, 96)
        self.log_scroll = 0

    def _toggle_heatmap(self) -> None:
        self.show_heatmap = not self.show_heatmap
        self.btn_srch_heatmap.is_active = self.show_heatmap
        self.on_toggle_heatmap()

    def _toggle_debug(self) -> None:
        self.debug_mode = not self.debug_mode
        self.btn_srch_debug.text = f"DEBUG: {'ON' if self.debug_mode else 'OFF'}"
        self.btn_srch_debug.is_active = self.debug_mode
        self.on_toggle_debug()

    def _cycle_search_policy(self) -> None:
        """Cycle through 5 search policy algorithms."""
        self.search_policy_idx = (self.search_policy_idx + 1) % len(SEARCH_POLICY_NAMES)
        name = SEARCH_POLICY_NAMES[self.search_policy_idx]
        self.btn_srch_algo.text = f"ALGO: {name}"
        # Rebuild search mission with new policy if one exists
        if self.search_mission is not None:
            policy = _make_search_policy(name)
            self.search_mission.policy = policy
            self.search_mission.reset()
            self.btn_srch_start.text = "START"
            self.btn_srch_start.bg_color = (39, 174, 96)

    def _toggle_search_mode(self) -> None:
        self.search_mode_active = not self.search_mode_active
        if self.search_mode_active:
            self.btn_mode.text = "MODE: SEARCH  [M to switch]"
            self.btn_mode.bg_color = (60, 40, 90)
        else:
            self.btn_mode.text = "MODE: COVERAGE  [M to switch]"
            self.btn_mode.bg_color = (40, 60, 90)
        self.on_toggle_search_mode()

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event) -> None:
        """Route mouse events to active buttons."""
        for btn in self.shared_buttons:
            btn.handle_event(event)
        if self.search_mode_active:
            for btn in self.search_buttons:
                btn.handle_event(event)
        else:
            for btn in self.coverage_buttons:
                btn.handle_event(event)

    # ------------------------------------------------------------------
    # State sync
    # ------------------------------------------------------------------

    def update_state(self) -> None:
        """Sync coverage button labels with mission state."""
        if self.mission.state == MissionState.RUNNING:
            self.btn_start_pause.text = "PAUSE"
            self.btn_start_pause.bg_color = (231, 76, 60)
        elif self.mission.state == MissionState.PAUSED:
            self.btn_start_pause.text = "RESUME"
            self.btn_start_pause.bg_color = (230, 126, 34)
        elif self.mission.state == MissionState.COMPLETED:
            self.btn_start_pause.text = "FINISHED"
            self.btn_start_pause.bg_color = (127, 140, 141)
        else:
            self.btn_start_pause.text = "START"
            self.btn_start_pause.bg_color = (39, 174, 96)

        # Sync search mode button state
        sm = self.search_mission
        if sm is not None:
            if sm.state == SearchMissionState.RUNNING:
                self.btn_srch_start.text = "PAUSE"
                self.btn_srch_start.bg_color = (231, 76, 60)
            elif sm.state in (SearchMissionState.TARGET_FOUND,
                               SearchMissionState.BATTERY_DEPLETED,
                               SearchMissionState.COMPLETED):
                self.btn_srch_start.text = "DONE"
                self.btn_srch_start.bg_color = (80, 80, 80)

    # ------------------------------------------------------------------
    # Main draw
    # ------------------------------------------------------------------

    def draw(
        self,
        surface: pygame.Surface,
        font_title: pygame.font.Font,
        font_main: pygame.font.Font,
        font_small: pygame.font.Font,
    ) -> None:
        self.update_state()

        # Background panel
        pygame.draw.rect(surface, COLOR_PANEL_BG, self.panel_rect)
        pygame.draw.line(surface, COLOR_PANEL_BORDER, (MAP_WIDTH, 0), (MAP_WIDTH, WINDOW_HEIGHT), 2)

        bx = MAP_WIDTH + 15

        # Header (y=12)
        title_surf = font_title.render("AI DRONE SAR SYSTEM", True, COLOR_TEXT_ACCENT)
        surface.blit(title_surf, (bx, 12))
        sub_surf = font_small.render("Intelligent Search Under Uncertainty", True, COLOR_TEXT_MUTED)
        surface.blit(sub_surf, (bx, 32))

        # Mode toggle button (y=50, h=24 -> bottom=74)
        for btn in self.shared_buttons:
            btn.draw(surface, font_small)

        if self.search_mode_active:
            self._draw_search_panel(surface, font_main, font_small, bx)
        else:
            self._draw_coverage_panel(surface, font_main, font_small, bx)

    # ------------------------------------------------------------------
    # Coverage panel
    # ------------------------------------------------------------------

    def _draw_coverage_panel(self, surface, font_main, font_small, bx):
        # State badge row (y=80, h=22 -> bottom=102)
        y = 80

        state_color = {
            MissionState.RUNNING: COLOR_SUCCESS,
            MissionState.PAUSED: COLOR_WARNING,
            MissionState.COMPLETED: COLOR_TEXT_ACCENT,
        }.get(self.mission.state, COLOR_TEXT_MUTED)
        badge_text = f"STATE: {self.mission.state.name}"
        badge_surf = font_small.render(badge_text, True, (20, 24, 32))
        badge_w = badge_surf.get_width() + 14
        badge_rect = pygame.Rect(bx, y, badge_w, 18)
        pygame.draw.rect(surface, state_color, badge_rect, border_radius=4)
        surface.blit(badge_surf, (bx + 7, y + 2))

        mins = int(self.mission.elapsed_time_sec) // 60
        secs = int(self.mission.elapsed_time_sec) % 60
        clock_surf = font_small.render(f"Time: {mins:02d}:{secs:02d}", True, COLOR_TEXT_PRIMARY)
        surface.blit(clock_surf, (bx + badge_w + 12, y + 2))

        # Telemetry card (y=104, h=148 -> bottom=252)
        y = 104
        card_rect = pygame.Rect(bx, y, SIDEBAR_WIDTH - 30, 148)
        pygame.draw.rect(surface, (20, 25, 35), card_rect, border_radius=6)
        pygame.draw.rect(surface, COLOR_PANEL_BORDER, card_rect, width=1, border_radius=6)
        cy, cx = y + 8, bx + 12

        cov_pct = self.mission.coverage_percentage
        surface.blit(
            font_small.render(
                f"Coverage: {cov_pct:.1f}% ({self.mission.explored_traversable}/{self.mission.total_reachable})",
                True, COLOR_TEXT_PRIMARY,
            ),
            (cx, cy),
        )
        cy += 16
        bar_bg = pygame.Rect(cx, cy, SIDEBAR_WIDTH - 54, 8)
        bar_fill = pygame.Rect(cx, cy, int((SIDEBAR_WIDTH - 54) * (cov_pct / 100.0)), 8)
        pygame.draw.rect(surface, (40, 48, 64), bar_bg, border_radius=4)
        pygame.draw.rect(surface, COLOR_SUCCESS, bar_fill, border_radius=4)
        cy += 14

        items = [
            ("Survivors Discovered:", f"{self.mission.survivors_discovered}", COLOR_WARNING),
            ("Survivors Reached:", f"{self.mission.survivors_reached}", COLOR_SUCCESS),
            ("Flight Distance:", f"{self.mission.drone.distance_traveled} steps", COLOR_TEXT_PRIMARY),
            ("Pathfinding Queries:", f"{self.mission.pathfinding_operations}", COLOR_TEXT_PRIMARY),
            ("Active Algorithm:", f"{self.mission.active_algorithm_name.upper()}", COLOR_TEXT_ACCENT),
            ("Dynamic Replans:", f"{self.mission.replanning_events}", COLOR_TEXT_PRIMARY),
        ]
        for label, val, val_color in items:
            lbl_surf = font_small.render(label, True, COLOR_TEXT_MUTED)
            val_surf = font_small.render(val, True, val_color)
            surface.blit(lbl_surf, (cx, cy))
            surface.blit(val_surf, (bx + SIDEBAR_WIDTH - 42 - val_surf.get_width(), cy))
            cy += 17

        # Buttons (drawn AFTER card — y anchors defined in _setup_buttons)
        for btn in self.coverage_buttons:
            btn.draw(surface, font_small)

        # Status message and legend (below speed buttons at BTN_Y_SPEED=374+26=400)
        self._draw_status_box(surface, font_small, bx, self.STATUS_Y, self.mission.status_message)
        self._draw_legend(surface, font_small, bx, self.LEGEND_COV_Y)

    # ------------------------------------------------------------------
    # Search panel
    # ------------------------------------------------------------------

    def _draw_search_panel(self, surface, font_main, font_small, bx):
        sm = self.search_mission

        if sm is None:
            msg = font_small.render("Search mission not initialized.", True, COLOR_DANGER)
            surface.blit(msg, (bx, 80))
            return

        # State badge row (y=80, h=18 -> bottom=98)
        y = 80
        state_colors = {
            SearchMissionState.RUNNING: COLOR_SUCCESS,
            SearchMissionState.PAUSED: COLOR_WARNING,
            SearchMissionState.TARGET_FOUND: COLOR_SUCCESS,
            SearchMissionState.BATTERY_DEPLETED: COLOR_BATTERY_LOW,
            SearchMissionState.COMPLETED: COLOR_TEXT_ACCENT,
        }
        state_color = state_colors.get(sm.state, COLOR_TEXT_MUTED)
        badge_text = f"STATE: {sm.state.name}"
        badge_surf = font_small.render(badge_text, True, (20, 24, 32))
        badge_w = badge_surf.get_width() + 14
        badge_rect = pygame.Rect(bx, y, badge_w, 18)
        pygame.draw.rect(surface, state_color, badge_rect, border_radius=4)
        surface.blit(badge_surf, (bx + 7, y + 2))

        mins = int(sm.elapsed_time_sec) // 60
        secs = int(sm.elapsed_time_sec) % 60
        clock_surf = font_small.render(f"Time: {mins:02d}:{secs:02d}", True, COLOR_TEXT_PRIMARY)
        surface.blit(clock_surf, (bx + badge_w + 12, y + 2))

        # Battery meter (y=102)
        y = 102
        batt = sm.battery_remaining
        batt_pct = batt * 100.0
        if batt > 0.5:
            bat_color = COLOR_BATTERY_HIGH
        elif batt > 0.25:
            bat_color = COLOR_BATTERY_MED
        else:
            bat_color = COLOR_BATTERY_LOW

        batt_label = font_small.render(f"Battery: {batt_pct:.1f}%", True, bat_color)
        surface.blit(batt_label, (bx, y))
        y += 14
        bar_w = SIDEBAR_WIDTH - 30
        bat_bg = pygame.Rect(bx, y, bar_w, 10)
        bat_fill = pygame.Rect(bx, y, int(bar_w * batt), 10)
        pygame.draw.rect(surface, (40, 48, 64), bat_bg, border_radius=5)
        pygame.draw.rect(surface, bat_color, bat_fill, border_radius=5)

        # Search telemetry card (y=132, h=110 -> bottom=242; battery bar ends at y=126)
        y = 132
        card_h = 110
        card_rect = pygame.Rect(bx, y, SIDEBAR_WIDTH - 30, card_h)
        pygame.draw.rect(surface, (20, 25, 35), card_rect, border_radius=6)
        pygame.draw.rect(surface, COLOR_PANEL_BORDER, card_rect, width=1, border_radius=6)
        cy, cx = y + 6, bx + 10

        cov_pct = sm.coverage_percentage
        surface.blit(font_small.render(f"Coverage: {cov_pct:.1f}%", True, COLOR_TEXT_PRIMARY), (cx, cy))
        cy += 14
        cov_bar_bg = pygame.Rect(cx, cy, SIDEBAR_WIDTH - 50, 6)
        cov_bar_fill = pygame.Rect(cx, cy, int((SIDEBAR_WIDTH - 50) * (cov_pct / 100.0)), 6)
        pygame.draw.rect(surface, (40, 48, 64), cov_bar_bg, border_radius=3)
        pygame.draw.rect(surface, (52, 152, 219), cov_bar_fill, border_radius=3)
        cy += 12

        policy_name = sm.policy.name if sm.policy else "-"
        items = [
            ("Policy:", policy_name, COLOR_TEXT_ACCENT),
            ("Steps:", f"{sm.step_count}", COLOR_TEXT_PRIMARY),
            ("Distance:", f"{sm.distance_traveled} cells", COLOR_TEXT_PRIMARY),
            ("Detection:", "CONFIRMED" if sm.detection_confirmed else "Searching...",
             COLOR_SUCCESS if sm.detection_confirmed else COLOR_TEXT_MUTED),
            ("Target:", f"Step {sm.belief_map.detection_step}" if sm.belief_map.detection_step else "-",
             COLOR_SUCCESS if sm.detection_confirmed else COLOR_TEXT_MUTED),
        ]
        for label, val, val_color in items:
            lbl_surf = font_small.render(label, True, COLOR_TEXT_MUTED)
            val_surf = font_small.render(val, True, val_color)
            surface.blit(lbl_surf, (cx, cy))
            surface.blit(val_surf, (bx + SIDEBAR_WIDTH - 42 - val_surf.get_width(), cy))
            cy += 16

        # Search buttons drawn AFTER card (anchored at BTN_Y_START=262)
        for btn in self.search_buttons:
            btn.draw(surface, font_small)

        # Mission log (LOG_Y=406)
        self._draw_mission_log(surface, font_small, bx, self.LOG_Y, sm)

    def _draw_mission_log(self, surface, font_small, bx, y, sm):
        """Render the recent mission log entries."""
        log_h = 148
        log_rect = pygame.Rect(bx, y, SIDEBAR_WIDTH - 30, log_h)
        pygame.draw.rect(surface, (16, 20, 30), log_rect, border_radius=6)
        pygame.draw.rect(surface, COLOR_PANEL_BORDER, log_rect, width=1, border_radius=6)

        header = font_small.render("MISSION LOG", True, COLOR_TEXT_PRIMARY)
        surface.blit(header, (bx + 8, y + 4))
        pygame.draw.line(surface, COLOR_PANEL_BORDER, (bx + 4, y + 17), (bx + SIDEBAR_WIDTH - 34, y + 17), 1)

        log = sm.log
        visible = MISSION_LOG_VISIBLE
        # Show most recent entries
        start_idx = max(0, len(log) - visible)
        ly = y + 20
        for entry in log[start_idx:]:
            # Truncate to fit sidebar width
            max_chars = 42
            display = entry[:max_chars] + ("…" if len(entry) > max_chars else "")
            color = COLOR_SUCCESS if "[DETECT" in entry else (
                COLOR_BATTERY_LOW if "[END]" in entry else (
                    COLOR_TEXT_ACCENT if "[PLAN]" in entry else COLOR_TEXT_MUTED
                )
            )
            surf = font_small.render(display, True, color)
            surface.blit(surf, (bx + 6, ly))
            ly += 16
            if ly > y + log_h - 8:
                break

        # Legend below log (LEGEND_SRCH_Y = LOG_Y + log_h + 8 = 406+148+8 = 562)
        self._draw_search_legend(surface, font_small, bx, self.LEGEND_SRCH_Y)

    def _draw_search_legend(self, surface, font_small, bx, y):
        """Compact legend for search mode."""
        legend_rect = pygame.Rect(bx, y, SIDEBAR_WIDTH - 30, 122)
        pygame.draw.rect(surface, (20, 25, 35), legend_rect, border_radius=6)
        pygame.draw.rect(surface, COLOR_PANEL_BORDER, legend_rect, width=1, border_radius=6)
        ly, lx = y + 6, bx + 10
        surface.blit(font_small.render("LEGEND", True, COLOR_TEXT_PRIMARY), (lx, ly))
        ly += 18

        items = [
            ("Base Station", COLOR_BASE, "rect"),
            ("Drone FOV", COLOR_DRONE, "circle"),
            ("Building", COLOR_OBSTACLE_BUILDING, "rect"),
            ("Debris", COLOR_OBSTACLE_DEBRIS, "debris"),
            ("High Probability", COLOR_PROB_HIGH, "rect"),
            ("Low Probability", COLOR_PROB_LOW, "rect"),
        ]
        for name, color, shape in items:
            sym_rect = pygame.Rect(lx, ly + 1, 10, 10)
            if shape == "rect":
                pygame.draw.rect(surface, color, sym_rect, border_radius=2)
            elif shape == "circle":
                pygame.draw.circle(surface, color, sym_rect.center, 5)
            elif shape == "debris":
                pygame.draw.polygon(surface, color, [(lx, ly + 10), (lx + 5, ly + 1), (lx + 10, ly + 8)])
            lbl = font_small.render(name, True, COLOR_TEXT_MUTED)
            surface.blit(lbl, (lx + 14, ly))
            ly += 16

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _draw_status_box(self, surface, font_small, bx, y, message):
        alert_rect = pygame.Rect(bx, y, SIDEBAR_WIDTH - 30, 46)
        pygame.draw.rect(surface, (23, 29, 41), alert_rect, border_radius=6)
        pygame.draw.rect(surface, COLOR_PANEL_BORDER, alert_rect, width=1, border_radius=6)
        words = message.split()
        lines = []
        cur_line = ""
        for word in words:
            test_line = f"{cur_line} {word}".strip()
            if font_small.size(test_line)[0] < (SIDEBAR_WIDTH - 46):
                cur_line = test_line
            else:
                lines.append(cur_line)
                cur_line = word
        if cur_line:
            lines.append(cur_line)
        sy = y + 6
        for line in lines[:2]:
            surface.blit(font_small.render(line, True, COLOR_TEXT_ACCENT), (bx + 8, sy))
            sy += 16

    def _draw_legend(self, surface, font_small, bx, y):
        legend_rect = pygame.Rect(bx, y, SIDEBAR_WIDTH - 30, 220)
        pygame.draw.rect(surface, (20, 25, 35), legend_rect, border_radius=6)
        pygame.draw.rect(surface, COLOR_PANEL_BORDER, legend_rect, width=1, border_radius=6)
        ly, lx = y + 8, bx + 12
        surface.blit(font_small.render("MAP LEGEND & SYMBOLS", True, COLOR_TEXT_PRIMARY), (lx, ly))
        ly += 20
        legend_items = [
            ("Base Station", COLOR_BASE, "rect"),
            ("Drone & FOV Sensor", COLOR_DRONE, "circle"),
            ("Building (Impassable)", COLOR_OBSTACLE_BUILDING, "rect"),
            ("Debris / Rubble", COLOR_OBSTACLE_DEBRIS, "debris"),
            ("Explored Cell", COLOR_EXPLORED, "rect"),
            ("Discovered Survivor", COLOR_SURVIVOR_DISCOVERED, "diamond"),
            ("Rescued Survivor", COLOR_SURVIVOR_RESCUED, "check"),
            ("Planned Route", COLOR_PLANNED_PATH, "line"),
            ("Flight Trail", COLOR_TRAIL, "line"),
        ]
        for name, color, shape in legend_items:
            sym_rect = pygame.Rect(lx, ly + 2, 12, 12)
            if shape == "rect":
                pygame.draw.rect(surface, color, sym_rect, border_radius=2)
            elif shape == "circle":
                pygame.draw.circle(surface, color, sym_rect.center, 6)
            elif shape == "diamond":
                cx, cy_c = sym_rect.center
                pts = [(cx, cy_c - 6), (cx + 6, cy_c), (cx, cy_c + 6), (cx - 6, cy_c)]
                pygame.draw.polygon(surface, color, pts)
            elif shape == "check":
                pygame.draw.circle(surface, color, sym_rect.center, 6)
                pygame.draw.circle(surface, (20, 25, 35), sym_rect.center, 3)
            elif shape == "debris":
                pygame.draw.polygon(surface, color, [(lx, ly + 12), (lx + 6, ly + 2), (lx + 12, ly + 10)])
            elif shape == "line":
                pygame.draw.line(surface, color, (lx, ly + 8), (lx + 12, ly + 8), 3)
            lbl = font_small.render(name, True, COLOR_TEXT_MUTED)
            surface.blit(lbl, (lx + 18, ly))
            ly += 21
