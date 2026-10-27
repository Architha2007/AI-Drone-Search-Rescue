"""Renderer for grid map, drone animations, obstacles, paths, heatmap, and sensor FOV."""
import math
import os
from typing import List, Optional, Set, Tuple
import pygame

from config import (
    CELL_SIZE,
    COLOR_BASE,
    COLOR_BATTERY_HIGH,
    COLOR_BATTERY_LOW,
    COLOR_BATTERY_MED,
    COLOR_BG,
    COLOR_DRONE,
    COLOR_DRONE_GLOW,
    COLOR_EXPLORED,
    COLOR_GRID_LINE,
    COLOR_OBSTACLE_BUILDING,
    COLOR_OBSTACLE_DEBRIS,
    COLOR_PLANNED_PATH,
    COLOR_PROB_HIGH,
    COLOR_PROB_LOW,
    COLOR_PROB_MED,
    COLOR_PROB_ZERO,
    COLOR_SUCCESS,
    COLOR_SURVIVOR_DISCOVERED,
    COLOR_SURVIVOR_RESCUED,
    COLOR_TEXT_ACCENT,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_PRIMARY,
    COLOR_TRAIL,
    COLOR_UNEXPLORED,
    COLOR_WARNING,
    GRID_COLS,
    GRID_ROWS,
    HEATMAP_ALPHA,
    MAP_WIDTH,
    SENSOR_RANGE,
    WINDOW_HEIGHT,
)
from detection.yolo_detector import BoundingBox, YOLODetector
from simulation.environment import CellType, DisasterEnvironment
from simulation.mission import MissionController, MissionState


def _lerp_color(
    c1: Tuple[int, int, int],
    c2: Tuple[int, int, int],
    t: float,
) -> Tuple[int, int, int]:
    """Linear interpolate between two RGB colors at factor t in [0, 1]."""
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def _prob_to_color(prob: float, max_prob: float) -> Tuple[int, int, int]:
    """Map a probability value to a heatmap color (blue → amber → hot-red)."""
    if max_prob <= 0.0:
        return COLOR_PROB_ZERO
    t = min(1.0, prob / max_prob)
    if t < 0.5:
        return _lerp_color(COLOR_PROB_LOW, COLOR_PROB_MED, t * 2.0)
    else:
        return _lerp_color(COLOR_PROB_MED, COLOR_PROB_HIGH, (t - 0.5) * 2.0)


class MapRenderer:
    """Renders the 2D disaster environment, drone animations, heatmap overlays, and sensor FOV."""

    def __init__(self, mission: MissionController):
        self.mission = mission
        self.animation_frame: int = 0
        self.yolo_detector: Optional[YOLODetector] = None
        self.yolo_annotated_surface: Optional[pygame.Surface] = None
        self.yolo_detections: List[BoundingBox] = []

        # Search-mode state (injected from main app)
        self.search_mission = None          # SearchMission instance or None
        self.show_heatmap: bool = False     # Toggle for probability overlay
        self.debug_show_target: bool = False  # Reveal hidden target (debug mode)

    def grid_to_screen_center(self, x: int, y: int) -> Tuple[int, int]:
        """Convert grid cell coordinates (x, y) to screen pixel center (px, py)."""
        return (
            int(x * CELL_SIZE + CELL_SIZE / 2),
            int(y * CELL_SIZE + CELL_SIZE / 2),
        )

    def screen_to_grid(self, px: int, py: int) -> Optional[Tuple[int, int]]:
        """Convert screen pixel coordinates to grid cell (x, y) if inside map."""
        if 0 <= px < MAP_WIDTH and 0 <= py < WINDOW_HEIGHT:
            return (px // CELL_SIZE, py // CELL_SIZE)
        return None

    # ------------------------------------------------------------------
    # Base terrain rendering
    # ------------------------------------------------------------------

    def draw_grid_cells(self, surface: pygame.Surface) -> None:
        """Draw background, unexplored/explored fog of war, and grid lines."""
        env = self._active_env()

        for y in range(env.rows):
            for x in range(env.cols):
                cell_rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pos = (x, y)

                if pos in env.explored_cells:
                    fill_color = COLOR_EXPLORED
                else:
                    fill_color = COLOR_UNEXPLORED

                pygame.draw.rect(surface, fill_color, cell_rect)
                pygame.draw.rect(surface, COLOR_GRID_LINE, cell_rect, 1)

    def draw_obstacles(self, surface: pygame.Surface) -> None:
        """Render buildings and debris with textured patterns."""
        env = self._active_env()

        for (x, y), cell_type in env.obstacles.items():
            cx = x * CELL_SIZE
            cy = y * CELL_SIZE
            rect = pygame.Rect(cx + 2, cy + 2, CELL_SIZE - 4, CELL_SIZE - 4)

            if cell_type == CellType.BUILDING:
                pygame.draw.rect(surface, COLOR_OBSTACLE_BUILDING, rect, border_radius=4)
                inner = rect.inflate(-6, -6)
                pygame.draw.rect(surface, (60, 68, 86), inner, border_radius=2)
                pygame.draw.line(surface, (45, 52, 68), inner.topleft, inner.bottomright, 1)
                pygame.draw.line(surface, (45, 52, 68), inner.topright, inner.bottomleft, 1)
            else:
                pygame.draw.rect(surface, (70, 30, 30), rect, border_radius=4)
                pts1 = [(cx + 4, cy + 18), (cx + 12, cy + 8), (cx + 18, cy + 20), (cx + 8, cy + 26)]
                pts2 = [(cx + 16, cy + 12), (cx + 28, cy + 6), (cx + 24, cy + 22)]
                pygame.draw.polygon(surface, COLOR_OBSTACLE_DEBRIS, pts1)
                pygame.draw.polygon(surface, (230, 80, 60), pts2)

    def draw_base_station(self, surface: pygame.Surface, font_small: pygame.font.Font) -> None:
        """Render base station pad and 'H' helipad symbol."""
        env = self._active_env()
        bx, by = env.base_pos
        cx = bx * CELL_SIZE
        cy = by * CELL_SIZE
        pad_rect = pygame.Rect(cx + 2, cy + 2, CELL_SIZE - 4, CELL_SIZE - 4)

        pygame.draw.rect(surface, (25, 75, 45), pad_rect, border_radius=4)
        pygame.draw.rect(surface, COLOR_BASE, pad_rect, width=2, border_radius=4)

        center = pad_rect.center
        pygame.draw.circle(surface, COLOR_BASE, center, (CELL_SIZE // 2) - 6, 2)
        h_surf = font_small.render("H", True, COLOR_BASE)
        surface.blit(h_surf, h_surf.get_rect(center=center))

    def draw_survivors(self, surface: pygame.Surface) -> None:
        """Render discovered and rescued survivors (coverage mode only)."""
        if self.search_mission is not None:
            return  # In search mode, survivors are hidden
        detector = self.mission.detector
        pulse = abs(math.sin(self.animation_frame * 0.1))

        for s_pos in detector.discovered_survivors:
            if s_pos in detector.reached_survivors:
                continue
            scx, scy = self.grid_to_screen_center(*s_pos)
            pulse_radius = int(8 + (pulse * 6))
            beacon_surf = pygame.Surface((pulse_radius * 2 + 4, pulse_radius * 2 + 4), pygame.SRCALPHA)
            alpha = int(180 * (1.0 - (pulse * 0.5)))
            pygame.draw.circle(
                beacon_surf,
                (*COLOR_SURVIVOR_DISCOVERED, alpha),
                (pulse_radius + 2, pulse_radius + 2),
                pulse_radius,
                2,
            )
            surface.blit(beacon_surf, (scx - pulse_radius - 2, scy - pulse_radius - 2))
            d_size = 7
            pts = [(scx, scy - d_size), (scx + d_size, scy), (scx, scy + d_size), (scx - d_size, scy)]
            pygame.draw.polygon(surface, COLOR_SURVIVOR_DISCOVERED, pts)
            pygame.draw.circle(surface, (255, 255, 255), (scx, scy), 2)

        for s_pos in detector.reached_survivors:
            scx, scy = self.grid_to_screen_center(*s_pos)
            pygame.draw.circle(surface, COLOR_SURVIVOR_RESCUED, (scx, scy), 8)
            pygame.draw.line(surface, (255, 255, 255), (scx - 4, scy), (scx + 4, scy), 2)
            pygame.draw.line(surface, (255, 255, 255), (scx, scy - 4), (scx, scy + 4), 2)

    # ------------------------------------------------------------------
    # Path drawing
    # ------------------------------------------------------------------

    def draw_paths(self, surface: pygame.Surface) -> None:
        """Draw traversed flight history trail and upcoming planned route."""
        sm = self.search_mission
        if sm is not None:
            # Search mode: use search_mission drone
            trail = sm.drone.path_history
            if len(trail) > 1:
                points = [self.grid_to_screen_center(*p) for p in trail]
                pygame.draw.lines(surface, (*COLOR_TRAIL, 120), False, points, 2)

            plan = sm.planned_path
            if plan:
                full_plan = [sm.drone.position] + plan
                points = [self.grid_to_screen_center(*p) for p in full_plan]
                pygame.draw.lines(surface, COLOR_PLANNED_PATH, False, points, 3)
                dest_cx, dest_cy = points[-1]
                pygame.draw.circle(surface, COLOR_PLANNED_PATH, (dest_cx, dest_cy), 5)
                pygame.draw.circle(surface, (20, 24, 32), (dest_cx, dest_cy), 2)
        else:
            # Coverage mode
            trail = self.mission.drone.path_history
            if len(trail) > 1:
                points = [self.grid_to_screen_center(*p) for p in trail]
                pygame.draw.lines(surface, (*COLOR_TRAIL, 120), False, points, 2)

            plan = self.mission.planned_path
            if plan:
                full_plan = [self.mission.drone.position] + plan
                points = [self.grid_to_screen_center(*p) for p in full_plan]
                pygame.draw.lines(surface, COLOR_PLANNED_PATH, False, points, 3)
                dest_cx, dest_cy = points[-1]
                pygame.draw.circle(surface, COLOR_PLANNED_PATH, (dest_cx, dest_cy), 5)
                pygame.draw.circle(surface, (20, 24, 32), (dest_cx, dest_cy), 2)

    # ------------------------------------------------------------------
    # Drone rendering
    # ------------------------------------------------------------------

    def draw_drone(self, surface: pygame.Surface) -> None:
        """Render animated quadcopter drone with rotating rotors and sensor halo."""
        sm = self.search_mission
        pos = sm.drone.position if sm is not None else self.mission.drone.position
        dx, dy = pos
        dcx, dcy = self.grid_to_screen_center(dx, dy)

        # Sensor Range Halo (FOV)
        fov_pixels = int((SENSOR_RANGE + 0.5) * CELL_SIZE)
        fov_surf = pygame.Surface((fov_pixels * 2, fov_pixels * 2), pygame.SRCALPHA)
        pygame.draw.circle(fov_surf, (0, 212, 255, 30), (fov_pixels, fov_pixels), fov_pixels)
        pygame.draw.circle(fov_surf, (0, 212, 255, 90), (fov_pixels, fov_pixels), fov_pixels, 1)
        surface.blit(fov_surf, (dcx - fov_pixels, dcy - fov_pixels))

        # Quadcopter Frame
        arm_len = 11
        angles = [45, 135, 225, 315]
        rotor_positions = []
        for deg in angles:
            rad = math.radians(deg)
            rx = dcx + int(arm_len * math.cos(rad))
            ry = dcy + int(arm_len * math.sin(rad))
            rotor_positions.append((rx, ry))
            pygame.draw.line(surface, (120, 140, 160), (dcx, dcy), (rx, ry), 2)

        # Spinning Propellers
        blade_angle = (self.animation_frame * 35) % 360
        blade_len = 5
        for rx, ry in rotor_positions:
            rad = math.radians(blade_angle)
            bx1 = rx + int(blade_len * math.cos(rad))
            by1 = ry + int(blade_len * math.sin(rad))
            bx2 = rx - int(blade_len * math.cos(rad))
            by2 = ry - int(blade_len * math.sin(rad))
            pygame.draw.line(surface, COLOR_DRONE, (bx1, by1), (bx2, by2), 2)
            pygame.draw.circle(surface, (255, 255, 255), (rx, ry), 2)

        # Central Body
        pygame.draw.circle(surface, (20, 30, 45), (dcx, dcy), 6)
        pygame.draw.circle(surface, COLOR_DRONE, (dcx, dcy), 5)
        led_color = (0, 255, 128) if (self.animation_frame // 15) % 2 == 0 else (0, 180, 255)
        pygame.draw.circle(surface, led_color, (dcx, dcy), 2)

    # ------------------------------------------------------------------
    # Probability heatmap overlay
    # ------------------------------------------------------------------

    def draw_probability_heatmap(self, surface: pygame.Surface) -> None:
        """
        Draw a semi-transparent heatmap showing P(target at cell) over the grid.
        Only drawn when show_heatmap is True and a search_mission is active.
        """
        if not self.show_heatmap or self.search_mission is None:
            return

        beliefs = self.search_mission.belief_map.get_all_beliefs()
        if not beliefs:
            return

        max_prob = max(beliefs.values()) if beliefs else 1.0
        if max_prob <= 0.0:
            return

        env = self._active_env()
        overlay = pygame.Surface((MAP_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)

        for (x, y), prob in beliefs.items():
            if not env.is_traversable(x, y):
                continue
            if prob < 1e-8:
                continue
            color = _prob_to_color(prob, max_prob)
            alpha = int(HEATMAP_ALPHA * min(1.0, prob / max_prob))
            cell_rect = pygame.Rect(x * CELL_SIZE + 1, y * CELL_SIZE + 1, CELL_SIZE - 2, CELL_SIZE - 2)
            pygame.draw.rect(overlay, (*color, alpha), cell_rect)

        surface.blit(overlay, (0, 0))

    def draw_probability_legend(self, surface: pygame.Surface, font_small: pygame.font.Font) -> None:
        """Draw a horizontal gradient legend for the probability heatmap."""
        if not self.show_heatmap or self.search_mission is None:
            return

        lx, ly = 6, WINDOW_HEIGHT - 28
        lw, lh = 120, 12

        # Gradient bar
        for i in range(lw):
            t = i / lw
            if t < 0.5:
                c = _lerp_color(COLOR_PROB_LOW, COLOR_PROB_MED, t * 2)
            else:
                c = _lerp_color(COLOR_PROB_MED, COLOR_PROB_HIGH, (t - 0.5) * 2)
            pygame.draw.line(surface, c, (lx + i, ly), (lx + i, ly + lh))

        # Border
        pygame.draw.rect(surface, (80, 100, 120), pygame.Rect(lx, ly, lw, lh), 1)

        # Labels
        lo_surf = font_small.render("Low", True, COLOR_TEXT_MUTED)
        hi_surf = font_small.render("High Prob", True, COLOR_TEXT_MUTED)
        surface.blit(lo_surf, (lx, ly + lh + 2))
        surface.blit(hi_surf, (lx + lw - hi_surf.get_width(), ly + lh + 2))

    def draw_debug_target(self, surface: pygame.Surface, font_small: pygame.font.Font) -> None:
        """Draw the hidden target location in debug mode only."""
        if not self.debug_show_target or self.search_mission is None:
            return

        target = self.search_mission.belief_map.debug_hidden_target
        if target is None:
            return

        tx, ty = self.grid_to_screen_center(*target)
        pulse = abs(math.sin(self.animation_frame * 0.08))
        r = int(10 + pulse * 5)

        surf = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(surf, (255, 50, 200, 180), (r + 2, r + 2), r, 2)
        surface.blit(surf, (tx - r - 2, ty - r - 2))

        # Star marker
        pygame.draw.line(surface, (255, 50, 200), (tx - 6, ty), (tx + 6, ty), 2)
        pygame.draw.line(surface, (255, 50, 200), (tx, ty - 6), (tx, ty + 6), 2)

        lbl = font_small.render("TARGET", True, (255, 80, 220))
        surface.blit(lbl, (tx + 8, ty - 8))

    # ------------------------------------------------------------------
    # Banner overlays
    # ------------------------------------------------------------------

    def draw_mission_completion_banner(
        self, surface: pygame.Surface,
        font_main: pygame.font.Font,
        font_small: pygame.font.Font,
    ) -> None:
        """Render banner overlay for coverage mission completion."""
        if self.search_mission is not None:
            return  # Search mode has its own banners via draw_search_banner
        if self.mission.state != MissionState.COMPLETED:
            return

        self._draw_banner(
            surface, font_main, font_small,
            title="COVERAGE MISSION COMPLETED",
            color=COLOR_SUCCESS,
            stats=(
                f"Coverage: {self.mission.coverage_percentage:.1f}%  |  "
                f"Survivors Rescued: {self.mission.survivors_reached}  |  "
                f"Distance: {self.mission.drone.distance_traveled} steps"
            ),
        )

    def draw_search_banner(
        self, surface: pygame.Surface,
        font_main: pygame.font.Font,
        font_small: pygame.font.Font,
    ) -> None:
        """Draw search-mode result banner."""
        from simulation.search_mission import SearchMissionState
        sm = self.search_mission
        if sm is None:
            return

        if sm.state == SearchMissionState.TARGET_FOUND:
            self._draw_banner(
                surface, font_main, font_small,
                title="TARGET DETECTED!",
                color=COLOR_SUCCESS,
                stats=(
                    f"Steps: {sm.step_count}  |  "
                    f"Battery: {sm.battery_percentage:.1f}%  |  "
                    f"Distance: {sm.distance_traveled}"
                ),
            )
        elif sm.state == SearchMissionState.BATTERY_DEPLETED:
            self._draw_banner(
                surface, font_main, font_small,
                title="BATTERY DEPLETED — MISSION ENDED",
                color=COLOR_BATTERY_LOW,
                stats=(
                    f"Steps: {sm.step_count}  |  "
                    f"Coverage: {sm.coverage_percentage:.1f}%  |  "
                    f"Target: {'FOUND' if sm.detection_confirmed else 'NOT FOUND'}"
                ),
            )
        elif sm.state == SearchMissionState.COMPLETED:
            self._draw_banner(
                surface, font_main, font_small,
                title="SEARCH AREA EXHAUSTED",
                color=COLOR_WARNING,
                stats=(
                    f"Steps: {sm.step_count}  |  "
                    f"Target: {'FOUND' if sm.detection_confirmed else 'NOT FOUND'}"
                ),
            )

    def _draw_banner(
        self, surface, font_main, font_small, title, color, stats
    ) -> None:
        banner_w = 480
        banner_h = 75
        bx = (MAP_WIDTH - banner_w) // 2
        by = 20
        banner_rect = pygame.Rect(bx, by, banner_w, banner_h)
        overlay = pygame.Surface((banner_w, banner_h), pygame.SRCALPHA)
        overlay.fill((16, 24, 38, 230))
        surface.blit(overlay, (bx, by))
        pygame.draw.rect(surface, color, banner_rect, width=2, border_radius=8)
        t1 = font_main.render(title, True, color)
        surface.blit(t1, t1.get_rect(center=(bx + banner_w // 2, by + 22)))
        t2 = font_small.render(stats, True, COLOR_TEXT_PRIMARY)
        surface.blit(t2, t2.get_rect(center=(bx + banner_w // 2, by + 50)))

    # ------------------------------------------------------------------
    # YOLO feed
    # ------------------------------------------------------------------

    def draw_yolo_feed_overlay(self, surface: pygame.Surface, font_small: pygame.font.Font) -> None:
        """Render YOLO camera picture-in-picture feed."""
        if self.yolo_annotated_surface is None:
            return

        w, h = 240, 180
        x = MAP_WIDTH - w - 12
        y = WINDOW_HEIGHT - h - 12
        frame_rect = pygame.Rect(x, y, w, h)

        pygame.draw.rect(surface, (15, 20, 28), frame_rect, border_radius=6)
        scaled_img = pygame.transform.scale(self.yolo_annotated_surface, (w - 4, h - 26))
        surface.blit(scaled_img, (x + 2, y + 22))
        pygame.draw.rect(surface, (0, 212, 255), frame_rect, width=2, border_radius=6)
        title = font_small.render(f"YOLO FEED ({len(self.yolo_detections)} Person Detected)", True, (0, 212, 255))
        surface.blit(title, (x + 6, y + 4))

    # ------------------------------------------------------------------
    # Main render entry
    # ------------------------------------------------------------------

    def render(
        self,
        surface: pygame.Surface,
        font_main: pygame.font.Font,
        font_small: pygame.font.Font,
        show_yolo: bool = False,
    ) -> None:
        """Complete frame render pipeline."""
        self.animation_frame += 1

        self.draw_grid_cells(surface)
        self.draw_probability_heatmap(surface)  # Heatmap before obstacles
        self.draw_obstacles(surface)
        self.draw_base_station(surface, font_small)
        self.draw_paths(surface)
        self.draw_survivors(surface)
        self.draw_debug_target(surface, font_small)
        self.draw_drone(surface)
        self.draw_mission_completion_banner(surface, font_main, font_small)
        self.draw_search_banner(surface, font_main, font_small)
        self.draw_probability_legend(surface, font_small)

        if show_yolo:
            self.draw_yolo_feed_overlay(surface, font_small)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _active_env(self) -> DisasterEnvironment:
        """Return the active environment (search mode or coverage mode)."""
        if self.search_mission is not None:
            return self.search_mission.env
        return self.mission.env
