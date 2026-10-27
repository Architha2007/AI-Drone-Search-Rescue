"""UI Control widgets: Buttons, Toggles, and Sliders for the Pygame interface."""
from typing import Callable, Optional, Tuple
import pygame

from config import (
    COLOR_BUTTON_ACTIVE,
    COLOR_BUTTON_BG,
    COLOR_BUTTON_HOVER,
    COLOR_PANEL_BORDER,
    COLOR_TEXT_PRIMARY,
)


class Button:
    """Clickable UI button with hover states and callback action."""

    def __init__(
        self,
        rect: Tuple[int, int, int, int],
        text: str,
        callback: Optional[Callable[[], None]] = None,
        bg_color: Tuple[int, int, int] = COLOR_BUTTON_BG,
        hover_color: Tuple[int, int, int] = COLOR_BUTTON_HOVER,
        active_color: Tuple[int, int, int] = COLOR_BUTTON_ACTIVE,
        text_color: Tuple[int, int, int] = COLOR_TEXT_PRIMARY,
        is_toggle: bool = False,
    ):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.callback = callback
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.active_color = active_color
        self.text_color = text_color
        self.is_toggle = is_toggle
        self.is_active = False
        self.is_hovered = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Process mouse movement and click events. Returns True if clicked."""
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
            return False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                if self.is_toggle:
                    self.is_active = not self.is_active
                if self.callback:
                    self.callback()
                return True
        return False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        """Render the button on the target surface."""
        # Determine background color
        if self.is_active:
            fill_color = self.active_color
        elif self.is_hovered:
            fill_color = self.hover_color
        else:
            fill_color = self.bg_color

        # Draw rounded rectangle
        pygame.draw.rect(surface, fill_color, self.rect, border_radius=6)
        pygame.draw.rect(surface, COLOR_PANEL_BORDER, self.rect, width=1, border_radius=6)

        # Render centered text
        text_surf = font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)
