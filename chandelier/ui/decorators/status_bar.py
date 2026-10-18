"""Compact camera status strip.

The strip is split into evenly spaced items with thin separators, a
translucent panel and an accent edge.  Items may be plain strings or
``(text, role)`` tuples where ``role`` selects a colour from the active menu
theme (``text``, ``dim``, ``accent``, ``success``, ``warning``, ``danger``).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from chandelier.ui.text_render import get_renderer
from chandelier.ui.themes import get_theme

_ROLE_ATTRS = {
    "text": "text",
    "dim": "text_dim",
    "accent": "accent",
    "success": "success",
    "warning": "warning",
    "danger": "danger",
}


@dataclass(frozen=True)
class _Item:
    text: str
    role: str = "text"


class StatusBar:
    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        font_height: int = 10,
        position: str = "top",
        alpha: float = 0.78,
    ):
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if font_height <= 0:
            raise ValueError("font_height must be positive")

        self._width = width
        self._height = height
        self._font_height = font_height
        self._renderer = get_renderer()
        self._position = "bottom" if position == "bottom" else "top"
        self._alpha = max(0.0, min(1.0, float(alpha)))
        self._items: list[_Item] = []

    @property
    def items(self) -> list[str]:
        return [item.text for item in self._items]

    def set_items(self, items: list) -> None:
        parsed: list[_Item] = []
        for item in items:
            if isinstance(item, str):
                text, role = item, "text"
            elif isinstance(item, (tuple, list)) and item:
                text = item[0]
                role = item[1] if len(item) > 1 else "text"
            else:
                continue
            text = str(text).strip() if text is not None else ""
            if text:
                parsed.append(_Item(text, str(role)))
        self._items = parsed

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------
    def _text_width(self, text: str) -> int:
        return self._renderer.measure(text, self._font_height)[0]

    def _fit(self, text: str, max_width: int | None = None) -> str:
        if max_width is None:
            max_width = self._width - 12
        if self._text_width(text) <= max_width:
            return text
        suffix = ".."
        while text:
            text = text[:-1]
            if self._text_width(text + suffix) <= max_width:
                return text + suffix
        return ""

    def _layout(self, width: int) -> list[tuple[_Item, int]]:
        """Return the items that fit and their drawn widths."""

        items = list(self._items)
        widths = [self._text_width(item.text) for item in items]
        available = max(1, width - 14)
        gap = 8

        while (
            len(items) > 1
            and sum(widths) + gap * (len(items) - 1) > available
        ):
            items.pop()
            widths.pop()

        if not items:
            return []

        if widths[0] > available:
            items[0] = _Item(
                self._fit(items[0].text, available), items[0].role
            )
            widths[0] = self._text_width(items[0].text)
        return list(zip(items, widths))

    def _role_color(self, theme, role: str):
        if role == "text" and theme.header_text is not None:
            return theme.header_text
        if role == "dim" and theme.header_text_dim is not None:
            return theme.header_text_dim
        if role == "accent" and theme.header_text is not None:
            return theme.header_text_dim or theme.header_text
        return getattr(theme, _ROLE_ATTRS.get(role, "text"), theme.text)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def decorate(self, frame, theme_id: str = "dark_pro") -> None:
        if not self._items or frame is None or frame.size == 0:
            return
        if frame.ndim != 3 or frame.shape[2] < 3:
            return

        theme = get_theme(theme_id)
        frame_height, frame_width = frame.shape[:2]
        width = min(self._width, frame_width)
        bar_height = max(20, self._font_height + 12)
        if self._position == "bottom":
            y0 = max(0, frame_height - bar_height)
        else:
            y0 = 0
        y1 = min(frame_height, y0 + bar_height)
        if y1 <= y0:
            return

        # Translucent header surface, blended in place over the region.
        region = frame[y0:y1, :width]
        surface = np.empty_like(region)
        color = theme.header_bg
        if region.shape[2] == 4:
            color = (*color, 255)
        surface[:] = color
        cv2.addWeighted(
            surface, self._alpha, region, 1.0 - self._alpha, 0, region
        )

        accent_y = y1 - 2 if self._position == "top" else y0
        cv2.line(
            frame,
            (0, accent_y),
            (width - 1, accent_y),
            theme.accent,
            2,
            cv2.LINE_AA,
        )

        layout = self._layout(width)
        if not layout:
            return

        total_width = sum(item_width for _, item_width in layout)
        gap = 8
        if len(layout) > 1:
            remaining = width - 14 - total_width
            gap = max(
                4, min(18, remaining // (len(layout) - 1))
            )
        block_width = total_width + gap * (len(layout) - 1)
        x = max(7, (width - block_width) // 2)
        baseline = y0 + (bar_height + self._font_height) // 2 - 1

        for index, (item, item_width) in enumerate(layout):
            self._renderer.draw(
                frame,
                item.text,
                (x, baseline),
                self._font_height,
                self._role_color(theme, item.role),
                1,
            )
            if index < len(layout) - 1:
                separator_x = x + item_width + gap // 2
                cv2.line(
                    frame,
                    (separator_x, y0 + 5),
                    (separator_x, y1 - 6),
                    theme.border,
                    1,
                    cv2.LINE_AA,
                )
            x += item_width + gap
