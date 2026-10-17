"""Paginated key/value text overlay."""

from __future__ import annotations

from collections.abc import Iterable
from math import ceil

import cv2
import numpy as np

from chandelier.i18n import tr
from chandelier.ui.text_render import get_renderer


class SimpleText:
    def __init__(
        self,
        func_list: list,
        height: int,
        padding: tuple = (5, 5, 5, 5),
        font_height: int = 24,
        color: tuple = (255, 255, 255),
        thickness: float = 1,
    ):
        if height <= 0:
            raise ValueError("height must be positive")
        if any(value < 0 for value in padding):
            raise ValueError("padding values must be non-negative")
        if not func_list:
            raise ValueError("func_list cannot be empty")

        self._font_height = font_height
        self._renderer = get_renderer()
        self._height = height
        self._padding = padding
        self._color = color
        self._thickness = thickness
        self._func_list = tuple(func_list)
        self._index = 0

    @property
    def current_page(self) -> int:
        return self._index

    @property
    def total_pages(self) -> int:
        return len(self._func_list)

    def next_page(self) -> None:
        self._index = min(self._index + 1, self.total_pages - 1)

    def previous_page(self) -> None:
        self._index = max(self._index - 1, 0)

    def set_page(self, page: int) -> None:
        if not 0 <= page < self.total_pages:
            raise IndexError(
                f"page index out of range [0, {self.total_pages - 1}]"
            )
        self._index = page

    def decorate(self, frame: np.ndarray, rotate: int = 0) -> np.ndarray:
        widget = self._func_list[self._index]()
        if not widget:
            return frame

        # Drawing directly avoids a full-frame temporary and an extra blend
        # on the common (unrotated) path.
        sketch = np.zeros_like(frame, dtype=np.uint8) if rotate else frame
        _, top_padding, _, bottom_padding = self._padding
        available_height = self._height - top_padding - bottom_padding
        text_count = len(widget)

        if text_count == 1:
            step = 0
        else:
            total_text_height = text_count * self._font_height
            step = ceil(
                (available_height - total_text_height) / (text_count - 1)
            )

        left_padding = self._padding[0]
        y_position = top_padding
        for key, value in widget.items():
            if isinstance(value, Iterable) and not isinstance(value, str):
                text = key.format(*value)
            else:
                text = key.format(value)

            self._renderer.draw(
                sketch,
                tr(text),
                (left_padding, y_position),
                self._font_height,
                self._color,
                self._thickness,
            )
            y_position += self._font_height + step

        if rotate:
            rotate_times = (-rotate // 90) % 4
            if rotate_times:
                sketch = np.rot90(sketch, rotate_times)
            cv2.addWeighted(sketch, 1, frame, 1, 0, frame)
        return frame
