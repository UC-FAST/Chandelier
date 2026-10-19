"""Red warning card used for empty galleries and similar states."""

from __future__ import annotations

import cv2
import numpy as np

from chandelier.i18n import tr
from chandelier.ui.decorators.colors import Colors
from chandelier.ui.text_render import get_renderer


class Warning:
    """Small red card with wrapped white text.

    Unlike most decorators this returns a new image instead of drawing in
    place: callers use the result as a standalone screen (empty gallery,
    missing media).  Words are split on whitespace and stacked; colours are
    BGR.
    """

    def __init__(
        self, width: int = 128, height: int = 128, font_height: int = 12
    ):
        self._width = width
        self._height = height
        self._font_height = font_height
        self._scale = cv2.getFontScaleFromHeight(
            cv2.FONT_ITALIC, font_height
        )
        self._renderer = get_renderer()

    def decorate(self, text: str, rotate: int = 0) -> np.ndarray:
        """Render ``text`` and return the rotated card image."""

        # A fresh sketch per call avoids text accumulating across calls.
        sketch = np.zeros((self._width, self._height, 3), np.uint8)
        # Two inset rectangles: dark red frame, dark blue body.
        cv2.rectangle(
            sketch, (0, 0), (self._width, self._height),
            Colors.darkred.value, -1,
        )
        cv2.rectangle(
            sketch,
            (self._width // 10, self._height // 10),
            (int(self._width * 0.9), int(self._height * 0.9)),
            Colors.darkblue.value,
            -1,
        )

        # Spread the words over 80 % of the card height; ``space`` is the
        # extra pixels inserted between consecutive baselines.
        text_list = text.split()
        space_count = len(text_list) - 1
        space_total = self._height * 0.8 - len(text_list) * self._font_height
        space = int(space_total // space_count) if space_count else 0

        for index, word in enumerate(text_list):
            self._renderer.draw(
                sketch,
                tr(word),
                (
                    self._width // 10,
                    (index + 1) * self._font_height
                    + index * space
                    + int(self._height // 10),
                ),
                self._font_height,
                (255, 255, 255),
                1,
            )

        # Negative count because np.rot90 rotates counter-clockwise while
        # the preview rotation is clockwise.
        return np.rot90(sketch, -rotate // 90)
