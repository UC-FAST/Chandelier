"""Red warning card used for empty galleries and similar states."""

from __future__ import annotations

import cv2
import numpy as np

from chandelier.i18n import tr
from chandelier.ui.decorators.colors import Colors
from chandelier.ui.text_render import get_renderer


class Warning:
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
        # A fresh sketch per call avoids text accumulating across calls.
        sketch = np.zeros((self._width, self._height, 3), np.uint8)
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

        return np.rot90(sketch, -rotate // 90)
