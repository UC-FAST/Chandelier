"""Camera watermark burned into saved pictures."""

from __future__ import annotations

import cv2
import numpy as np


class Watermark:
    def __init__(
        self, width: int = 128, height: int = 128, font_height: int = 0
    ):
        self._width = width
        self._height = height
        self._font_height = font_height or height // 35
        self._font_scale = cv2.getFontScaleFromHeight(
            cv2.FONT_HERSHEY_DUPLEX, self._font_height
        )

    def decorate(self, frame: np.ndarray) -> None:
        center = (
            int(self._width / 20.6) + self._font_height // 2,
            self._height - (int(self._height / 39.3) + self._font_height // 2),
        )
        sketch = np.zeros(frame.shape, frame.dtype)
        cv2.circle(
            sketch,
            center,
            int(self._font_height / 2.8),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.circle(
            sketch,
            center,
            int(self._font_height / 1.8),
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            sketch,
            "CEEE",
            (
                center[0] + self._font_height // 2 + self._width // 70,
                center[1] + self._font_height // 2,
            ),
            cv2.FONT_HERSHEY_DUPLEX,
            self._font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.addWeighted(sketch, 0.6, frame, 1, 0, frame)
