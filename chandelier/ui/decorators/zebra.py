"""Overexposure zebra stripes."""

from __future__ import annotations

import cv2
import numpy as np


class ZebraOverlay:
    def __init__(
        self,
        threshold: int = 235,
        color: tuple[int, int, int] = (0, 0, 255),
        stripe_width: int = 2,
    ):
        self._threshold = max(0, min(255, threshold))
        self._color = color
        self._stripe_width = max(1, stripe_width)
        self._stripe_cache: tuple[tuple[int, int], np.ndarray] | None = None
        self._color_cache: tuple[tuple[int, ...], np.ndarray] | None = None

    @property
    def threshold(self) -> int:
        return self._threshold

    def set_threshold(self, threshold: int) -> None:
        self._threshold = max(0, min(255, int(threshold)))

    def _stripe_mask(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if (
            self._stripe_cache is None
            or self._stripe_cache[0] != (height, width)
        ):
            ys, xs = np.indices((height, width))
            mask = (
                (xs // self._stripe_width + ys // self._stripe_width) % 2
                == 0
            ).astype(np.uint8) * 255
            self._stripe_cache = ((height, width), mask)
        return self._stripe_cache[1]

    def _solid_color(self, frame: np.ndarray) -> np.ndarray:
        shape = frame.shape
        if self._color_cache is None or self._color_cache[0] != shape:
            layer = np.empty(shape, dtype=np.uint8)
            color = self._color
            if shape[2] == 4:
                color = (*color, 255)
            layer[:] = color
            self._color_cache = (shape, layer)
        return self._color_cache[1]

    def decorate(self, frame: np.ndarray) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        highlight = cv2.compare(gray, self._threshold, cv2.CMP_GE)
        mask = cv2.bitwise_and(highlight, self._stripe_mask(frame))
        if cv2.countNonZero(mask) == 0:
            return
        cv2.copyTo(self._solid_color(frame), mask, frame)
