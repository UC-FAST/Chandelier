"""Overexposure zebra stripes."""

from __future__ import annotations

import cv2
import numpy as np


class ZebraOverlay:
    """Diagonal stripes over pixels at or above the exposure threshold.

    The stripe pattern and the solid colour layer are frame-shaped buffers;
    both are cached and rebuilt only when the frame geometry changes, which
    keeps per-frame work to a threshold, an AND and a masked copy.  Colours
    are BGR.
    """

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
        """Clamp and store the luma threshold (0-255)."""

        self._threshold = max(0, min(255, int(threshold)))

    def _stripe_mask(self, frame: np.ndarray) -> np.ndarray:
        """Return the cached checkerboard-ish diagonal stripe mask."""

        height, width = frame.shape[:2]
        if (
            self._stripe_cache is None
            or self._stripe_cache[0] != (height, width)
        ):
            # Integer division by the stripe width creates square cells;
            # alternating cells produce a diagonal pattern when summed.
            ys, xs = np.indices((height, width))
            mask = (
                (xs // self._stripe_width + ys // self._stripe_width) % 2
                == 0
            ).astype(np.uint8) * 255
            self._stripe_cache = ((height, width), mask)
        return self._stripe_cache[1]

    def _solid_color(self, frame: np.ndarray) -> np.ndarray:
        """Return a cached frame-filled layer of the stripe colour."""

        shape = frame.shape
        if self._color_cache is None or self._color_cache[0] != shape:
            # Cache by full shape, not just (h, w): a BGRA source needs an
            # extra opacity byte that an RGB layer does not provide.
            layer = np.empty(shape, dtype=np.uint8)
            color = self._color
            if shape[2] == 4:
                color = (*color, 255)
            layer[:] = color
            self._color_cache = (shape, layer)
        return self._color_cache[1]

    def decorate(self, frame: np.ndarray) -> None:
        """Overlay stripes on overexposed pixels in place."""

        # Work on luma so all channels of a bright pixel are covered, then
        # intersect with the stripe pattern.  copyTo leaves the thresholded
        # pixels that fall between stripes untouched.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        highlight = cv2.compare(gray, self._threshold, cv2.CMP_GE)
        mask = cv2.bitwise_and(highlight, self._stripe_mask(frame))
        if cv2.countNonZero(mask) == 0:
            return
        cv2.copyTo(self._solid_color(frame), mask, frame)
