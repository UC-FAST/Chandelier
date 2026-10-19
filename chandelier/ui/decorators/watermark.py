"""Camera watermark burned into saved pictures."""

from __future__ import annotations

import cv2
import numpy as np


class Watermark:
    """Camera badge burned into saved stills.

    Mark geometry is scaled from the configured image size (the divisors
    below match the original 128 px preview layout) so the badge keeps its
    proportions on larger captures.  The mark itself is white and drawn on
    a scratch layer before being added to the frame.
    """

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
        """Add the badge near the bottom-left corner of ``frame``."""

        # Proportional insets keep the badge in the same visual spot no
        # matter what resolution the camera captured.
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
        # Frame weight 1.0 makes this additive: white strokes brighten the
        # photo while the black background leaves it unchanged.
        cv2.addWeighted(sketch, 0.6, frame, 1, 0, frame)
