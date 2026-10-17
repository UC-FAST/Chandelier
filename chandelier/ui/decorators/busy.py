"""Small corner indicator shown while the camera is busy."""

from __future__ import annotations

import cv2


class Busy:
    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        color: tuple = (0, 0, 255),
    ):
        self._width = width
        self._height = height
        self._color = color

    def decorate(self, frame, rotate: int = 0) -> None:
        width = self._width
        height = self._height
        if rotate == 90:
            top_left = (width - width // 16, height - height // 16)
            bottom_right = (width, height)
        elif rotate == 180:
            top_left = (0, height - height // 16)
            bottom_right = (width // 16, height)
        elif rotate == 270:
            top_left = (0, 0)
            bottom_right = (width // 16, height // 16)
        else:
            top_left = (width - width // 16, 0)
            bottom_right = (width, height // 16)

        cv2.rectangle(
            frame, top_left, bottom_right, color=self._color, thickness=-1
        )
