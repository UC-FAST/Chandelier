"""Small corner indicator shown while the camera is busy."""

from __future__ import annotations

import cv2


class Busy:
    """Small filled square in a screen corner while the camera is working.

    The corner is chosen from ``rotate`` so the indicator stays in the same
    visual corner after the controller rotates the preview.  ``color`` is a
    BGR tuple; the rectangle is drawn directly into the frame.
    """

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
        """Fill the rotated top-right corner of the logical image."""

        width = self._width
        height = self._height
        # Rotating the content clockwise moves the top-right corner around
        # the screen; pick the matching physical corner here.
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
