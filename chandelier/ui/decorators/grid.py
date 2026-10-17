"""Composition guides."""

from __future__ import annotations

import cv2
import numpy as np


class RuleOfThirds:
    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        color: tuple[int, int, int] = (230, 230, 230),
        alpha: float = 0.35,
        show_center: bool = False,
    ):
        self._width = width
        self._height = height
        self._color = color
        self._alpha = max(0.0, min(1.0, alpha))
        self._show_center = show_center
        self._overlay: np.ndarray | None = None

    def _make_overlay(self, frame: np.ndarray) -> np.ndarray:
        overlay = np.zeros_like(frame)
        height, width = frame.shape[:2]
        for fraction in (1 / 3, 2 / 3):
            x = int(width * fraction)
            y = int(height * fraction)
            cv2.line(overlay, (x, 0), (x, height), self._color, 1, cv2.LINE_AA)
            cv2.line(overlay, (0, y), (width, y), self._color, 1, cv2.LINE_AA)
        if self._show_center:
            cx, cy = width // 2, height // 2
            cv2.line(
                overlay, (cx, cy - 6), (cx, cy + 6), self._color, 1,
                cv2.LINE_AA,
            )
            cv2.line(
                overlay, (cx - 6, cy), (cx + 6, cy), self._color, 1,
                cv2.LINE_AA,
            )
        return overlay

    def decorate(self, frame: np.ndarray) -> None:
        if self._overlay is None or self._overlay.shape != frame.shape:
            self._overlay = self._make_overlay(frame)
        cv2.addWeighted(self._overlay, self._alpha, frame, 1.0, 0, frame)


class CenterMark:
    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        color: tuple[int, int, int] = (0, 190, 255),
        size: int = 10,
    ):
        self._width = width
        self._height = height
        self._color = color
        self._size = max(3, size)

    def decorate(self, frame: np.ndarray) -> None:
        cx, cy = self._width // 2, self._height // 2
        for start, end in (
            ((cx - self._size, cy), (cx + self._size, cy)),
            ((cx, cy - self._size), (cx, cy + self._size)),
        ):
            cv2.line(frame, start, end, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.line(frame, start, end, self._color, 1, cv2.LINE_AA)
