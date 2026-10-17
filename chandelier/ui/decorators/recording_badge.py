"""Recording indicator and self-timer countdown."""

from __future__ import annotations

import math
import time

import cv2

from chandelier.ui.decorators.player_hud import format_time


class RecordingBadge:
    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        font_height: int = 12,
    ):
        self._width = width
        self._height = height
        self._font_height = font_height
        self._font_scale = cv2.getFontScaleFromHeight(
            cv2.FONT_ITALIC, font_height
        )

    def decorate(self, frame, elapsed: float | None) -> None:
        if elapsed is None:
            return
        blink = int(time.monotonic() * 2) % 2 == 0
        text = f"REC {format_time(elapsed)}"
        (text_width, _), _ = cv2.getTextSize(
            text, cv2.FONT_ITALIC, self._font_scale, 1
        )
        y = 8
        x = max(8, self._width - text_width - 22)
        if blink:
            cv2.circle(frame, (x, y + 6), 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            text,
            (x + 8, y + 11),
            cv2.FONT_ITALIC,
            self._font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


class Countdown:
    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        font_height: int = 36,
    ):
        self._width = width
        self._height = height
        self._font_height = font_height
        self._font_scale = cv2.getFontScaleFromHeight(
            cv2.FONT_HERSHEY_DUPLEX, font_height
        )

    def decorate(self, frame, remaining: float | None) -> None:
        if remaining is None or remaining <= 0:
            return
        text = str(max(1, int(math.ceil(remaining))))
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_DUPLEX, self._font_scale, 2
        )
        cx = self._width // 2
        cy = self._height // 2
        radius = max(text_width, text_height) // 2 + 12
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), radius, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(
            frame,
            text,
            (cx - text_width // 2, cy + text_height // 2),
            cv2.FONT_HERSHEY_DUPLEX,
            self._font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
