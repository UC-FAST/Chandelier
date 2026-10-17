"""Video player HUD: play/pause, progress and time."""

from __future__ import annotations

import cv2
import numpy as np

from chandelier.i18n import tr
from chandelier.ui.text_render import get_renderer


def format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


class PlayerHud:
    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        font_height: int = 10,
    ):
        self._width = width
        self._height = height
        self._font_height = font_height
        self._font_scale = cv2.getFontScaleFromHeight(
            cv2.FONT_ITALIC, font_height
        )
        self._renderer = get_renderer()
        self._position = 0.0
        self._duration = 0.0
        self._playing = False
        self._loop = False
        self._visible = False

    @property
    def visible(self) -> bool:
        return self._visible

    def set_state(
        self,
        position: float,
        duration: float,
        playing: bool,
        loop: bool,
    ) -> None:
        self._position = max(0.0, position)
        self._duration = max(0.0, duration)
        self._playing = playing
        self._loop = loop
        self._visible = True

    def decorate(self, frame) -> None:
        if not self._visible:
            return
        bar_height = 24
        y0 = self._height - bar_height
        cv2.rectangle(
            frame, (0, y0), (self._width, self._height), (16, 16, 16), -1
        )
        cv2.line(
            frame, (0, y0), (self._width, y0), (80, 80, 80), 1, cv2.LINE_AA
        )

        icon_x = 8
        icon_cy = y0 + 12
        if self._playing:
            cv2.rectangle(
                frame, (icon_x, icon_cy - 6), (icon_x + 3, icon_cy + 6),
                (255, 255, 255), -1,
            )
            cv2.rectangle(
                frame, (icon_x + 6, icon_cy - 6), (icon_x + 9, icon_cy + 6),
                (255, 255, 255), -1,
            )
        else:
            cv2.fillPoly(
                frame,
                [
                    np.array(
                        [
                            (icon_x, icon_cy - 6),
                            (icon_x, icon_cy + 6),
                            (icon_x + 10, icon_cy),
                        ],
                        dtype="int32",
                    )
                ],
                (255, 255, 255),
            )

        time_text = (
            f"{format_time(self._position)} / {format_time(self._duration)}"
        )
        cv2.putText(
            frame,
            time_text,
            (24, y0 + 16),
            cv2.FONT_ITALIC,
            self._font_scale,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

        track_x1 = 98
        track_x2 = self._width - 10
        track_y = y0 + 12
        cv2.line(
            frame, (track_x1, track_y), (track_x2, track_y),
            (70, 70, 70), 3, cv2.LINE_AA,
        )
        denominator = self._duration if self._duration > 0 else 1.0
        fraction = max(0.0, min(1.0, self._position / denominator))
        knob_x = int(track_x1 + (track_x2 - track_x1) * fraction)
        cv2.line(
            frame, (track_x1, track_y), (knob_x, track_y),
            (0, 190, 255), 3, cv2.LINE_AA,
        )
        cv2.circle(frame, (knob_x, track_y), 4, (0, 190, 255), -1, cv2.LINE_AA)

        if self._loop:
            self._renderer.draw(
                frame,
                tr("LOOP"),
                (self._width - 38, y0 + 9),
                self._font_height,
                (0, 190, 255),
                1,
            )
