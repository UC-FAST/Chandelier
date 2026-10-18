"""Sound-source marker overlays.

Each tracked source keeps a small time window of recent detections.  A marker
is drawn at the median position (robust against a single-frame outlier), with
a short trail showing recent movement, and disappears after ``ttl`` seconds
without a new detection.  A source whose centre falls outside the frame is
shown as an arrow on the screen edge, pointing towards the off-screen
direction.

Coordinates are in the frame's pixel space (the controller draws this after
the camera preview has been scaled to the LCD resolution).
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

# High-contrast BGR colours: red, green, cyan, orange, magenta, yellow.
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (0, 0, 255),
    (0, 255, 0),
    (255, 255, 0),
    (0, 165, 255),
    (255, 0, 255),
    (0, 255, 255),
)


@dataclass
class _Sample:
    x: int
    y: int
    text: str
    timestamp: float


class MarkerBox:
    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        color: tuple[int, int, int] | None = (0, 0, 255),
        size: tuple[int, int] = (24, 24),
        font_height: int = 12,
        cache: int = 5,
        ttl: float | None = 3.0,
        trail: int = 3,
    ):
        self._width = width
        self._height = height
        self._box_size = size
        self._font_height = font_height
        self._font_scale = cv2.getFontScaleFromHeight(
            cv2.FONT_ITALIC, self._font_height
        )
        self._ttl = ttl
        self._trail = max(0, trail)
        self._cache = max(1, cache)
        self._palette = self._build_palette(color)
        self._tracks: dict[int, deque[_Sample]] = {}

    @staticmethod
    def _build_palette(
        color: tuple[int, int, int] | None,
    ) -> tuple[tuple[int, int, int], ...]:
        if color is None:
            return _PALETTE
        return (color,) + tuple(item for item in _PALETTE if item != color)

    # ------------------------------------------------------------------
    # Sample management
    # ------------------------------------------------------------------
    def clear(self) -> None:
        self._tracks.clear()

    @property
    def active(self) -> bool:
        return bool(self._tracks)

    @property
    def track_count(self) -> int:
        return len(self._tracks)

    def update(
        self,
        track_id: int,
        coordinate: tuple[int, int] | None,
        text: str = "",
        timestamp: float | None = None,
    ) -> None:
        if coordinate is None:
            return
        # Keep the raw position: the drawing code needs to know whether the
        # source is still on screen so it can render an edge arrow instead.
        x = int(coordinate[0])
        y = int(coordinate[1])
        samples = self._tracks.setdefault(
            track_id, deque(maxlen=self._cache)
        )
        samples.append(
            _Sample(
                x=x,
                y=y,
                text=text,
                timestamp=time.monotonic() if timestamp is None else timestamp,
            )
        )

    def update_many(
        self,
        targets: Iterable[tuple[int, tuple[int, int] | None, str]],
        timestamp: float | None = None,
    ) -> None:
        for track_id, coordinate, text in targets:
            self.update(track_id, coordinate, text, timestamp)

    def _prune(self, now: float) -> None:
        if self._ttl is None:
            return
        for track_id in list(self._tracks):
            samples = self._tracks[track_id]
            while samples and now - samples[0].timestamp > self._ttl:
                samples.popleft()
            if not samples:
                del self._tracks[track_id]

    @staticmethod
    def _smoothed_position(
        samples: deque[_Sample],
    ) -> tuple[int, int]:
        xs = sorted(sample.x for sample in samples)
        ys = sorted(sample.y for sample in samples)
        return xs[len(xs) // 2], ys[len(ys) // 2]

    def _track_color(self, track_id: int) -> tuple[int, int, int]:
        return self._palette[track_id % len(self._palette)]

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def _draw_trail(
        self, frame, samples: deque[_Sample], color: tuple[int, int, int]
    ) -> None:
        if self._trail < 2 or len(samples) < 2:
            return
        points = [
            (sample.x, sample.y)
            for sample in list(samples)[-self._trail :]
        ]
        for start, end in zip(points, points[1:]):
            cv2.line(frame, start, end, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.line(frame, start, end, color, 1, cv2.LINE_AA)

    def _draw_label(
        self, frame, center_x: int, center_y: int, text: str
    ) -> None:
        if not text:
            return

        thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_ITALIC, self._font_scale, thickness
        )
        padding = 3
        block_height = text_height + 2 * padding
        half_height = self._box_size[1] // 2

        # Prefer above the box; fall back below, and finally clamp inside the
        # frame so the label never becomes an inverted rectangle.
        above_y = center_y - half_height - block_height
        if above_y >= 0:
            background_y1 = above_y
        else:
            below_y = center_y + half_height
            background_y1 = min(
                max(0, below_y), max(0, self._height - block_height - 1)
            )
        background_y2 = min(
            self._height - 1, background_y1 + block_height
        )
        baseline = background_y1 + text_height + padding

        background_x1 = max(
            0, min(self._width - 1, center_x - text_width // 2 - padding)
        )
        background_x2 = max(
            0, min(self._width - 1, center_x + text_width // 2 + padding)
        )
        text_x = max(
            0, min(self._width - text_width - 1, center_x - text_width // 2)
        )
        text_y = max(text_height, min(self._height - 1, baseline))

        cv2.rectangle(
            frame,
            (background_x1, background_y1),
            (background_x2, background_y2),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            cv2.FONT_ITALIC,
            self._font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    def _is_offscreen(self, x: int, y: int) -> bool:
        return not (0 <= x < self._width and 0 <= y < self._height)

    def _edge_tip(
        self, x: int, y: int, margin: int
    ) -> tuple[float, float]:
        """Intersect the centre-to-point ray with an inset screen border."""

        center_x = self._width / 2.0
        center_y = self._height / 2.0
        dx = x - center_x
        dy = y - center_y

        left = float(margin)
        top = float(margin)
        right = max(left, float(self._width - 1 - margin))
        bottom = max(top, float(self._height - 1 - margin))

        factors: list[float] = []
        if dx > 0:
            factors.append((right - center_x) / dx)
        elif dx < 0:
            factors.append((left - center_x) / dx)
        if dy > 0:
            factors.append((bottom - center_y) / dy)
        elif dy < 0:
            factors.append((top - center_y) / dy)
        positive = [factor for factor in factors if factor > 0]
        if not positive:
            return center_x, center_y

        factor = min(positive)
        return center_x + dx * factor, center_y + dy * factor

    def _draw_arrow(
        self, frame, x: int, y: int, color: tuple[int, int, int], text: str
    ) -> None:
        """Draw a border arrow pointing towards an off-screen position."""

        center_x = self._width / 2.0
        center_y = self._height / 2.0
        dx = x - center_x
        dy = y - center_y
        norm = math.hypot(dx, dy)
        if norm == 0:
            return
        unit_x, unit_y = dx / norm, dy / norm

        tip_x, tip_y = self._edge_tip(x, y, margin=3)
        length = min(
            max(12, min(self._box_size)),
            max(6, min(self._width, self._height) // 3),
        )
        base_x = tip_x - unit_x * length
        base_y = tip_y - unit_y * length
        half = max(3.0, length * 0.45)
        perp_x, perp_y = -unit_y, unit_x

        triangle = np.array(
            [
                (round(tip_x), round(tip_y)),
                (
                    round(base_x + perp_x * half),
                    round(base_y + perp_y * half),
                ),
                (
                    round(base_x - perp_x * half),
                    round(base_y - perp_y * half),
                ),
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(frame, triangle, color, cv2.LINE_AA)
        cv2.polylines(frame, [triangle], True, (0, 0, 0), 2, cv2.LINE_AA)
        self._draw_label(frame, round(base_x), round(base_y), text)

    def _draw_track(
        self, frame, track_id: int, samples: deque[_Sample]
    ) -> None:
        center_x, center_y = self._smoothed_position(samples)
        color = self._track_color(track_id)

        if self._is_offscreen(center_x, center_y):
            self._draw_arrow(
                frame, center_x, center_y, color, samples[-1].text
            )
            return

        half_width = self._box_size[0] // 2
        half_height = self._box_size[1] // 2
        x1 = max(0, min(self._width - 1, center_x - half_width))
        y1 = max(0, min(self._height - 1, center_y - half_height))
        x2 = max(0, min(self._width - 1, center_x + half_width))
        y2 = max(0, min(self._height - 1, center_y + half_height))

        self._draw_trail(frame, samples, color)

        # High-contrast box: black outline plus the marker colour.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        # Centre crosshair makes small targets easier to read.
        for start, end in (
            ((center_x - 4, center_y), (center_x + 4, center_y)),
            ((center_x, center_y - 4), (center_x, center_y + 4)),
        ):
            cv2.line(frame, start, end, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.line(frame, start, end, color, 1, cv2.LINE_AA)

        self._draw_label(frame, center_x, center_y, samples[-1].text)

    def decorate(
        self,
        frame,
        coordinate: tuple[int, int] | None = None,
        text: str | None = None,
        timestamp: float | None = None,
    ) -> None:
        """Draw every live track.

        Passing ``coordinate`` is a convenience for single-track callers that
        do not track detection generations; the controller updates tracks
        explicitly and calls this without coordinates every frame.
        """

        now = time.monotonic() if timestamp is None else timestamp
        if coordinate is not None:
            self.update(0, coordinate, text or "", now)
        self._prune(now)
        for track_id in sorted(self._tracks):
            self._draw_track(frame, track_id, self._tracks[track_id])
