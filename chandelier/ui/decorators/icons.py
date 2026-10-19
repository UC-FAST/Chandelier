"""Small ASCII-friendly vector icons drawn with OpenCV primitives.

Hershey fonts cannot render emoji or most Unicode, so menu icons are drawn as
simple line art instead.
"""

from __future__ import annotations

import cv2
import numpy as np


def draw_icon(
    frame,
    name: str,
    x: int,
    y: int,
    size: int,
    color: tuple[int, int, int],
    thickness: int | None = None,
) -> None:
    """Draw a named line-art icon with OpenCV primitives.

    ``(x, y)`` is the top-left corner of a ``size`` x ``size`` box and
    ``color`` is a BGR triple.  All coordinates below are normalised to
    0..1 inside that box, so one definition scales from small status icons
    to full-screen menu glyphs.  ``thickness`` defaults to a
    size-proportional stroke.  Unknown names draw a placeholder square
    instead of raising, so a stale menu definition cannot break the UI.
    """

    if size <= 0:
        return
    thickness = thickness or max(1, size // 9)
    aa = cv2.LINE_AA

    # Normalised design-space helper: 0..1 coordinates to pixels.
    def point(px: float, py: float) -> tuple[int, int]:
        return x + int(px * size), y + int(py * size)

    # Capture, display and camera-control glyphs.
    if name in {"camera", "video", "record"}:
        cv2.rectangle(frame, point(0.1, 0.3), point(0.9, 0.85), color, thickness, aa)
        cv2.circle(frame, point(0.5, 0.58), max(2, size // 5), color, thickness, aa)
        cv2.line(frame, point(0.3, 0.3), point(0.4, 0.16), color, thickness, aa)
        cv2.line(frame, point(0.4, 0.16), point(0.6, 0.16), color, thickness, aa)
        cv2.line(frame, point(0.6, 0.16), point(0.7, 0.3), color, thickness, aa)
        if name == "record":
            cv2.circle(frame, point(0.5, 0.58), max(1, size // 10), color, -1, aa)
    elif name == "image":
        cv2.rectangle(frame, point(0.12, 0.2), point(0.88, 0.8), color, thickness, aa)
        cv2.circle(frame, point(0.3, 0.38), max(1, size // 9), color, -1, aa)
        cv2.line(frame, point(0.18, 0.72), point(0.42, 0.46), color, thickness, aa)
        cv2.line(frame, point(0.42, 0.46), point(0.62, 0.66), color, thickness, aa)
        cv2.line(frame, point(0.55, 0.64), point(0.72, 0.5), color, thickness, aa)
        cv2.line(frame, point(0.72, 0.5), point(0.84, 0.68), color, thickness, aa)
    elif name == "monitor":
        cv2.rectangle(frame, point(0.1, 0.16), point(0.9, 0.7), color, thickness, aa)
        cv2.line(frame, point(0.5, 0.7), point(0.5, 0.85), color, thickness, aa)
        cv2.line(frame, point(0.32, 0.86), point(0.68, 0.86), color, thickness, aa)
    elif name == "display":
        cv2.rectangle(frame, point(0.12, 0.18), point(0.88, 0.78), color, thickness, aa)
        cv2.line(frame, point(0.2, 0.34), point(0.6, 0.34), color, thickness, aa)
        cv2.line(frame, point(0.2, 0.5), point(0.72, 0.5), color, thickness, aa)
        cv2.line(frame, point(0.2, 0.66), point(0.52, 0.66), color, thickness, aa)
    elif name in {"sliders", "manual"}:
        for row, knob in ((0.25, 0.35), (0.5, 0.7), (0.75, 0.45)):
            cv2.line(frame, point(0.12, row), point(0.88, row), color, thickness, aa)
            cv2.circle(frame, point(knob, row), max(1, size // 9), color, -1, aa)
    elif name == "sun":
        center = point(0.5, 0.5)
        radius = max(2, size // 5)
        cv2.circle(frame, center, radius, color, thickness, aa)
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, -1), (-1, 1), (1, 1)):
            cv2.line(
                frame,
                point(0.5 + dx * 0.3, 0.5 + dy * 0.3),
                point(0.5 + dx * 0.42, 0.5 + dy * 0.42),
                color,
                thickness,
                aa,
            )
    elif name == "wb":
        cv2.circle(frame, point(0.5, 0.5), max(2, size // 3), color, thickness, aa)
        cv2.line(frame, point(0.5, 0.2), point(0.5, 0.8), color, thickness, aa)
        cv2.line(frame, point(0.2, 0.5), point(0.8, 0.5), color, thickness, aa)
    elif name == "grid":
        for fraction in (0.33, 0.66):
            cv2.line(frame, point(fraction, 0.12), point(fraction, 0.88), color, thickness, aa)
            cv2.line(frame, point(0.12, fraction), point(0.88, fraction), color, thickness, aa)
        cv2.rectangle(frame, point(0.12, 0.12), point(0.88, 0.88), color, thickness, aa)
    elif name == "zebra":
        for offset in (0.15, 0.38, 0.61, 0.84):
            cv2.line(frame, point(0.1, offset), point(offset, 0.1), color, thickness, aa)
    elif name == "focus":
        length = 0.25
        for px, py, dx, dy in (
            (0.12, 0.12, 1, 1),
            (0.88, 0.12, -1, 1),
            (0.12, 0.88, 1, -1),
            (0.88, 0.88, -1, -1),
        ):
            cv2.line(frame, point(px, py), point(px + dx * length, py), color, thickness, aa)
            cv2.line(frame, point(px, py), point(px, py + dy * length), color, thickness, aa)
        cv2.circle(frame, point(0.5, 0.5), max(1, size // 9), color, thickness, aa)
    # Shooting-mode and menu glyphs.
    elif name == "timer":
        cv2.circle(frame, point(0.5, 0.56), max(2, size // 3), color, thickness, aa)
        cv2.line(frame, point(0.5, 0.56), point(0.5, 0.38), color, thickness, aa)
        cv2.line(frame, point(0.5, 0.56), point(0.65, 0.62), color, thickness, aa)
        cv2.line(frame, point(0.38, 0.14), point(0.62, 0.14), color, thickness, aa)
    elif name == "burst":
        for offset in (0.0, 0.12, 0.24):
            cv2.rectangle(
                frame,
                point(0.12 + offset, 0.2 + offset),
                point(0.62 + offset, 0.7 + offset),
                color,
                thickness,
                aa,
            )
    elif name == "hdr":
        cv2.rectangle(frame, point(0.1, 0.25), point(0.9, 0.75), color, thickness, aa)
        cv2.line(frame, point(0.22, 0.5), point(0.78, 0.5), color, thickness, aa)
        cv2.line(frame, point(0.35, 0.34), point(0.35, 0.66), color, thickness, aa)
        cv2.line(frame, point(0.65, 0.34), point(0.65, 0.66), color, thickness, aa)
    elif name == "theme":
        cv2.circle(frame, point(0.5, 0.5), max(2, size // 3), color, thickness, aa)
        cv2.ellipse(frame, point(0.5, 0.5), (max(2, size // 3), max(2, size // 3)), 0, 270, 90, color, -1, aa)
    elif name == "delete":
        cv2.rectangle(frame, point(0.25, 0.3), point(0.75, 0.85), color, thickness, aa)
        cv2.line(frame, point(0.18, 0.24), point(0.82, 0.24), color, thickness, aa)
        cv2.line(frame, point(0.38, 0.14), point(0.62, 0.14), color, thickness, aa)
        for offset in (0.4, 0.5, 0.6):
            cv2.line(frame, point(offset, 0.4), point(offset, 0.75), color, thickness, aa)
    elif name == "hist":
        for index, height in enumerate((0.3, 0.55, 0.4, 0.7)):
            left = 0.18 + index * 0.17
            cv2.rectangle(
                frame,
                point(left, 0.86 - height),
                point(left + 0.1, 0.86),
                color,
                -1,
                aa,
            )
    # Playback controls.
    elif name == "play":
        cv2.fillPoly(
            frame,
            [np.array(
                [point(0.28, 0.18), point(0.28, 0.82), point(0.8, 0.5)],
                dtype="int32",
            )],
            color,
            aa,
        )
    elif name == "pause":
        cv2.rectangle(frame, point(0.28, 0.18), point(0.44, 0.82), color, -1, aa)
        cv2.rectangle(frame, point(0.56, 0.18), point(0.72, 0.82), color, -1, aa)
    elif name == "loop":
        cv2.ellipse(frame, point(0.5, 0.5), (max(2, size // 3), max(2, size // 3)), 0, 20, 320, color, thickness, aa)
        cv2.line(frame, point(0.5, 0.12), point(0.62, 0.24), color, thickness, aa)
        cv2.line(frame, point(0.5, 0.12), point(0.38, 0.24), color, thickness, aa)
    # System and hardware glyphs (used by the system monitor).
    elif name == "power":
        radius = max(3, size // 3)
        cv2.ellipse(
            frame,
            point(0.5, 0.55),
            (radius, radius),
            0,
            300,
            240,
            color,
            thickness,
            aa,
        )
        cv2.line(
            frame, point(0.5, 0.12), point(0.5, 0.5), color, thickness, aa
        )
    elif name == "cpu":
        cv2.rectangle(
            frame, point(0.24, 0.24), point(0.76, 0.76), color, thickness, aa
        )
        cv2.rectangle(
            frame, point(0.4, 0.4), point(0.6, 0.6), color, thickness, aa
        )
        for offset in (0.36, 0.5, 0.64):
            cv2.line(
                frame, point(offset, 0.24), point(offset, 0.12),
                color, thickness, aa,
            )
            cv2.line(
                frame, point(offset, 0.76), point(offset, 0.88),
                color, thickness, aa,
            )
            cv2.line(
                frame, point(0.24, offset), point(0.12, offset),
                color, thickness, aa,
            )
            cv2.line(
                frame, point(0.76, offset), point(0.88, offset),
                color, thickness, aa,
            )
    elif name == "memory":
        cv2.rectangle(
            frame, point(0.1, 0.3), point(0.9, 0.7), color, thickness, aa
        )
        for offset in (0.24, 0.38, 0.52, 0.66):
            cv2.line(
                frame, point(offset, 0.36), point(offset, 0.64),
                color, thickness, aa,
            )
        for offset in (0.2, 0.4, 0.6, 0.8):
            cv2.line(
                frame, point(offset, 0.7), point(offset, 0.82),
                color, thickness, aa,
            )
    elif name == "disk":
        cv2.rectangle(
            frame, point(0.18, 0.2), point(0.82, 0.8), color, thickness, aa
        )
        cv2.line(
            frame, point(0.18, 0.56), point(0.82, 0.56),
            color, thickness, aa,
        )
        cv2.circle(
            frame, point(0.68, 0.68), max(1, size // 10), color, -1, aa
        )
    elif name == "temperature":
        cv2.line(
            frame, point(0.5, 0.16), point(0.5, 0.62), color, thickness, aa
        )
        cv2.circle(
            frame, point(0.5, 0.72), max(2, size // 6),
            color, thickness, aa,
        )
        cv2.line(
            frame, point(0.5, 0.72), point(0.5, 0.48),
            color, thickness, aa,
        )
        for offset in (0.3, 0.42, 0.54):
            cv2.line(
                frame, point(0.58, offset), point(0.68, offset),
                color, thickness, aa,
            )
    elif name == "battery":
        cv2.rectangle(
            frame, point(0.12, 0.3), point(0.78, 0.7), color, thickness, aa
        )
        cv2.rectangle(
            frame, point(0.78, 0.42), point(0.88, 0.58), color, -1, aa
        )
        for left in (0.18, 0.34, 0.5):
            cv2.rectangle(
                frame,
                point(left, 0.4),
                point(left + 0.1, 0.6),
                color,
                -1,
                aa,
            )
    elif name == "network":
        for cx, cy in ((0.18, 0.5), (0.82, 0.24), (0.82, 0.76)):
            cv2.circle(
                frame, point(cx, cy), max(1, size // 10), color, -1, aa
            )
        cv2.line(
            frame, point(0.26, 0.46), point(0.74, 0.28),
            color, thickness, aa,
        )
        cv2.line(
            frame, point(0.26, 0.54), point(0.74, 0.72),
            color, thickness, aa,
        )
        cv2.line(
            frame, point(0.82, 0.32), point(0.82, 0.68),
            color, thickness, aa,
        )
    elif name == "wifi":
        for radius in (0.18, 0.32, 0.46):
            axes = max(1, int(radius * size))
            cv2.ellipse(
                frame, point(0.5, 0.82), (axes, axes),
                0, 225, 315, color, thickness, aa,
            )
        cv2.circle(
            frame, point(0.5, 0.82), max(1, size // 12), color, -1, aa
        )
    elif name == "signal":
        for index, height in enumerate((0.22, 0.4, 0.58, 0.76)):
            left = 0.16 + index * 0.19
            cv2.rectangle(
                frame,
                point(left, 0.82 - height),
                point(left + 0.1, 0.82),
                color,
                -1,
                aa,
            )
    elif name == "speed":
        axes = max(2, int(0.34 * size))
        cv2.ellipse(
            frame, point(0.5, 0.68), (axes, axes),
            0, 180, 360, color, thickness, aa,
        )
        cv2.line(
            frame, point(0.5, 0.68), point(0.66, 0.42),
            color, thickness, aa,
        )
        cv2.circle(
            frame, point(0.5, 0.68), max(1, size // 12), color, -1, aa
        )
    elif name == "clock":
        cv2.circle(
            frame, point(0.5, 0.55), max(2, size // 3),
            color, thickness, aa,
        )
        cv2.line(
            frame, point(0.5, 0.55), point(0.5, 0.35),
            color, thickness, aa,
        )
        cv2.line(
            frame, point(0.5, 0.55), point(0.66, 0.62),
            color, thickness, aa,
        )
    # Settings, services and capture-assist glyphs.
    elif name == "warning":
        cv2.polylines(
            frame,
            [np.array(
                [point(0.5, 0.12), point(0.92, 0.86), point(0.08, 0.86)],
                dtype="int32",
            )],
            True,
            color,
            thickness,
            aa,
        )
        cv2.line(
            frame, point(0.5, 0.38), point(0.5, 0.64),
            color, thickness, aa,
        )
        cv2.circle(
            frame, point(0.5, 0.74), max(1, size // 12), color, -1, aa
        )
    elif name == "service":
        cv2.circle(
            frame, point(0.5, 0.5), max(2, size // 3),
            color, thickness, aa,
        )
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            cv2.line(
                frame,
                point(0.5 + dx * 0.34, 0.5 + dy * 0.34),
                point(0.5 + dx * 0.48, 0.5 + dy * 0.48),
                color,
                thickness,
                aa,
            )
        cv2.circle(
            frame, point(0.5, 0.5), max(1, size // 10), color, -1, aa
        )
    elif name in {"settings", "gear"}:
        center = point(0.5, 0.5)
        cv2.circle(
            frame, center, max(2, size // 4), color, thickness, aa
        )
        cv2.circle(
            frame, center, max(1, size // 10), color, thickness, aa
        )
        for dx, dy in (
            (0, -1), (0, 1), (-1, 0), (1, 0),
            (-0.7, -0.7), (0.7, -0.7), (-0.7, 0.7), (0.7, 0.7),
        ):
            cv2.line(
                frame,
                point(0.5 + dx * 0.28, 0.5 + dy * 0.28),
                point(0.5 + dx * 0.44, 0.5 + dy * 0.44),
                color,
                thickness,
                aa,
            )
    elif name == "lock":
        cv2.rectangle(
            frame, point(0.26, 0.45), point(0.74, 0.86),
            color, thickness, aa,
        )
        axes = max(2, size // 5)
        cv2.ellipse(
            frame, point(0.5, 0.45), (axes, axes),
            0, 180, 360, color, thickness, aa,
        )
        cv2.circle(
            frame, point(0.5, 0.64), max(1, size // 12), color, -1, aa
        )
    elif name in {"shutter", "aperture"}:
        radius = max(2, size // 3)
        cv2.circle(
            frame, point(0.5, 0.5), radius, color, thickness, aa
        )
        for px, py in ((0.5, 0.17), (0.21, 0.67), (0.79, 0.67)):
            cv2.line(
                frame, point(0.5, 0.5), point(px, py),
                color, thickness, aa,
            )
    elif name == "gain":
        cv2.line(
            frame, point(0.14, 0.82), point(0.86, 0.82),
            color, thickness, aa,
        )
        cv2.line(
            frame, point(0.2, 0.72), point(0.74, 0.22),
            color, thickness, aa,
        )
        cv2.line(
            frame, point(0.74, 0.22), point(0.74, 0.42),
            color, thickness, aa,
        )
        cv2.line(
            frame, point(0.74, 0.22), point(0.54, 0.22),
            color, thickness, aa,
        )
        for left, height in ((0.22, 0.14), (0.38, 0.24), (0.54, 0.36)):
            cv2.line(
                frame, point(left, 0.82), point(left, 0.82 - height),
                color, thickness, aa,
            )
    elif name == "flicker":
        points = [
            point(0.1, 0.5), point(0.27, 0.22), point(0.44, 0.78),
            point(0.61, 0.22), point(0.78, 0.78), point(0.9, 0.5),
        ]
        cv2.polylines(
            frame, [np.array(points, dtype="int32")],
            False, color, thickness, aa,
        )
    # Overlay and metadata glyphs.
    elif name in {"frame", "aspect"}:
        cv2.rectangle(
            frame, point(0.1, 0.16), point(0.9, 0.84),
            color, thickness, aa,
        )
        cv2.line(
            frame, point(0.1, 0.34), point(0.9, 0.34),
            color, thickness, aa,
        )
        cv2.line(
            frame, point(0.1, 0.66), point(0.9, 0.66),
            color, thickness, aa,
        )
    elif name == "tag":
        points = [
            point(0.14, 0.38), point(0.48, 0.1), point(0.88, 0.5),
            point(0.52, 0.9), point(0.14, 0.52),
        ]
        cv2.polylines(
            frame, [np.array(points, dtype="int32")],
            True, color, thickness, aa,
        )
        cv2.circle(
            frame, point(0.33, 0.35), max(1, size // 12), color, -1, aa
        )
    elif name == "watermark":
        cv2.rectangle(
            frame, point(0.1, 0.18), point(0.9, 0.82),
            color, thickness, aa,
        )
        points = [
            point(0.24, 0.34), point(0.34, 0.68), point(0.5, 0.46),
            point(0.66, 0.68), point(0.76, 0.34),
        ]
        cv2.polylines(
            frame, [np.array(points, dtype="int32")],
            False, color, thickness, aa,
        )
    # Audio glyphs.
    elif name == "mic":
        cv2.ellipse(
            frame, point(0.5, 0.32),
            (max(2, size // 9), max(3, size // 5)),
            0, 0, 360, color, thickness, aa,
        )
        cv2.line(
            frame, point(0.5, 0.52), point(0.5, 0.8),
            color, thickness, aa,
        )
        cv2.line(
            frame, point(0.36, 0.8), point(0.64, 0.8),
            color, thickness, aa,
        )
    elif name in {"sound", "speaker"}:
        cv2.rectangle(
            frame, point(0.1, 0.42), point(0.28, 0.58),
            color, thickness, aa,
        )
        points = [
            point(0.28, 0.42), point(0.46, 0.22),
            point(0.46, 0.78), point(0.28, 0.58),
        ]
        cv2.polylines(
            frame, [np.array(points, dtype="int32")],
            False, color, thickness, aa,
        )
        for radius in (0.14, 0.24):
            axes = max(1, int(radius * size))
            cv2.ellipse(
                frame, point(0.54, 0.5), (axes, axes),
                0, -45, 45, color, thickness, aa,
            )
    # Calibration and colour glyphs.
    elif name == "angle":
        cv2.line(
            frame, point(0.12, 0.82), point(0.88, 0.82),
            color, thickness, aa,
        )
        cv2.line(
            frame, point(0.12, 0.82), point(0.72, 0.2),
            color, thickness, aa,
        )
        axes = max(2, int(0.3 * size))
        cv2.ellipse(
            frame, point(0.12, 0.82), (axes, axes),
            0, 314, 360, color, thickness, aa,
        )
    elif name == "globe":
        cv2.circle(
            frame, point(0.5, 0.5), max(2, size // 3),
            color, thickness, aa,
        )
        cv2.ellipse(
            frame, point(0.5, 0.5),
            (max(1, size // 7), max(2, size // 3)),
            0, 0, 360, color, thickness, aa,
        )
        cv2.line(
            frame, point(0.18, 0.5), point(0.82, 0.5),
            color, thickness, aa,
        )
    elif name == "contrast":
        radius = max(2, size // 3)
        cv2.circle(
            frame, point(0.5, 0.5), radius, color, thickness, aa
        )
        cv2.ellipse(
            frame, point(0.5, 0.5), (radius, radius),
            0, 90, 270, color, -1, aa,
        )
    elif name in {"droplet", "color"}:
        points = [
            point(0.5, 0.08), point(0.28, 0.5), point(0.25, 0.66),
            point(0.35, 0.82), point(0.5, 0.88), point(0.65, 0.82),
            point(0.75, 0.66), point(0.72, 0.5),
        ]
        cv2.polylines(
            frame, [np.array(points, dtype="int32")],
            True, color, thickness, aa,
        )
    # Transport, storage and framing helpers.
    elif name in {"seek", "skip"}:
        for left in (0.18, 0.5):
            points = np.array(
                [
                    point(left, 0.2),
                    point(left, 0.8),
                    point(left + 0.34, 0.5),
                ],
                dtype="int32",
            )
            cv2.fillPoly(frame, [points], color, aa)
    elif name == "server":
        for top in (0.18, 0.54):
            cv2.rectangle(
                frame, point(0.16, top), point(0.84, top + 0.24),
                color, thickness, aa,
            )
            cv2.circle(
                frame, point(0.72, top + 0.12),
                max(1, size // 14), color, -1, aa,
            )
    elif name in {"file", "document"}:
        points = [
            point(0.25, 0.1), point(0.62, 0.1), point(0.78, 0.28),
            point(0.78, 0.9), point(0.25, 0.9),
        ]
        cv2.polylines(
            frame, [np.array(points, dtype="int32")],
            True, color, thickness, aa,
        )
        cv2.line(
            frame, point(0.62, 0.1), point(0.62, 0.28),
            color, thickness, aa,
        )
        cv2.line(
            frame, point(0.62, 0.28), point(0.78, 0.28),
            color, thickness, aa,
        )
        for row in (0.55, 0.7):
            cv2.line(
                frame, point(0.35, row), point(0.68, row),
                color, thickness, aa,
            )
    elif name in {"crosshair", "target"}:
        cv2.circle(
            frame, point(0.5, 0.5), max(2, size // 4),
            color, thickness, aa,
        )
        for start, end in (
            ((0.5, 0.06), (0.5, 0.26)),
            ((0.5, 0.74), (0.5, 0.94)),
            ((0.06, 0.5), (0.26, 0.5)),
            ((0.74, 0.5), (0.94, 0.5)),
        ):
            cv2.line(
                frame, point(*start), point(*end),
                color, thickness, aa,
            )
        cv2.circle(
            frame, point(0.5, 0.5), max(1, size // 14), color, -1, aa
        )
    elif name in {"sharp", "sparkle"}:
        for dx, dy in ((1, 0), (0, 1), (0.7, 0.7), (0.7, -0.7)):
            cv2.line(
                frame,
                point(0.5 - dx * 0.34, 0.5 - dy * 0.34),
                point(0.5 + dx * 0.34, 0.5 + dy * 0.34),
                color,
                thickness,
                aa,
            )
        cv2.circle(
            frame, point(0.5, 0.5), max(1, size // 12), color, -1, aa
        )
    elif name == "ruler":
        cv2.rectangle(
            frame, point(0.08, 0.34), point(0.92, 0.66),
            color, thickness, aa,
        )
        for index, left in enumerate((0.2, 0.32, 0.44, 0.56, 0.68, 0.8)):
            length = 0.18 if index % 2 == 0 else 0.1
            cv2.line(
                frame, point(left, 0.34), point(left, 0.34 + length),
                color, thickness, aa,
            )
    # Unknown names fall back to a visible placeholder square.
    else:
        cv2.rectangle(frame, point(0.2, 0.2), point(0.8, 0.8), color, thickness, aa)
