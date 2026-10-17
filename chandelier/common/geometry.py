"""Mapping from spherical sound-source coordinates to screen pixels.

The microphone array reports an azimuth in ``[0, 360)`` and an elevation in
``[0, 90]``.  The camera preview is displayed on a rectangular screen; the
mapping below projects the direction onto an ellipse whose horizontal and
vertical radii peak at ``ele_const_w`` and ``ele_const_h`` respectively.
"""

from __future__ import annotations

import math

DEFAULT_WIDTH = 320
DEFAULT_HEIGHT = 240
DEFAULT_ELE_CONST_W = 65.0
DEFAULT_ELE_CONST_H = 68.0


def _radius(elevation: float, const: float, maximum: float) -> float:
    if elevation <= const:
        t = elevation / const
        return maximum * math.sin(t * math.pi / 2)
    t = (elevation - const) / (90.0 - const)
    return maximum * math.cos(t * math.pi / 2)


def aziel_to_screen_spherical(
    azi: float,
    ele: float,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    ele_const_w: float = DEFAULT_ELE_CONST_W,
    ele_const_h: float = DEFAULT_ELE_CONST_H,
) -> tuple[int, int]:
    """Convert ``(azimuth, elevation)`` degrees to a clamped ``(x, y)`` pixel.

    Following the original device convention, ``azi=0`` maps to the left
    edge, ``azi=90`` to the top, ``azi=180`` to the right and ``azi=270`` to
    the bottom.  The screen origin is the top-left corner.
    """

    if width <= 0 or height <= 0:
        raise ValueError("screen dimensions must be positive")
    if not 0.0 <= ele_const_w < 90.0 or not 0.0 <= ele_const_h < 90.0:
        raise ValueError("elevation constants must be in [0, 90)")

    azi = float(azi) % 360.0
    ele = min(90.0, max(0.0, float(ele)))

    center_x = width / 2.0
    center_y = height / 2.0
    if ele == 90.0:
        return int(round(center_x)), int(round(center_y))

    radius_h = _radius(ele, ele_const_w, width / 2.0)
    radius_v = _radius(ele, ele_const_h, height / 2.0 - 1.0)

    azimuth_rad = math.radians(azi)
    offset_x = radius_h * math.cos(azimuth_rad)
    offset_y = -radius_v * math.sin(azimuth_rad)

    x = max(1, min(width, int(round(center_x - offset_x))))
    y = max(1, min(height, int(round(center_y + offset_y))))
    return x, y
