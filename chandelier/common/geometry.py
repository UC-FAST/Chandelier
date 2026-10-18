"""Mapping from spherical sound-source coordinates to screen pixels.

The microphone array is mounted with its plane parallel to the camera image
plane, so the array normal is the camera's optical axis.  ``azimuth`` rotates
around the optical axis while ``elevation`` measures the angle out of the
array plane: ``elevation=90`` points along the optical axis (screen centre)
and ``elevation=0`` lies in the image plane (infinitely far away).

The projection is the rectilinear (pinhole) model of a camera with square
pixels.  No clamping is applied: a direction outside the camera's field of
view yields coordinates outside the frame, which the marker overlay renders
as a screen-edge arrow.
"""

from __future__ import annotations

import math

DEFAULT_WIDTH = 320
DEFAULT_HEIGHT = 240
DEFAULT_FOV = 62.0

# Keeps the horizon (elevation 0) finite while staying far outside any frame.
MAX_OFFSCREEN_RADIUS = 10_000.0


def aziel_to_screen_spherical(
    azi: float,
    ele: float,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fov: float = DEFAULT_FOV,
    azimuth_offset: float = 0.0,
) -> tuple[int, int]:
    """Convert ``(azimuth, elevation)`` degrees to an unclamped pixel.

    ``fov`` is the horizontal field of view in degrees; the same focal length
    is used on both axes because the pixels are square.  ``azimuth_offset``
    is subtracted from the azimuth, which rotates the whole marker overlay
    around the screen centre (a positive value turns it counter-clockwise).

    Following the original device convention, ``azi=0`` maps to the left,
    ``azi=90`` to the top, ``azi=180`` to the right and ``azi=270`` to the
    bottom.  ``ele=90`` maps to the screen centre.  ``ele=0`` projects to
    infinity and is capped at ``MAX_OFFSCREEN_RADIUS`` pixels from the
    centre.  The result is *not* clamped to the frame.
    """

    if width <= 0 or height <= 0:
        raise ValueError("screen dimensions must be positive")
    if not 0.0 < float(fov) < 180.0:
        raise ValueError("horizontal FOV must be in (0, 180) degrees")

    azi = float(azi) % 360.0
    ele = min(90.0, max(0.0, float(ele)))

    focal = (width / 2.0) / math.tan(math.radians(float(fov)) / 2.0)
    radians_ele = math.radians(ele)
    sin_ele = math.sin(radians_ele)
    if sin_ele <= 1e-12:
        radius = MAX_OFFSCREEN_RADIUS
    else:
        radius = min(
            focal * math.cos(radians_ele) / sin_ele,
            MAX_OFFSCREEN_RADIUS,
        )

    azimuth_rad = math.radians((azi - float(azimuth_offset)) % 360.0)
    x = width / 2.0 - radius * math.cos(azimuth_rad)
    y = height / 2.0 - radius * math.sin(azimuth_rad)
    return int(round(x)), int(round(y))
