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

When the array and the lens are not co-located, the direction measured by
the array is not the direction seen by the camera.  ``mic_offset`` gives the
array centre in camera coordinates (x right, y up, z forward, in metres) and
``source_distance`` the distance from the camera to the source.  The source
is placed on the array's ray and re-projected from the camera origin, which
removes the parallax for sources at that distance.
"""

from __future__ import annotations

import math

DEFAULT_WIDTH = 320
DEFAULT_HEIGHT = 240

# Horizontal field of view of the camera preview.  The IMX708 wide module
# (2.75 mm lens, 1.4 um pixels) is cropped by libcamera from its native 16:9
# sensor to the 4:3 preview, which keeps the full sensor height and 3/4 of its
# width.  At 320 px wide this gives a focal length of ~181 px, i.e. a 41.3
# degree half-angle and an 83 degree horizontal FOV.
DEFAULT_FOV = 83.0

# Keeps the horizon (elevation 0) finite while staying far outside any frame.
MAX_OFFSCREEN_RADIUS = 10_000.0


def _unit_direction(azimuth_rad: float, elevation_rad: float):
    """Return the array direction in camera axes (x right, y up, z forward).

    The array frame uses ``x`` towards the left of the image and ``y`` towards
    the top, which is why the x component is negated here.
    """

    transverse = math.cos(elevation_rad)
    # x is negated so azimuth 0 appears on the left; y points up and z
    # points forward along the optical axis.
    return (
        -transverse * math.cos(azimuth_rad),
        transverse * math.sin(azimuth_rad),
        math.sin(elevation_rad),
    )


def aziel_to_screen_spherical(
    azi: float,
    ele: float,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fov: float = DEFAULT_FOV,
    azimuth_offset: float = 0.0,
    elevation_offset: float = 0.0,
    mic_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    source_distance: float = 0.0,
) -> tuple[int, int]:
    """Convert ``(azimuth, elevation)`` degrees to an unclamped pixel.

    ``fov`` is the horizontal field of view in degrees; the same focal length
    is used on both axes because the pixels are square.  ``azimuth_offset``
    is subtracted from the azimuth, which rotates the whole marker overlay
    around the screen centre (a positive value turns it counter-clockwise).
    ``elevation_offset`` is added to the elevation before projecting, which
    compensates a microphone array that is pitched relative to the camera.

    ``mic_offset`` is the microphone-array centre relative to the camera
    (x right, y up, z forward, metres).  When ``source_distance`` (metres,
    from the camera) is positive the parallax between the array and the lens
    is compensated; a value of zero keeps the direction-only projection.

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
    # The estimator only reports the front hemisphere; clamp to it before
    # projecting so a stray negative elevation cannot flip the marker.
    ele = min(90.0, max(0.0, float(ele) + float(elevation_offset)))

    # Pinhole focal length in pixels from the horizontal field of view.
    focal = (width / 2.0) / math.tan(math.radians(float(fov)) / 2.0)
    azimuth_rad = math.radians((azi - float(azimuth_offset)) % 360.0)
    radians_ele = math.radians(ele)

    offset_x, offset_y, offset_z = (float(value) for value in mic_offset)
    if float(source_distance) > 0.0:
        ux, uy, uz = _unit_direction(azimuth_rad, radians_ele)
        distance = float(source_distance)
        # Place the source on the array's ray at ``distance`` from the
        # camera: solve |offset + t * u|^2 = distance^2 for t > 0.
        linear = 2.0 * (
            offset_x * ux + offset_y * uy + offset_z * uz
        )
        constant = (
            offset_x * offset_x
            + offset_y * offset_y
            + offset_z * offset_z
            - distance * distance
        )
        discriminant = linear * linear - 4.0 * constant
        if discriminant >= 0.0:
            # The positive root is the intersection in front of the array.
            travel = (-linear + math.sqrt(discriminant)) / 2.0
            if travel > 0.0:
                # Source position relative to the camera.
                sx = offset_x + travel * ux
                sy = offset_y + travel * uy
                sz = offset_z + travel * uz
                if sz > 1e-9:
                    # Rectilinear projection; image rows grow downwards,
                    # hence the minus on y.
                    x = width / 2.0 + focal * sx / sz
                    y = height / 2.0 - focal * sy / sz
                    return _cap_offscreen(x, y, width, height)

    sin_ele = math.sin(radians_ele)
    # Direction-only projection: the radius is focal / tan(elevation).
    # At the horizon this tends to infinity, so cap it at a finite radius;
    # the 1e-12 guard avoids a division by zero at elevation 0.
    if sin_ele <= 1e-12:
        radius = MAX_OFFSCREEN_RADIUS
    else:
        radius = min(
            focal * math.cos(radians_ele) / sin_ele,
            MAX_OFFSCREEN_RADIUS,
        )

    # Mirrored axes match the device convention documented above.
    x = width / 2.0 - radius * math.cos(azimuth_rad)
    y = height / 2.0 - radius * math.sin(azimuth_rad)
    return int(round(x)), int(round(y))


def _cap_offscreen(x: float, y: float, width: int, height: int):
    """Keep a projected point finite without clamping it into the frame."""

    dx = x - width / 2.0
    dy = y - height / 2.0
    radius = math.hypot(dx, dy)
    if radius <= MAX_OFFSCREEN_RADIUS:
        return int(round(x)), int(round(y))
    scale = MAX_OFFSCREEN_RADIUS / radius
    return (
        int(round(width / 2.0 + dx * scale)),
        int(round(height / 2.0 + dy * scale)),
    )


def apply_screen_gain(
    x: int,
    y: int,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    gain_x: float = 0.0,
    gain_y: float = 0.0,
) -> tuple[int, int]:
    """Scale a marker's distance from the screen centre per axis.

    ``gain_x``/``gain_y`` are multiplicative coefficients; zero disables the
    correction.  ``gain=1`` doubles the distance from the centre, ``gain=0.1``
    expands it by 10 percent.  The offset is applied as
    ``position += gain * (position - centre)``.
    """

    if width <= 0 or height <= 0:
        raise ValueError("screen dimensions must be positive")

    center_x = width / 2.0
    center_y = height / 2.0
    # Per-axis expansion about the centre; the axes are independent so
    # lens/geometry calibration does not have to be isotropic.
    corrected_x = x + float(gain_x) * (x - center_x)
    corrected_y = y + float(gain_y) * (y - center_y)
    return int(round(corrected_x)), int(round(corrected_y))
