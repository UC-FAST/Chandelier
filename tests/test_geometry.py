import math

import pytest

from chandelier.common.geometry import (
    apply_screen_gain,
    aziel_to_screen_spherical,
)


@pytest.mark.parametrize(
    ("azi", "ele", "expected"),
    [
        (0, 90, (160, 120)),
        (123, 90, (160, 120)),
        (0, 80, (128, 120)),
        (90, 80, (160, 88)),
        (180, 80, (192, 120)),
        (270, 80, (160, 152)),
        (0, 65, (76, 120)),
        (90, 65, (160, 36)),
        (180, 65, (244, 120)),
        (270, 65, (160, 204)),
        (0, 60, (56, 120)),
        (90, 60, (160, 16)),
        (0, 59, (51, 120)),
        (180, 59, (269, 120)),
        (90, 56.43397776801855, (160, 0)),
        (270, 56.43397776801855, (160, 240)),
    ],
)
def test_known_mapping_points(azi, ele, expected):
    assert aziel_to_screen_spherical(azi, ele) == expected


def test_equal_off_axis_angles_map_to_equal_pixel_radius():
    # The old mapping stretched the image into an ellipse; the pinhole model
    # must use the same focal length on both axes.
    for ele in (70, 75, 80, 85):
        horizontal_x, _ = aziel_to_screen_spherical(0, ele)
        _, vertical_y = aziel_to_screen_spherical(90, ele)
        assert abs(horizontal_x - 160) == pytest.approx(
            abs(vertical_y - 120), abs=1
        )


def test_horizontal_fov_edges_are_not_clamped():
    # The default 83 degree horizontal FOV reaches the left/right border at
    # 90 - 83 / 2 = 48.5 degrees elevation.
    assert aziel_to_screen_spherical(0, 48.5) == (0, 120)
    assert aziel_to_screen_spherical(180, 48.5) == (320, 120)


def test_offscreen_points_are_not_clamped():
    x, y = aziel_to_screen_spherical(90, 45)
    assert x == 160 and y < 0

    x, y = aziel_to_screen_spherical(270, 45)
    assert x == 160 and y >= 240


def test_horizon_projects_far_offscreen_in_each_direction():
    assert aziel_to_screen_spherical(0, 0)[0] < 0
    assert aziel_to_screen_spherical(180, 0)[0] > 320
    assert aziel_to_screen_spherical(90, 0)[1] < 0
    assert aziel_to_screen_spherical(270, 0)[1] > 240


def test_azimuth_offset_rotates_the_projection():
    assert aziel_to_screen_spherical(
        0, 80, azimuth_offset=90
    ) == aziel_to_screen_spherical(270, 80)
    assert aziel_to_screen_spherical(
        15, 70, azimuth_offset=360
    ) == aziel_to_screen_spherical(15, 70)


def test_azimuth_wraps_and_elevation_clamps():
    assert aziel_to_screen_spherical(
        360, 65
    ) == aziel_to_screen_spherical(0, 65)
    assert aziel_to_screen_spherical(
        -90, 65
    ) == aziel_to_screen_spherical(270, 65)
    assert aziel_to_screen_spherical(
        0, -10
    ) == aziel_to_screen_spherical(0, 0)
    assert aziel_to_screen_spherical(
        0, 120
    ) == aziel_to_screen_spherical(0, 90)


def test_fov_changes_the_scale():
    assert aziel_to_screen_spherical(0, 60, fov=120) == (107, 120)


def test_custom_screen_size():
    assert aziel_to_screen_spherical(
        0, 80, width=160, height=120
    ) == (64, 60)
    assert aziel_to_screen_spherical(
        0, 80, width=64, height=48
    ) == (26, 24)


@pytest.mark.parametrize("fov", [0, -1, 180, 360])
def test_invalid_fov_raises(fov):
    with pytest.raises(ValueError):
        aziel_to_screen_spherical(0, 45, fov=fov)


@pytest.mark.parametrize(
    ("width", "height"),
    [(0, 240), (-1, 240), (320, 0), (320, -1)],
)
def test_invalid_dimensions_raise(width, height):
    with pytest.raises(ValueError):
        aziel_to_screen_spherical(0, 45, width=width, height=height)


def test_elevation_offset_shifts_the_projection():
    assert aziel_to_screen_spherical(
        0, 80, elevation_offset=-20
    ) == aziel_to_screen_spherical(0, 60)
    assert aziel_to_screen_spherical(
        90, 40, elevation_offset=20
    ) == aziel_to_screen_spherical(90, 60)


def test_parallax_disabled_matches_direction_projection():
    assert aziel_to_screen_spherical(
        10,
        73,
        mic_offset=(0.0, -0.05, 0.0),
        source_distance=0.0,
    ) == aziel_to_screen_spherical(10, 73)


def test_parallax_projection_round_trip():
    # Array mounted 5 cm below the lens; sources at 40 cm in front of the
    # camera.  Projecting the array direction must recover the camera pixel
    # of the real source position.
    fov = 83.0
    focal = (320 / 2.0) / math.tan(math.radians(fov) / 2.0)
    mic_offset = (0.0, -0.05, 0.0)
    distance = 0.4
    for sx, sy, sz in [
        (0.0, 0.0, 1.0),
        (0.3, 0.1, 0.9),
        (-0.35, -0.08, 0.95),
    ]:
        scale = distance / math.hypot(sx, sy, sz)
        sx, sy, sz = sx * scale, sy * scale, sz * scale

        # Direction from the array, expressed in the array frame
        # (x towards the left of the image, y towards the top).
        vx = sx - mic_offset[0]
        vy = sy - mic_offset[1]
        vz = sz - mic_offset[2]
        ax, ay, az = -vx, vy, vz
        azi = math.degrees(math.atan2(ay, ax)) % 360.0
        ele = math.degrees(math.atan2(abs(az), math.hypot(ax, ay)))

        expected = (
            round(320 / 2.0 + focal * sx / sz),
            round(240 / 2.0 - focal * sy / sz),
        )
        assert aziel_to_screen_spherical(
            azi,
            ele,
            fov=fov,
            mic_offset=mic_offset,
            source_distance=distance,
        ) == expected


def test_screen_gain_scales_distance_from_centre():
    assert apply_screen_gain(200, 60, 320, 240, 0.0, 0.0) == (200, 60)
    assert apply_screen_gain(160, 120, 320, 240, 3.0, 3.0) == (160, 120)
    assert apply_screen_gain(200, 60, 320, 240, 0.5, 0.0) == (220, 60)
    assert apply_screen_gain(200, 60, 320, 240, 0.0, 1.0) == (200, 0)
    assert apply_screen_gain(200, 60, 320, 240, 0.5, 0.5) == (220, 30)
