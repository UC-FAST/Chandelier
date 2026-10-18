import pytest

from chandelier.common.geometry import aziel_to_screen_spherical


@pytest.mark.parametrize(
    ("azi", "ele", "expected"),
    [
        (0, 90, (160, 120)),
        (123, 90, (160, 120)),
        (0, 80, (113, 120)),
        (90, 80, (160, 73)),
        (180, 80, (207, 120)),
        (270, 80, (160, 167)),
        (0, 65, (36, 120)),
        (90, 65, (160, -4)),
        (180, 65, (284, 120)),
        (270, 65, (160, 244)),
        (0, 60, (6, 120)),
        (90, 60, (160, -34)),
        (0, 59, (0, 120)),
        (180, 59, (320, 120)),
        (90, 65.74150753509974, (160, 0)),
        (270, 65.74150753509974, (160, 240)),
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
    # default 62 degree horizontal FOV reaches the left/right border at 59.
    assert aziel_to_screen_spherical(0, 59) == (0, 120)
    assert aziel_to_screen_spherical(180, 59) == (320, 120)


def test_offscreen_points_are_not_clamped():
    x, y = aziel_to_screen_spherical(90, 65)
    assert x == 160 and y < 0

    x, y = aziel_to_screen_spherical(270, 65)
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
    ) == (57, 60)
    assert aziel_to_screen_spherical(
        0, 80, width=64, height=48
    ) == (23, 24)


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
