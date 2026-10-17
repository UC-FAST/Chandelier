import pytest

from chandelier.common.geometry import aziel_to_screen_spherical


@pytest.mark.parametrize(
    ("azi", "ele", "expected"),
    [
        (0, 65, (1, 120)),
        (180, 65, (320, 120)),
        (90, 68, (160, 1)),
        (270, 68, (160, 239)),
        (0, 90, (160, 120)),
        (123, 90, (160, 120)),
    ],
)
def test_known_mapping_points(azi, ele, expected):
    assert aziel_to_screen_spherical(azi, ele) == expected


def test_coordinates_are_always_inside_the_screen():
    for azi in range(0, 360, 7):
        for ele in range(0, 91, 3):
            x, y = aziel_to_screen_spherical(azi, ele)
            assert 1 <= x <= 320
            assert 1 <= y <= 240


def test_custom_screen_size():
    x, y = aziel_to_screen_spherical(0, 65, width=160, height=120)
    assert 1 <= x <= 160
    assert 1 <= y <= 120


def test_invalid_dimensions_raise():
    with pytest.raises(ValueError):
        aziel_to_screen_spherical(0, 0, width=0)
    with pytest.raises(ValueError):
        aziel_to_screen_spherical(0, 0, ele_const_w=90)


def test_azimuth_wraps():
    assert aziel_to_screen_spherical(360, 65) == aziel_to_screen_spherical(0, 65)
    assert aziel_to_screen_spherical(-90, 65) == aziel_to_screen_spherical(270, 65)
