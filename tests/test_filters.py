import pytest

from chandelier.common.filters import SlidingWindowFilter


def test_moving_average():
    window = SlidingWindowFilter(3)
    assert window.add_data(2).add_data(4).add_data(6).calc() == 4
    window.add_data(8)
    assert window.calc() == 6


def test_empty_window_raises():
    with pytest.raises(ValueError):
        SlidingWindowFilter(3).calc()


def test_invalid_size():
    with pytest.raises(ValueError):
        SlidingWindowFilter(0)


def test_clear():
    window = SlidingWindowFilter(2)
    window.add_data(1)
    window.clear()
    assert len(window) == 0
    with pytest.raises(ValueError):
        window.calc()
