import numpy as np

from chandelier.localizer.geometry import mic_array_positions
from chandelier.localizer.peaks import find_peak_indices


def test_finds_two_peaks_in_1d():
    values = np.array([0, 1, 0, 0, 2, 0, 0, 0])
    peaks = find_peak_indices(values, n_peaks=2, min_dist=1, threshold=0.1)
    assert peaks == [4, 1]


def test_finds_peak_in_2d():
    values = np.zeros((5, 5))
    values[1, 3] = 1.0
    values[4, 0] = 0.8
    peaks = find_peak_indices(values, n_peaks=2, min_dist=1, threshold=0.1)
    assert peaks[0] == [1, 3]


def test_mic_geometry_rotation_preserves_distances():
    original = mic_array_positions()
    rotated = mic_array_positions(45)
    assert original.shape == (4, 3)
    assert np.allclose(
        np.linalg.norm(original, axis=1),
        np.linalg.norm(rotated, axis=1),
    )
    assert not np.allclose(original, rotated)
