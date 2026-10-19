"""Microphone-array geometry for the ReSpeaker 4-mic array."""

from __future__ import annotations

import numpy as np

# Coordinates in metres, matching the ReSpeaker array used by the project.
# x/y span the square array plane (about 45.7 mm on a side) and z is the
# outward normal; the capsules sit 5 mm above the board origin.
_RESPEAKER_4MIC = np.array(
    [
        [+0.02285, +0.02285, +0.005],
        [-0.02285, +0.02285, +0.005],
        [-0.02285, -0.02285, +0.005],
        [+0.02285, -0.02285, +0.005],
    ],
    dtype=np.float64,
)


def mic_array_positions(rotation: float = 0.0) -> np.ndarray:
    """Return the four microphone coordinates, optionally rotated (degrees)."""

    positions = _RESPEAKER_4MIC.copy()
    if not rotation:
        return positions

    radians = np.deg2rad(rotation)
    sin, cos = np.sin(radians), np.cos(radians)
    # Row-vector rotation about z; positive values rotate the array
    # counter-clockwise when viewed from +z (the array normal).
    rotation_matrix = np.array(
        [[cos, sin, 0.0], [-sin, cos, 0.0], [0.0, 0.0, 1.0]]
    )
    return positions @ rotation_matrix
