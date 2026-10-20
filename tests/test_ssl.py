"""Tests for the SRP direction-of-arrival estimator."""

import math

import numpy as np
import pytest

from chandelier.localizer.geometry import mic_array_positions
from chandelier.localizer.ssl import (
    _local_cap,
    _score_directions,
    _score_directions_batch,
    doa_detection,
)

SAMPLE_RATE = 16000
DURATION = 1.2
SPEED_OF_SOUND = 343.0


def _direction(azimuth: float, elevation: float) -> np.ndarray:
    azimuth = math.radians(azimuth)
    elevation = math.radians(elevation)
    return np.array(
        [
            math.cos(azimuth) * math.cos(elevation),
            math.sin(azimuth) * math.cos(elevation),
            math.sin(elevation),
        ]
    )


def _synthetic_source(
    azimuth: float,
    elevation: float,
    *,
    distance: float | None = None,
    band: tuple[float, float] | None = (300.0, 3400.0),
    noise: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    """Return a ``(1, time, channels)`` speech-band source in a direction."""

    rng = np.random.default_rng(seed)
    samples = rng.standard_normal(int(SAMPLE_RATE * DURATION))
    if band is not None:
        spectrum = np.fft.rfft(samples)
        frequencies = np.fft.rfftfreq(samples.size, 1.0 / SAMPLE_RATE)
        spectrum[(frequencies < band[0]) | (frequencies > band[1])] = 0
        samples = np.fft.irfft(spectrum, samples.size)

    microphones = mic_array_positions()
    direction = _direction(azimuth, elevation)
    if distance is None:
        delays = -(microphones @ direction) / SPEED_OF_SOUND
    else:
        source = distance * direction
        ranges = np.linalg.norm(source - microphones, axis=1)
        delays = ranges / SPEED_OF_SOUND
    delays = delays - delays.min()

    source_spectrum = np.fft.rfft(samples)
    frequencies = np.fft.rfftfreq(samples.size, 1.0 / SAMPLE_RATE)
    channels = [
        np.fft.irfft(
            source_spectrum * np.exp(-2j * np.pi * frequencies * delay),
            samples.size,
        )
        for delay in delays
    ]
    waveform = np.stack(channels, axis=1)
    waveform += noise * rng.standard_normal(waveform.shape)
    return waveform.astype(np.float32)[None, ...]


def _circular_error(actual: float, expected: float) -> float:
    return abs((actual - expected + 180.0) % 360.0 - 180.0)


@pytest.mark.parametrize(
    ("azimuth", "elevation", "distance"),
    [
        (10.0, 73.0, None),
        (154.0, 73.0, None),
        (270.0, 45.0, None),
        (10.0, 73.0, 0.2),
        (180.0, 30.0, 0.15),
    ],
)
def test_speech_band_source_is_localised(azimuth, elevation, distance):
    # Regression: pure PHAT weights every frequency bin equally, so bands
    # that only contain microphone noise pull the result towards the array
    # normal.  Speech-band input used to collapse to elevation ~0.
    waveform = _synthetic_source(
        azimuth, elevation, distance=distance
    )
    result = doa_detection(
        mic_array_positions(), waveform, sample_rate=SAMPLE_RATE
    ).numpy()

    assert result.shape == (1, 2)
    assert _circular_error(float(result[0, 0]), azimuth) < 3.0
    assert abs(float(result[0, 1]) - elevation) < 3.0


def test_empty_waveform_returns_no_directions():
    result = doa_detection(
        mic_array_positions(),
        np.zeros((1, 0, 4), dtype=np.float32),
        sample_rate=SAMPLE_RATE,
    )
    assert result.shape == (1, 0)


def _local_cap_reference(center, radius_deg, step_deg):
    """Scalar implementation kept as a correctness reference."""

    import torch

    center = center / center.norm()
    helper = (
        torch.tensor([1.0, 0.0, 0.0])
        if abs(float(center[0])) < 0.9
        else torch.tensor([0.0, 1.0, 0.0])
    )
    tangent_u = torch.linalg.cross(center, helper)
    tangent_u = tangent_u / tangent_u.norm()
    tangent_v = torch.linalg.cross(center, tangent_u)

    radius = math.radians(radius_deg)
    step = math.radians(step_deg)
    rings = max(1, int(round(radius / step)))
    points = [center]
    for ring in range(1, rings + 1):
        ring_radius = ring * step
        count = max(6, int(round(2.0 * math.pi * ring_radius / step)))
        for index in range(count):
            phi = 2.0 * math.pi * index / count
            point = (
                math.cos(ring_radius) * center
                + math.sin(ring_radius)
                * (math.cos(phi) * tangent_u + math.sin(phi) * tangent_v)
            )
            if float(point[2]) >= -1e-6:
                points.append(point)
    return torch.stack(points)


@pytest.mark.parametrize(
    "center",
    [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.3, 0.4, 0.866),
        (-0.2, -0.5, 0.84),
    ],
)
def test_local_cap_matches_scalar_reference(center):
    # The vectorised cap must produce the same direction set as the original
    # per-point loop; only the construction strategy changed.
    import torch

    point = torch.tensor(center, dtype=torch.float32)
    actual = _local_cap(point, 6.5, 0.5)
    expected = _local_cap_reference(point, 6.5, 0.5)

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-5)


def test_batched_scoring_matches_per_source():
    import torch

    torch.manual_seed(0)
    directions = torch.nn.functional.normalize(
        torch.randn(200, 3), dim=1
    )
    deltas = torch.randn(6, 3) * 1e-4
    freqs = torch.linspace(100.0, 7800.0, 199)
    weights = torch.randn(3, 6, 199, dtype=torch.complex64)

    batch = _score_directions_batch(
        directions, weights, deltas, freqs
    )
    per_source = torch.stack(
        [
            _score_directions(directions, weights[source], deltas, freqs)
            for source in range(weights.shape[0])
        ]
    )

    assert batch.shape == per_source.shape
    assert torch.allclose(batch, per_source, atol=2e-3, rtol=1e-4)
