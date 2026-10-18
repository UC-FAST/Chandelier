"""Tests for the SRP direction-of-arrival estimator."""

import math

import numpy as np
import pytest

from chandelier.localizer.geometry import mic_array_positions
from chandelier.localizer.ssl import doa_detection

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
