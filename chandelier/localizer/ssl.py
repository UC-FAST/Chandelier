"""Steered Response Power direction-of-arrival estimation.

The estimator steers a plane-wave model over a hemisphere of candidate
directions and picks the direction whose predicted inter-microphone phases
best match the observed cross-spectra.  Two details matter for accuracy with
speech-band audio:

* Frequency bins are weighted by their magnitude (generalised PHAT with
  ``beta < 1``).  Classic PHAT (``beta = 1``) whitens every bin, including
  bands that only contain microphone noise, and those noisy bins then pull
  the estimate towards the array normal.
* A coarse hemisphere search is followed by a local refinement.  The coarse
  grid alone quantises the result to several degrees, which is very visible
  when the direction is projected onto the camera image.

The array is planar, so the front/back sign of the elevation is inherently
ambiguous; the estimator reports the source in front of the array
(``elevation`` in ``[0, 90]``).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

_EPSILON = 1e-12
_SPEED_OF_SOUND = 343.0

# STFT resolution: 25 ms window, 10 ms hop (same as the previous estimator).
_FRAME_SECONDS = 0.025
_HOP_SECONDS = 0.010

# Generalised PHAT exponent.  0 is plain SRP (magnitude-weighted cross
# spectra), 1 is classic PHAT.  Measured on recorded speech-band material,
# values around 0.3 keep the direct path dominant without amplifying noise.
_BETA = 0.3

# 600 Fibonacci points leave a worst-case nearest-neighbour gap of 5.8°.
# The refinement cap is slightly larger so the true peak always lies inside
# it; the 0.5° refinement step then removes the remaining quantisation.
# Measured on the recorded `res/data` set, this matches the previous 2000
# point grid within 0.6° while halving the detection time.
_COARSE_POINTS = 600
_REFINE_RADIUS_DEG = 6.5
_REFINE_STEP_DEG = 0.5


def _fibonacci_hemisphere(count: int):
    """Return ``count`` quasi-uniform unit vectors with ``z >= 0``."""

    import torch

    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    # Half-sample offset avoids a point exactly on the pole and makes the
    # spiral symmetric; z runs from ~1 down to ~0 (upper hemisphere only).
    index = torch.arange(count, dtype=torch.float32) + 0.5
    z = 1.0 - index / count
    radius = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    theta = golden_angle * index
    return torch.stack(
        (radius * torch.cos(theta), radius * torch.sin(theta), z), dim=1
    )


def _local_cap(center, radius_deg: float, step_deg: float):
    """Return refinement directions inside a cone around ``center``."""

    import torch

    # Build the whole cap with NumPy.  A per-point torch loop produced the
    # same 300-odd directions but cost ~20 ms per call, which dominated the
    # DOA estimate; the vectorised version takes well under 1 ms.
    centre = np.asarray(center.detach().cpu().numpy(), dtype=np.float64)
    centre = centre / np.linalg.norm(centre)
    # Pick a helper axis that is not nearly parallel to the centre so the
    # tangent basis is well conditioned.
    helper = (
        np.array([1.0, 0.0, 0.0])
        if abs(float(centre[0])) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    tangent_u = np.cross(centre, helper)
    tangent_u = tangent_u / np.linalg.norm(tangent_u)
    tangent_v = np.cross(centre, tangent_u)

    radius = math.radians(radius_deg)
    step = math.radians(step_deg)
    # One ring per step of angular radius; each ring is sampled at roughly
    # constant arc length, with at least six points to avoid degeneracy.
    rings = max(1, int(round(radius / step)))
    points = [centre[None, :]]
    for ring in range(1, rings + 1):
        ring_radius = ring * step
        count = max(6, int(round(2.0 * math.pi * ring_radius / step)))
        phi = 2.0 * math.pi * np.arange(count) / count
        ring_points = (
            math.cos(ring_radius) * centre
            + math.sin(ring_radius)
            * (
                np.cos(phi)[:, None] * tangent_u
                + np.sin(phi)[:, None] * tangent_v
            )
        )
        # The planar array cannot tell front from back; stay in front.
        points.append(ring_points[ring_points[:, 2] >= -1e-6])

    return torch.as_tensor(
        np.concatenate(points), dtype=torch.float32, device=center.device
    )


def _score_directions(directions, weights, deltas, freqs):
    """Correlate candidate directions with the measured cross-spectra."""

    import torch

    scores = torch.zeros(directions.shape[0], dtype=torch.float32)
    omega = 2.0 * math.pi * freqs
    for pair in range(deltas.shape[0]):
        # Project the candidate direction onto the pair's displacement
        # vector: this is the TDOA for a unit-amplitude plane wave.
        delays = directions @ deltas[pair]
        angle = delays[:, None] * omega[None, :]
        weight = weights[pair]
        # Correlate with the measured cross-spectrum: Re(W * e^{j angle})
        # is the phase alignment of that pair at this direction.
        scores += (
            weight.real[None, :] * torch.cos(angle)
            + weight.imag[None, :] * torch.sin(angle)
        ).sum(dim=1)
    return scores


def _score_directions_batch(directions, weights, deltas, freqs):
    """Score every source against the coarse grid in one pass.

    ``weights`` has shape ``(n_sources, n_pairs, n_freq)`` and the result is
    ``(n_sources, n_directions)``.  Sharing the per-pair delay/sin/cos work
    across sources avoids re-launching the same kernels once per source;
    measured on the Pi this is ~1.8x faster for three sources.
    """

    import torch

    scores = torch.zeros(
        (weights.shape[0], directions.shape[0]),
        dtype=torch.float32,
        device=directions.device,
    )
    omega = 2.0 * math.pi * freqs
    for pair in range(deltas.shape[0]):
        # The direction-to-pair projection is source-independent, so compute
        # it once and broadcast over the source axis below.
        delays = directions @ deltas[pair]
        angle = delays[:, None] * omega[None, :]
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        weight = weights[:, pair]
        scores += (
            weight.real[:, None, :] * cos_angle[None, :, :]
            + weight.imag[:, None, :] * sin_angle[None, :, :]
        ).sum(dim=2)
    return scores


def doa_detection(mics: Any, waveform: Any, sample_rate: int = 16000):
    """Estimate ``(azimuth, elevation)`` for every source in ``waveform``.

    Parameters
    ----------
    mics:
        ``(n_mics, 3)`` microphone coordinates in metres.  A numpy array or a
        torch tensor is accepted.
    waveform:
        ``(n_sources, time_step, n_channels)`` tensor/array.
    sample_rate:
        Sample rate of ``waveform``.

    Returns
    -------
    torch.Tensor
        ``(n_sources, 2)`` tensor of azimuth/elevation pairs in degrees.
    """

    import torch

    # Accept numpy arrays or tensors; flatten to (n_mics, 3).
    mics_t = torch.as_tensor(mics, dtype=torch.float32).reshape(-1, 3)
    wave = torch.as_tensor(waveform, dtype=torch.float32)
    if wave.ndim != 3:
        raise ValueError(
            "waveform must have shape (n_sources, time_step, n_channels)"
        )

    n_sources, time_steps, n_channels = wave.shape
    # A single channel or no samples carries no spatial information.
    if n_channels < 2 or time_steps == 0:
        return torch.empty((n_sources, 0), dtype=torch.float32)

    win_length = max(16, int(round(sample_rate * _FRAME_SECONDS)))
    hop_length = max(1, int(round(sample_rate * _HOP_SECONDS)))
    if time_steps < win_length:
        # Pad to one full analysis window so torch.stft returns a frame.
        wave = torch.nn.functional.pad(
            wave, (0, 0, 0, win_length - time_steps)
        )

    # One batched STFT for all sources and channels, then reshape back.
    spectrum = torch.stft(
        wave.transpose(1, 2).reshape(n_sources * n_channels, -1),
        n_fft=win_length,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hamming_window(
            win_length, periodic=True, dtype=torch.float32
        ),
        center=True,
        onesided=True,
        return_complex=True,
    ).reshape(n_sources, n_channels, win_length // 2 + 1, -1)
    # spectrum: (n_sources, n_channels, n_freq, n_frames)
    # Drop DC: its phase is always zero and would bias the phase fit.
    freqs = torch.fft.rfftfreq(
        win_length, d=1.0 / float(sample_rate)
    )[1:]
    spectrum = spectrum[..., 1:, :]

    # Unique microphone pairs; self-pairs would add only constant terms.
    pair_i, pair_j = torch.triu_indices(
        n_channels, n_channels, offset=1
    )
    # Cross-spectrum per pair, averaged over frames to reduce noise.
    cross = spectrum[:, pair_i] * spectrum[:, pair_j].conj()
    weights = cross.mean(dim=-1)
    if _BETA:
        # Generalised PHAT: divide by |W|^beta to keep phase information
        # while taming low-energy bins.
        weights = weights / (weights.abs().pow(_BETA) + _EPSILON)
    # Pair displacement / speed of sound: seconds of delay per unit of
    # projection onto a candidate direction.
    deltas = (mics_t[pair_i] - mics_t[pair_j]) / _SPEED_OF_SOUND

    # Fibonacci hemisphere gives near-uniform coverage without grid bias.
    coarse = _fibonacci_hemisphere(_COARSE_POINTS)
    # Score all sources before refining: the coarse pass is the expensive
    # part and its direction-to-pair projections are shared by every source.
    coarse_scores = _score_directions_batch(
        coarse, weights, deltas, freqs
    )
    detections = []
    for source in range(n_sources):
        source_weights = weights[source]
        best = coarse[int(torch.argmax(coarse_scores[source]))]

        # Refine inside a small cap to remove the coarse grid quantisation.
        refinement = _local_cap(
            best, _REFINE_RADIUS_DEG, _REFINE_STEP_DEG
        )
        refined_scores = _score_directions(
            refinement, source_weights, deltas, freqs
        )
        best = refinement[int(torch.argmax(refined_scores))]

        x, y, z = (float(value) for value in best)
        # z >= 0 by construction, so elevation stays in [0, 90] degrees.
        azimuth = math.degrees(math.atan2(y, x)) % 360.0
        elevation = math.degrees(
            math.atan2(z, math.hypot(x, y))
        )
        detections.append((azimuth, elevation))

    return torch.tensor(detections, dtype=torch.float32)
