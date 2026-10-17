"""Steered Response Power - PHAT direction-of-arrival estimation."""

from __future__ import annotations

from typing import Any


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
    from speechbrain.processing.features import STFT
    from speechbrain.processing.multi_mic import Covariance, SrpPhat

    if not isinstance(mics, torch.Tensor):
        mics = torch.as_tensor(mics, dtype=torch.float32)
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform, dtype=torch.float32)

    stft = STFT(sample_rate=sample_rate)
    covariance = Covariance()
    srp_phat = SrpPhat(mics=mics)

    spectra = stft(waveform)
    covariance_matrix = covariance(spectra)
    directions = srp_phat(covariance_matrix)
    directions = directions[:, 0, :]
    directions[:, 2] = directions[:, 2].abs()

    radius = (directions[:, 0] ** 2 + directions[:, 1] ** 2).sqrt()
    azimuth = torch.atan2(directions[:, 1], directions[:, 0]).rad2deg()
    elevation = torch.atan2(directions[:, 2], radius).rad2deg()

    return torch.column_stack((azimuth, elevation))
