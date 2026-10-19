"""Online and offline sound-source detection pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator, Protocol

import numpy as np

from chandelier.config import LocalizerConfig
from chandelier.localizer.bss import Duet
from chandelier.localizer.geometry import mic_array_positions
from chandelier.localizer.ssl import doa_detection
from chandelier.localizer.vad import SpeechModel, get_microphone_chunks


class RuntimeSettings(Protocol):
    """Mutable localizer fields read once per audio chunk."""

    n_sources: int
    confidence_threshold: float
    max_to_cumulate: int
    speech_threshold: float
    mic_rotation: float
    delay_max: float
    n_delay_bins: int


@dataclass(frozen=True)
class Detection:
    """One detected source, ordered by decreasing histogram prominence."""

    confidence: float
    azimuth: float
    elevation: float
    rank: int = 0


@dataclass(frozen=True)
class DetectionBatch:
    """Detections from one captured speech chunk."""

    confidence: float
    sample_rate: int
    detections: tuple[Detection, ...]

    @property
    def strongest(self) -> Detection | None:
        return self.detections[0] if self.detections else None


def detect_sources(
    waveform: np.ndarray,
    sample_rate: int,
    n_sources: int,
    *,
    rotation: float = 0.0,
    delay_max: float = 0.0,
    n_delay_bins: int = 50,
) -> tuple[Detection, ...]:
    """Separate ``waveform`` with DUET and estimate a DOA per source.

    ``waveform`` is expected in ``(time_step, n_channels)`` layout.
    """

    # Imported lazily: torch is only needed when a detection actually runs.
    import torch

    if n_sources == 1:
        # DUET's single-source mask keeps every time-frequency bin, so the
        # separation is a costly no-op.  Feeding the raw multichannel signal
        # to the DOA estimator is equivalent and removes a source of lag.
        estimates = np.asarray(waveform, dtype=np.float32)[None, ...]
    else:
        # Duet expects (n_channels, time_step); the pipeline is time-major.
        duet = Duet(
            waveform.transpose(),
            n_sources=n_sources,
            sample_rate=sample_rate,
            delay_max=delay_max,
            n_delay_bins=n_delay_bins,
            # Keep every channel so the DOA estimator can use the array.
            output_all_channels=True,
        )
        estimates = duet().astype(np.float32)
    if estimates.shape[0] == 0:
        return ()

    directions = doa_detection(
        mic_array_positions(rotation),
        torch.from_numpy(estimates),
        sample_rate=sample_rate,
    )
    if isinstance(directions, torch.Tensor):
        directions = directions.detach().cpu().numpy()

    # Batch confidence is a cheap loudness proxy (peak sample), not a
    # calibrated probability; the worker thresholds it for the overlay.
    confidence = (
        float(np.max(np.abs(waveform))) if waveform.size else 0.0
    )
    detections = []
    # Directions arrive ordered by decreasing histogram prominence; the
    # enumerate index becomes the marker rank downstream.
    for rank, (azimuth, elevation) in enumerate(directions):
        detections.append(
            Detection(
                confidence=confidence,
                azimuth=float(azimuth) % 360.0,
                elevation=float(elevation),
                rank=rank,
            )
        )
    return tuple(detections)


def iter_detections(
    microphone_stream,
    config: LocalizerConfig,
    *,
    model: SpeechModel | None = None,
    model_loader: Callable[[], SpeechModel] | None = None,
    stop_event=None,
    settings_provider: Callable[[], RuntimeSettings] | None = None,
) -> Generator[DetectionBatch, None, None]:
    """Continuously capture speech and yield detection batches.

    ``settings_provider`` may return a :class:`LocalizerSettings` snapshot;
    when present it is consulted for every audio chunk, so menu changes apply
    without restarting the microphone stream.
    """

    # Build the model lazily: loading Silero takes seconds and must happen
    # inside the worker process, not at import time.
    loader = model_loader
    if model is None and loader is None:
        from chandelier.localizer.vad import load_silero_model

        if config.silero_model_dir is not None:
            # Bind the configured checkout path into a zero-argument loader.
            loader = lambda: load_silero_model(config.silero_model_dir)
        else:
            loader = load_silero_model

    # `loader` is None only when a ready-made `model` was supplied; the
    # fallback keeps the chunker's callable parameter non-None.
    chunks = get_microphone_chunks(
        microphone_stream,
        max_to_cumulate=config.max_to_cumulate,
        speech_threshold=config.speech_threshold,
        model=model,
        model_loader=loader or load_silero_model,
        stop_event=stop_event,
        settings_provider=settings_provider,
    )
    for chunk in chunks:
        # Re-read the shared settings for every chunk so menu changes take
        # effect without reopening the microphone stream.
        settings = settings_provider() if settings_provider else None
        # Fall back to the startup config when no live settings exist.
        n_sources = (
            settings.n_sources if settings is not None else config.n_sources
        )
        rotation = (
            settings.mic_rotation
            if settings is not None
            else config.mic_rotation
        )
        delay_max = (
            settings.delay_max
            if settings is not None
            else config.delay_max
        )
        n_delay_bins = (
            settings.n_delay_bins
            if settings is not None
            else config.n_delay_bins
        )
        detections = detect_sources(
            chunk.waveform,
            chunk.sample_rate,
            n_sources,
            rotation=rotation,
            delay_max=delay_max,
            n_delay_bins=n_delay_bins,
        )
        yield DetectionBatch(
            confidence=float(np.max(np.abs(chunk.waveform))),
            sample_rate=chunk.sample_rate,
            detections=detections,
        )


def detect_file(
    path: str | Path,
    config: LocalizerConfig,
    *,
    n_sources: int | None = None,
) -> tuple[Detection, ...]:
    """Run the pipeline on a WAV file (offline CLI helper)."""

    # scipy is only needed for offline use, so import it here.
    from scipy.io.wavfile import read

    sample_rate, samples = read(str(path))
    samples = np.asarray(samples)

    if samples.ndim == 1:
        # Treat mono as a (time_step, 1) array to reuse the live path.
        samples = samples[:, None]

    # Infer the channel count from the file; an offline recording may have a
    # different layout from the live six-channel array.
    channels = samples.shape[1]
    mask = np.ones(channels, dtype=bool)
    ignored = [
        channel
        for channel in config.ignored_channels
        if 0 <= channel < channels
    ]
    if ignored:
        mask[ignored] = False

    # Boolean selection preserves channel order while dropping ignored
    # channels.
    waveform = samples[:, mask]
    return detect_sources(
        waveform,
        sample_rate,
        n_sources if n_sources is not None else config.n_sources,
        rotation=config.mic_rotation,
        delay_max=config.delay_max,
        n_delay_bins=config.n_delay_bins,
    )
