"""Voice-activity-based chunking of a microphone stream.

The Silero VAD loader is injectable so the chunking state machine can be
unit-tested without downloading the model.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generator, Protocol

import numpy as np


class SpeechModel(Protocol):
    def __call__(self, chunk: Any, sample_rate: int) -> Any: ...


@dataclass(frozen=True)
class VoiceChunk:
    sample_rate: int
    waveform: np.ndarray


def load_silero_model(model_dir: str | Path | None = None) -> SpeechModel:
    """Load the Silero VAD model from a local torch-hub checkout."""

    import torch

    torch.set_num_threads(4)
    if model_dir is None:
        model_dir = (
            Path.home()
            / ".cache"
            / "torch"
            / "hub"
            / "snakers4_silero-vad_master"
        )
    model, _ = torch.hub.load(
        repo_or_dir=str(model_dir),
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
        source="local",
    )
    return model


def get_microphone_chunks(
    microphone_stream,
    *,
    min_to_cumulate: int = 5,
    max_to_cumulate: int = 20,
    deque_size: int = 5,
    speech_threshold: float = 0.8,
    model: SpeechModel | None = None,
    model_loader: Callable[[], SpeechModel] = load_silero_model,
    stop_event=None,
    settings_provider: Callable[[], Any] | None = None,
) -> Generator[VoiceChunk, None, None]:
    """Yield complete speech segments from ``microphone_stream``.

    Both ``precumulated`` (audio just before speech) and ``cumulated`` (the
    speech itself) are concatenated, matching the behaviour of the original
    implementation.
    """

    import torch

    if model is None:
        model = model_loader()

    cumulated: list[np.ndarray] = []
    precumulated: deque[np.ndarray] = deque(maxlen=deque_size)

    chunk_size = 512 if microphone_stream.rate == 16000 else 256
    with microphone_stream as stream:
        for chunk in stream:
            if stop_event is not None and stop_event.is_set():
                return

            confidence = model(
                torch.from_numpy(np.copy(chunk[:chunk_size, 0])),
                stream.rate,
            ).item()

            settings = (
                settings_provider() if settings_provider is not None else None
            )
            threshold = (
                settings.speech_threshold
                if settings is not None
                else speech_threshold
            )
            max_chunks = (
                settings.max_to_cumulate
                if settings is not None
                else max_to_cumulate
            )
            is_speech = confidence >= threshold

            if is_speech or cumulated:
                cumulated.append(chunk)
            else:
                precumulated.append(chunk)

            finished = (
                (not is_speech and len(cumulated) >= min_to_cumulate)
                or len(cumulated) >= max_chunks
            )
            if finished:
                waveform = np.concatenate(
                    (list(precumulated) + cumulated), axis=0
                )
                yield VoiceChunk(sample_rate=stream.rate, waveform=waveform)
                cumulated.clear()
                precumulated.clear()
