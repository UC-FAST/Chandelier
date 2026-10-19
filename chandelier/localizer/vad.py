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
    """Callable returning a speech probability for a mono audio chunk."""

    def __call__(self, chunk: Any, sample_rate: int) -> Any: ...


@dataclass(frozen=True)
class VoiceChunk:
    """One complete speech segment and the rate it was captured at."""

    sample_rate: int
    waveform: np.ndarray


def load_silero_model(model_dir: str | Path | None = None) -> SpeechModel:
    """Load the Silero VAD model from a local torch-hub checkout."""

    import torch

    # Cap intra-op threads: the Pi's four cores also run audio and the UI.
    torch.set_num_threads(4)
    if model_dir is None:
        # torch.hub's cache location; "source=local" keeps loading offline.
        model_dir = (
            Path.home()
            / ".cache"
            / "torch"
            / "hub"
            / "snakers4_silero-vad_master"
        )
    # The hub loader also returns utility helpers; only the model is used.
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

    # Silero expects exactly 512 samples per call at 16 kHz (256 at 8 kHz).
    chunk_size = 512 if microphone_stream.rate == 16000 else 256
    # The context manager guarantees the device is released on exit.
    with microphone_stream as stream:
        for chunk in stream:
            # Poll between chunks so shutdown is prompt.
            if stop_event is not None and stop_event.is_set():
                return

            # np.copy gives torch a contiguous buffer instead of a view
            # into the stream's reused chunk buffer.
            confidence = model(
                torch.from_numpy(np.copy(chunk[:chunk_size, 0])),
                stream.rate,
            ).item()

            # Re-read settings per chunk so live menu changes apply.
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

            # Keep a rolling pre-roll of non-speech audio; once speech has
            # started, keep accumulating through short pauses so an
            # utterance is not split.
            if is_speech or cumulated:
                cumulated.append(chunk)
            else:
                precumulated.append(chunk)

            # End after a pause once the segment is long enough, or at the
            # hard cap that bounds latency.
            finished = (
                (not is_speech and len(cumulated) >= min_to_cumulate)
                or len(cumulated) >= max_chunks
            )
            if finished:
                # Emit pre-roll + speech as one segment and reset both
                # buffers for the next utterance.
                waveform = np.concatenate(
                    (list(precumulated) + cumulated), axis=0
                )
                yield VoiceChunk(sample_rate=stream.rate, waveform=waveform)
                cumulated.clear()
                precumulated.clear()
