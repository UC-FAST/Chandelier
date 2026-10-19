"""PyAudio microphone stream used by the localiser.

Based on the PyTorch audio example, modified to support selecting which
channels of a multi-channel array to keep.
"""

from __future__ import annotations

import queue
from typing import List

import numpy as np


class MicrophoneStream:
    """Record from a multi-channel device, yielding ``(chunk, channels)``."""

    def __init__(
        self,
        rate: int,
        chunk: int,
        channels: int = 1,
        ignored_channels: List[int] | None = None,
        device=None,
    ):
        import pyaudio

        self._audio_interface = pyaudio.PyAudio()
        self._rate = rate
        self._chunk = chunk
        self._device = device
        self._channels = channels
        self._ignored_channels = ignored_channels
        # Unbounded queue written by PortAudio's callback thread; the
        # consumer below drains any backlog, so it never blocks the callback.
        self._buff: queue.Queue = queue.Queue()
        # Closed until __enter__ opens the stream, so iteration terminates
        # immediately if used without the context manager.
        self._closed = True
        self._terminated = False
        self._format_type = pyaudio.paFloat32
        self._audio_stream = None

    @property
    def rate(self) -> int:
        return self._rate

    @property
    def chunk(self) -> int:
        return self._chunk

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def ignored_channels(self) -> List[int] | None:
        return self._ignored_channels

    def __enter__(self) -> "MicrophoneStream":
        """Open the PortAudio input stream and start delivering chunks."""
        # ``device=None`` selects PortAudio's default input device.
        self._audio_stream = self._audio_interface.open(
            format=self._format_type,
            channels=self._channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            input_device_index=self._device,
            stream_callback=self._fill_buffer,
        )
        self._closed = False
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Idempotently stop the stream and release the audio device."""

        if self._closed and self._terminated:
            return
        if self._audio_stream is not None:
            try:
                self._audio_stream.stop_stream()
            finally:
                self._audio_stream.close()
                self._audio_stream = None
        self._closed = True
        # Wake a consumer blocked in __iter__; it exits on this sentinel.
        self._buff.put(None)
        if not self._terminated:
            self._audio_interface.terminate()
            self._terminated = True

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """PortAudio callback: enqueue one block of interleaved samples."""
        import pyaudio

        # Runs on PortAudio's real-time thread; only enqueue and return.
        self._buff.put(np.frombuffer(in_data, dtype=np.float32))
        return None, pyaudio.paContinue

    def __iter__(self):
        """Yield the newest buffered chunk, discarding any backlog."""
        while not self._closed:
            chunk = self._buff.get()
            if chunk is None:
                return

            # Consume everything currently buffered and keep the newest chunk.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                except queue.Empty:
                    break

            # PortAudio delivers interleaved samples; reshape to
            # (frames, channel) before dropping unwanted channels.
            chunk = np.reshape(chunk, (-1, self._channels))
            if self._ignored_channels:
                mask = np.ones(self._channels, dtype=bool)
                mask[self._ignored_channels] = False
                chunk = chunk[:, mask]

            yield chunk
