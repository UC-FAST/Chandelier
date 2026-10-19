"""Multiprocessing worker that keeps the latest localisation results.

The child process owns the microphone stream and the heavy ML libraries.  The
parent reads a small shared snapshot (up to ``MAX_SOURCES`` detected sources),
which avoids the per-frame ``Manager().list`` round-trips of the original
implementation.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Any, Iterable

from chandelier.common.geometry import (
    apply_screen_gain,
    aziel_to_screen_spherical,
)
from chandelier.config import LocalizerConfig

# Fixed capacity of the shared snapshot; the UI never allocates per read.
MAX_SOURCES = 8

# Layout of the shared double array.  Keep in sync with LocalizerSettings:
# values are packed positionally, not by name.
_FIELD_ORDER = (
    "n_sources",
    "confidence_threshold",
    "max_to_cumulate",
    "speech_threshold",
    "mic_rotation",
    "delay_max",
    "n_delay_bins",
    "marker_fov",
    "marker_azimuth_offset",
    "marker_elevation_offset",
    "marker_x_gain",
    "marker_y_gain",
    "marker_mic_offset_x",
    "marker_mic_offset_y",
    "marker_mic_offset_z",
    "marker_source_distance",
)
_INTEGER_FIELDS = frozenset({"n_sources", "max_to_cumulate", "n_delay_bins"})
# Numeric bounds applied to values arriving from the UI.  Angles that are
# wrapped (mic_rotation, marker_azimuth_offset) are handled in _normalize
# instead of being clamped.
_LIMITS = {
    "n_sources": (1.0, float(MAX_SOURCES)),
    "confidence_threshold": (0.0, 1.0),
    "max_to_cumulate": (1.0, 50.0),
    "speech_threshold": (0.0, 1.0),
    "delay_max": (0.0, 64.0),
    "n_delay_bins": (2.0, 512.0),
    "marker_fov": (1.0, 179.0),
    "marker_elevation_offset": (-90.0, 90.0),
    "marker_x_gain": (0.0, 3.0),
    "marker_y_gain": (0.0, 3.0),
    "marker_mic_offset_x": (-10.0, 10.0),
    "marker_mic_offset_y": (-10.0, 10.0),
    "marker_mic_offset_z": (-10.0, 10.0),
    "marker_source_distance": (0.0, 100.0),
}


@dataclass(frozen=True)
class LocalizerSettings:
    """Runtime-tunable subset of :class:`LocalizerConfig`.

    Stream parameters (device, rate, channels, ...) cannot change without
    reopening the microphone and are deliberately not part of this snapshot.
    """

    n_sources: int = 3
    confidence_threshold: float = 0.03
    max_to_cumulate: int = 5
    speech_threshold: float = 0.0
    mic_rotation: float = 0.0
    delay_max: float = 0.0
    n_delay_bins: int = 50
    marker_fov: float = 83.0
    marker_azimuth_offset: float = 0.0
    marker_elevation_offset: float = 0.0
    marker_x_gain: float = 0.0
    marker_y_gain: float = 0.0
    marker_mic_offset_x: float = 0.0
    marker_mic_offset_y: float = 0.0
    marker_mic_offset_z: float = 0.0
    marker_source_distance: float = 0.0

    @classmethod
    def from_config(cls, config: LocalizerConfig) -> "LocalizerSettings":
        """Copy the runtime-tunable fields out of a config object."""

        return cls(
            n_sources=config.n_sources,
            confidence_threshold=config.confidence_threshold,
            max_to_cumulate=config.max_to_cumulate,
            speech_threshold=config.speech_threshold,
            mic_rotation=config.mic_rotation,
            delay_max=config.delay_max,
            n_delay_bins=config.n_delay_bins,
            marker_fov=config.marker_fov,
            marker_azimuth_offset=config.marker_azimuth_offset,
            marker_elevation_offset=config.marker_elevation_offset,
            marker_x_gain=config.marker_x_gain,
            marker_y_gain=config.marker_y_gain,
            marker_mic_offset_x=config.marker_mic_offset_x,
            marker_mic_offset_y=config.marker_mic_offset_y,
            marker_mic_offset_z=config.marker_mic_offset_z,
            marker_source_distance=config.marker_source_distance,
        )


class SharedLocalizerSettings:
    """Lock-protected settings shared between the UI and worker processes."""

    def __init__(
        self,
        config: LocalizerConfig,
        context: Any | None = None,
    ):
        context = context or mp.get_context()
        initial = LocalizerSettings.from_config(config)
        # One shared array of doubles holds the whole snapshot; callers
        # must create the worker process from the same context so the
        # child can inherit this memory.
        self._values = context.Array(
            "d", [float(getattr(initial, field)) for field in _FIELD_ORDER]
        )
        self._generation = context.Value("i", 0)

    @property
    def generation(self) -> int:
        return int(self._generation.value)

    def update(self, **values: Any) -> bool:
        """Apply the provided fields; unknown names raise ``KeyError``.

        Values are normalised (angles wrapped, integers rounded, numeric
        limits clamped) before they are stored.  Returns True when the
        snapshot actually changed.
        """

        unknown = set(values).difference(_FIELD_ORDER)
        if unknown:
            raise KeyError(
                f"unknown localizer setting(s): {', '.join(sorted(unknown))}"
            )

        changed = False
        with self._values.get_lock():
            for index, field in enumerate(_FIELD_ORDER):
                if field not in values:
                    continue
                value = _normalize(field, values[field])
                # Only bump the generation when something really changed,
                # so readers do not reprocess identical settings.
                if self._values[index] != value:
                    self._values[index] = value
                    changed = True
            if changed:
                self._generation.value += 1
        return changed

    def snapshot(self) -> LocalizerSettings:
        """Return a consistent copy of the shared values."""

        # Copy while holding the lock, then convert outside it.
        with self._values.get_lock():
            values = [
                self._values[index] for index in range(len(_FIELD_ORDER))
            ]
        result = {}
        for field, value in zip(_FIELD_ORDER, values):
            result[field] = int(value) if field in _INTEGER_FIELDS else float(value)
        return LocalizerSettings(**result)


def _normalize(field: str, value: Any) -> float:
    """Coerce a UI value into the range and type the field expects."""

    number = float(value)
    if field in {"mic_rotation", "marker_azimuth_offset"}:
        # Angles are periodic, so wrap instead of clamping.
        return number % 360.0
    if field in _INTEGER_FIELDS:
        # Shared storage is double; round so int fields stay integral.
        number = float(round(number))
    low, high = _LIMITS[field]
    # Keep out-of-range values from reaching the DSP stages.
    return min(high, max(low, number))


@dataclass(frozen=True)
class SourceSnapshot:
    confidence: float
    azimuth: float
    elevation: float
    screen_x: int
    screen_y: int
    rank: int = 0


@dataclass(frozen=True)
class LocalizationBatch:
    confidence: float
    generation: int
    sources: tuple[SourceSnapshot, ...]

    @property
    def strongest(self) -> SourceSnapshot | None:
        return self.sources[0] if self.sources else None


class LatestLocalization:
    """Shared, lock-protected "latest detection batch" cell."""

    def __init__(self, context: Any | None = None):
        context = context or mp.get_context()
        self._count = context.Value("i", 0)
        self._confidence = context.Value("d", 0.0)
        self._angles = context.Array("d", [0.0] * (2 * MAX_SOURCES))
        self._screen = context.Array("i", [0] * (2 * MAX_SOURCES))
        self._generation = context.Value("i", 0)
        self._lock = context.Lock()

    def write(
        self, confidence: float, sources: Iterable[SourceSnapshot]
    ) -> None:
        """Publish a batch; sources beyond MAX_SOURCES are dropped."""

        source_list = list(sources)[:MAX_SOURCES]
        # Everything is written under one lock so a reader never sees a
        # half-updated batch.
        with self._lock:
            self._count.value = len(source_list)
            self._confidence.value = float(confidence)
            for index, source in enumerate(source_list):
                self._angles[2 * index] = float(source.azimuth)
                self._angles[2 * index + 1] = float(source.elevation)
                self._screen[2 * index] = int(source.screen_x)
                self._screen[2 * index + 1] = int(source.screen_y)
            self._generation.value += 1

    def read(self) -> LocalizationBatch | None:
        """Return the latest batch, or None until the first write."""

        with self._lock:
            # generation == 0 means that no batch has ever been published.
            if self._generation.value == 0:
                return None

            # Clamp defensively in case capacity ever changes.
            count = min(int(self._count.value), MAX_SOURCES)
            confidence = float(self._confidence.value)
            sources = tuple(
                SourceSnapshot(
                    confidence=confidence,
                    azimuth=float(self._angles[2 * index]),
                    elevation=float(self._angles[2 * index + 1]),
                    screen_x=int(self._screen[2 * index]),
                    screen_y=int(self._screen[2 * index + 1]),
                    rank=index,
                )
                for index in range(count)
            )
            return LocalizationBatch(
                confidence=confidence,
                generation=int(self._generation.value),
                sources=sources,
            )

    def clear(self) -> None:
        """Mark the snapshot empty and advance the generation."""

        with self._lock:
            self._count.value = 0
            self._confidence.value = 0.0
            for index in range(MAX_SOURCES):
                self._angles[2 * index] = 0.0
                self._angles[2 * index + 1] = 0.0
                self._screen[2 * index] = 0
                self._screen[2 * index + 1] = 0
            # Bump the generation so readers observe the cleared state.
            self._generation.value += 1


@dataclass
class LocalizerHandle:
    """Parent-side handle bundling the worker process and shared state."""

    process: Any
    latest: LatestLocalization
    stop_event: Any
    settings: SharedLocalizerSettings | None = None

    def update_settings(self, **values: Any) -> bool:
        """Forward live menu changes to the worker process."""

        if self.settings is None:
            return False
        return self.settings.update(**values)

    def stop(self, timeout: float = 2.0) -> None:
        """Ask the worker to exit, terminating it if it overruns."""

        self.stop_event.set()
        self.process.join(timeout=timeout)
        if self.process.is_alive():
            # Escalate if the worker ignored the event (for example while
            # blocked in a native audio or ML call).
            self.process.terminate()
            self.process.join(timeout=timeout)


def run_worker(
    config: LocalizerConfig,
    screen_size: tuple[int, int],
    latest: LatestLocalization,
    stop_event,
    settings: SharedLocalizerSettings | None = None,
) -> None:
    """Child-process entry point.  Must stay importable and picklable."""

    # Imported here so the parent process does not pull in torch or the
    # audio hardware just to spawn the worker.
    from chandelier.hardware.microphone import MicrophoneStream
    from chandelier.localizer.pipeline import iter_detections
    from chandelier.log import configure_logging, get_logger

    configure_logging()
    logger = get_logger(__name__)
    stream = None
    # Bind the bound method once; it is called per chunk and per batch.
    settings_provider = (
        settings.snapshot if settings is not None else None
    )

    try:
        # Open the stream in the child: the parent never touches ALSA.
        stream = MicrophoneStream(
            rate=config.rate,
            chunk=config.chunk,
            channels=config.channels,
            ignored_channels=list(config.ignored_channels),
            device=config.device,
        )
        for batch in iter_detections(
            stream,
            config,
            stop_event=stop_event,
            settings_provider=settings_provider,
        ):
            if stop_event.is_set():
                break
            # Re-read settings per batch so menu changes apply without a
            # worker restart.
            current = (
                settings_provider()
                if settings_provider is not None
                else None
            )
            # Drop quiet or empty batches before paying for the projection.
            threshold = (
                current.confidence_threshold
                if current is not None
                else config.confidence_threshold
            )
            if not batch.detections or batch.confidence <= threshold:
                continue

            marker_fov = (
                current.marker_fov if current is not None else config.marker_fov
            )
            marker_azimuth_offset = (
                current.marker_azimuth_offset
                if current is not None
                else config.marker_azimuth_offset
            )
            marker_elevation_offset = (
                current.marker_elevation_offset
                if current is not None
                else config.marker_elevation_offset
            )
            mic_offset = (
                (
                    current.marker_mic_offset_x,
                    current.marker_mic_offset_y,
                    current.marker_mic_offset_z,
                )
                if current is not None
                else (
                    config.marker_mic_offset_x,
                    config.marker_mic_offset_y,
                    config.marker_mic_offset_z,
                )
            )
            source_distance = (
                current.marker_source_distance
                if current is not None
                else config.marker_source_distance
            )
            x_gain = (
                current.marker_x_gain
                if current is not None
                else config.marker_x_gain
            )
            y_gain = (
                current.marker_y_gain
                if current is not None
                else config.marker_y_gain
            )
            # Project directions to screen pixels here so the parent only
            # has to copy integers.
            sources = []
            for rank, detection in enumerate(batch.detections[:MAX_SOURCES]):
                screen_x, screen_y = aziel_to_screen_spherical(
                    detection.azimuth,
                    detection.elevation,
                    width=screen_size[0],
                    height=screen_size[1],
                    fov=marker_fov,
                    azimuth_offset=marker_azimuth_offset,
                    elevation_offset=marker_elevation_offset,
                    mic_offset=mic_offset,
                    source_distance=source_distance,
                )
                screen_x, screen_y = apply_screen_gain(
                    screen_x,
                    screen_y,
                    width=screen_size[0],
                    height=screen_size[1],
                    gain_x=x_gain,
                    gain_y=y_gain,
                )
                sources.append(
                    SourceSnapshot(
                        confidence=batch.confidence,
                        azimuth=detection.azimuth,
                        elevation=detection.elevation,
                        screen_x=screen_x,
                        screen_y=screen_y,
                        rank=rank,
                    )
                )
            # Publish the whole batch atomically for the UI thread.
            latest.write(batch.confidence, sources)
    except KeyboardInterrupt:
        pass
    except Exception:
        logger.exception("localisation worker stopped unexpectedly")
    finally:
        # Always release the microphone, even after an unexpected error.
        if stream is not None:
            try:
                stream.close()
            except Exception:
                logger.debug(
                    "failed to close microphone stream", exc_info=True
                )


def _context_name() -> str:
    # fork is cheaper and avoids re-importing torch on Linux; Windows has no
    # fork, so fall back to spawn there.
    return "fork" if os.name != "nt" else "spawn"


def start_worker(
    config: LocalizerConfig,
    screen_size: tuple[int, int],
    *,
    context_name: str | None = None,
) -> LocalizerHandle:
    """Create the shared state and start the localizer child process."""

    # All primitives must come from the same context as the process so
    # they can be inherited by the child.
    context = mp.get_context(context_name or _context_name())
    latest = LatestLocalization(context)
    stop_event = context.Event()
    settings = SharedLocalizerSettings(config, context)
    # Daemon worker: it must not outlive the process that owns the display.
    process = context.Process(
        target=run_worker,
        args=(config, screen_size, latest, stop_event, settings),
        daemon=True,
        name="chandelier-localizer",
    )
    process.start()
    return LocalizerHandle(
        process=process,
        latest=latest,
        stop_event=stop_event,
        settings=settings,
    )
