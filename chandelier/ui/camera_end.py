"""Camera preview screen with pro capture modes and overlays."""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np

from chandelier.common.filters import SlidingWindowFilter
from chandelier.config import AppConfig
from chandelier.constants import EndId, MenuOptionId
from chandelier.hardware.camera import CameraDevice
from chandelier.i18n import tr
from chandelier.hardware.leds import Leds
from chandelier.ui.capture_session import CaptureSession, CaptureSettings
from chandelier.ui.controlled_end import ControlledEnd
from chandelier.ui.decorators import (
    BarChart,
    Busy,
    CenterMark,
    Colors,
    Countdown,
    Histogram,
    RecordingBadge,
    RuleOfThirds,
    SimpleText,
    StatusBar,
    StatusItem,
    Toast,
    Watermark,
    ZebraOverlay,
    corner_padding,
)


# The 320x240 / 30 fps preview mode exposes 1..66666 us.  Longer presets are
# applied only while the still configuration is active.
_PREVIEW_MAX_EXPOSURE = 66_666

# Standard third-stop shutter speeds (seconds) and ISO values.  Auto
# exposure reports a slightly different exposure/gain every frame; snapping
# the status-bar text to these stops keeps adjacent fractions from flickering.
_SHUTTER_STOPS = (
    1 / 100000,
    1 / 80000,
    1 / 64000,
    1 / 50000,
    1 / 40000,
    1 / 32000,
    1 / 25000,
    1 / 20000,
    1 / 16000,
    1 / 12500,
    1 / 10000,
    1 / 8000,
    1 / 6400,
    1 / 5000,
    1 / 4000,
    1 / 3200,
    1 / 2500,
    1 / 2000,
    1 / 1600,
    1 / 1250,
    1 / 1000,
    1 / 800,
    1 / 640,
    1 / 500,
    1 / 400,
    1 / 320,
    1 / 250,
    1 / 200,
    1 / 160,
    1 / 125,
    1 / 100,
    1 / 80,
    1 / 60,
    1 / 50,
    1 / 40,
    1 / 30,
    1 / 25,
    1 / 20,
    1 / 15,
    1 / 13,
    1 / 10,
    1 / 8,
    1 / 6,
    1 / 5,
    1 / 4,
    0.3,
    0.4,
    0.5,
    0.6,
    0.8,
    1.0,
    1.3,
    1.6,
    2.0,
    2.5,
    3.2,
    4.0,
    5.0,
    6.0,
    8.0,
    10.0,
    13.0,
    15.0,
    20.0,
    25.0,
    30.0,
)
_ISO_STOPS = (
    100,
    125,
    160,
    200,
    250,
    320,
    400,
    500,
    640,
    800,
    1000,
    1250,
    1600,
    2000,
    2500,
    3200,
    4000,
    5000,
    6400,
    8000,
    10000,
    12800,
    16000,
    20000,
    25600,
    32000,
)
_SHUTTER_LOG_STOPS = tuple(math.log(value) for value in _SHUTTER_STOPS)
_ISO_LOG_STOPS = tuple(math.log(value) for value in _ISO_STOPS)
# Half of a third-stop interval (the nearest-stop boundary) plus a 1/12-stop
# hysteresis margin.  The displayed value only changes after the measurement
# moves a little past the midpoint between two standard stops.
_STOP_DEADBAND = math.log(2.0) / 6.0 + math.log(2.0) / 12.0
# A different nearest stop must persist for this many status refreshes before
# it replaces the text.  At 30 fps this is ~0.2 s, which ignores alternating
# readings (HDR short/long frames) while still following real scene changes.
_STABLE_FRAMES = 6

# Manual-focus peaking: binary Canny edges coloured by cycling through the
# RGB channels, one colour per preview frame.
_MF_RGB_CYCLE = ((0, 0, 255), (255, 0, 0), (0, 255, 0))


def _nearest_stop_index(value: float, log_stops: tuple) -> int:
    """Return the index of the nearest standard stop in log space."""

    level = math.log(value)
    best = 0
    best_distance = abs(log_stops[0] - level)
    for index in range(1, len(log_stops)):
        distance = abs(log_stops[index] - level)
        if distance < best_distance:
            best = index
            best_distance = distance
    return best


@dataclass
class _StableStop:
    """Hysteresis and debounce for one displayed third-stop value.

    ``shown`` is the index currently rendered.  A different nearest stop is
    adopted only after it has been observed for ``_STABLE_FRAMES``
    consecutive refreshes; readings that alternate between two values never
    satisfy that condition, so the text stays put.
    """

    stops: tuple
    log_stops: tuple
    shown: int | None = None
    candidate: int | None = None
    frames: int = 0

    def update(self, value):
        """Return the stop value to display, or ``None`` when unavailable."""

        if value is None or value <= 0:
            self.shown = self.candidate = None
            self.frames = 0
            return None
        level = math.log(value)
        if (
            self.shown is not None
            and abs(level - self.log_stops[self.shown])
            <= _STOP_DEADBAND
        ):
            # Inside the deadband: keep the current text and forget any
            # pending candidate.
            self.candidate, self.frames = None, 0
            return self.stops[self.shown]
        nearest = _nearest_stop_index(value, self.log_stops)
        if self.shown is None:
            # First reading: show it immediately instead of blanking the
            # field for the debounce interval.
            self.shown, self.candidate, self.frames = nearest, None, 0
            return self.stops[nearest]
        if nearest == self.shown:
            self.candidate, self.frames = None, 0
            return self.stops[self.shown]
        if nearest == self.candidate:
            self.frames += 1
        else:
            self.candidate, self.frames = nearest, 1
        if self.frames >= _STABLE_FRAMES:
            self.shown, self.candidate, self.frames = nearest, None, 0
        return self.stops[self.shown]


# (menu option id, localizer worker field, cast, optional scale factor).
# The trailing factor converts menu units into the worker's expected units.
_LOCALIZER_OPTIONS = (
    (MenuOptionId.LOC_N_SOURCES.value, "n_sources", int),
    (MenuOptionId.LOC_CONFIDENCE.value, "confidence_threshold", float),
    (MenuOptionId.LOC_MAX_CHUNKS.value, "max_to_cumulate", int),
    (MenuOptionId.LOC_SPEECH_THRESHOLD.value, "speech_threshold", float),
    (MenuOptionId.LOC_MIC_ROTATION.value, "mic_rotation", float),
    (MenuOptionId.LOC_DELAY_MAX.value, "delay_max", float),
    (MenuOptionId.LOC_DELAY_BINS.value, "n_delay_bins", int),
    (MenuOptionId.LOC_MARKER_FOV.value, "marker_fov", float),
    (MenuOptionId.LOC_MARKER_AZIMUTH.value, "marker_azimuth_offset", float),
    (MenuOptionId.LOC_MARKER_ELEVATION.value, "marker_elevation_offset", float),
    (MenuOptionId.LOC_MARKER_X_GAIN.value, "marker_x_gain", float),
    (MenuOptionId.LOC_MARKER_Y_GAIN.value, "marker_y_gain", float),
    (MenuOptionId.LOC_MARKER_MIC_X.value, "marker_mic_offset_x", float, 0.01),
    (MenuOptionId.LOC_MARKER_MIC_Y.value, "marker_mic_offset_y", float, 0.01),
    (MenuOptionId.LOC_MARKER_MIC_Z.value, "marker_mic_offset_z", float, 0.01),
    (MenuOptionId.LOC_TALK_DISTANCE.value, "marker_source_distance", float, 0.01),
)


@dataclass
class CameraState:
    """Runtime state of the preview screen.

    Values are edited by buttons and menu options, then pushed to
    libcamera by the ``_apply_*`` helpers.  Keeping them here lets the
    preview loop read a consistent snapshot without querying the camera.
    """

    zoom: float = 1.0
    brightness: float = 0.0
    show_hist: bool = False
    is_busy: bool = False
    mf_assist: bool = False
    decorate_enabled: bool = False
    zoom_hold: bool = False
    bright_hold: bool = False
    record_timestamp: float | None = None

    # Pro overlay flags
    show_grid: bool = False
    show_center: bool = False
    show_zebra: bool = False
    zebra_threshold: int = 235
    status_position: str = "off"
    menu_theme: str = "dark_pro"

    # Capture / recording modes
    capture_mode: str = "single"
    burst_count: int = 3
    burst_interval: float = 0.5
    interval_count: int = 10
    interval_seconds: float = 5.0
    self_timer: int = 0
    record_mode: bool = False
    record_resolution: tuple[int, int] = (1280, 720)

    # Focus
    focus_mode: int = 2  # 0 manual, 1 single AF, 2 continuous AF
    focus_position: float = 500.0
    af_speed: int = 0
    af_range: int = 0
    af_pause: bool = False
    touch_focus: bool = True
    manual_focus: bool = False  # derived from focus_mode

    # Manual controls
    exp_compensation: float = 0.0
    ae_lock: bool = False
    awb_lock: bool = False
    saturation: float = 1.0
    contrast: float = 1.0
    sharpness: float = 1.0
    hdr_mode: int = 0
    exposure_priority: int = 0
    noise_reduction: int = -1  # Auto: use per-mode Picamera2 defaults
    preview_fps: int = 30
    colour_temperature: int = 0  # 0 = AWB; >0 fixed Kelvin
    isp_stats: bool = False

    countdown: float = 0.0


class CameraControlledEnd(ControlledEnd):
    """Preview end that owns capture/recording and libcamera settings.

    The class sequences the still-capture session and video recording,
    applies menu options, and decorates frames from
    :meth:`CameraDevice.preview`.  Long-running sensor work stays in the
    camera backend; this end only decides what to do and when.
    """

    def __init__(
        self,
        camera: CameraDevice,
        leds: Leds,
        config: AppConfig,
        _id: str = EndId.CAMERA.value,
        verbose_console: int = logging.INFO,
        localizer=None,
        preview_server=None,
        stats=None,
    ):
        """Create the decorators once; per-frame overlays reuse them."""
        super().__init__(_id)
        self._camera = camera
        self._leds = leds
        self._config = config
        self._screen = config.screen
        self._preview_server = preview_server
        self._stats = stats
        self._state = CameraState()
        self._menu_tree: dict | None = None
        self._localizer = localizer
        self._focus_distance: float | None = None
        # Smooths the autofocus figure of merit so the metadata page and
        # bar chart do not twitch with every frame.
        self._filter = SlidingWindowFilter(10)
        # Debounced third-stop displays: auto-exposure jitter and alternating
        # HDR frames must not make the status text flicker.
        self._shutter_display = _StableStop(
            _SHUTTER_STOPS, _SHUTTER_LOG_STOPS
        )
        self._iso_display = _StableStop(_ISO_STOPS, _ISO_LOG_STOPS)
        self._capture_session: CaptureSession | None = None
        self._mf_edges: np.ndarray | None = None

        self._bar_chart = BarChart(
            self._screen.width,
            self._screen.height,
            fill=True,
            alpha=0.7,
        )
        self._toast = Toast(self._screen.width, self._screen.height)
        self._decorator = SimpleText(
            [self._metadata_page, self._diagnostics_page],
            height=self._screen.height,
            padding=(10, 20, 0, 0),
            font_height=10,
            color=Colors.gold.value,
        )
        self._busy = Busy(self._screen.width, self._screen.height)
        self._hist = Histogram(
            self._screen.width,
            self._screen.height,
            padding=corner_padding(
                self._screen.width,
                self._screen.height,
                corner="bottom-right",
            ),
        )
        self._grid = RuleOfThirds(
            self._screen.width, self._screen.height
        )
        self._center = CenterMark(
            self._screen.width, self._screen.height
        )
        self._zebra = ZebraOverlay()
        self._status = StatusBar(
            self._screen.width, self._screen.height
        )
        self._recording_badge = RecordingBadge(
            self._screen.width, self._screen.height
        )
        self._countdown = Countdown(
            self._screen.width, self._screen.height
        )
        self._logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Menu option helpers
    # ------------------------------------------------------------------
    def _find_option_value(self, target: str):
        """Return the raw value of the menu option identified by ``target``."""
        if self._menu_tree is None:
            raise RuntimeError("camera has not received the menu tree yet")
        for section in self._menu_tree.values():
            for option in section.get("options", []):
                if option.get("id") == target and "value" in option:
                    return option["value"]
        raise LookupError(target)

    def _option_value(self, target: str):
        """Return a scalar option value, unwrapping dict-style entries."""
        value = self._find_option_value(target)
        if isinstance(value, dict):
            return value.get("value")
        return value

    def _option_value_or(self, target: str, default=None):
        """Like :meth:`_option_value`, but tolerate missing or bad options."""
        try:
            return self._option_value(target)
        except (RuntimeError, LookupError, TypeError, ValueError):
            return default

    # ------------------------------------------------------------------
    # Metadata overlay
    # ------------------------------------------------------------------
    def _metadata_page(self):
        """Build the metadata overlay page shown when decoration is on.

        Report keys double as format strings (``"EPTime {}"``), matching
        the convention used by the menu's metadata page.
        """
        metadata = self._camera.metadata
        if not metadata:
            return None
        report = {
            "EPTime {}": metadata["ExposureTime"],
            "FocusFoM {}": self._camera.frame_quality,
            "AF {}": {0: "MF", 1: "AF-S", 2: "AF-C"}.get(
                self._state.focus_mode, "AF"
            ),
            "FrameDur {}": metadata["FrameDuration"],
            "AnGain {}": round(metadata["AnalogueGain"], 2),
            "DigGain {}": round(metadata["DigitalGain"], 2),
            "Lux {}": round(metadata["Lux"], 2),
            "ClrTemp {}": metadata["ColourTemperature"],
            "FPS {}": round(self._camera.frame_per_second, 1),
            "FocusAvg {}": int(self._filter.calc()) if len(self._filter) else 0,
        }
        iso = self._iso_value(metadata)
        if iso is not None:
            report["ISO {}"] = iso
        lens_position = metadata.get("LensPosition")
        if lens_position is not None:
            report["Lens {}"] = round(lens_position, 2)
        return report

    @staticmethod
    def _format_timestamp(value, wall: bool = False) -> str:
        """Format a libcamera nanosecond timestamp for the diagnostics page."""

        if value is None:
            return "n/a"
        try:
            seconds = float(value) / 1_000_000_000.0
        except (TypeError, ValueError):
            return str(value)
        if not wall:
            # SensorTimestamp is monotonic nanoseconds; seconds are easier
            # to compare across frames than a huge raw integer.
            return f"{seconds:.2f}s"
        try:
            clock = time.strftime(
                "%H:%M:%S", time.localtime(seconds)
            )
        except (OverflowError, OSError, ValueError):
            return str(value)
        millis = int((seconds % 1.0) * 1000)
        return f"{clock}.{millis:03d}"

    def _diagnostics_page(self):
        """Second metadata page with AF/AE state and sensor timestamps."""

        metadata = self._camera.metadata
        if not metadata:
            return None
        report = {
            "AeState {}": {
                0: "Idle",
                1: "Searching",
                2: "Converged",
            }.get(metadata.get("AeState"), metadata.get("AeState", "n/a")),
            "AfState {}": {
                0: "Idle",
                1: "Scanning",
                2: "Focused",
                3: "Failed",
            }.get(metadata.get("AfState"), metadata.get("AfState", "n/a")),
            "AfPause {}": {
                0: "Running",
                1: "Pausing",
                2: "Paused",
            }.get(
                metadata.get("AfPauseState"),
                metadata.get("AfPauseState", "n/a"),
            ),
            "Stats {}": "ON" if self._state.isp_stats else "OFF",
        }
        temperature = metadata.get("SensorTemperature")
        if temperature is not None:
            try:
                report["SensorT {}"] = f"{float(temperature):.1f}C"
            except (TypeError, ValueError):
                report["SensorT {}"] = str(temperature)
        if metadata.get("SensorTimestamp") is not None:
            report["SensorTs {}"] = self._format_timestamp(
                metadata["SensorTimestamp"]
            )
        if metadata.get("FrameWallClock") is not None:
            report["WallClock {}"] = self._format_timestamp(
                metadata["FrameWallClock"], wall=True
            )
        stats = metadata.get("PispStatsOutput")
        if stats is None:
            stats = metadata.get("Bcm2835StatsOutput")
        if stats is not None:
            try:
                report["StatsBlob {}"] = len(stats)
            except TypeError:
                report["StatsBlob {}"] = "on"
        return report

    @staticmethod
    def _iso_value(metadata) -> int | None:
        """Approximate ISO the same way Picamera2 writes EXIF ISO.

        Libcamera exposes gain, not ISO.  Picamera2 stores
        ``ISOSpeedRatings = (AnalogueGain * DigitalGain) * 100``; using the
        same formula keeps the overlay consistent with saved photos.
        """

        try:
            analogue = float(metadata["AnalogueGain"])
            digital = float(metadata.get("DigitalGain", 1.0))
        except (KeyError, TypeError, ValueError):
            return None
        total_gain = analogue * digital
        if total_gain <= 0:
            return None
        return max(1, int(round(total_gain * 100)))

    @staticmethod
    def _format_shutter_stop(seconds: float) -> str:
        """Render one standard shutter stop in the status-bar style."""

        if seconds >= 1.0:
            return f"{seconds:g}s"
        return f"1/{max(1, round(1.0 / seconds))}"

    @staticmethod
    def _format_shutter(microseconds: float) -> str:
        """Render an exposure time as the nearest standard shutter stop."""

        seconds = float(microseconds) / 1_000_000.0
        if seconds <= 0:
            return "0"
        index = _nearest_stop_index(seconds, _SHUTTER_LOG_STOPS)
        return CameraControlledEnd._format_shutter_stop(
            _SHUTTER_STOPS[index]
        )

    # ------------------------------------------------------------------
    # Button actions
    # ------------------------------------------------------------------
    def up_press_action(self) -> None:
        if self._state.decorate_enabled:
            self._decorator.previous_page()
            time.sleep(0.3)
            return
        self._state.bright_hold = True
        self._state.brightness = min(1.0, self._state.brightness + 0.01)
        self._toast.set_text(f"{tr('BRT')} {int(self._state.brightness * 100)}")
        self._camera.brightness(self._state.brightness)
        time.sleep(0.05)

    def up_release_action(self) -> None:
        self._state.bright_hold = False

    def down_press_action(self) -> None:
        if self._state.decorate_enabled:
            self._decorator.next_page()
            time.sleep(0.3)
            return
        self._state.bright_hold = True
        self._state.brightness = max(-1.0, self._state.brightness - 0.01)
        self._toast.set_text(f"{tr('BRT')} {int(self._state.brightness * 100)}")
        self._camera.brightness(self._state.brightness)
        time.sleep(0.05)

    def down_release_action(self) -> None:
        self._state.bright_hold = False

    def left_press_action(self) -> None:
        self._state.zoom_hold = True
        self._state.zoom = max(1.0, self._state.zoom - 0.2)
        self._toast.set_text(f"X {round(self._state.zoom, 1)}")
        self._camera.zoom(self._state.zoom)
        time.sleep(0.05)

    def left_release_action(self) -> None:
        self._state.zoom_hold = False

    def right_press_action(self) -> None:
        self._state.zoom_hold = True
        self._state.zoom += 0.2
        self._toast.set_text(f"X {round(self._state.zoom, 1)}")
        self._camera.zoom(self._state.zoom)
        time.sleep(0.05)

    def right_release_action(self) -> None:
        self._state.zoom_hold = False

    def shutter_press_action(self) -> None:
        """Dispatch the shutter by current state.

        Priority matters: stop an active recording first, then cancel a
        running burst/interval session, then start whatever the menu mode
        selects.  This makes one button both start and cancel.
        """
        if self._state.record_timestamp is not None:
            self._stop_recording()
            return

        if self._capture_session is not None and self._capture_session.active:
            self._capture_session.cancel()
            self._capture_session = None
            self._state.countdown = 0.0
            self._state.is_busy = False
            self._toast.set_text("CANCELED")
            return

        if self._state.record_mode:
            self._start_recording()
            return

        self._start_capture_session()

    def _start_capture_session(self) -> None:
        """Create the capture session for the configured still mode."""
        mode = self._state.capture_mode
        if mode == "burst":
            settings = CaptureSettings(
                mode="burst",
                count=max(2, self._state.burst_count),
                interval=self._state.burst_interval,
                initial_delay=self._state.self_timer,
            )
        elif mode == "interval":
            settings = CaptureSettings(
                mode="interval",
                count=max(1, self._state.interval_count),
                interval=self._state.interval_seconds,
                initial_delay=self._state.self_timer,
            )
        else:
            settings = CaptureSettings(
                mode="single",
                count=1,
                interval=0.0,
                initial_delay=self._state.self_timer,
            )
        # The preview loop drives the session; there is no timer thread.
        self._capture_session = CaptureSession(settings, time.monotonic())
        if self._state.self_timer:
            self._toast.set_text(f"{tr('TIMER')} {self._state.self_timer}")
        else:
            self._toast.set_text("CAPTURE")

    def _resolution(self) -> tuple[int, int]:
        """Return the still resolution from the menu, or ``(0, 0)``."""
        value = self._option_value_or(MenuOptionId.RESOLUTION.value, [0, 0])
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError, IndexError):
            return 0, 0

    def _capture_photo(self, index: int) -> None:
        """Save one still using the menu's format and exposure settings.

        Invoked from :meth:`CaptureSession.tick` inside the preview loop,
        so it blocks frame delivery while the camera switches modes.
        """
        multi = self._state.capture_mode in {"burst", "interval"}
        suffix = f"_{index + 1}" if multi else ""
        self._state.is_busy = True
        self._leds.green.on()
        self._toast.set_text(
                f"{tr('SHOT')} {index + 1}" if multi else "PROCESSING"
            )

        try:
            width, height = self._resolution()
            extension = str(
                self._option_value_or(
                    MenuOptionId.PICT_FORMAT.value, "jpeg"
                )
                or "jpeg"
            )
            path = os.path.join(
                str(self._config.camera.path),
                f"{int(time.time())}{suffix}",
            )
            auto_expose = bool(
                self._option_value_or(
                    MenuOptionId.AUTO_EXPOSE.value, True
                )
            )
            # Manual exposure/gain controls are only valid with AE off;
            # libcamera ignores or rejects them while AE is enabled.
            exposure_time = None
            analogue_gain = None
            exposure_priority = 0
            if not auto_expose:
                exposure_time = int(
                    self._option_value_or(
                        MenuOptionId.EXPOSURE_TIME.value, 10000
                    )
                )
                analogue_gain = float(
                    self._option_value_or(
                        MenuOptionId.ANALOGUE_GAIN.value, 1.0
                    )
                )
            elif not self._state.ae_lock:
                # Mirror the semi-auto preview mode in the still.
                exposure_priority = self._state.exposure_priority
                if exposure_priority == 1:
                    exposure_time = int(
                        self._option_value_or(
                            MenuOptionId.EXPOSURE_TIME.value, 10000
                        )
                    )
                elif exposure_priority == 2:
                    analogue_gain = float(
                        self._option_value_or(
                            MenuOptionId.ANALOGUE_GAIN.value, 1.0
                        )
                    )
            apply_watermark = bool(
                self._option_value_or(MenuOptionId.WATERMARK.value, False)
            )
            self._camera.save_frame(
                file_path=path,
                extension=extension,
                width=width,
                height=height,
                save_metadata=bool(
                    self._option_value_or(
                        MenuOptionId.SAVE_METADATA.value, False
                    )
                ),
                save_raw=bool(
                    self._option_value_or(
                        MenuOptionId.DNG_ENABLE.value, False
                    )
                ),
                exposure_time=exposure_time,
                analogue_gain=analogue_gain,
                exposure_priority=exposure_priority,
                # The callback receives the actual encoded frame, so the
                # badge also works for full-resolution captures where the
                # menu reports (0, 0) instead of a pixel size.
                watermark=(
                    lambda image: Watermark(
                        image.shape[1], image.shape[0]
                    ).decorate(image)
                ) if apply_watermark else None,
            )
        except Exception:
            self._logger.exception("photo capture failed")
            self._toast.set_text("CAPTURE FAILED")
        finally:
            self._leds.green.off()
            self._state.is_busy = False

    def _start_recording(self) -> None:
        """Switch to the video configuration and start the MP4 encoder."""
        if self._state.is_busy or (
            self._capture_session is not None
            and self._capture_session.active
        ):
            return
        width, height = self._state.record_resolution
        self._config.camera.video_path.mkdir(parents=True, exist_ok=True)
        path = os.path.join(
            str(self._config.camera.video_path), f"{int(time.time())}.mp4"
        )
        # Video reconfigures libcamera and restarts the preview on stop,
        # so it must not overlap a burst/interval still session.
        try:
            self._camera.start_recording(int(width), int(height), path)
        except Exception:
            self._logger.exception("failed to start recording")
            self._toast.set_text("REC FAILED")
            return
        self._leds.blue.on()
        self._state.record_timestamp = time.time()
        self._toast.set_text("REC")

    def _stop_recording(self) -> None:
        """Stop the encoder and restore the preview configuration."""
        try:
            self._camera.stop_recording()
        except Exception:
            self._logger.exception("failed to stop recording")
        finally:
            self._leds.blue.off()
            self._state.record_timestamp = None
            self._toast.set_text("STOPPED")

    def shutter_long_press_action(self) -> None:
        """Alternate recording entry point; input long-press is not wired."""

        if self._state.record_timestamp is not None:
            self._stop_recording()
        elif (
            not self._state.is_busy
            and not (
                self._capture_session is not None
                and self._capture_session.active
            )
        ):
            self._start_recording()

    def square_press_action(self) -> None:
        if self._state.record_timestamp is not None:
            return
        if self._capture_session is not None and self._capture_session.active:
            return
        if not self._state.is_busy:
            self._irq(EndId.MENU.value)

    def circle_press_action(self) -> None:
        self._state.decorate_enabled = not self._state.decorate_enabled

    def cross_press_action(self) -> None:
        pass

    def _trigger_focus(self) -> None:
        """Request a one-shot AF scan unless capture owns the camera."""
        if self._state.is_busy or (
            self._capture_session is not None
            and self._capture_session.active
        ):
            return
        self._camera.trigger_autofocus()
        self._toast.set_text("AF")

    def center_press_action(self) -> None:
        self._trigger_focus()

    def handle_touch(self, x: float, y: float) -> bool:
        """Focus on a normalized tap position from the web preview."""

        state = self._state
        if not state.touch_focus or state.focus_mode == 0:
            return False
        if state.is_busy or (
            self._capture_session is not None
            and self._capture_session.active
        ) or state.record_timestamp is not None:
            return False

        window = self._camera.focus_window_at(x, y)
        if not self._camera.set_af_window(window):
            # Fall back to the default metering region on older pipelines.
            self._camera.set_af_window(None)
        if state.focus_mode == 1:
            self._camera.trigger_autofocus()
        self._toast.set_text("AF")
        return True

    def rotary_encoder_clockwise(self) -> None:
        pass

    def rotary_encoder_counter_clockwise(self) -> None:
        pass

    def rotary_encoder_select(self) -> None:
        self._trigger_focus()

    # ------------------------------------------------------------------
    # Camera settings
    # ------------------------------------------------------------------
    def _apply_hdr(self) -> None:
        """Push the HDR mode, downgrading to off when AE is disabled."""
        auto_expose = bool(
            self._option_value_or(MenuOptionId.AUTO_EXPOSE.value, True)
        )
        # HDR needs Auto Exposure; fail safe if an old menu still contains
        # the invalid combination (the menu controller normally clears HDR).
        self._state.hdr_mode = self._state.hdr_mode if auto_expose else 0
        self._camera.set_hdr(self._state.hdr_mode)

    def _apply_exposure(self) -> None:
        """Apply either the AE configuration or fixed exposure and gain.

        The branches are mutually exclusive: libcamera exposes EV,
        metering and flicker controls only while AE is enabled.
        """
        exposure = int(
            self._option_value_or(
                MenuOptionId.EXPOSURE_TIME.value, 10000
            )
        )
        gain = float(
            self._option_value_or(
                MenuOptionId.ANALOGUE_GAIN.value, 1.0
            )
        )
        if bool(self._option_value_or(MenuOptionId.AUTO_EXPOSE.value, True)):
            self._camera.set_ae_enable(True)
            self._camera.set_ae_constraint_mode(
                self._option_value_or(
                    MenuOptionId.CONSTRAINT_MODE.value, 0
                )
            )
            self._camera.set_ae_exposure_mode(
                self._option_value_or(MenuOptionId.EXPOSURE_MODE.value, 0)
            )
            self._camera.set_ae_metering_mode(
                self._option_value_or(MenuOptionId.METERING_MODE.value, 0)
            )
            self._camera.set_ae_flicker_mode(
                self._option_value_or(MenuOptionId.FLICKER_MODE.value, 0)
            )
            self._camera.set_ae_flicker_period(
                self._option_value_or(MenuOptionId.FLICKER_PERIOD.value, 0)
            )
            self._camera.set_exposure_value(self._state.exp_compensation)
            # Shutter/gain priority keeps AE active for the other half.
            if not self._camera.set_exposure_priority(
                self._state.exposure_priority,
                min(exposure, _PREVIEW_MAX_EXPOSURE),
                gain,
            ):
                self._camera.set_exposure_priority(0, exposure, gain)
        else:
            self._camera.set_ae_enable(False)
            # Keep the live preview responsive; longer presets are applied
            # when the still configuration is active.
            self._camera.set_manual_exposure(
                min(exposure, _PREVIEW_MAX_EXPOSURE), gain
            )

    def _apply_awb(self) -> None:
        """Apply automatic white balance or fixed red/blue colour gains."""
        # An explicit Kelvin value takes precedence over AWB modes and
        # manual gains; fall through when the pipeline rejects it.
        if self._state.colour_temperature > 0:
            if self._camera.set_colour_temperature(
                self._state.colour_temperature
            ):
                return
        if bool(self._option_value_or(MenuOptionId.AWB.value, True)):
            self._camera.set_awb_enable(True)
            self._camera.set_awb_mode(
                self._option_value_or(MenuOptionId.AWB_MODE.value, 0)
            )
        else:
            self._camera.set_awb_enable(False)
            self._camera.set_colour_gains(
                self._option_value_or(MenuOptionId.RED_GAIN.value, 1.0),
                self._option_value_or(MenuOptionId.BLUE_GAIN.value, 1.0),
            )

    def _apply_focus(self) -> None:
        """Push the AF mode plus its speed, range and pause state."""
        state = self._state
        self._camera.set_af_speed(state.af_speed)
        self._camera.set_af_range(state.af_range)
        self._camera.set_focus_mode(
            state.focus_mode, int(state.focus_position)
        )
        # AfPause is only valid for continuous AF; sending Resume in the
        # other modes makes libcamera log a warning.
        if state.focus_mode == 2:
            self._camera.set_af_pause(state.af_pause)
        # A custom AF window only makes sense while touch focus and AF are
        # active; otherwise restore full-frame metering.
        if not state.touch_focus or state.focus_mode == 0:
            self._camera.set_af_window(None)

    def _apply_preview_fps(self) -> None:
        """Cap the preview frame rate from the Display menu."""
        self._camera.set_frame_rate(self._state.preview_fps)

    def _apply_noise_reduction(self) -> None:
        """Apply the selected denoise mode (negative means Auto)."""
        self._camera.set_noise_reduction(self._state.noise_reduction)

    def _apply_stats_output(self) -> None:
        """Enable or disable ISP statistics in metadata."""
        self._camera.set_stats_output(self._state.isp_stats)

    def _apply_manual_controls(self) -> None:
        """Apply colour controls and the optional AE/AWB locks."""
        self._camera.set_image_controls(
            saturation=self._state.saturation,
            contrast=self._state.contrast,
            sharpness=self._state.sharpness,
        )
        # Locks only make sense in the corresponding automatic mode.  In
        # manual mode the lock would overwrite the user's exposure/gain
        # values with the last frame's metadata.
        auto_expose = bool(
            self._option_value_or(MenuOptionId.AUTO_EXPOSE.value, True)
        )
        awb = bool(self._option_value_or(MenuOptionId.AWB.value, True))
        if self._state.ae_lock and auto_expose:
            self._camera.set_ae_lock(True)
        if (
            self._state.awb_lock
            and awb
            and self._state.colour_temperature <= 0
        ):
            self._camera.set_awb_lock(True)

    def _apply_preview_server(self, value=None) -> None:
        """Start or stop the HTTP preview server from the System menu."""

        if self._preview_server is None:
            return
        try:
            enabled = (
                bool(
                    self._option_value_or(
                        MenuOptionId.WEB_SERVER.value, True
                    )
                )
                if value is None
                else bool(value)
            )
            self._preview_server.set_enabled(enabled)
        except Exception:
            self._logger.exception(
                "failed to apply preview server setting"
            )

    def _apply_localizer_settings(self) -> None:
        """Forward the localizer menu page to the worker process.

        Missing options (older menu files) and an unavailable localizer are
        both harmless: the worker keeps its previous values.
        """

        if self._localizer is None:
            return
        values = {}
        for option_id, field, cast, *rest in _LOCALIZER_OPTIONS:
            raw = self._option_value_or(option_id, None)
            if raw is None:
                continue
            try:
                values[field] = cast(raw) * (rest[0] if rest else 1.0)
            except (TypeError, ValueError):
                continue
        if values:
            try:
                self._localizer.update_settings(**values)
            except Exception:
                self._logger.exception(
                    "failed to apply localizer settings %s", values
                )

    def _sync_source_distance(self) -> None:
        """Use the autofocus lens position as the source distance.

        ``LensPosition`` is reported in dioptres, so the focused object is
        ``1 / LensPosition`` metres away.  The camera is normally focused on
        the person who is speaking, which is exactly the distance needed for
        the array/camera parallax correction.  A positive "Talk Distance"
        menu value overrides the estimate.
        """

        if self._localizer is None:
            return
        if self._state.focus_mode == 0:
            # In manual focus LensPosition is a user setting, not a measurement
            # of where the subject is.
            return
        manual = self._option_value_or(
            MenuOptionId.LOC_TALK_DISTANCE.value, 0.0
        )
        try:
            if manual is not None and float(manual) > 0.0:
                return
        except (TypeError, ValueError):
            pass

        lens_position = (self._camera.metadata or {}).get("LensPosition")
        if not lens_position:
            return
        try:
            distance = 1.0 / float(lens_position)
        except (TypeError, ValueError, ZeroDivisionError):
            return
        if not 0.05 <= distance <= 20.0:
            return

        if self._focus_distance is None:
            self._focus_distance = distance
        else:
            self._focus_distance = (
                0.8 * self._focus_distance + 0.2 * distance
            )
        try:
            self._localizer.update_settings(
                marker_source_distance=self._focus_distance
            )
        except Exception:
            self._logger.debug(
                "failed to apply focus distance", exc_info=True
            )

    def _read_overlay_settings(self) -> None:
        """Copy overlay-related menu options into :class:`CameraState`."""
        state = self._state
        state.show_grid = bool(
            self._option_value_or(MenuOptionId.SHOW_GRID.value, False)
        )
        state.show_center = bool(
            self._option_value_or(MenuOptionId.SHOW_CENTER.value, False)
        )
        state.show_zebra = bool(
            self._option_value_or(MenuOptionId.SHOW_ZEBRA.value, False)
        )
        state.zebra_threshold = int(
            self._option_value_or(MenuOptionId.ZEBRA_THRESHOLD.value, 235)
        )
        self._zebra.set_threshold(state.zebra_threshold)
        state.status_position = str(
            self._option_value_or(
                MenuOptionId.STATUS_BAR_POSITION.value, "off"
            )
        )
        state.menu_theme = str(
            self._option_value_or(MenuOptionId.THEME.value, "dark_pro")
        )
        state.mf_assist = bool(
            self._option_value_or(MenuOptionId.MF_ASSIST.value, False)
        )
        state.show_hist = bool(
            self._option_value_or(MenuOptionId.SHOW_HIST.value, False)
        )

    def _read_capture_settings(self) -> None:
        """Copy capture/recording menu options into :class:`CameraState`."""
        state = self._state
        mode = int(
            self._option_value_or(MenuOptionId.CAPTURE_MODE.value, 0)
        )
        state.capture_mode = {
            0: "single",
            1: "burst",
            2: "interval",
        }.get(mode, "single")
        state.burst_count = int(
            self._option_value_or(MenuOptionId.BURST_COUNT.value, 3)
        )
        state.burst_interval = float(
            self._option_value_or(MenuOptionId.BURST_INTERVAL.value, 0.5)
        )
        state.interval_count = int(
            self._option_value_or(MenuOptionId.INTERVAL_COUNT.value, 10)
        )
        state.interval_seconds = float(
            self._option_value_or(MenuOptionId.INTERVAL_SECONDS.value, 5)
        )
        state.self_timer = int(
            self._option_value_or(MenuOptionId.DELAY.value, 0) or 0
        )
        state.record_mode = bool(
            self._option_value_or(MenuOptionId.RECORD_MODE.value, False)
        )
        resolution = self._option_value_or(
            MenuOptionId.RECORD_RESOLUTION.value, [1280, 720]
        )
        try:
            state.record_resolution = (int(resolution[0]), int(resolution[1]))
        except (TypeError, ValueError, IndexError):
            state.record_resolution = (1280, 720)

    def load_settings(self) -> None:
        """Read every camera menu option and apply it to libcamera.

        Called on entry and whenever the menu sends a new tree, so it must
        be idempotent and tolerate older menus with missing options.
        """
        state = self._state
        state.exp_compensation = float(
            self._option_value_or(
                MenuOptionId.EXP_COMPENSATION.value, 0.0
            )
        )
        state.ae_lock = bool(
            self._option_value_or(MenuOptionId.AE_LOCK.value, False)
        )
        state.awb_lock = bool(
            self._option_value_or(MenuOptionId.AWB_LOCK.value, False)
        )
        focus_mode = self._option_value_or(
            MenuOptionId.AF_MODE.value, None
        )
        if focus_mode is None:
            # Older menus only had the manual-focus bool.
            legacy_manual = bool(
                self._option_value_or(
                    MenuOptionId.MANUAL_FOCUS.value, False
                )
            )
            state.focus_mode = 0 if legacy_manual else 2
        else:
            state.focus_mode = int(focus_mode)
        state.manual_focus = state.focus_mode == 0
        state.focus_position = float(
            self._option_value_or(MenuOptionId.FOCUS_POSITION.value, 500)
        )
        state.af_speed = int(
            self._option_value_or(MenuOptionId.AF_SPEED.value, 0)
        )
        state.af_range = int(
            self._option_value_or(MenuOptionId.AF_RANGE.value, 0)
        )
        state.saturation = float(
            self._option_value_or(MenuOptionId.SATURATION.value, 1.0)
        )
        state.contrast = float(
            self._option_value_or(MenuOptionId.CONTRAST.value, 1.0)
        )
        state.sharpness = float(
            self._option_value_or(MenuOptionId.SHARPNESS.value, 1.0)
        )
        state.hdr_mode = int(
            self._option_value_or(MenuOptionId.HDR.value, 0)
        )
        state.exposure_priority = int(
            self._option_value_or(MenuOptionId.EXPOSURE_PRIORITY.value, 0)
        )
        state.noise_reduction = int(
            self._option_value_or(MenuOptionId.NOISE_REDUCTION.value, -1)
        )
        state.af_pause = bool(
            self._option_value_or(MenuOptionId.AF_PAUSE.value, False)
        )
        state.touch_focus = bool(
            self._option_value_or(MenuOptionId.TOUCH_FOCUS.value, True)
        )
        state.preview_fps = int(
            self._option_value_or(MenuOptionId.PREVIEW_FPS.value, 30)
        )
        state.colour_temperature = int(
            self._option_value_or(
                MenuOptionId.COLOUR_TEMPERATURE.value, 0
            )
        )
        state.isp_stats = bool(
            self._option_value_or(MenuOptionId.ISP_STATS.value, False)
        )

        self._read_overlay_settings()
        self._read_capture_settings()
        self._apply_preview_fps()
        self._apply_hdr()
        self._apply_exposure()
        self._apply_noise_reduction()
        self._apply_awb()
        self._apply_manual_controls()
        self._apply_stats_output()
        self._apply_focus()
        self._apply_localizer_settings()
        self._apply_preview_server()

    # ------------------------------------------------------------------
    # Messages and lifecycle
    # ------------------------------------------------------------------
    def msg_receiver(self, sender: str, payload) -> None:
        """Handle menu-tree updates, preview toggles and AF requests.

        Menu broadcasts are ``(action, tree)`` sequences; a bare bool
        enables or disables the web preview server.
        """
        if isinstance(payload, bool):
            self._apply_preview_server(payload)
            return
        action = None
        if isinstance(payload, (tuple, list)) and len(payload) > 1:
            action, menu_tree = payload[0], payload[1]
        else:
            menu_tree = self._menu_tree
        if menu_tree is not None:
            self._menu_tree = menu_tree
            self.load_settings()
        if action == "autofocus":
            self._trigger_focus()

    def on_enter(self, last_id: str | None) -> None:
        """Prepare media folders and request the current menu settings."""
        self._config.camera.path.mkdir(parents=True, exist_ok=True)
        self._config.camera.video_path.mkdir(parents=True, exist_ok=True)
        self.send_message(EndId.MENU.value, self.id)
        if self._menu_tree is not None:
            self.load_settings()

    def on_exit(self) -> None:
        """Cancel capture/recording and turn off the indicator LEDs."""
        if self._capture_session is not None:
            self._capture_session.cancel()
            self._capture_session = None
            self._state.countdown = 0.0
        if self._state.record_timestamp is not None:
            self._stop_recording()
        self._leds.green.off()
        self._leds.blue.off()

    # ------------------------------------------------------------------
    # Preview loop
    # ------------------------------------------------------------------
    def _histogram_padding(self) -> tuple[int, int, int, int]:
        """Keep the histogram box clear of the status bar."""

        width = self._screen.width
        height = self._screen.height
        position = self._state.status_position
        if position == "sides":
            # The side panels hug both edges; the bottom centre stays free.
            return corner_padding(width, height, corner="bottom-center")

        left, top, right, bottom = corner_padding(
            width, height, corner="bottom-right"
        )
        if position in {"bottom", "top_bottom"}:
            band = self._status.bar_height
            top = max(0, top - band)
            bottom += band
        return left, top, right, bottom

    def _build_mf_mask(self, frame: np.ndarray) -> np.ndarray:
        """Build the binary Canny peaking mask from the raw preview frame."""

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.Canny(blurred, threshold1=70, threshold2=400)

    def _status_items(self) -> list:
        """Build status-bar items from the capture mode and metadata."""
        metadata = self._camera.metadata or {}
        # Side panels are narrow, so they use label/value pairs and
        # abbreviated values rather than inline tokens.
        side_mode = self._state.status_position == "sides"
        if self._state.record_timestamp is not None:
            mode = "REC"
            mode_role = "danger"
        else:
            mode = {
                "single": "SGL",
                "burst": "BRST",
                "interval": "INT",
            }.get(self._state.capture_mode, "SGL")
            mode_role = "success"
        if side_mode:
            items: list = [
                StatusItem("DRV", mode, value_role=mode_role)
            ]
        else:
            items = [(mode, mode_role)]
        focus_code = {0: "MF", 1: "AF-S", 2: "AF-C"}.get(
            self._state.focus_mode, "AF"
        )
        focus_role = (
            "warning" if self._state.focus_mode == 0 else "success"
        )
        if self._state.focus_mode != 0:
            # libcamera's live AF state is more informative than the static
            # mode token: show scanning/focused/failed directly.
            af_state = metadata.get("AfState")
            try:
                af_state = int(af_state)
            except (TypeError, ValueError):
                af_state = None
            if af_state is not None:
                focus_role = {
                    0: "dim",
                    1: "warning",
                    2: "success",
                    3: "danger",
                }.get(af_state, focus_role)
            if self._state.af_pause:
                focus_role = "warning"
        if side_mode:
            focus_value = {
                "MF": "MF",
                "AF-S": "S",
                "AF-C": "C",
            }.get(focus_code, focus_code)
            items.append(
                StatusItem("AF", focus_value, value_role=focus_role)
            )
        else:
            items.append((focus_code, focus_role))
        if self._state.focus_mode == 0:
            items.append(
                StatusItem(
                    "L",
                    str(int(self._state.focus_position)),
                    label_role="dim",
                    value_role="dim",
                )
            )
        exposure = metadata.get("ExposureTime")
        seconds = None
        if exposure is not None:
            try:
                seconds = float(exposure) / 1_000_000.0
            except (TypeError, ValueError):
                pass
        shutter_stop = self._shutter_display.update(seconds)
        if shutter_stop is not None:
            shutter = self._format_shutter_stop(shutter_stop)
            if side_mode:
                items.append(StatusItem("SH", shutter))
            else:
                items.append((shutter, "text"))
        iso = self._iso_value(metadata)
        iso_stop = self._iso_display.update(iso)
        if iso_stop is not None:
            items.append(StatusItem("ISO", str(iso_stop)))
        if metadata.get("ColourTemperature") is not None:
            kelvin = int(metadata["ColourTemperature"])
            if side_mode:
                # "WB" already implies Kelvin; drop the unit to stay narrow.
                items.append(StatusItem("WB", str(kelvin)))
            else:
                items.append((f"{kelvin}K", "text"))
        if self._stats is not None:
            try:
                cpu, temperature = self._stats.cpu_snapshot()
            except Exception:
                cpu = temperature = None
            if cpu is not None:
                # Match the system monitor's green/amber/red bands.
                role = (
                    "success"
                    if cpu < 60
                    else "warning"
                    if cpu < 85
                    else "danger"
                )
                items.append(
                    StatusItem("CPU", f"{cpu:.0f}%", value_role=role)
                )
            if temperature is not None:
                # Same warning bands the system monitor uses.
                role = (
                    "success"
                    if temperature < 60
                    else "warning"
                    if temperature < 75
                    else "danger"
                )
                items.append(
                    StatusItem("T", f"{temperature:.0f}C", value_role=role)
                )
        items.append(
            StatusItem("FPS", f"{self._camera.frame_per_second:.0f}")
        )
        items.append(StatusItem("Z", f"{self._state.zoom:.1f}"))
        if abs(self._state.exp_compensation) > 1e-6:
            items.append(
                StatusItem(
                    "EV",
                    f"{self._state.exp_compensation:+.1f}",
                    value_role="accent",
                )
            )
        if self._state.hdr_mode:
            items.append(
                StatusItem(
                    "HDR",
                    str(self._state.hdr_mode),
                    value_role="warning",
                )
            )
        return items

    def main_loop(self):
        """Yield decorated preview frames for the controller.

        Overlay order is deliberate: focus peaking and the histogram both
        read the raw scene, then the peaking colour is composited before the
        UI overlays so grid, zebra, text and the status bar are never
        mistaken for scene edges.  Informational overlays paint last.
        """
        # preview() blocks until libcamera delivers a frame, so the loop
        # is paced by the sensor rather than by a sleep interval.
        for index, frame in enumerate(self._camera.preview()):
            now = time.monotonic()
            # Refreshing the focus-derived distance at ~4 Hz is plenty; the
            # lens itself moves much more slowly than the preview frame rate.
            if index % 5 == 0:
                self._sync_source_distance()
            # tick() captures at most one shot per preview frame, which
            # serializes still saves with preview delivery.
            session = self._capture_session
            if session is not None:
                session.tick(now, self._capture_photo)
                if session.active:
                    self._state.countdown = session.countdown_remaining(now)
                else:
                    self._capture_session = None
                    self._state.countdown = 0.0
                    self._toast.set_text("DONE")
            else:
                self._state.countdown = 0.0

            # Focus statistics are only consumed by the metadata overlay and
            # focus chart; do not update them when those are hidden.
            if self._state.decorate_enabled:
                quality = self._camera.frame_quality
                if quality is not None:
                    self._filter.add_data(quality)
                if len(self._filter):
                    self._bar_chart.add_data(int(self._filter.calc()))

            # Focus peaking must read the raw scene, before any UI overlay
            # is drawn; otherwise grid lines, zebra marks and text become
            # "edges" and light up together with the image.
            edges = None
            if self._state.mf_assist:
                # Refreshing the edge map at 10 fps is plenty and saves two
                # out of three Canny passes.
                if (
                    self._mf_edges is None
                    or self._mf_edges.shape != frame.shape[:2]
                    or index % 3 == 0
                ):
                    self._mf_edges = self._build_mf_mask(frame)
                edges = self._mf_edges
            else:
                self._mf_edges = None

            # Compute the histogram from the raw preview, before UI overlays
            # (grid, zebra, status text) skew the pixel distribution.
            if self._state.show_hist:
                self._hist.set_padding(self._histogram_padding())
                self._hist.decorate(frame)

            # Composite peaking after the histogram so scene edges stay
            # visible across the histogram box.  The edge map still comes
            # from the raw frame, so the box and its bars never create
            # peaking; grid/status overlays are drawn later and cover the
            # coloured lines instead of being highlighted by them.
            if (
                self._state.mf_assist
                and edges is not None
                and cv2.countNonZero(edges)
            ):
                colorful_edges = np.zeros(
                    (edges.shape[0], edges.shape[1], 3),
                    dtype=np.uint8,
                )
                colorful_edges[edges != 0] = _MF_RGB_CYCLE[
                    index % len(_MF_RGB_CYCLE)
                ]
                frame = cv2.addWeighted(frame, 1, colorful_edges, 1.0, 0)

            # Geometry overlays are cheap and pixel-independent, so they
            # can be redrawn every frame.
            if self._state.show_grid:
                self._grid.decorate(frame)
            if self._state.show_center:
                self._center.decorate(frame)
            if self._state.show_zebra:
                self._zebra.decorate(frame)

            # Suppress the chart/metadata while zooming or recording so
            # those deliberate actions keep an unobstructed view.
            if (
                self._state.decorate_enabled
                and not self._state.zoom_hold
                and self._state.record_timestamp is None
            ):
                self._bar_chart.decorate(frame)
                self._decorator.decorate(frame)
            if self._state.is_busy:
                self._busy.decorate(frame)

            if self._state.status_position in {
                "top",
                "bottom",
                "top_bottom",
                "sides",
            }:
                self._status.set_position(self._state.status_position)
                self._status.set_items(self._status_items())
                self._status.decorate(
                    frame, theme_id=self._state.menu_theme
                )

            # The zoom toast replaces the badge while zooming.
            if (
                self._state.record_timestamp is not None
                and not self._state.zoom_hold
            ):
                self._recording_badge.decorate(
                    frame, time.time() - self._state.record_timestamp
                )
            if self._state.countdown > 0:
                self._countdown.decorate(frame, self._state.countdown)

            if (
                self._state.zoom_hold
                or self._state.bright_hold
                or self._toast.visible
            ):
                self._toast.decorate(frame)

            yield frame
