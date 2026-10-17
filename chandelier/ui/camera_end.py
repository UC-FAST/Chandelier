"""Camera preview screen with pro capture modes and overlays."""

from __future__ import annotations

import logging
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
    Toast,
    Watermark,
    ZebraOverlay,
)


_LOCALIZER_OPTIONS = (
    (MenuOptionId.LOC_N_SOURCES.value, "n_sources", int),
    (MenuOptionId.LOC_CONFIDENCE.value, "confidence_threshold", float),
    (MenuOptionId.LOC_MAX_CHUNKS.value, "max_to_cumulate", int),
    (MenuOptionId.LOC_SPEECH_THRESHOLD.value, "speech_threshold", float),
    (MenuOptionId.LOC_MIC_ROTATION.value, "mic_rotation", float),
    (MenuOptionId.LOC_DELAY_MAX.value, "delay_max", float),
    (MenuOptionId.LOC_DELAY_BINS.value, "n_delay_bins", int),
)


@dataclass
class CameraState:
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
    show_status_bar: bool = False
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

    # Manual controls
    exp_compensation: float = 0.0
    ae_lock: bool = False
    awb_lock: bool = False
    manual_focus: bool = False
    focus_position: float = 500.0
    saturation: float = 1.0
    contrast: float = 1.0
    sharpness: float = 1.0
    hdr_mode: int = 0

    countdown: float = 0.0


class CameraControlledEnd(ControlledEnd):
    def __init__(
        self,
        camera: CameraDevice,
        leds: Leds,
        config: AppConfig,
        _id: str = EndId.CAMERA.value,
        verbose_console: int = logging.INFO,
        localizer=None,
    ):
        super().__init__(_id)
        self._camera = camera
        self._leds = leds
        self._config = config
        self._screen = config.screen
        self._state = CameraState()
        self._menu_tree: dict | None = None
        self._localizer = localizer
        self._filter = SlidingWindowFilter(10)
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
            [self._metadata_page],
            height=self._screen.height,
            padding=(10, 20, 0, 0),
            font_height=10,
            color=Colors.gold.value,
        )
        self._busy = Busy(self._screen.width, self._screen.height)
        self._hist = Histogram()
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
        if self._menu_tree is None:
            raise RuntimeError("camera has not received the menu tree yet")
        for section in self._menu_tree.values():
            for option in section.get("options", []):
                if option.get("id") == target and "value" in option:
                    return option["value"]
        raise LookupError(target)

    def _option_value(self, target: str):
        value = self._find_option_value(target)
        if isinstance(value, dict):
            return value.get("value")
        return value

    def _option_value_or(self, target: str, default=None):
        try:
            return self._option_value(target)
        except (RuntimeError, LookupError, TypeError, ValueError):
            return default

    # ------------------------------------------------------------------
    # Metadata overlay
    # ------------------------------------------------------------------
    def _metadata_page(self):
        metadata = self._camera.metadata
        if not metadata:
            return None
        return {
            "EPTime {}": metadata["ExposureTime"],
            "FocusFoM {}": self._camera.frame_quality,
            "FrameDur {}": metadata["FrameDuration"],
            "AnGain {}": round(metadata["AnalogueGain"], 2),
            "DigGain {}": round(metadata["DigitalGain"], 2),
            "Lux {}": round(metadata["Lux"], 2),
            "ClrTemp {}": metadata["ColourTemperature"],
            "FPS {}": round(self._camera.frame_per_second, 1),
            "FocusAvg {}": int(self._filter.calc()) if len(self._filter) else 0,
        }

    @staticmethod
    def _format_shutter(microseconds: float) -> str:
        if microseconds >= 1_000_000:
            return f"{microseconds / 1_000_000:.1f}s"
        if microseconds >= 1000:
            return f"1/{max(1, round(1_000_000 / microseconds))}"
        return f"{microseconds:.0f}us"

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
        self._capture_session = CaptureSession(settings, time.monotonic())
        if self._state.self_timer:
            self._toast.set_text(f"{tr('TIMER')} {self._state.self_timer}")
        else:
            self._toast.set_text("CAPTURE")

    def _resolution(self) -> tuple[int, int]:
        value = self._option_value_or(MenuOptionId.RESOLUTION.value, [0, 0])
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError, IndexError):
            return 0, 0

    def _capture_photo(self, index: int) -> None:
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
            )
            if self._option_value_or(MenuOptionId.WATERMARK.value, False):
                frame = cv2.imread(f"{path}.{extension}")
                if frame is not None and width > 0 and height > 0:
                    Watermark(width, height).decorate(frame)
                    cv2.imwrite(f"{path}.{extension}", frame)
        except Exception:
            self._logger.exception("photo capture failed")
            self._toast.set_text("CAPTURE FAILED")
        finally:
            self._leds.green.off()
            self._state.is_busy = False

    def _start_recording(self) -> None:
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

    def center_press_action(self) -> None:
        pass

    def rotary_encoder_clockwise(self) -> None:
        pass

    def rotary_encoder_counter_clockwise(self) -> None:
        pass

    def rotary_encoder_select(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Camera settings
    # ------------------------------------------------------------------
    def _apply_hdr(self) -> None:
        self._camera.set_hdr(self._state.hdr_mode)

    def _apply_exposure(self) -> None:
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
        else:
            self._camera.set_ae_enable(False)
            self._camera.set_manual_exposure(
                self._option_value_or(MenuOptionId.EXPOSURE_TIME.value, 10000),
                self._option_value_or(MenuOptionId.ANALOGUE_GAIN.value, 1.0),
            )

    def _apply_awb(self) -> None:
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

    def _apply_manual_controls(self) -> None:
        self._camera.set_image_controls(
            saturation=self._state.saturation,
            contrast=self._state.contrast,
            sharpness=self._state.sharpness,
        )
        self._camera.set_focus(
            self._state.manual_focus, int(self._state.focus_position)
        )
        # Locks are applied last so they override the automatic settings.
        if self._state.ae_lock:
            self._camera.set_ae_lock(True)
        if self._state.awb_lock:
            self._camera.set_awb_lock(True)

    def _apply_localizer_settings(self) -> None:
        """Forward the localizer menu page to the worker process.

        Missing options (older menu files) and an unavailable localizer are
        both harmless: the worker keeps its previous values.
        """

        if self._localizer is None:
            return
        values = {}
        for option_id, field, cast in _LOCALIZER_OPTIONS:
            raw = self._option_value_or(option_id, None)
            if raw is None:
                continue
            try:
                values[field] = cast(raw)
            except (TypeError, ValueError):
                continue
        if values:
            try:
                self._localizer.update_settings(**values)
            except Exception:
                self._logger.exception(
                    "failed to apply localizer settings %s", values
                )

    def _read_overlay_settings(self) -> None:
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
        state.show_status_bar = bool(
            self._option_value_or(MenuOptionId.SHOW_STATUS_BAR.value, False)
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
        state.manual_focus = bool(
            self._option_value_or(MenuOptionId.MANUAL_FOCUS.value, False)
        )
        state.focus_position = float(
            self._option_value_or(MenuOptionId.FOCUS_POSITION.value, 500)
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

        self._read_overlay_settings()
        self._read_capture_settings()
        self._apply_hdr()
        self._apply_exposure()
        self._apply_awb()
        self._apply_manual_controls()
        self._apply_localizer_settings()

    # ------------------------------------------------------------------
    # Messages and lifecycle
    # ------------------------------------------------------------------
    def msg_receiver(self, sender: str, payload) -> None:
        self._menu_tree = payload[1]
        self.load_settings()

    def on_enter(self, last_id: str | None) -> None:
        self._config.camera.path.mkdir(parents=True, exist_ok=True)
        self._config.camera.video_path.mkdir(parents=True, exist_ok=True)
        self.send_message(EndId.MENU.value, self.id)
        if self._menu_tree is not None:
            self.load_settings()

    def on_exit(self) -> None:
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
    def _status_items(self) -> list[str | tuple[str, str]]:
        metadata = self._camera.metadata or {}
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
        items: list[str | tuple[str, str]] = [(mode, mode_role)]
        if metadata.get("ExposureTime") is not None:
            items.append(
                (self._format_shutter(metadata["ExposureTime"]), "text")
            )
        if metadata.get("AnalogueGain") is not None:
            items.append((f"G{float(metadata['AnalogueGain']):.1f}", "text"))
        if metadata.get("ColourTemperature") is not None:
            items.append((f"{int(metadata['ColourTemperature'])}K", "text"))
        items.append((f"{self._camera.frame_per_second:.0f}fps", "dim"))
        items.append((f"Z{self._state.zoom:.1f}", "text"))
        if abs(self._state.exp_compensation) > 1e-6:
            items.append(
                (f"EV{self._state.exp_compensation:+.1f}", "accent")
            )
        if self._state.hdr_mode:
            items.append((f"HDR{self._state.hdr_mode}", "warning"))
        return items

    def main_loop(self):
        for index, frame in enumerate(self._camera.preview()):
            now = time.monotonic()
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

            # Compute the histogram from the raw preview, before UI overlays
            # (grid, zebra, status text) skew the pixel distribution.
            if self._state.show_hist:
                self._hist.decorate(frame)

            if self._state.show_grid:
                self._grid.decorate(frame)
            if self._state.show_center:
                self._center.decorate(frame)
            if self._state.show_zebra:
                self._zebra.decorate(frame)

            if (
                self._state.decorate_enabled
                and not self._state.zoom_hold
                and self._state.record_timestamp is None
            ):
                self._bar_chart.decorate(frame)
                self._decorator.decorate(frame)
            if self._state.is_busy:
                self._busy.decorate(frame)

            if self._state.show_status_bar:
                self._status.set_items(self._status_items())
                self._status.decorate(
                    frame, theme_id=self._state.menu_theme
                )

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

            edges = None
            if self._state.mf_assist:
                # Focus peaking is a visual aid; refreshing the edge map at
                # 10 fps is plenty and saves two out of three Canny passes.
                if self._mf_edges is None or index % 3 == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    self._mf_edges = cv2.Canny(
                        blurred, threshold1=70, threshold2=400
                    )
                edges = self._mf_edges
            else:
                self._mf_edges = None

            if (
                self._state.mf_assist
                and edges is not None
                and cv2.countNonZero(edges)
            ):
                colorful_edges = np.zeros(
                    (edges.shape[0], edges.shape[1], 3), dtype=np.uint8
                )
                if index % 3 == 0:
                    colorful_edges[edges != 0] = (0, 0, 255)
                elif index % 3 == 1:
                    colorful_edges[edges != 0] = (255, 0, 0)
                else:
                    colorful_edges[edges != 0] = (0, 255, 0)
                frame = cv2.addWeighted(frame, 1, colorful_edges, 1.0, 0)

            yield frame
