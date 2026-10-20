"""Picamera2 wrapper.

Frames leave this module as BGR ``numpy`` arrays, matching OpenCV and the LCD
driver.  All calls that switch sensor modes restore the preview configuration
and re-apply the current digital zoom/crop.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import threading
import time
from typing import Callable, Iterator

import cv2
import numpy as np

from chandelier.config import CameraConfig, ScreenConfig

logger = logging.getLogger(__name__)


class CameraDevice:
    """Picamera2-backed camera with preview, still and video modes.

    Every Picamera2 call that changes mode or captures is serialised behind
    ``_lock``.  Switching modes reprograms the sensor and may drop per-mode
    controls, so cached controls and the digital-zoom crop are re-applied
    after each switch.
    """

    def __init__(
        self,
        config: CameraConfig,
        screen: ScreenConfig,
        verbose_console: int = logging.INFO,
    ):
        # Imported lazily so this module remains importable on non-Pi hosts.
        import picamera2
        from picamera2.encoders import H264Encoder

        self._screen = screen
        self._config = config

        # A tuning file overrides libcamera's default ISP tuning; None asks
        # libcamera to pick the tuning that matches the detected sensor.
        self._cam = picamera2.Picamera2(tuning=_load_tuning(config))
        self._cam.set_logging(verbose_console)
        # Use the main stream for the preview.  RGB888 on main is supported on
        # every Pi model (lores is effectively YUV420-only before Pi 5).
        # libcamera's "RGB888" is BGR in memory, so make_array() already
        # returns an OpenCV-compatible (height, width, 3) BGR array.
        self._preview_config = self._cam.create_preview_configuration(
            main={
                "size": (config.preview_width, config.preview_height),
                "format": "RGB888",
            },
        )
        self._cam.configure(self._preview_config)
        # The encoder bitrate is in bits per second.
        self._encoder = H264Encoder(config.video_bitrate)

        # Guards mode switches and capture requests; the preview generator
        # holds it only while a request is alive.
        self._lock = threading.Lock()
        self._frame_per_second = 0.0
        self._digital_zoom = 1.0
        self._brightness = 0.0
        # Last successfully applied controls, so unchanged values can be
        # skipped after they survive a mode switch.
        self._controls: dict = {}
        # Controls a sensor/PiSP variant rejected; avoid retrying them.
        self._unsupported_controls: set[str] = set()
        self._focus_mode: int | None = None
        # None means "Auto": preview/still/video use Picamera2's per-mode
        # defaults (Minimal / HighQuality / Fast).
        self._noise_reduction_mode: int | None = None
        # StatsOutputEnable must be present when the stream is configured;
        # a runtime set_controls() alone does not produce PispStatsOutput.
        self._stats_output = False
        self._metadata: dict | None = None
        # Placeholder frame, replaced by the first preview() iteration.
        self._frame = np.zeros(
            (screen.height, screen.width, 3), dtype=np.uint8
        )

        # Preview.NULL keeps libcamera from opening a desktop preview window;
        # frames are pulled with capture_request() instead.
        self._cam.start_preview(picamera2.Preview.NULL)
        self._cam.start()
        # ScalerCrop of the running sensor is the rectangle digital zoom
        # shrinks from.
        self._base_crop = tuple(self._cam.capture_metadata()["ScalerCrop"])
        logger.info("camera initialised")

    # ------------------------------------------------------------------
    # Capture properties and simple controls
    # ------------------------------------------------------------------
    @property
    def frame_per_second(self) -> float:
        return self._frame_per_second

    @property
    def metadata(self) -> dict | None:
        return self._metadata

    @property
    def frame_quality(self) -> float | None:
        # FocusFoM is libcamera's focus figure of merit; higher is sharper.
        if not self._metadata:
            return None
        return float(self._metadata.get("FocusFoM", 0))

    def zoom(self, zoom: float) -> None:
        """Set digital zoom, where 1.0 keeps the full sensor crop."""
        # Zoom in only: below 1.0 would request pixels outside the sensor.
        self._digital_zoom = max(1.0, round(float(zoom), 2))
        self._apply_zoom()
        logger.debug("zoom set to %.2f", self._digital_zoom)

    def current_crop(self) -> tuple[int, int, int, int]:
        """Return the active sensor crop as ``(left, top, width, height)``."""

        w_offset, h_offset, f_width, f_height = self._base_crop
        p_width = max(1, int(f_width // self._digital_zoom))
        p_height = max(1, int(f_height // self._digital_zoom))
        # Shrink the base crop symmetrically around its centre.  ScalerCrop
        # expects [left, top, width, height] in sensor pixels.
        left = int((f_width - p_width) // 2 + w_offset)
        top = int((f_height - p_height) // 2 + h_offset)
        return left, top, p_width, p_height

    def _apply_zoom(self) -> None:
        self._set_controls({"ScalerCrop": list(self.current_crop())})

    def focus_window_at(
        self, nx: float, ny: float, fraction: float = 0.2
    ) -> tuple[int, int, int, int]:
        """Map a normalized preview tap to a sensor-pixel AF window.

        The preview is a straight resize of the current crop, so the
        normalized coordinates map linearly into ``current_crop()``.
        """

        left, top, width, height = self.current_crop()
        nx = max(0.0, min(1.0, float(nx)))
        ny = max(0.0, min(1.0, float(ny)))
        window_width = max(1, int(round(width * fraction)))
        window_height = max(1, int(round(height * fraction)))
        centre_x = left + int(round(nx * width))
        centre_y = top + int(round(ny * height))
        window_left = max(
            left, min(centre_x - window_width // 2, left + width - window_width)
        )
        window_top = max(
            top, min(centre_y - window_height // 2, top + height - window_height)
        )
        return window_left, window_top, window_width, window_height

    def brightness(self, brightness: float) -> None:
        """Set the brightness control, clipped to libcamera's -1..1 range."""
        self._brightness = max(-1.0, min(1.0, round(float(brightness), 2)))
        self._set_controls({"Brightness": self._brightness})
        logger.debug("brightness set to %.2f", self._brightness)

    def _control_supported(self, name: str) -> bool:
        # Older picamera2 releases do not expose camera_controls; in that
        # case assume the control exists and let set_controls() decide.
        available = getattr(self._cam, "camera_controls", None)
        return available is None or name in available

    # Commands that must be sent every time even when the stored value is
    # unchanged (``AfTrigger`` is a one-shot event, not a state).
    _VOLATILE_CONTROLS = frozenset({"AfTrigger"})

    def _set_controls(self, controls: dict, force: bool = False) -> None:
        """Apply controls one key at a time so one unsupported key cannot
        discard the whole batch (sensors/PiSP variants expose different
        control sets)."""

        for name, value in controls.items():
            # Skip values already applied: every call crosses into
            # libcamera and may disturb the running pipeline.
            if (
                not force
                and name not in self._VOLATILE_CONTROLS
                and name in self._controls
                and self._controls[name] == value
            ):
                continue
            if name in self._unsupported_controls:
                continue
            if not self._control_supported(name):
                logger.warning("camera control %s is not supported", name)
                self._unsupported_controls.add(name)
                continue
            try:
                self._cam.set_controls({name: value})
            except Exception as exc:  # libcamera raises RuntimeError variants
                logger.warning("camera control %s failed: %s", name, exc)
                self._unsupported_controls.add(name)
                continue
            if name not in self._VOLATILE_CONTROLS:
                self._controls[name] = value

    def _refresh_base_crop(self) -> None:
        """Re-read the sensor crop after a mode switch."""
        self._base_crop = tuple(self._cam.capture_metadata()["ScalerCrop"])

    # ------------------------------------------------------------------
    # Auto exposure / white balance
    # ------------------------------------------------------------------
    # The *_mode setters pass libcamera enum codes straight from the menu
    # configuration; no validation here so new libcamera values keep working.

    def set_ae_enable(self, enable: bool) -> None:
        """Enable or disable auto-exposure."""
        self._set_controls({"AeEnable": enable})

    def set_ae_exposure_mode(self, code: int) -> None:
        self._set_controls({"AeExposureMode": code})

    def set_ae_constraint_mode(self, code: int) -> None:
        self._set_controls({"AeConstraintMode": code})

    def set_ae_metering_mode(self, code: int) -> None:
        self._set_controls({"AeMeteringMode": code})

    def set_ae_flicker_mode(self, code: int) -> None:
        self._set_controls({"AeFlickerMode": code})

    def set_ae_flicker_period(self, code: int) -> None:
        self._set_controls({"AeFlickerPeriod": code})

    def set_manual_exposure(
        self, exposure_time: int, analogue_gain: float
    ) -> None:
        """Disable AE and use a fixed exposure (us) and analogue gain."""
        self._set_controls(
            {
                "AeEnable": False,
                "ExposureTime": exposure_time,
                "AnalogueGain": analogue_gain,
                # libcamera 0.7 uses the per-control modes; keep the
                # deprecated AeEnable for older pipelines.
                "ExposureTimeMode": 1,
                "AnalogueGainMode": 1,
            }
        )

    def set_awb_enable(self, enable: bool) -> None:
        self._set_controls({"AwbEnable": enable})

    def set_awb_mode(self, code: int) -> None:
        self._set_controls({"AwbMode": code})

    def set_colour_gains(self, red_gain: float, blue_gain: float) -> None:
        """Disable AWB and apply fixed red/blue channel gains."""
        self._set_controls(
            {"AwbEnable": False, "ColourGains": (red_gain, blue_gain)}
        )

    def set_hdr(self, mode: int) -> None:
        """Set the HDR mode using libcamera's ``HdrMode`` enum."""
        self._set_controls({"HdrMode": int(mode)})

    def set_exposure_value(self, value: float) -> None:
        """Apply an exposure-compensation offset in stops."""
        self._set_controls({"ExposureValue": float(value)})

    def set_exposure_priority(
        self,
        priority: int,
        exposure_time: int,
        analogue_gain: float,
    ) -> bool:
        """Fix shutter or gain while letting the other half track AE.

        ``1`` locks the exposure time (shutter priority), ``2`` locks the
        analogue gain (gain priority).  Returns False when the pipeline does
        not expose the per-control mode needed for the request.
        """

        priority = int(priority)
        if priority == 1:
            if not self._control_supported("ExposureTimeMode"):
                logger.warning("shutter priority is not supported")
                return False
            self._set_controls(
                {
                    "AeEnable": True,
                    "AnalogueGainMode": 0,
                    "ExposureTime": max(1, int(exposure_time)),
                    "ExposureTimeMode": 1,
                }
            )
            return "ExposureTimeMode" in self._controls
        if priority == 2:
            if not self._control_supported("AnalogueGainMode"):
                logger.warning("gain priority is not supported")
                return False
            self._set_controls(
                {
                    "AeEnable": True,
                    "ExposureTimeMode": 0,
                    "AnalogueGain": float(analogue_gain),
                    "AnalogueGainMode": 1,
                }
            )
            return "AnalogueGainMode" in self._controls
        self._set_controls(
            {
                "AeEnable": True,
                "ExposureTimeMode": 0,
                "AnalogueGainMode": 0,
            }
        )
        return True

    def set_frame_rate(self, fps: float) -> None:
        """Cap the preview frame rate via ``FrameDurationLimits``."""

        if fps <= 0:
            return
        period = max(33_333, int(round(1_000_000 / float(fps))))
        # Leave the upper bound at Picamera2's preview default so AE can
        # still use longer frames in low light.
        self._set_controls(
            {"FrameDurationLimits": (period, max(period, 83_333))}
        )

    def set_noise_reduction(self, mode: int) -> None:
        """Select NoiseReductionMode, or Auto (negative) for mode defaults."""

        mode = int(mode)
        if mode < 0:
            self._noise_reduction_mode = None
            # Clear an explicit value; stills/video pop the key so their
            # own configuration default applies.
            self._controls.pop("NoiseReductionMode", None)
            self._set_controls({"NoiseReductionMode": 3})
            return
        self._noise_reduction_mode = mode
        self._set_controls({"NoiseReductionMode": mode})

    def set_colour_temperature(self, kelvin: int) -> bool:
        """Disable AWB and request a fixed colour temperature."""

        kelvin = int(kelvin)
        if kelvin <= 0:
            return False
        if not self._control_supported("ColourTemperature"):
            logger.warning("fixed colour temperature is not supported")
            return False
        self._set_controls({"AwbEnable": False})
        self._set_controls({"ColourTemperature": kelvin})
        if "ColourTemperature" not in self._controls:
            # Rejected by the pipeline: hand white balance back to AWB.
            self._set_controls({"AwbEnable": True})
            return False
        return True

    def set_stats_output(self, enable: bool) -> None:
        """Enable or disable ISP statistics in the request metadata.

        The control only takes effect when it is part of the stream
        configuration, so changing it reconfigures the preview briefly.
        """

        enable = bool(enable)
        if enable == self._stats_output:
            return
        self._stats_output = enable
        if not enable:
            # The pipeline is sticky: a runtime False clears the output
            # without reconfiguring the stream.
            self._set_controls({"StatsOutputEnable": False})
            return
        # Enabling requires the control at stream-configuration time.
        # Drop any cached False first so re-applying controls after the
        # switch cannot immediately turn the output off again.
        self._controls.pop("StatsOutputEnable", None)
        self._preview_config = self._cam.create_preview_configuration(
            main={
                "size": (
                    self._config.preview_width,
                    self._config.preview_height,
                ),
                "format": "RGB888",
            },
            controls={"StatsOutputEnable": True},
        )
        with self._lock:
            self._cam.switch_mode(self._preview_config)
            self._refresh_base_crop()
            self._cam.set_controls(self._controls)
            self._apply_zoom()

    def set_image_controls(
        self,
        *,
        saturation: float | None = None,
        contrast: float | None = None,
        sharpness: float | None = None,
    ) -> None:
        """Apply any subset of the tuning controls; None skips that one."""
        controls = {}
        if saturation is not None:
            controls["Saturation"] = float(saturation)
        if contrast is not None:
            controls["Contrast"] = float(contrast)
        if sharpness is not None:
            controls["Sharpness"] = float(sharpness)
        if controls:
            self._set_controls(controls)

    def set_ae_lock(self, enable: bool) -> None:
        """Freeze the current exposure/gain, or hand control back to AE."""
        if not enable:
            self._set_controls(
                {
                    "AeEnable": True,
                    "ExposureTimeMode": 0,
                    "AnalogueGainMode": 0,
                }
            )
            return
        # Use the latest preview metadata; capturing a fresh metadata set is
        # a fallback for callers that lock before the first preview frame.
        metadata = self._metadata
        if not metadata:
            try:
                metadata = self._cam.capture_metadata()
            except Exception:
                metadata = {}
        controls = {
            "AeEnable": False,
            "ExposureTimeMode": 1,
            "AnalogueGainMode": 1,
        }
        if "ExposureTime" in metadata:
            controls["ExposureTime"] = metadata["ExposureTime"]
        if "AnalogueGain" in metadata:
            controls["AnalogueGain"] = metadata["AnalogueGain"]
        self._set_controls(controls)

    def set_awb_lock(self, enable: bool) -> None:
        """Freeze the current white-balance gains, or re-enable AWB."""
        if not enable:
            self._set_controls({"AwbEnable": True})
            return
        # Fall back to the last gains we requested when metadata does not
        # report ColourGains yet.
        gains = (self._metadata or {}).get("ColourGains")
        if gains is None:
            gains = self._controls.get("ColourGains")
        controls = {"AwbEnable": False}
        if gains:
            controls["ColourGains"] = tuple(gains)
        self._set_controls(controls)

    @staticmethod
    def _lens_position(position: float) -> float:
        # LensPosition is in dioptres; the menu uses 0..1000, so divide by
        # 100 and clamp to libcamera's 0..10 dioptre range.
        return max(0.0, min(10.0, float(position) / 100.0))

    def set_focus_mode(self, mode: int, position: float = 500) -> None:
        """Select ``0=manual``, ``1=single AF`` or ``2=continuous AF``."""

        # Normalise any unexpected menu value to continuous AF.
        mode = 0 if int(mode) <= 0 else 1 if int(mode) == 1 else 2
        previous = self._focus_mode
        changed = mode != previous
        if previous == 2 and mode != 2:
            # Leaving AF-C must not leave the AF algorithm paused.
            self._set_controls({"AfPause": 2}, force=True)
        self._focus_mode = mode
        if mode == 0:
            self._set_controls(
                {
                    "AfMode": 0,
                    "LensPosition": self._lens_position(position),
                }
            )
        elif mode == 1:
            self._set_controls({"AfMode": 1})
            if changed:
                self.trigger_autofocus()
        else:
            self._set_controls({"AfMode": 2})

    def set_focus(self, manual: bool, position: int) -> None:
        """Select manual focus (``True``) or continuous AF (``False``)."""
        self.set_focus_mode(0 if manual else 2, position)

    def set_focus_position(self, position: float) -> None:
        """Move the lens to a menu position (0..1000) without AF."""
        self._set_controls(
            {"LensPosition": self._lens_position(position)}
        )

    def set_af_speed(self, speed: int) -> None:
        """Set libcamera's ``AfSpeed`` enum (fast or slow scan)."""
        self._set_controls({"AfSpeed": int(speed)})

    def set_af_range(self, af_range: int) -> None:
        """Set libcamera's ``AfRange`` enum (normal, macro or full)."""
        self._set_controls({"AfRange": int(af_range)})

    def set_af_pause(self, paused: bool) -> None:
        """Pause continuous AF or resume it (one-shot transitions)."""

        self._set_controls(
            {"AfPause": 0 if paused else 2}, force=True
        )

    def set_af_window(self, window: tuple | None) -> bool:
        """Focus on a sensor-pixel rectangle, or restore full-frame metering."""

        if window is None:
            self._set_controls({"AfMetering": 0})
            return True
        if not self._control_supported("AfWindows"):
            logger.warning("AF windows are not supported")
            return False
        x, y, width, height = (int(value) for value in window)
        self._set_controls(
            {
                "AfMetering": 1,
                # Picamera2 converts each 4-tuple to a libcamera Rectangle.
                "AfWindows": [(x, y, width, height)],
            }
        )
        return "AfWindows" in self._controls

    def trigger_autofocus(self) -> None:
        """Request a single AF scan from the tuning algorithm."""
        # AfTrigger is a one-shot command; force it even if it was already 0.
        self._set_controls({"AfTrigger": 0}, force=True)

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------
    def preview(self) -> Iterator[np.ndarray]:
        """Yield screen-sized BGR frames indefinitely.

        Each iteration yields a freshly allocated array.  Frames are pulled
        only as fast as the consumer iterates; there is no internal queue.
        """
        previous = time.monotonic()
        while True:
            with self._lock:
                # capture_request() must not overlap a mode switch.
                request = self._cam.capture_request()
                try:
                    # libcamera's "RGB888" is BGR in memory (picamera2 maps it
                    # to bgr24), so the array is already in OpenCV order.
                    array = request.make_array("main")
                    if array.ndim == 3 and array.shape[2] == 4:
                        # Some pipelines pad RGB888 to XRGB; drop the pad.
                        array = array[..., :3]

                    # Resize straight from the request buffer instead of
                    # copying the full preview frame first.  The request stays
                    # alive until cv2.resize has consumed the pixels.
                    if (
                        array.shape[1] != self._screen.width
                        or array.shape[0] != self._screen.height
                    ):
                        bgr = cv2.resize(
                            array,
                            (self._screen.width, self._screen.height),
                            interpolation=cv2.INTER_AREA,
                        )
                    else:
                        bgr = np.array(array, dtype=np.uint8, copy=True)
                    self._metadata = request.get_metadata()
                finally:
                    request.release()

            self._frame = bgr

            # Instantaneous interval between yields approximates the preview
            # frame rate; a single slow consumer temporarily lowers it.
            now = time.monotonic()
            elapsed = now - previous
            previous = now
            if elapsed > 0:
                self._frame_per_second = 1.0 / elapsed

            yield self._frame

    # ------------------------------------------------------------------
    # Still capture
    # ------------------------------------------------------------------
    def save_frame(
        self,
        file_path: str,
        extension: str,
        width: int,
        height: int,
        save_metadata: bool = False,
        save_raw: bool = False,
        exposure_time: int | None = None,
        analogue_gain: float | None = None,
        exposure_priority: int = 0,
        watermark: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        """Capture a still image to ``file_path``.

        A non-positive ``width``/``height`` selects the full sensor
        resolution.  ``save_raw`` also writes a DNG, and ``save_metadata``
        writes libcamera metadata as JSON next to the image.  Manual
        exposure is in microseconds and disables AE for this capture.
        ``exposure_priority`` selects shutter priority (1) or gain priority
        (2), matching the preview menu.  ``watermark`` is applied to the
        rendered frame before it is encoded, so JPEG EXIF survives; it is
        never applied to the raw DNG.
        """
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        if width <= 0 or height <= 0:
            width, height = self._cam.sensor_resolution

        # Manual exposure is applied as part of the still configuration so
        # exposures longer than the preview limit are honoured from the first
        # frame.  libcamera needs the frame duration to be at least as long
        # as the exposure.
        still_controls = {}
        priority = int(exposure_priority)
        if priority == 1 and exposure_time is not None:
            exposure_time = max(1, int(exposure_time))
            frame_duration = max(33_333, exposure_time + 50_000)
            if self._control_supported("ExposureTimeMode"):
                still_controls = {
                    "AeEnable": True,
                    "ExposureTimeMode": 1,
                    "AnalogueGainMode": 0,
                    "ExposureTime": exposure_time,
                    "FrameDurationLimits": (
                        frame_duration,
                        frame_duration,
                    ),
                }
            else:
                logger.warning("shutter priority still is not supported")
        elif priority == 2 and analogue_gain is not None:
            if self._control_supported("AnalogueGainMode"):
                still_controls = {
                    "AeEnable": True,
                    "ExposureTimeMode": 0,
                    "AnalogueGainMode": 1,
                    "AnalogueGain": float(analogue_gain),
                }
            else:
                logger.warning("gain priority still is not supported")
        elif exposure_time is not None:
            exposure_time = max(1, int(exposure_time))
            # The frame duration must cover the exposure plus readout and
            # scheduling overhead; 33.3 ms is a 30 fps floor.
            frame_duration = max(33_333, exposure_time + 50_000)
            still_controls = {
                "AeEnable": False,
                "ExposureTime": exposure_time,
                "AnalogueGain": float(
                    analogue_gain if analogue_gain is not None else 1.0
                ),
                "ExposureTimeMode": 1,
                "AnalogueGainMode": 1,
                "FrameDurationLimits": (
                    frame_duration,
                    frame_duration,
                ),
            }

        config_args = {
            "main": {"size": (width, height), "format": "RGB888"},
        }
        if save_raw:
            config_args["raw"] = {"size": self._cam.sensor_resolution}
        if still_controls:
            config_args["controls"] = still_controls
        still_config = self._cam.create_still_configuration(**config_args)

        with self._lock:
            request = None
            try:
                # A mode switch may drop per-mode controls, so re-apply the
                # cached ones afterwards.
                self._cam.switch_mode(still_config)
                self._refresh_base_crop()
                controls = dict(self._controls)
                # The preview frame-rate cap must never clamp a still.
                controls.pop("FrameDurationLimits", None)
                if self._noise_reduction_mode is None:
                    # Let the still configuration's HighQuality default win.
                    controls.pop("NoiseReductionMode", None)
                if still_controls:
                    # The still configuration already carries these; do not
                    # overwrite them with the stale cached preview values.
                    for key in (
                        "AeEnable",
                        "ExposureTime",
                        "AnalogueGain",
                        "ExposureTimeMode",
                        "AnalogueGainMode",
                        "FrameDurationLimits",
                    ):
                        controls.pop(key, None)
                self._cam.set_controls(controls)
                self._apply_zoom()
                # Give AE/AF a moment to settle in the new mode before the
                # single request that produces the still.
                time.sleep(1)
                request = self._cam.capture_request()

                if extension:
                    # "RGB888" is BGR in memory; write it as-is.
                    frame = request.make_array("main")
                    if frame.ndim == 3 and frame.shape[2] == 4:
                        frame = frame[..., :3]
                    if watermark is not None:
                        # Burn the badge in before encoding.  Re-reading and
                        # rewriting the file afterwards would strip EXIF and
                        # could leave a truncated image if the second write
                        # fails.
                        try:
                            watermark(frame)
                        except Exception:
                            logger.exception("watermark failed")
                    self._write_still(
                        frame,
                        f"{file_path}.{extension}",
                        extension,
                        request.get_metadata(),
                    )
                if save_metadata:
                    with open(f"{file_path}.json", "w", encoding="utf-8") as f:
                        json.dump(request.get_metadata(), f, indent=4)
                if save_raw:
                    # The DNG carries the raw Bayer frame and its metadata,
                    # including a JSON description of the full metadata set.
                    self._save_dng_with_metadata(
                        request, f"{file_path}.dng"
                    )
            finally:
                # Always return to preview mode, even if the capture failed.
                if request is not None:
                    request.release()
                self._restore_preview()

    def _camera_model(self) -> str:
        """Return the sensor/model name used in EXIF Make/Model tags."""

        properties = getattr(self._cam, "camera_properties", None) or {}
        model = (
            properties.get("Model")
            if isinstance(properties, dict)
            else None
        )
        if not model:
            model = getattr(self._cam, "camera", None)
        return str(model or "Raspberry Pi Camera")

    @staticmethod
    def _exif_bytes(
        metadata: dict, model: str, width: int, height: int
    ) -> bytes | None:
        """Build standard EXIF tags for a still, or ``None`` on failure.

        ``piexif`` is a Raspberry Pi dependency, but importing it lazily
        keeps this module usable on hosts that only run the UI tests.  EXIF
        is best effort: a malformed metadata value must never prevent the
        photo from being written.
        """

        try:
            import piexif
        except ImportError:
            logger.warning("piexif is not installed; saving without EXIF")
            return None

        timestamp = metadata.get("DateTime")
        if not isinstance(timestamp, str) or not timestamp:
            timestamp = time.strftime("%Y:%m:%d %H:%M:%S")

        zeroth = {
            piexif.ImageIFD.Make: "Raspberry Pi",
            piexif.ImageIFD.Model: model,
            piexif.ImageIFD.Software: "Chandelier",
            piexif.ImageIFD.DateTime: timestamp,
            piexif.ImageIFD.Orientation: 1,
        }
        exif: dict[int, object] = {
            piexif.ExifIFD.DateTimeOriginal: timestamp,
            piexif.ExifIFD.DateTimeDigitized: timestamp,
            piexif.ExifIFD.ColorSpace: 1,
            piexif.ExifIFD.PixelXDimension: int(width),
            piexif.ExifIFD.PixelYDimension: int(height),
            piexif.ExifIFD.Flash: 0,
        }

        try:
            exposure = int(metadata.get("ExposureTime", 0))
        except (TypeError, ValueError):
            exposure = 0
        if exposure > 0:
            divisor = math.gcd(exposure, 1_000_000)
            exif[piexif.ExifIFD.ExposureTime] = (
                exposure // divisor,
                1_000_000 // divisor,
            )

        try:
            total_gain = float(metadata.get("AnalogueGain", 0)) * float(
                metadata.get("DigitalGain", 1.0)
            )
        except (TypeError, ValueError):
            total_gain = 0.0
        if total_gain > 0:
            exif[piexif.ExifIFD.ISOSpeedRatings] = int(
                round(total_gain * 100)
            )

        try:
            return piexif.dump(
                {
                    "0th": zeroth,
                    "Exif": exif,
                    "GPS": {},
                    "1st": {},
                    "thumbnail": None,
                }
            )
        except Exception:
            logger.warning("failed to build still EXIF", exc_info=True)
            return None

    def _write_still(
        self,
        frame: np.ndarray,
        path: str,
        extension: str,
        metadata: dict,
    ) -> None:
        """Write ``frame`` with EXIF when the container supports it.

        JPEG goes through OpenCV's encoder and gets EXIF spliced in with
        ``piexif``.  PNG/TIFF/BMP use Pillow, which writes an eXIf chunk or
        TIFF tags directly; BMP carries no EXIF.  Writes go to a temporary
        file first so a failed encode cannot truncate an existing photo.
        """

        extension = str(extension).lower().lstrip(".")
        exif = self._exif_bytes(
            metadata,
            self._camera_model(),
            frame.shape[1],
            frame.shape[0],
        )
        options = getattr(self._cam, "options", None) or {}
        temp = f"{path}.part"
        try:
            if extension in {"jpg", "jpeg"}:
                quality = int(options.get("quality", 95))
                ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), quality],
                )
                if not ok:
                    raise OSError("JPEG encoding failed")
                data = encoded.tobytes()
                if exif:
                    try:
                        import piexif

                        tagged = io.BytesIO()
                        piexif.insert(exif, data, tagged)
                        data = tagged.getvalue()
                    except Exception:
                        logger.warning(
                            "failed to insert JPEG EXIF", exc_info=True
                        )
                with open(temp, "wb") as handle:
                    handle.write(data)
            elif extension in {"png", "tif", "tiff", "bmp"}:
                from PIL import Image

                image = Image.fromarray(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )
                format_name = {
                    "png": "PNG",
                    "tif": "TIFF",
                    "tiff": "TIFF",
                    "bmp": "BMP",
                }[extension]
                save_args: dict = {"format": format_name}
                # BMP has no EXIF container; Pillow rejects the kwarg.
                if exif and extension != "bmp":
                    save_args["exif"] = exif
                if extension == "png":
                    save_args["compress_level"] = int(
                        options.get("compress_level", 3)
                    )
                try:
                    image.save(temp, **save_args)
                except Exception:
                    if not save_args.get("exif"):
                        raise
                    # Never lose the photo because the metadata writer
                    # rejected a tag.
                    logger.warning(
                        "EXIF write failed for %s; retrying without it",
                        extension,
                        exc_info=True,
                    )
                    image.save(temp, format=format_name)
            else:
                temp = f"{path}.part.{extension}"
                if not cv2.imwrite(temp, frame):
                    raise OSError("image write failed")
            os.replace(temp, path)
        except Exception:
            try:
                os.unlink(temp)
            except OSError:
                pass
            raise

    @staticmethod
    def _metadata_description(metadata: dict) -> str:
        """Encode the full metadata dict as an ASCII TIFF description."""

        # ensure_ascii keeps the TIFF Ascii tag encodable even if a future
        # metadata value contains non-ASCII text.
        return json.dumps(
            metadata,
            default=str,
            ensure_ascii=True,
            sort_keys=True,
        )

    def _save_dng_with_metadata(self, request, file_path: str) -> None:
        """Write the raw stream as DNG with the metadata dict embedded.

        Picamera2's own ``save_dng`` only writes exposure/ISO/gain tags.
        This mirrors its writer and adds ``ImageDescription`` (the full
        metadata as ASCII JSON) plus standard DateTime tags.
        """

        # Imported lazily so this module remains importable off-device.
        from picamera2.sensor_format import SensorFormat
        from pidng.camdefs import Picamera2Camera
        from pidng.core import PICAM2DNG
        from pidng.dng import Tag

        metadata = request.get_metadata()
        config = dict(request.config["raw"])
        buffer = request.make_buffer("raw")
        raw = self._cam.helpers._make_array_shared(buffer, config)
        fmt = SensorFormat(config["format"])
        if fmt.packing == "PISP_COMP1":
            raw = self._cam.helpers.decompress(raw)
            fmt.bit_depth = 16
            config["format"] = fmt.unpacked
            config["stride"] = raw.shape[1]
            config["framesize"] = raw.shape[0] * raw.shape[1]

        model = self._cam.camera_properties.get("Model") or "Picamera2"
        camera = Picamera2Camera(config, metadata, model)
        camera.tags.set(
            Tag.ImageDescription, self._metadata_description(metadata)
        )
        timestamp = time.strftime("%Y:%m:%d %H:%M:%S", time.localtime())
        camera.tags.set(Tag.DateTime, timestamp)
        camera.tags.set(Tag.DateTimeOriginal, timestamp)

        writer = PICAM2DNG(camera)
        writer.options(
            compress=self._cam.options.get("compress_level", 0)
        )
        with open(file_path, "wb") as handle:
            writer.convert(raw, file=handle)

    def exposure_capture(
        self, exposure_time: int, width: int, height: int
    ) -> tuple[int, np.ndarray]:
        """Capture with a fixed exposure; return (applied exposure, frame).

        Oversized or non-positive dimensions fall back to 1920x1080 to keep
        the capture time and memory use bounded.
        """
        if width <= 0 or height <= 0 or width > 1920 or height > 1920:
            width, height = 1920, 1080

        config = self._cam.create_still_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        with self._lock:
            self._cam.switch_mode(config)
            self._refresh_base_crop()
            controls = dict(self._controls)
            # The preview FPS cap (and Auto noise reduction) must not clamp
            # or override the still configuration used for metering.
            controls.pop("FrameDurationLimits", None)
            if self._noise_reduction_mode is None:
                controls.pop("NoiseReductionMode", None)
            self._cam.set_controls(controls)
            self._apply_zoom()
            # Gain 1.0 isolates the exposure time; metadata reports the
            # value the sensor actually applied.
            self.set_manual_exposure(exposure_time, 1)
            time.sleep(1)
            request = self._cam.capture_request()
            try:
                frame = request.make_array("main")
                metadata = request.get_metadata()
            finally:
                request.release()
            self._restore_preview()

        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[..., :3]
        return metadata["ExposureTime"], frame

    # ------------------------------------------------------------------
    # Video recording
    # ------------------------------------------------------------------
    def start_recording(
        self, width: int, height: int, file_path: str
    ) -> None:
        """Start H.264 recording to ``file_path``.

        Video mode differs from preview mode, so the camera is reconfigured
        and left recording; call :meth:`stop_recording` to return to
        preview.  Oversized or non-positive dimensions fall back to
        1920x1080.
        """
        from picamera2.outputs import FfmpegOutput

        if width <= 0 or height <= 0 or width > 1920 or height > 1920:
            width, height = 1920, 1080

        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # lores is kept at LCD size alongside the recorded main stream.
        video_config = self._cam.create_video_configuration(
            main={"size": (int(width), int(height))},
            lores={"size": (self._screen.width, self._screen.height)},
        )
        temp_config = self._cam.create_preview_configuration(
            main={"size": (int(width), int(height))},
            lores={"size": (self._screen.width, self._screen.height)},
        )

        with self._lock:
            # This helper reads ScalerCrop for the new size and leaves the
            # camera stopped, ready for configure() below.
            request = self._cam.switch_mode_capture_request_and_stop(
                temp_config
            )
            try:
                self._base_crop = tuple(request.get_metadata()["ScalerCrop"])
            finally:
                request.release()
            self._cam.configure(video_config)
            controls = dict(self._controls)
            # Video uses its own fixed 30 fps and Fast NR defaults.
            controls.pop("FrameDurationLimits", None)
            if self._noise_reduction_mode is None:
                controls.pop("NoiseReductionMode", None)
            self._cam.set_controls(controls)
            self._apply_zoom()
            # Single AF does not track on its own; re-run it in video mode.
            if self._focus_mode == 1:
                self.trigger_autofocus()
            # FfmpegOutput muxes the encoded H.264 stream into its container.
            self._cam.start_recording(
                self._encoder, FfmpegOutput(file_path)
            )
        logger.info("recording started: %s", file_path)

    def stop_recording(self) -> None:
        """Stop recording and restore the preview pipeline."""
        with self._lock:
            self._cam.stop_recording()
            # configure() replaces the running mode and stops the camera,
            # so rebuild preview mode and start it again.
            self._cam.configure(self._preview_config)
            self._cam.start()
            self._refresh_base_crop()
            self._apply_zoom()
            self._cam.set_controls(self._controls)
            if self._focus_mode == 1:
                self.trigger_autofocus()
        logger.info("recording stopped")

    def _restore_preview(self) -> None:
        """Return to preview mode and re-apply cached controls and zoom."""
        self._cam.switch_mode(self._preview_config)
        self._cam.set_controls(self._controls)
        self._refresh_base_crop()
        self._apply_zoom()
        if self._focus_mode == 1:
            self.trigger_autofocus()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the sensor in its current mode."""
        with self._lock:
            self._cam.start()
        logger.info("camera started")

    def stop(self) -> None:
        """Stop the sensor without releasing the device."""
        with self._lock:
            self._cam.stop()
        logger.info("camera stopped")

    def close(self) -> None:
        """Release the Picamera2 device; no calls are valid afterwards."""
        self._cam.close()
        logger.info("camera closed")


def _load_tuning(config: CameraConfig):
    """Load a libcamera tuning JSON file, or None for the default tuning."""
    if config.tuning_file is None:
        return None
    with open(config.tuning_file, encoding="utf-8") as handle:
        return json.load(handle)
