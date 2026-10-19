"""Picamera2 wrapper.

Frames leave this module as BGR ``numpy`` arrays, matching OpenCV and the LCD
driver.  All calls that switch sensor modes restore the preview configuration
and re-apply the current digital zoom/crop.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Iterator

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

    def _apply_zoom(self) -> None:
        w_offset, h_offset, f_width, f_height = self._base_crop
        p_width = max(1, int(f_width // self._digital_zoom))
        p_height = max(1, int(f_height // self._digital_zoom))
        # Shrink the base crop symmetrically around its centre.  ScalerCrop
        # expects [left, top, width, height] in sensor pixels.
        offset = [
            int((f_width - p_width) // 2 + w_offset),
            int((f_height - p_height) // 2 + h_offset),
        ]
        self._set_controls({"ScalerCrop": offset + [p_width, p_height]})

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
            self._set_controls({"AeEnable": True})
            return
        # Use the latest preview metadata; capturing a fresh metadata set is
        # a fallback for callers that lock before the first preview frame.
        metadata = self._metadata
        if not metadata:
            try:
                metadata = self._cam.capture_metadata()
            except Exception:
                metadata = {}
        controls = {"AeEnable": False}
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
        changed = mode != self._focus_mode
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
    ) -> None:
        """Capture a still image to ``file_path``.

        A non-positive ``width``/``height`` selects the full sensor
        resolution.  ``save_raw`` also writes a DNG, and ``save_metadata``
        writes libcamera metadata as JSON next to the image.  Manual
        exposure is in microseconds and disables AE for this capture.
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
        if exposure_time is not None:
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
                if still_controls:
                    # The still configuration already carries these; do not
                    # overwrite them with the stale cached preview values.
                    for key in (
                        "AeEnable",
                        "ExposureTime",
                        "AnalogueGain",
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
                    cv2.imwrite(f"{file_path}.{extension}", frame)
                if save_metadata:
                    with open(f"{file_path}.json", "w", encoding="utf-8") as f:
                        json.dump(request.get_metadata(), f, indent=4)
                if save_raw:
                    # The DNG carries the raw Bayer frame and its metadata.
                    request.save_dng(f"{file_path}.dng")
            finally:
                # Always return to preview mode, even if the capture failed.
                if request is not None:
                    request.release()
                self._restore_preview()

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
            self._cam.set_controls(self._controls)
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
            self._cam.set_controls(self._controls)
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
