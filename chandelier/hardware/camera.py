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
    def __init__(
        self,
        config: CameraConfig,
        screen: ScreenConfig,
        verbose_console: int = logging.INFO,
    ):
        import picamera2
        from picamera2.encoders import H264Encoder

        self._screen = screen
        self._config = config

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
        self._encoder = H264Encoder(config.video_bitrate)

        self._lock = threading.Lock()
        self._frame_per_second = 0.0
        self._digital_zoom = 1.0
        self._brightness = 0.0
        self._controls: dict = {}
        self._focus_mode: int | None = None
        self._metadata: dict | None = None
        self._frame = np.zeros(
            (screen.height, screen.width, 3), dtype=np.uint8
        )

        self._cam.start_preview(picamera2.Preview.NULL)
        self._cam.start()
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
        if not self._metadata:
            return None
        return float(self._metadata.get("FocusFoM", 0))

    def zoom(self, zoom: float) -> None:
        self._digital_zoom = max(1.0, round(float(zoom), 2))
        self._apply_zoom()
        logger.debug("zoom set to %.2f", self._digital_zoom)

    def _apply_zoom(self) -> None:
        w_offset, h_offset, f_width, f_height = self._base_crop
        p_width = max(1, int(f_width // self._digital_zoom))
        p_height = max(1, int(f_height // self._digital_zoom))
        offset = [
            int((f_width - p_width) // 2 + w_offset),
            int((f_height - p_height) // 2 + h_offset),
        ]
        self._set_controls({"ScalerCrop": offset + [p_width, p_height]})

    def brightness(self, brightness: float) -> None:
        self._brightness = max(-1.0, min(1.0, round(float(brightness), 2)))
        self._set_controls({"Brightness": self._brightness})
        logger.debug("brightness set to %.2f", self._brightness)

    def _control_supported(self, name: str) -> bool:
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
            if (
                not force
                and name not in self._VOLATILE_CONTROLS
                and name in self._controls
                and self._controls[name] == value
            ):
                continue
            if not self._control_supported(name):
                logger.warning("camera control %s is not supported", name)
                continue
            try:
                self._cam.set_controls({name: value})
            except Exception as exc:  # libcamera raises RuntimeError variants
                logger.warning("camera control %s failed: %s", name, exc)
                continue
            if name not in self._VOLATILE_CONTROLS:
                self._controls[name] = value

    def _refresh_base_crop(self) -> None:
        self._base_crop = tuple(self._cam.capture_metadata()["ScalerCrop"])

    # ------------------------------------------------------------------
    # Auto exposure / white balance
    # ------------------------------------------------------------------
    def set_ae_enable(self, enable: bool) -> None:
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
        self._set_controls(
            {"AwbEnable": False, "ColourGains": (red_gain, blue_gain)}
        )

    def set_hdr(self, mode: int) -> None:
        self._set_controls({"HdrMode": int(mode)})

    def set_exposure_value(self, value: float) -> None:
        self._set_controls({"ExposureValue": float(value)})

    def set_image_controls(
        self,
        *,
        saturation: float | None = None,
        contrast: float | None = None,
        sharpness: float | None = None,
    ) -> None:
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
        if not enable:
            self._set_controls({"AeEnable": True})
            return
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
        if not enable:
            self._set_controls({"AwbEnable": True})
            return
        gains = (self._metadata or {}).get("ColourGains")
        if gains is None:
            gains = self._controls.get("ColourGains")
        controls = {"AwbEnable": False}
        if gains:
            controls["ColourGains"] = tuple(gains)
        self._set_controls(controls)

    def set_focus(self, manual: bool, position: int) -> None:
        if manual:
            # LensPosition is in dioptres; the menu uses 0..1000.
            lens_position = max(0.0, min(10.0, float(position) / 100.0))
            self._set_controls(
                {"AfMode": 0, "LensPosition": lens_position}
            )
        else:
            self._set_controls({"AfMode": 2})

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------
    def preview(self) -> Iterator[np.ndarray]:
        previous = time.monotonic()
        while True:
            with self._lock:
                request = self._cam.capture_request()
                try:
                    # libcamera's "RGB888" is BGR in memory (picamera2 maps it
                    # to bgr24), so the array is already in OpenCV order.
                    array = request.make_array("main")
                    if array.ndim == 3 and array.shape[2] == 4:
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
    ) -> None:
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        if width <= 0 or height <= 0:
            width, height = self._cam.sensor_resolution

        if save_raw:
            still_config = self._cam.create_still_configuration(
                main={"size": (width, height), "format": "RGB888"},
                raw={"size": self._cam.sensor_resolution},
            )
        else:
            still_config = self._cam.create_still_configuration(
                main={"size": (width, height), "format": "RGB888"}
            )

        with self._lock:
            request = None
            try:
                self._cam.switch_mode(still_config)
                self._refresh_base_crop()
                self._cam.set_controls(self._controls)
                self._apply_zoom()
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
                    request.save_dng(f"{file_path}.dng")
            finally:
                if request is not None:
                    request.release()
                self._restore_preview()

    def exposure_capture(
        self, exposure_time: int, width: int, height: int
    ) -> tuple[int, np.ndarray]:
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
        from picamera2.outputs import FfmpegOutput

        if width <= 0 or height <= 0 or width > 1920 or height > 1920:
            width, height = 1920, 1080

        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        video_config = self._cam.create_video_configuration(
            main={"size": (int(width), int(height))},
            lores={"size": (self._screen.width, self._screen.height)},
        )
        temp_config = self._cam.create_preview_configuration(
            main={"size": (int(width), int(height))},
            lores={"size": (self._screen.width, self._screen.height)},
        )

        with self._lock:
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
            self._cam.start_recording(
                self._encoder, FfmpegOutput(file_path)
            )
        logger.info("recording started: %s", file_path)

    def stop_recording(self) -> None:
        with self._lock:
            self._cam.stop_recording()
            self._cam.configure(self._preview_config)
            self._cam.start()
            self._refresh_base_crop()
            self._apply_zoom()
            self._cam.set_controls(self._controls)
        logger.info("recording stopped")

    def _restore_preview(self) -> None:
        self._cam.switch_mode(self._preview_config)
        self._cam.set_controls(self._controls)
        self._refresh_base_crop()
        self._apply_zoom()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            self._cam.start()
        logger.info("camera started")

    def stop(self) -> None:
        with self._lock:
            self._cam.stop()
        logger.info("camera stopped")

    def close(self) -> None:
        self._cam.close()
        logger.info("camera closed")


def _load_tuning(config: CameraConfig):
    if config.tuning_file is None:
        return None
    with open(config.tuning_file, encoding="utf-8") as handle:
        return json.load(handle)
