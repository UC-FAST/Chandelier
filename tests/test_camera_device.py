"""Camera device tests with a fake picamera2 package."""

import json
import sys
import types

import numpy as np
import pytest

from chandelier.config import load_config
from chandelier.hardware.camera import CameraDevice


class FakeRequest:
    def __init__(self, metadata):
        self.metadata = metadata
        self.released = False
        self.saved_dng = None

    def make_buffer(self, name):
        return b"buffer"

    def get_metadata(self):
        return dict(self.metadata)

    def make_array(self, name):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def save_dng(self, path):
        self.saved_dng = path

    def release(self):
        self.released = True


class FakePicamera2:
    sensor_resolution = (4608, 2592)

    def __init__(self, tuning=None):
        self.tuning = tuning
        self.configs = []
        self.controls = []
        self.requests = []
        self.started = False
        self.recording = False
        self.closed = False
        self.reject_controls: set[str] = set()
        self.camera_controls = {
            "AeEnable": None,
            "AwbEnable": None,
            "ExposureTime": None,
            "AnalogueGain": None,
            "ColourGains": None,
            "ExposureValue": None,
            "Saturation": None,
            "Contrast": None,
            "Sharpness": None,
            "AfMode": None,
            "AfTrigger": None,
            "AfSpeed": None,
            "AfRange": None,
            "LensPosition": None,
            "HdrMode": None,
            "ScalerCrop": None,
            "Brightness": None,
            "AeConstraintMode": None,
            "AeExposureMode": None,
            "AeMeteringMode": None,
            "AeFlickerMode": None,
            "AeFlickerPeriod": None,
            "AwbMode": None,
            "ExposureTimeMode": None,
            "AnalogueGainMode": None,
            "NoiseReductionMode": None,
            "AfMetering": None,
            "AfWindows": None,
            "AfPause": None,
            "FrameDurationLimits": None,
            "ColourTemperature": None,
            "StatsOutputEnable": None,
        }

    def set_logging(self, level):
        self.log_level = level

    def create_preview_configuration(self, **kwargs):
        return ("preview", kwargs)

    def create_still_configuration(self, **kwargs):
        return ("still", kwargs)

    def create_video_configuration(self, **kwargs):
        return ("video", kwargs)

    def configure(self, config):
        self.configs.append(config)

    def start_preview(self, preview):
        self.preview = preview

    def start(self):
        self.started = True

    def capture_metadata(self):
        return {"ScalerCrop": (0, 0, 4608, 2592)}

    def capture_request(self):
        request = FakeRequest(
            {
                "ScalerCrop": (0, 0, 4608, 2592),
                "ExposureTime": 1000,
                "FrameDuration": 33333,
                "AnalogueGain": 1.0,
                "DigitalGain": 1.0,
                "ColourTemperature": 5000,
                "Lux": 100.0,
                "FocusFoM": 12,
                "ColourGains": (1.8, 1.5),
            }
        )
        self.requests.append(request)
        return request

    def switch_mode(self, config):
        self.configs.append(("switch", config))

    def set_controls(self, controls):
        if self.reject_controls.intersection(controls):
            raise RuntimeError("control rejected")
        self.controls.append(dict(controls))

    def switch_mode_capture_request_and_stop(self, config):
        return self.capture_request()

    def start_recording(self, encoder, output):
        self.recording = True

    def stop_recording(self):
        self.recording = False

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True


def _install_fake_picamera2(monkeypatch):
    package = types.ModuleType("picamera2")
    package.Picamera2 = FakePicamera2
    package.YUV420_to_RGB = lambda buffer, size: np.zeros(
        (size[1], size[0], 3), dtype=np.uint8
    )
    package.Preview = types.SimpleNamespace(NULL=0)

    encoders = types.ModuleType("picamera2.encoders")
    encoders.H264Encoder = lambda bitrate: ("encoder", bitrate)
    outputs = types.ModuleType("picamera2.outputs")
    outputs.FfmpegOutput = lambda path: ("output", path)

    monkeypatch.setitem(sys.modules, "picamera2", package)
    monkeypatch.setitem(sys.modules, "picamera2.encoders", encoders)
    monkeypatch.setitem(sys.modules, "picamera2.outputs", outputs)
    return package


def test_camera_preview_capture_and_recording(monkeypatch, tmp_path):
    _install_fake_picamera2(monkeypatch)
    monkeypatch.setattr("chandelier.hardware.camera.time.sleep", lambda _: None)

    config = load_config()
    camera = CameraDevice(config.camera, config.screen)

    assert camera._preview_config[1]["main"]["size"] == (
        config.camera.preview_width,
        config.camera.preview_height,
    )
    frame = next(camera.preview())
    assert frame.shape == (config.screen.height, config.screen.width, 3)
    assert camera.metadata["ExposureTime"] == 1000
    assert camera.frame_quality == 12

    camera.zoom(2.0)
    assert any("ScalerCrop" in controls for controls in camera._cam.controls)

    target = tmp_path / "shots" / "image"
    camera.save_frame(str(target), "jpeg", 640, 480)
    assert (tmp_path / "shots" / "image.jpeg").exists()
    assert camera._cam.requests

    camera.start_recording(640, 480, str(tmp_path / "video.mp4"))
    assert camera._cam.recording is True
    camera.stop_recording()
    assert camera._cam.recording is False

    camera.close()
    assert camera._cam.closed is True


def test_preview_preserves_bgr_channel_order(monkeypatch):
    _install_fake_picamera2(monkeypatch)

    class ColoredRequest(FakeRequest):
        def make_array(self, name):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            frame[..., 0] = 10  # B
            frame[..., 1] = 20  # G
            frame[..., 2] = 30  # R
            return frame

    config = load_config()
    camera = CameraDevice(config.camera, config.screen)
    monkeypatch.setattr(
        camera._cam,
        "capture_request",
        lambda: ColoredRequest({"ScalerCrop": (0, 0, 4608, 2592)}),
    )

    frame = next(camera.preview())
    # picamera2's "RGB888" is BGR in memory; no channel swap is expected.
    assert tuple(frame[0, 0]) == (10, 20, 30)
    camera.close()


def test_new_pro_controls(monkeypatch):
    _install_fake_picamera2(monkeypatch)
    monkeypatch.setattr("chandelier.hardware.camera.time.sleep", lambda _: None)
    config = load_config()
    camera = CameraDevice(config.camera, config.screen)
    next(camera.preview())

    camera.set_hdr(2)
    camera.set_exposure_value(1.5)
    camera.set_image_controls(saturation=1.2, contrast=0.9, sharpness=1.4)
    camera.set_focus(True, 500)
    camera.set_af_speed(1)
    camera.set_af_range(2)
    camera.set_ae_lock(True)
    camera.set_awb_lock(True)

    applied = {}
    for controls in camera._cam.controls:
        applied.update(controls)
    assert applied["HdrMode"] == 2
    assert applied["ExposureValue"] == 1.5
    assert applied["Saturation"] == 1.2
    assert applied["Contrast"] == 0.9
    assert applied["Sharpness"] == 1.4
    assert applied["AfMode"] == 0
    assert applied["LensPosition"] == 5.0
    assert applied["AfSpeed"] == 1
    assert applied["AfRange"] == 2
    assert applied["AeEnable"] is False
    assert applied["ExposureTime"] == 1000
    assert applied["AnalogueGain"] == 1.0
    assert applied["AwbEnable"] is False
    assert applied["ColourGains"] == (1.8, 1.5)


def test_focus_modes_and_trigger(monkeypatch):
    _install_fake_picamera2(monkeypatch)
    monkeypatch.setattr("chandelier.hardware.camera.time.sleep", lambda _: None)
    config = load_config()
    camera = CameraDevice(config.camera, config.screen)

    camera.set_focus_mode(1, 500)
    applied = {}
    for controls in camera._cam.controls:
        applied.update(controls)
    assert applied["AfMode"] == 1
    assert applied["AfTrigger"] == 0

    camera.trigger_autofocus()
    assert camera._cam.controls[-1]["AfTrigger"] == 0

    camera.set_focus_mode(0, 250)
    applied = {}
    for controls in camera._cam.controls:
        applied.update(controls)
    assert applied["AfMode"] == 0
    assert applied["LensPosition"] == 2.5

    camera.set_focus_mode(2, 250)
    assert camera._cam.controls[-1]["AfMode"] == 2


def test_unsupported_and_rejected_controls_are_skipped(monkeypatch):
    _install_fake_picamera2(monkeypatch)
    monkeypatch.setattr("chandelier.hardware.camera.time.sleep", lambda _: None)
    config = load_config()
    camera = CameraDevice(config.camera, config.screen)

    del camera._cam.camera_controls["Contrast"]
    camera._cam.reject_controls.add("Sharpness")
    camera.set_image_controls(saturation=1.1, contrast=1.3, sharpness=2.0)

    applied = {}
    for controls in camera._cam.controls:
        applied.update(controls)
    assert applied["Saturation"] == 1.1
    assert "Contrast" not in applied
    assert "Sharpness" not in applied
    assert "Contrast" not in camera._controls
    assert "Sharpness" not in camera._controls


def test_save_frame_applies_long_exposure_to_still_config(
    monkeypatch, tmp_path
):
    _install_fake_picamera2(monkeypatch)
    monkeypatch.setattr("chandelier.hardware.camera.time.sleep", lambda _: None)
    config = load_config()
    camera = CameraDevice(config.camera, config.screen)

    camera.save_frame(
        file_path=str(tmp_path / "long"),
        extension="jpeg",
        width=640,
        height=480,
        exposure_time=4_000_000,
        analogue_gain=2.0,
    )

    still_configs = [
        cfg
        for kind, cfg in camera._cam.configs
        if kind == "switch" and cfg[0] == "still"
    ]
    assert still_configs
    controls = still_configs[-1][1]["controls"]
    assert controls["AeEnable"] is False
    assert controls["ExposureTime"] == 4_000_000
    assert controls["AnalogueGain"] == 2.0
    assert controls["ExposureTimeMode"] == 1
    assert controls["AnalogueGainMode"] == 1
    assert controls["FrameDurationLimits"] == (4_050_000, 4_050_000)


def _merged_controls(camera):
    applied = {}
    for controls in camera._cam.controls:
        applied.update(controls)
    return applied


def test_exposure_priority_control_sequences(monkeypatch):
    _install_fake_picamera2(monkeypatch)
    monkeypatch.setattr("chandelier.hardware.camera.time.sleep", lambda _: None)
    config = load_config()
    camera = CameraDevice(config.camera, config.screen)

    assert camera.set_exposure_priority(1, 8000, 2.0) is True
    applied = _merged_controls(camera)
    assert applied["AeEnable"] is True
    assert applied["ExposureTimeMode"] == 1
    assert applied["AnalogueGainMode"] == 0
    assert applied["ExposureTime"] == 8000

    assert camera.set_exposure_priority(2, 8000, 3.0) is True
    applied = _merged_controls(camera)
    assert applied["ExposureTimeMode"] == 0
    assert applied["AnalogueGainMode"] == 1
    assert applied["AnalogueGain"] == 3.0

    assert camera.set_exposure_priority(0, 8000, 3.0) is True
    applied = _merged_controls(camera)
    assert applied["ExposureTimeMode"] == 0
    assert applied["AnalogueGainMode"] == 0


def test_exposure_priority_falls_back_when_modes_unsupported(monkeypatch):
    _install_fake_picamera2(monkeypatch)
    monkeypatch.setattr("chandelier.hardware.camera.time.sleep", lambda _: None)
    config = load_config()
    camera = CameraDevice(config.camera, config.screen)

    del camera._cam.camera_controls["ExposureTimeMode"]
    assert camera.set_exposure_priority(1, 8000, 2.0) is False
    del camera._cam.camera_controls["AnalogueGainMode"]
    assert camera.set_exposure_priority(2, 8000, 2.0) is False


def test_noise_reduction_pause_rate_and_stats_controls(monkeypatch):
    _install_fake_picamera2(monkeypatch)
    monkeypatch.setattr("chandelier.hardware.camera.time.sleep", lambda _: None)
    config = load_config()
    camera = CameraDevice(config.camera, config.screen)

    camera.set_noise_reduction(2)
    assert _merged_controls(camera)["NoiseReductionMode"] == 2
    camera.set_noise_reduction(-1)
    assert camera._noise_reduction_mode is None
    assert _merged_controls(camera)["NoiseReductionMode"] == 3

    camera.set_af_pause(True)
    assert _merged_controls(camera)["AfPause"] == 0
    camera.set_af_pause(False)
    assert _merged_controls(camera)["AfPause"] == 2

    camera.set_frame_rate(30)
    assert _merged_controls(camera)["FrameDurationLimits"] == (33333, 83333)
    camera.set_frame_rate(15)
    assert _merged_controls(camera)["FrameDurationLimits"] == (66667, 83333)
    camera.set_frame_rate(5)
    assert _merged_controls(camera)["FrameDurationLimits"] == (200000, 200000)

    camera.set_stats_output(True)
    assert camera._stats_output is True
    assert camera._preview_config[1]["controls"] == {
        "StatsOutputEnable": True
    }
    assert camera._cam.configs[-1][0] == "switch"
    camera.set_stats_output(False)
    assert camera._stats_output is False
    assert camera._controls["StatsOutputEnable"] is False
    # Re-enabling drops the cached False and reconfigures the stream.
    camera.set_stats_output(True)
    assert "StatsOutputEnable" not in camera._controls


def test_colour_temperature_control(monkeypatch):
    _install_fake_picamera2(monkeypatch)
    monkeypatch.setattr("chandelier.hardware.camera.time.sleep", lambda _: None)
    config = load_config()
    camera = CameraDevice(config.camera, config.screen)

    assert camera.set_colour_temperature(4000) is True
    applied = _merged_controls(camera)
    assert applied["AwbEnable"] is False
    assert applied["ColourTemperature"] == 4000

    del camera._cam.camera_controls["ColourTemperature"]
    assert camera.set_colour_temperature(5000) is False


def test_focus_window_maps_through_zoom(monkeypatch):
    _install_fake_picamera2(monkeypatch)
    monkeypatch.setattr("chandelier.hardware.camera.time.sleep", lambda _: None)
    config = load_config()
    camera = CameraDevice(config.camera, config.screen)
    camera.zoom(2.0)

    left, top, width, height = camera.current_crop()
    assert (left, top) == (1152, 648)
    assert (width, height) == (2304, 1296)

    window = camera.focus_window_at(0.5, 0.5)
    assert window[0] + window[2] // 2 == 2304
    assert window[1] + window[3] // 2 == 1296

    corner = camera.focus_window_at(0.0, 0.0)
    assert corner[0] == left
    assert corner[1] == top

    assert camera.set_af_window(window) is True
    applied = _merged_controls(camera)
    assert applied["AfMetering"] == 1
    assert applied["AfWindows"] == [window]

    assert camera.set_af_window(None) is True
    assert _merged_controls(camera)["AfMetering"] == 0


def test_save_frame_semi_auto_still_controls(monkeypatch, tmp_path):
    _install_fake_picamera2(monkeypatch)
    monkeypatch.setattr("chandelier.hardware.camera.time.sleep", lambda _: None)
    config = load_config()
    camera = CameraDevice(config.camera, config.screen)
    camera.set_frame_rate(5)

    camera.save_frame(
        file_path=str(tmp_path / "shutter"),
        extension="jpeg",
        width=640,
        height=480,
        exposure_time=8000,
        analogue_gain=None,
        exposure_priority=1,
    )
    still = [
        cfg
        for kind, cfg in camera._cam.configs
        if kind == "switch" and cfg[0] == "still"
    ][-1][1]
    controls = still["controls"]
    assert controls["ExposureTimeMode"] == 1
    assert controls["AnalogueGainMode"] == 0
    assert controls["ExposureTime"] == 8000
    assert "FrameDurationLimits" in controls

    camera.save_frame(
        file_path=str(tmp_path / "gain"),
        extension="jpeg",
        width=640,
        height=480,
        exposure_time=None,
        analogue_gain=3.0,
        exposure_priority=2,
    )
    still = [
        cfg
        for kind, cfg in camera._cam.configs
        if kind == "switch" and cfg[0] == "still"
    ][-1][1]
    controls = still["controls"]
    assert controls["ExposureTimeMode"] == 0
    assert controls["AnalogueGainMode"] == 1
    assert controls["AnalogueGain"] == 3.0


def test_metadata_description_is_ascii_json():
    description = CameraDevice._metadata_description(
        {
            "ExposureTime": 1000,
            "ColourGains": (1.5, 2.0),
            "ScalerCrop": object(),
            "Note": "非 ASCII",
        }
    )
    assert description.isascii()
    decoded = json.loads(description)
    assert decoded["ExposureTime"] == 1000
    assert decoded["ColourGains"] == [1.5, 2.0]
    assert "ScalerCrop" in decoded


def test_save_raw_routes_through_metadata_dng_writer(monkeypatch, tmp_path):
    _install_fake_picamera2(monkeypatch)
    monkeypatch.setattr("chandelier.hardware.camera.time.sleep", lambda _: None)
    config = load_config()
    camera = CameraDevice(config.camera, config.screen)

    written = []
    monkeypatch.setattr(
        camera,
        "_save_dng_with_metadata",
        lambda request, path: written.append(path),
    )

    camera.save_frame(
        file_path=str(tmp_path / "raw"),
        extension="jpeg",
        width=640,
        height=480,
        save_raw=True,
    )

    assert written == [str(tmp_path / "raw.dng")]


def test_save_frame_embeds_jpeg_exif(monkeypatch, tmp_path):
    piexif = pytest.importorskip("piexif")
    _install_fake_picamera2(monkeypatch)
    monkeypatch.setattr("chandelier.hardware.camera.time.sleep", lambda _: None)
    config = load_config()
    camera = CameraDevice(config.camera, config.screen)

    camera.save_frame(
        file_path=str(tmp_path / "exif"),
        extension="jpeg",
        width=640,
        height=480,
    )

    exif = piexif.load(str(tmp_path / "exif.jpeg"))
    assert exif["0th"][piexif.ImageIFD.Make] == b"Raspberry Pi"
    assert exif["0th"][piexif.ImageIFD.Software] == b"Chandelier"
    assert exif["Exif"][piexif.ExifIFD.ExposureTime] == (1, 1000)
    assert exif["Exif"][piexif.ExifIFD.ISOSpeedRatings] == 100
    assert exif["Exif"][piexif.ExifIFD.PixelXDimension] == 640
    assert exif["Exif"][piexif.ExifIFD.PixelYDimension] == 480


@pytest.mark.parametrize("extension", ["png", "tiff"])
def test_save_frame_embeds_exif_and_applies_watermark(
    monkeypatch, tmp_path, extension
):
    piexif = pytest.importorskip("piexif")
    from PIL import Image

    _install_fake_picamera2(monkeypatch)
    monkeypatch.setattr("chandelier.hardware.camera.time.sleep", lambda _: None)
    config = load_config()
    camera = CameraDevice(config.camera, config.screen)
    seen = {}

    def watermark(frame):
        seen["shape"] = frame.shape
        frame[:8, :8] = (0, 0, 255)  # BGR red

    camera.save_frame(
        file_path=str(tmp_path / "marked"),
        extension=extension,
        width=640,
        height=480,
        watermark=watermark,
    )

    assert seen["shape"] == (480, 640, 3)
    path = tmp_path / f"marked.{extension}"
    with Image.open(path) as image:
        assert image.getpixel((0, 0))[:3] == (255, 0, 0)
        exif = image.getexif()
        assert exif[piexif.ImageIFD.Make] == "Raspberry Pi"
        sub = exif.get_ifd(0x8769)
        assert sub[piexif.ExifIFD.ISOSpeedRatings] == 100
