"""Camera device tests with a fake picamera2 package."""

import sys
import types

import numpy as np

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

    assert camera._preview_config[1]["main"]["size"] == (640, 480)
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
    assert applied["AeEnable"] is False
    assert applied["ExposureTime"] == 1000
    assert applied["AnalogueGain"] == 1.0
    assert applied["AwbEnable"] is False
    assert applied["ColourGains"] == (1.8, 1.5)


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
