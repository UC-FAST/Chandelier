"""Hardware fakes used by the unit tests."""

from __future__ import annotations

import time

import numpy as np


class FakeDisplay:
    def __init__(self, width: int = 320, height: int = 240):
        self.width = width
        self.height = height
        self.frames: list[np.ndarray] = []
        self.backlight_state = True
        self.cleared = False
        self.closed = False

    def show(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def clear(self) -> None:
        self.cleared = True

    def backlight(self, state: bool) -> None:
        self.backlight_state = state

    def close(self) -> None:
        self.closed = True


class FakeLed:
    def __init__(self):
        self.is_on = False
        self.closed = False

    def on(self) -> None:
        self.is_on = True

    def off(self) -> None:
        self.is_on = False

    def toggle(self) -> None:
        self.is_on = not self.is_on

    def close(self) -> None:
        self.closed = True


class FakeLeds:
    def __init__(self):
        self.green = FakeLed()
        self.blue = FakeLed()

    def close(self) -> None:
        self.green.close()
        self.blue.close()


class FakeCamera:
    def __init__(self, width: int = 320, height: int = 240):
        self.width = width
        self.height = height
        self.metadata = {
            "ExposureTime": 1000,
            "FrameDuration": 33333,
            "AnalogueGain": 1.5,
            "DigitalGain": 1.0,
            "Lux": 120.5,
            "ColourTemperature": 5000,
            "FocusFoM": 42,
            "AeState": 2,
            "AfState": 2,
            "AfPauseState": 0,
            "SensorTemperature": 45.0,
            "SensorTimestamp": 1_234_567_890_000,
            "FrameWallClock": 1_700_000_000_000_000_000,
        }
        self.frame_per_second = 30.0
        self.calls: list[tuple[str, dict]] = []
        self.saved: list[dict] = []
        self.closed = False
        self._zoom = 1.0
        self._brightness = 0.0

    @property
    def frame_quality(self) -> float:
        return float(self.metadata["FocusFoM"])

    def _record(self, name: str, **kwargs) -> None:
        self.calls.append((name, kwargs))

    def preview(self):
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        while not self.closed:
            yield frame.copy()

    def zoom(self, zoom: float) -> None:
        self._zoom = zoom
        self._record("zoom", zoom=zoom)

    def brightness(self, brightness: float) -> None:
        self._brightness = brightness
        self._record("brightness", brightness=brightness)

    def set_ae_enable(self, enable: bool) -> None:
        self._record("set_ae_enable", enable=enable)

    def set_ae_exposure_mode(self, code: int) -> None:
        self._record("set_ae_exposure_mode", code=code)

    def set_ae_constraint_mode(self, code: int) -> None:
        self._record("set_ae_constraint_mode", code=code)

    def set_ae_metering_mode(self, code: int) -> None:
        self._record("set_ae_metering_mode", code=code)

    def set_ae_flicker_mode(self, code: int) -> None:
        self._record("set_ae_flicker_mode", code=code)

    def set_ae_flicker_period(self, code: int) -> None:
        self._record("set_ae_flicker_period", code=code)

    def set_manual_exposure(self, exposure_time, analogue_gain) -> None:
        self._record(
            "set_manual_exposure",
            exposure_time=exposure_time,
            analogue_gain=analogue_gain,
        )

    def set_awb_enable(self, enable: bool) -> None:
        self._record("set_awb_enable", enable=enable)

    def set_awb_mode(self, code: int) -> None:
        self._record("set_awb_mode", code=code)

    def set_colour_gains(self, red_gain, blue_gain) -> None:
        self._record(
            "set_colour_gains", red_gain=red_gain, blue_gain=blue_gain
        )

    def set_hdr(self, mode) -> None:
        self._record("set_hdr", mode=mode)

    def set_exposure_value(self, value) -> None:
        self._record("set_exposure_value", value=value)

    def set_image_controls(self, **kwargs) -> None:
        self._record("set_image_controls", **kwargs)

    def set_ae_lock(self, enable) -> None:
        self._record("set_ae_lock", enable=enable)

    def set_awb_lock(self, enable) -> None:
        self._record("set_awb_lock", enable=enable)

    def set_exposure_priority(
        self, priority, exposure_time, analogue_gain
    ) -> bool:
        self._record(
            "set_exposure_priority",
            priority=priority,
            exposure_time=exposure_time,
            analogue_gain=analogue_gain,
        )
        return True

    def set_noise_reduction(self, mode) -> None:
        self._record("set_noise_reduction", mode=mode)

    def set_af_pause(self, paused) -> None:
        self._record("set_af_pause", paused=paused)

    def set_frame_rate(self, fps) -> None:
        self._record("set_frame_rate", fps=fps)

    def set_colour_temperature(self, kelvin) -> bool:
        self._record("set_colour_temperature", kelvin=kelvin)
        return True

    def set_stats_output(self, enable) -> None:
        self._record("set_stats_output", enable=enable)

    def focus_window_at(self, nx, ny, fraction=0.2):
        self._record(
            "focus_window_at", nx=nx, ny=ny, fraction=fraction
        )
        return (10, 10, 64, 64)

    def set_af_window(self, window) -> bool:
        self._record("set_af_window", window=window)
        return True

    def set_focus(self, manual, position) -> None:
        self._record("set_focus", manual=manual, position=position)

    def set_focus_mode(self, mode, position=500) -> None:
        self._record("set_focus_mode", mode=mode, position=position)

    def set_focus_position(self, position) -> None:
        self._record("set_focus_position", position=position)

    def set_af_speed(self, speed) -> None:
        self._record("set_af_speed", speed=speed)

    def set_af_range(self, af_range) -> None:
        self._record("set_af_range", af_range=af_range)

    def trigger_autofocus(self) -> None:
        self._record("trigger_autofocus")

    def save_frame(self, **kwargs) -> None:
        self.saved.append(kwargs)

    def start_recording(self, width, height, file_path) -> None:
        self._record(
            "start_recording",
            width=width,
            height=height,
            file_path=file_path,
        )

    def stop_recording(self) -> None:
        self._record("stop_recording")

    def start(self) -> None:
        self._record("start")

    def stop(self) -> None:
        self._record("stop")

    def close(self) -> None:
        self.closed = True


class FakeStats:
    def hardware_report(self):
        return {"CPU {}%": 10}

    def power_report(self):
        return {"BAT {}v {}%": (4.0, 50.0)}

    def network_report(self):
        return {"IP {}": "127.0.0.1"}


class FakeSensor:
    def __init__(self, value=4.0):
        self.value = value
        self.closed = False

    def get_bat(self):
        return self.value

    def get_battery_percent(self, voltage):
        return 50.0

    def read_voltage(self):
        return 5.0

    def read_shunt_voltage(self):
        return 0.01

    def read_current(self):
        return 0.5

    def read_power(self):
        return 2.5

    def read_time(self):
        return time.time()

    def close(self):
        self.closed = True
