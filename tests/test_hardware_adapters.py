"""Tests for the lazy hardware adapters using fake gpiozero/spidev modules."""

import sys
import types

import numpy as np

from chandelier.config import load_config
from chandelier.events import Action, Phase
from chandelier.hardware.display import St7735Display
from chandelier.hardware.gpio_input import GpioInputPanel
from chandelier.hardware.leds import Leds


class FakeDigitalOutputDevice:
    def __init__(self, pin, active_high=True, initial_value=False):
        self.pin = pin
        self.value = initial_value
        self.closed = False

    def on(self):
        self.value = True

    def off(self):
        self.value = False

    def toggle(self):
        self.value = not self.value

    def close(self):
        self.closed = True


class FakeDigitalInputDevice:
    def __init__(self, pin):
        self.pin = pin
        self.when_activated = None
        self.when_deactivated = None
        self.closed = False
        self.press = lambda: self.when_activated and self.when_activated()
        self.release = lambda: self.when_deactivated and self.when_deactivated()

    def close(self):
        self.closed = True


class FakeRotaryEncoder:
    def __init__(self, pin_a, pin_b):
        self.pin_a = pin_a
        self.pin_b = pin_b
        self.when_rotated_clockwise = None
        self.when_rotated_counter_clockwise = None
        self.closed = False

    def close(self):
        self.closed = True


class FakeSpiDev:
    instances = []

    def __init__(self, bus, device):
        self.bus = bus
        self.device = device
        self.max_speed_hz = 0
        self.mode = 0
        self.writes = []
        FakeSpiDev.instances.append(self)

    def writebytes(self, data):
        self.writes.append(("bytes", list(data)))

    def writebytes2(self, data):
        self.writes.append(("bytes2", bytes(data)))

    def close(self):
        self.closed = True


def _install_fake_hardware(monkeypatch):
    gpiozero = types.ModuleType("gpiozero")
    gpiozero.DigitalOutputDevice = FakeDigitalOutputDevice
    gpiozero.DigitalInputDevice = FakeDigitalInputDevice
    gpiozero.RotaryEncoder = FakeRotaryEncoder
    spidev = types.ModuleType("spidev")
    spidev.SpiDev = FakeSpiDev
    monkeypatch.setitem(sys.modules, "gpiozero", gpiozero)
    monkeypatch.setitem(sys.modules, "spidev", spidev)
    return gpiozero


def test_display_initialises_and_writes_rgb565(monkeypatch):
    _install_fake_hardware(monkeypatch)
    display = St7735Display(320, 240, load_config().pin)
    display.init()

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    display.show(frame)
    writes = FakeSpiDev.instances[-1].writes
    assert any(kind == "bytes2" for kind, _ in writes)
    payload = next(data for kind, data in writes if kind == "bytes2")
    assert len(payload) == 320 * 240 * 2
    # First pixel is black; check a known blue pixel separately.
    frame[0, 0] = (255, 0, 0)  # BGR
    display.show(frame)
    payload = next(
        data
        for kind, data in FakeSpiDev.instances[-1].writes[::-1]
        if kind == "bytes2"
    )
    assert payload[:2] == b"\x00\x1f"  # RGB565 blue, big-endian

    display.clear()
    payload = next(
        data
        for kind, data in FakeSpiDev.instances[-1].writes[::-1]
        if kind == "bytes2"
    )
    assert len(payload) == 320 * 240 * 2
    assert payload == bytes(len(payload))

    display.close()
    assert FakeSpiDev.instances[-1].closed is True


def test_gpio_input_maps_press_and_release(monkeypatch):
    _install_fake_hardware(monkeypatch)
    events = []
    panel = GpioInputPanel(load_config().pin, lambda a, p: events.append((a, p)))

    # The first device is the shutter button.
    panel._devices[0].press()
    panel._devices[0].release()
    assert events[0] == (Action.SHUTTER, Phase.PRESS)

    # Find the centre device and exercise both edges.
    centre = next(device for device in panel._devices if getattr(device, "pin", None) == 5)
    centre.press()
    centre.release()
    assert (Action.CENTER, Phase.PRESS) in events
    assert (Action.CENTER, Phase.RELEASE) in events

    rotary = next(
        device for device in panel._devices if isinstance(device, FakeRotaryEncoder)
    )
    rotary.when_rotated_clockwise()
    assert (Action.ROTARY_CLOCKWISE, Phase.PRESS) in events

    panel.close()
    assert all(device.closed for device in panel._devices)


def test_leds(monkeypatch):
    _install_fake_hardware(monkeypatch)
    leds = Leds(load_config().led)
    leds.green.on()
    assert leds.green.is_on is True
    leds.green.toggle()
    assert leds.green.is_on is False
    leds.blue.on()
    leds.close()
    assert leds.blue.is_on is False
