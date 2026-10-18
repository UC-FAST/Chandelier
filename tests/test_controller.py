import threading
import time
import types
from dataclasses import replace

import numpy as np

from chandelier.config import load_config
from chandelier.constants import EndId
from chandelier.controller import Controller
from chandelier.events import Action, Phase
from chandelier.ui.controlled_end import ControlledEnd
from tests.fakes import FakeDisplay


class RecordingEnd(ControlledEnd):
    def __init__(self, _id):
        super().__init__(_id)
        self.entered = []
        self.exit_count = 0
        self.actions = []
        self.running = True
        self.received = []

    def on_enter(self, last_id):
        self.entered.append(last_id)

    def on_exit(self):
        self.exit_count += 1

    def main_loop(self):
        while self.running:
            yield np.zeros((240, 320, 3), dtype=np.uint8)
            time.sleep(0.005)

    def center_press_action(self):
        self.actions.append("center")

    def up_press_action(self):
        self.actions.append("up")

    def msg_receiver(self, sender, payload):
        self.received.append((sender, payload))


def _controller():
    config = load_config()
    display = FakeDisplay(config.screen.width, config.screen.height)
    first = RecordingEnd(EndId.CAMERA.value)
    second = RecordingEnd(EndId.GALLERY.value)
    controller = Controller(display, [first, second], config)
    return controller, display, first, second


def test_switch_to_changes_current_end():
    controller, _, first, second = _controller()
    controller.switch_to(EndId.GALLERY.value)

    assert controller.current_end is second
    assert first.exit_count == 1
    assert controller._last_end_id == EndId.CAMERA.value


def test_switch_to_unknown_end_raises():
    controller, _, _, _ = _controller()
    try:
        controller.switch_to("missing")
    except LookupError:
        pass
    else:
        raise AssertionError("expected LookupError")


def test_action_gating_matches_original_behaviour():
    controller, _, _, second = _controller()
    controller.switch_to(EndId.GALLERY.value)

    assert controller.handle_action(Action.CENTER, Phase.PRESS) is True
    assert second.actions == ["center"]

    controller.enabled = False
    assert controller.handle_action(Action.CENTER, Phase.PRESS) is False
    assert controller.handle_action(Action.UP, Phase.PRESS) is True
    assert second.actions == ["center", "up"]


def test_message_routing():
    controller, _, _, second = _controller()
    controller.switch_to(EndId.GALLERY.value)
    controller.route_message(EndId.CAMERA.value, EndId.GALLERY.value, "hello")
    assert second.received == [("camera", "hello")]


def test_controller_message_route_is_accepted():
    controller, _, _, _ = _controller()
    # Unknown controller payloads are ignored rather than raising.
    controller.route_message(EndId.CAMERA.value, EndId.CONTROLLER.value, None)


class FakeLocalization:
    def __init__(self, sources=None):
        self._sources = sources or [
            types.SimpleNamespace(
                screen_x=160,
                screen_y=120,
                azimuth=90.0,
                elevation=45.0,
            )
        ]

    def read(self):
        return types.SimpleNamespace(
            confidence=0.5,
            generation=1,
            sources=tuple(self._sources),
        )


def test_localization_marker_is_drawn_on_camera_only():
    config = load_config()
    display = FakeDisplay(config.screen.width, config.screen.height)
    camera_end = RecordingEnd(EndId.CAMERA.value)
    controller = Controller(display, [camera_end], config)

    thread = threading.Thread(
        target=controller.main_loop,
        kwargs={"localization": FakeLocalization()},
        daemon=True,
    )
    thread.start()

    deadline = time.time() + 2
    while not display.frames and time.time() < deadline:
        time.sleep(0.01)
    controller.request_stop()
    thread.join(timeout=2)

    assert display.frames
    # The crosshair is drawn in the marker colour (BGR red).
    assert any(
        int(frame[..., 2].max()) > 200 for frame in display.frames
    )


def test_localization_marker_is_skipped_on_non_camera_screens():
    config = load_config()
    display = FakeDisplay(config.screen.width, config.screen.height)
    gallery_end = RecordingEnd(EndId.GALLERY.value)
    controller = Controller(display, [gallery_end], config)

    thread = threading.Thread(
        target=controller.main_loop,
        kwargs={"localization": FakeLocalization()},
        daemon=True,
    )
    thread.start()

    deadline = time.time() + 2
    while not display.frames and time.time() < deadline:
        time.sleep(0.01)
    controller.request_stop()
    thread.join(timeout=2)

    assert display.frames
    assert all(int(frame[..., 2].max()) == 0 for frame in display.frames)


def test_render_loop_switches_and_shuts_down():
    controller, display, first, second = _controller()
    thread = threading.Thread(
        target=controller.main_loop, kwargs={"localization": None}, daemon=True
    )
    thread.start()

    deadline = time.time() + 2
    while not display.frames and time.time() < deadline:
        time.sleep(0.01)
    assert display.frames

    controller.switch_to(EndId.GALLERY.value)
    deadline = time.time() + 2
    while second.entered == [] and time.time() < deadline:
        time.sleep(0.01)
    assert second.entered

    controller.request_stop()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert display.cleared is True
    assert display.backlight_state is False


def test_multiple_source_markers_use_distinct_colours():
    config = load_config()
    config = replace(
        config,
        marker=replace(config.marker, mode="multiple", max_sources=2),
    )
    display = FakeDisplay(config.screen.width, config.screen.height)
    camera_end = RecordingEnd(EndId.CAMERA.value)
    controller = Controller(display, [camera_end], config)

    localization = FakeLocalization(
        [
            types.SimpleNamespace(
                screen_x=100, screen_y=100, azimuth=90.0, elevation=45.0
            ),
            types.SimpleNamespace(
                screen_x=220, screen_y=100, azimuth=180.0, elevation=30.0
            ),
        ]
    )
    thread = threading.Thread(
        target=controller.main_loop,
        kwargs={"localization": localization},
        daemon=True,
    )
    thread.start()

    deadline = time.time() + 2
    while not display.frames and time.time() < deadline:
        time.sleep(0.01)
    controller.request_stop()
    thread.join(timeout=2)

    frame = display.frames[-1]
    red_blue, red_green, red_red = (int(v) for v in frame[100, 100])
    green_blue, green_green, green_red = (int(v) for v in frame[100, 220])
    assert red_red > 200 and red_blue < 20 and red_green < 20
    assert green_green > 200 and green_blue < 20 and green_red < 20


def test_controller_accepts_menu_reboot_message(monkeypatch):
    calls = []
    monkeypatch.setattr(
        Controller, "_reboot", staticmethod(lambda: calls.append("reboot"))
    )
    controller = Controller.__new__(Controller)
    controller._handle_controller_message(("restart", {}))
    assert calls == ["reboot"]


def test_controller_accepts_power_option_messages(monkeypatch):
    calls = []
    monkeypatch.setattr(
        Controller, "_reboot", staticmethod(lambda: calls.append("reboot"))
    )
    monkeypatch.setattr(
        Controller,
        "_shutdown",
        staticmethod(lambda: calls.append("shutdown")),
    )
    controller = Controller.__new__(Controller)

    controller._handle_controller_message(
        {"id": "power", "value": {"content": "Reboot", "value": "reboot"}}
    )
    controller._handle_controller_message(
        {"id": "power", "value": {"content": "Shutdown", "value": "shutdown"}}
    )
    assert calls == ["reboot", "shutdown"]


def test_shutdown_prefers_passwordless_sudo(monkeypatch):
    calls = []

    class Result:
        returncode = 0
        stderr = ""

    def fake_run(command, **kwargs):
        calls.append(command)
        return Result()

    monkeypatch.setattr("chandelier.controller.subprocess.run", fake_run)
    Controller._shutdown()
    assert calls == [["sudo", "-n", "poweroff"]]


def test_reboot_prefers_passwordless_sudo(monkeypatch):
    calls = []

    class Result:
        returncode = 0
        stderr = ""

    def fake_run(command, **kwargs):
        calls.append(command)
        return Result()

    monkeypatch.setattr(
        "chandelier.controller.subprocess.run", fake_run
    )
    Controller._reboot()
    assert calls == [["sudo", "-n", "reboot"]]


def test_reboot_falls_back_to_systemctl(monkeypatch):
    calls = []

    class Result:
        def __init__(self, returncode):
            self.returncode = returncode
            self.stderr = "denied"

    def fake_run(command, **kwargs):
        calls.append(command)
        return Result(1 if command[0] == "sudo" else 0)

    monkeypatch.setattr(
        "chandelier.controller.subprocess.run", fake_run
    )
    Controller._reboot()
    assert calls == [
        ["sudo", "-n", "reboot"],
        ["systemctl", "reboot"],
    ]
