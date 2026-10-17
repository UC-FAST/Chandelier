import pytest
import numpy as np

from chandelier.config import load_config, load_menu_config
from chandelier.ui.camera_end import CameraControlledEnd
from tests.fakes import FakeCamera, FakeLeds


def _make_end(localizer=None):
    config = load_config()
    tree = load_menu_config()["camera"]
    camera = FakeCamera(config.screen.width, config.screen.height)
    end = CameraControlledEnd(
        camera, FakeLeds(), config, localizer=localizer
    )
    end._msg_sender = lambda *args: None
    end.msg_receiver("menu", ("0", tree))
    return end, camera, tree


def test_menu_settings_are_applied():
    end, camera, tree = _make_end()
    names = [name for name, _ in camera.calls]
    assert "set_ae_enable" in names
    assert "set_awb_enable" in names

    values = {}
    for section in tree.values():
        for option in section.get("options", []):
            values[option.get("id")] = option.get("value")
    assert end._state.mf_assist is bool(values["mf_assist"])
    assert end._state.show_hist is bool(values["show_hist"])


def test_localizer_menu_settings_are_forwarded():
    class FakeLocalizer:
        def __init__(self):
            self.updates = []

        def update_settings(self, **values):
            self.updates.append(values)
            return True

    localizer = FakeLocalizer()
    end, _, tree = _make_end(localizer=localizer)

    initial_sources = next(
        option["value"]
        for section in tree.values()
        for option in section.get("options", [])
        if option.get("id") == "loc_n_sources"
    )
    assert localizer.updates
    assert localizer.updates[-1]["n_sources"] == initial_sources

    for section in tree.values():
        for option in section.get("options", []):
            if option.get("id") == "loc_n_sources":
                option["value"] = 6
            elif option.get("id") == "loc_mic_rotation":
                option["value"] = 90
            elif option.get("id") == "loc_confidence":
                option["value"] = 0.2

    end.load_settings()
    update = localizer.updates[-1]
    assert update["n_sources"] == 6
    assert update["mic_rotation"] == 90.0
    assert update["confidence_threshold"] == 0.2


def test_show_hist_state_is_read_from_menu():
    end, _, tree = _make_end()
    for section in tree.values():
        for option in section.get("options", []):
            if option.get("id") == "show_hist":
                option["value"] = True
    end.load_settings()
    assert end._state.show_hist is True


def test_shutter_saves_with_menu_values():
    end, camera, _ = _make_end()
    end.shutter_press_action()
    assert end._capture_session is not None
    next(end.main_loop())

    assert len(camera.saved) == 1
    saved = camera.saved[0]
    assert saved["extension"] == "jpeg"
    assert saved["width"] == 4608
    assert saved["height"] == 2592
    assert saved["save_metadata"] is False
    assert saved["save_raw"] is False
    assert end._state.is_busy is False


def test_brightness_and_zoom_actions():
    end, camera, _ = _make_end()
    end.up_press_action()
    assert end._state.bright_hold is True
    assert any(name == "brightness" for name, _ in camera.calls)
    end.up_release_action()
    assert end._state.bright_hold is False

    end.right_press_action()
    assert end._state.zoom_hold is True
    assert any(name == "zoom" for name, _ in camera.calls)
    end.right_release_action()
    assert end._state.zoom_hold is False


def test_preview_loop_yields_bgr_frame():
    end, camera, _ = _make_end()
    frame = next(end.main_loop())
    assert frame.shape == (240, 320, 3)
    assert frame.dtype == np.uint8
    camera.stop()


def test_metadata_page():
    end, _, _ = _make_end()
    page = end._metadata_page()
    assert page["EPTime {}"] == 1000
    assert page["FPS {}"] == 30.0


def _set_option(tree, option_id, value):
    for section in tree.values():
        for option in section.get("options", []):
            if option.get("id") == option_id:
                option["value"] = value


def test_pro_controls_are_applied():
    end, camera, tree = _make_end()
    _set_option(tree, "hdr", 2)
    _set_option(tree, "exp_compensation", 1.5)
    _set_option(tree, "ae_lock", True)
    _set_option(tree, "awb_lock", True)
    _set_option(tree, "manual_focus", True)
    _set_option(tree, "focus_position", 300)
    end.load_settings()

    calls = dict(camera.calls)
    assert calls["set_hdr"]["mode"] == 2
    assert calls["set_exposure_value"]["value"] == 1.5
    assert calls["set_ae_lock"]["enable"] is True
    assert calls["set_awb_lock"]["enable"] is True
    assert calls["set_focus"]["manual"] is True
    assert calls["set_focus"]["position"] == 300
    assert "set_image_controls" in calls


def test_display_and_capture_settings_are_read():
    end, _, tree = _make_end()
    _set_option(tree, "show_grid", True)
    _set_option(tree, "show_center", True)
    _set_option(tree, "show_zebra", True)
    _set_option(tree, "zebra_threshold", 200)
    _set_option(tree, "show_status_bar", True)
    _set_option(tree, "capture_mode", {"content": "Burst", "value": 1})
    _set_option(tree, "burst_count", 4)
    _set_option(tree, "record_mode", True)
    _set_option(tree, "record_resolution", {"content": "640x480", "value": [640, 480]})
    end.load_settings()

    assert end._state.show_grid is True
    assert end._state.show_center is True
    assert end._state.show_zebra is True
    assert end._state.zebra_threshold == 200
    assert end._state.show_status_bar is True
    assert end._state.capture_mode == "burst"
    assert end._state.burst_count == 4
    assert end._state.record_mode is True
    assert end._state.record_resolution == (640, 480)


def test_burst_capture_driven_by_preview_loop(monkeypatch):
    end, camera, tree = _make_end()
    _set_option(tree, "capture_mode", {"content": "Burst", "value": 1})
    _set_option(tree, "burst_count", 2)
    _set_option(tree, "burst_interval", 0.1)
    end.load_settings()

    clock = [0.0]
    monkeypatch.setattr(
        "chandelier.ui.camera_end.time.monotonic", lambda: clock[0]
    )
    end.shutter_press_action()
    next(end.main_loop())
    assert len(camera.saved) == 1

    clock[0] = 0.2
    next(end.main_loop())
    assert len(camera.saved) == 2
    assert end._capture_session is None


def test_self_timer_countdown_and_capture(monkeypatch):
    end, camera, tree = _make_end()
    _set_option(tree, "delay", 2)
    end.load_settings()

    clock = [0.0]
    monkeypatch.setattr(
        "chandelier.ui.camera_end.time.monotonic", lambda: clock[0]
    )
    end.shutter_press_action()
    next(end.main_loop())
    assert camera.saved == []
    assert end._state.countdown == pytest.approx(2.0)

    clock[0] = 2.0
    next(end.main_loop())
    assert len(camera.saved) == 1


def test_record_mode_starts_and_stops_recording():
    end, camera, tree = _make_end()
    _set_option(tree, "record_mode", True)
    end.load_settings()

    end.shutter_press_action()
    names = [name for name, _ in camera.calls]
    assert "start_recording" in names
    assert end._state.record_timestamp is not None
    assert end._leds.blue.is_on is True

    end.shutter_press_action()
    names = [name for name, _ in camera.calls]
    assert "stop_recording" in names
    assert end._state.record_timestamp is None
    assert end._leds.blue.is_on is False


def test_all_overlays_render_without_errors():
    end, _, tree = _make_end()
    for option_id in (
        "show_grid",
        "show_center",
        "show_zebra",
        "show_hist",
        "show_status_bar",
    ):
        _set_option(tree, option_id, True)
    _set_option(tree, "record_mode", True)
    end.load_settings()
    end.shutter_press_action()

    frame = next(end.main_loop())
    assert frame.shape == (240, 320, 3)
    end.on_exit()
