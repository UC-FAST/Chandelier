import pytest
import numpy as np

from chandelier.config import load_config, load_menu_config
from chandelier.ui.camera_end import CameraControlledEnd
from chandelier.ui.decorators import StatusItem
from tests.fakes import FakeCamera, FakeLeds


def _make_end(localizer=None, preview_server=None, stats=None):
    config = load_config()
    tree = load_menu_config()["camera"]
    camera = FakeCamera(config.screen.width, config.screen.height)
    end = CameraControlledEnd(
        camera,
        FakeLeds(),
        config,
        localizer=localizer,
        preview_server=preview_server,
        stats=stats,
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


def test_histogram_box_avoids_the_status_bar():
    end, _, _ = _make_end()

    end._state.status_position = "off"
    assert end._histogram_padding() == (214, 170, 6, 6)

    end._state.status_position = "top_bottom"
    band = end._status.bar_height
    assert band == 22
    assert end._histogram_padding() == (214, 170 - band, 6, 6 + band)

    end._state.status_position = "bottom"
    assert end._histogram_padding() == (214, 170 - band, 6, 6 + band)

    end._state.status_position = "sides"
    assert end._histogram_padding() == (110, 170, 110, 6)


def test_localizer_menu_settings_are_forwarded():
    class FakeLocalizer:
        def __init__(self):
            self.updates = []

        def update_settings(self, **values):
            self.updates.append(values)
            return True

    localizer = FakeLocalizer()
    end, _, tree = _make_end(localizer=localizer)

    def initial_value(option_id):
        return next(
            option["value"]
            for section in tree.values()
            for option in section.get("options", [])
            if option.get("id") == option_id
        )

    assert localizer.updates
    assert localizer.updates[-1]["n_sources"] == initial_value(
        "loc_n_sources"
    )
    assert localizer.updates[-1]["marker_fov"] == pytest.approx(
        initial_value("loc_marker_fov")
    )
    assert localizer.updates[-1]["marker_azimuth_offset"] == pytest.approx(
        initial_value("loc_marker_azimuth")
    )
    assert localizer.updates[-1]["marker_elevation_offset"] == pytest.approx(
        initial_value("loc_marker_elevation")
    )
    assert localizer.updates[-1]["marker_x_gain"] == initial_value(
        "loc_marker_x_gain"
    )
    assert localizer.updates[-1]["marker_y_gain"] == initial_value(
        "loc_marker_y_gain"
    )
    assert localizer.updates[-1]["marker_mic_offset_x"] == pytest.approx(
        initial_value("loc_marker_mic_x") * 0.01
    )
    assert localizer.updates[-1]["marker_source_distance"] == pytest.approx(
        initial_value("loc_talk_distance") * 0.01
    )

    for section in tree.values():
        for option in section.get("options", []):
            if option.get("id") == "loc_n_sources":
                option["value"] = 6
            elif option.get("id") == "loc_mic_rotation":
                option["value"] = 90
            elif option.get("id") == "loc_confidence":
                option["value"] = 0.2
            elif option.get("id") == "loc_marker_fov":
                option["value"] = 70
            elif option.get("id") == "loc_marker_azimuth":
                option["value"] = 15
            elif option.get("id") == "loc_marker_elevation":
                option["value"] = -18.0
            elif option.get("id") == "loc_marker_x_gain":
                option["value"] = 0.5
            elif option.get("id") == "loc_marker_y_gain":
                option["value"] = 2.5
            elif option.get("id") == "loc_marker_mic_x":
                option["value"] = 3.0
            elif option.get("id") == "loc_marker_mic_y":
                option["value"] = -5.0
            elif option.get("id") == "loc_marker_mic_z":
                option["value"] = 1.0
            elif option.get("id") == "loc_talk_distance":
                option["value"] = 40.0

    end.load_settings()
    update = localizer.updates[-1]
    assert update["n_sources"] == 6
    assert update["mic_rotation"] == 90.0
    assert update["confidence_threshold"] == 0.2
    assert update["marker_fov"] == pytest.approx(70.0)
    assert update["marker_azimuth_offset"] == pytest.approx(15.0)
    assert update["marker_elevation_offset"] == pytest.approx(-18.0)
    assert update["marker_x_gain"] == pytest.approx(0.5)
    assert update["marker_y_gain"] == pytest.approx(2.5)
    assert update["marker_mic_offset_x"] == pytest.approx(0.03)
    assert update["marker_mic_offset_y"] == pytest.approx(-0.05)
    assert update["marker_mic_offset_z"] == pytest.approx(0.01)
    assert update["marker_source_distance"] == pytest.approx(0.4)


def test_focus_distance_is_forwarded_to_the_localizer():
    class FakeLocalizer:
        def __init__(self):
            self.updates = []

        def update_settings(self, **values):
            self.updates.append(values)
            return True

    localizer = FakeLocalizer()
    end, camera, tree = _make_end(localizer=localizer)
    for section in tree.values():
        for option in section.get("options", []):
            if option.get("id") == "loc_talk_distance":
                option["value"] = 0.0
    camera.metadata["LensPosition"] = 2.5  # 1 / 2.5 dpt = 0.4 m
    end._focus_distance = None
    end._sync_source_distance()
    assert localizer.updates[-1]["marker_source_distance"] == pytest.approx(
        0.4
    )


def test_manual_talk_distance_disables_focus_estimate():
    class FakeLocalizer:
        def __init__(self):
            self.updates = []

        def update_settings(self, **values):
            self.updates.append(values)
            return True

    localizer = FakeLocalizer()
    end, camera, tree = _make_end(localizer=localizer)
    for section in tree.values():
        for option in section.get("options", []):
            if option.get("id") == "loc_talk_distance":
                option["value"] = 40.0
    camera.metadata["LensPosition"] = 2.5
    end._focus_distance = None
    count = len(localizer.updates)
    end._sync_source_distance()
    assert len(localizer.updates) == count


def test_web_server_menu_setting_is_forwarded():
    class FakePreviewServer:
        def __init__(self):
            self.states = []

        def set_enabled(self, enabled):
            self.states.append(enabled)
            return True

    preview = FakePreviewServer()
    end, _, tree = _make_end(preview_server=preview)

    expected = True
    for section in tree.values():
        for option in section.get("options", []):
            if option.get("id") == "web_server":
                expected = bool(option["value"])
    assert preview.states
    assert preview.states[-1] is expected

    for section in tree.values():
        for option in section.get("options", []):
            if option.get("id") == "web_server":
                option["value"] = False
    end.load_settings()
    assert preview.states[-1] is False


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
    assert saved["exposure_time"] is None
    assert saved["analogue_gain"] is None
    assert end._state.is_busy is False


def test_shutter_passes_manual_exposure_to_still_capture():
    end, camera, tree = _make_end()
    _set_option(tree, "auto_expose", False)
    _set_option(
        tree,
        "exposure_time",
        {"content": "4s (Still)", "value": 4_000_000},
    )
    _set_option(tree, "analogue_gain", 2.0)
    end.load_settings()

    end.shutter_press_action()
    next(end.main_loop())

    saved = camera.saved[-1]
    assert saved["exposure_time"] == 4_000_000
    assert saved["analogue_gain"] == 2.0


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
    assert page["ISO {}"] == 150


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
    _set_option(tree, "af_mode", {"content": "Manual", "value": 0})
    _set_option(tree, "focus_position", 300)
    _set_option(tree, "af_speed", {"content": "Fast", "value": 1})
    _set_option(tree, "af_range", {"content": "Macro", "value": 1})
    end.load_settings()

    calls = dict(camera.calls)
    assert calls["set_hdr"]["mode"] == 2
    assert calls["set_exposure_value"]["value"] == 1.5
    assert calls["set_ae_lock"]["enable"] is True
    assert calls["set_awb_lock"]["enable"] is True
    assert calls["set_focus_mode"]["mode"] == 0
    assert calls["set_focus_mode"]["position"] == 300
    assert calls["set_af_speed"]["speed"] == 1
    assert calls["set_af_range"]["af_range"] == 1
    assert "set_image_controls" in calls


def test_focus_message_and_center_trigger_autofocus():
    end, camera, tree = _make_end()
    # A normal tree update from the Focus section must not trigger AF.
    end.msg_receiver("menu", ("focus", tree))
    assert not any(name == "trigger_autofocus" for name, _ in camera.calls)

    end.msg_receiver("menu", ("autofocus", tree))
    assert any(name == "trigger_autofocus" for name, _ in camera.calls)

    before = sum(name == "trigger_autofocus" for name, _ in camera.calls)
    end.center_press_action()
    after = sum(name == "trigger_autofocus" for name, _ in camera.calls)
    assert after == before + 1


def test_display_and_capture_settings_are_read():
    end, _, tree = _make_end()
    _set_option(tree, "show_grid", True)
    _set_option(tree, "show_center", True)
    _set_option(tree, "show_zebra", True)
    _set_option(tree, "zebra_threshold", 200)
    _set_option(
        tree, "status_bar_position", {"content": "Sides", "value": "sides"}
    )
    _set_option(tree, "capture_mode", {"content": "Burst", "value": 1})
    _set_option(tree, "burst_count", 4)
    _set_option(tree, "record_mode", True)
    _set_option(tree, "record_resolution", {"content": "640x480", "value": [640, 480]})
    end.load_settings()

    assert end._state.show_grid is True
    assert end._state.show_center is True
    assert end._state.show_zebra is True
    assert end._state.zebra_threshold == 200
    assert end._state.status_position == "sides"
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


def test_status_bar_off_is_read_from_menu():
    end, _, tree = _make_end()
    _set_option(
        tree, "status_bar_position", {"content": "Off", "value": "off"}
    )
    end.load_settings()
    assert end._state.status_position == "off"


def test_all_overlays_render_without_errors():
    end, _, tree = _make_end()
    for option_id in (
        "show_grid",
        "show_center",
        "show_zebra",
        "show_hist",
    ):
        _set_option(tree, option_id, True)
    _set_option(
        tree, "status_bar_position", {"content": "Sides", "value": "sides"}
    )
    _set_option(tree, "record_mode", True)
    end.load_settings()
    end.shutter_press_action()

    frame = next(end.main_loop())
    assert frame.shape == (240, 320, 3)
    end.on_exit()


def test_web_server_bool_message_is_applied():
    class FakePreviewServer:
        def __init__(self):
            self.states = []

        def set_enabled(self, enabled):
            self.states.append(enabled)
            return True

    preview = FakePreviewServer()
    end, _, _ = _make_end(preview_server=preview)

    end.msg_receiver("camera", False)
    assert preview.states[-1] is False
    end.msg_receiver("camera", True)
    assert preview.states[-1] is True


def test_preview_server_errors_do_not_break_settings():
    class BrokenPreviewServer:
        def set_enabled(self, enabled):
            raise RuntimeError("boom")

    end, _, _ = _make_end(preview_server=BrokenPreviewServer())
    end.load_settings()


def test_locks_are_skipped_in_manual_modes():
    end, camera, tree = _make_end()
    _set_option(tree, "auto_expose", False)
    _set_option(tree, "ae_lock", True)
    _set_option(tree, "awb", False)
    _set_option(tree, "awb_lock", True)

    camera.calls.clear()
    end.load_settings()

    names = [name for name, _ in camera.calls]
    assert "set_manual_exposure" in names
    assert "set_colour_gains" in names
    assert "set_ae_lock" not in names
    assert "set_awb_lock" not in names


def test_locks_apply_in_auto_modes():
    end, camera, tree = _make_end()
    _set_option(tree, "auto_expose", True)
    _set_option(tree, "ae_lock", True)
    _set_option(tree, "awb", True)
    _set_option(tree, "awb_lock", True)

    camera.calls.clear()
    end.load_settings()

    names = [name for name, _ in camera.calls]
    assert "set_ae_lock" in names
    assert "set_awb_lock" in names


def test_sides_status_items_use_two_line_labels():
    end, camera, _ = _make_end()
    end._state.status_position = "sides"

    items = end._status_items()
    labels = {
        item.label: item.value
        for item in items
        if isinstance(item, StatusItem)
    }

    assert labels["DRV"] == "SGL"
    assert labels["AF"] in {"MF", "S", "C"}
    assert labels["SH"] == end._format_shutter(
        camera.metadata["ExposureTime"]
    )
    assert labels["WB"] == "5000"
    assert labels["ISO"] == "150"


def test_status_bar_fps_is_a_label_value_item():
    end, _, _ = _make_end()
    items = end._status_items()
    assert any(
        isinstance(item, StatusItem)
        and item.label == "FPS"
        and item.value == f"{end._camera.frame_per_second:.0f}"
        and item.value_role == "text"
        for item in items
    )


def test_status_bar_shows_iso():
    end, camera, _ = _make_end()
    camera.metadata["AnalogueGain"] = 8.0
    camera.metadata["DigitalGain"] = 1.25

    items = end._status_items()

    assert any(
        isinstance(item, StatusItem)
        and item.label == "ISO"
        and item.value == "1000"
        for item in items
    )


def test_iso_is_hidden_when_gain_is_unavailable():
    end, camera, _ = _make_end()
    camera.metadata["AnalogueGain"] = None

    items = end._status_items()

    assert not any(
        isinstance(item, StatusItem) and item.label == "ISO"
        for item in items
    )


def test_status_bar_shows_cpu_and_temperature():
    class FakeStats:
        def cpu_snapshot(self):
            return (85.0, 65.2)

    end, _, _ = _make_end(stats=FakeStats())
    items = end._status_items()
    assert any(
        isinstance(item, StatusItem)
        and item.label == "CPU"
        and item.value == "85%"
        and item.value_role == "danger"
        for item in items
    )
    assert any(
        isinstance(item, StatusItem)
        and item.label == "T"
        and item.value == "65C"
        and item.value_role == "warning"
        for item in items
    )
