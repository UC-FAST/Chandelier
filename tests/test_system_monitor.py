"""Tests for the redesigned system monitor screen."""

import types

import numpy as np
import pytest

from chandelier.config import load_config
from chandelier.ui.dashboard_render import DashboardRenderer
from chandelier.ui.system_monitor_end import SystemMonitor
from chandelier.ui.text_render import get_renderer
from chandelier.ui.themes import THEMES, get_theme, set_active_theme
from tests.fakes import FakeStats


@pytest.fixture(autouse=True)
def _restore_default_theme():
    yield
    set_active_theme("dark_pro")


def _monitor(stats=None):
    return SystemMonitor(stats or FakeStats(), load_config())


def test_monitor_renders_and_navigates_pages():
    monitor = _monitor()
    first = monitor._compose_frame()
    assert first.shape == (240, 320, 3)
    assert first.any()

    monitor.down_press_action()
    second = monitor._compose_frame()
    assert monitor.current_page == 1
    assert not np.array_equal(first, second)

    monitor.down_press_action()
    monitor.down_press_action()
    assert monitor.current_page == 3
    assert monitor.total_pages == 4
    monitor.down_press_action()
    assert monitor.current_page == 3

    monitor.up_press_action()
    monitor.up_press_action()
    monitor.up_press_action()
    assert monitor.current_page == 0


def test_monitor_web_up_press_pages_back_from_last_page():
    from chandelier.events import Action, Phase

    monitor = _monitor()
    for _ in range(3):
        monitor.down_press_action()
    assert monitor.current_page == 3

    # The web UI sends only PRESS events; up must still page back.
    assert monitor.handle_action(Action.UP, Phase.PRESS)
    assert monitor.current_page == 2


def test_monitor_cross_returns_to_camera():
    monitor = _monitor()
    targets = []
    monitor.irq(targets.append)
    monitor.cross_press_action()
    assert targets == ["camera"]


def test_monitor_exit_wakes_render_loop():
    monitor = _monitor()
    generator = monitor.main_loop()
    next(generator)

    monitor.on_exit()
    sentinel = next(generator)
    assert not sentinel.any()
    with pytest.raises(StopIteration):
        next(generator)


def test_monitor_survives_failing_reports():
    class BrokenStats:
        def hardware_report(self):
            raise OSError("sensor failed")

        def power_report(self):
            raise OSError("sensor failed")

        def network_report(self):
            raise OSError("sensor failed")

    monitor = _monitor(BrokenStats())
    for _ in range(3):
        assert monitor._compose_frame().any()
        monitor.down_press_action()


def test_monitor_shows_preview_server_status():
    class FakePreviewServer:
        def __init__(self, enabled, clients=0):
            self.enabled = enabled
            self.port = 5989
            self.client_count = clients

    config = load_config()
    running = SystemMonitor(
        FakeStats(),
        config,
        preview_server=FakePreviewServer(True, clients=2),
    )
    assert running._network_report()["Web {}"] == "ON :5989 2cl"

    stopped = SystemMonitor(
        FakeStats(),
        config,
        preview_server=FakePreviewServer(False),
    )
    assert stopped._network_report()["Web {}"] == "OFF"


def test_monitor_service_page_reports_components():
    class Process:
        def is_alive(self):
            return True

    class Latest:
        def read(self):
            return types.SimpleNamespace(
                generation=1,
                confidence=0.75,
                sources=[1, 2],
            )

    class Localizer:
        process = Process()
        latest = Latest()

    class PreviewServer:
        enabled = True
        port = 5989
        client_count = 2

    class Camera:
        frame_per_second = 30.0

    monitor = SystemMonitor(
        FakeStats(),
        load_config(),
        preview_server=PreviewServer(),
        localizer=Localizer(),
        camera=Camera(),
    )
    report = monitor._service_report()

    assert report["Localizer {}"] == "RUNNING"
    assert report["Sources {}"] == 2
    assert report["Confidence {}"] == "0.75"
    assert report["Last DOA {}"].endswith("s")
    assert report["Web {}"] == "ON :5989"
    assert report["Clients {}"] == 2
    assert report["URL {}"] == "http://127.0.0.1:5989"
    assert report["Camera {}"] == "30fps"


def test_monitor_service_colours():
    monitor = _monitor()
    theme = monitor._theme
    assert monitor._value_color("Throttle {}", "OK") == theme.success
    assert monitor._value_color("Throttle {}", "UV") == theme.danger
    assert monitor._value_color("Localizer {}", "RUNNING") == theme.success
    assert monitor._value_color("Localizer {}", "STOPPED") == theme.danger
    assert monitor._value_color("Last DOA {}", "1.0s") == theme.success
    assert monitor._value_color("Last DOA {}", "60.0s") == theme.danger
    assert monitor._value_color("Confidence {}", "0.75") == theme.success
    assert monitor._value_color("Confidence {}", "0.05") == theme.danger
    assert monitor._value_color("Swap {}%", "90%") == theme.danger


def test_monitor_rows_parse_label_value_and_units():
    rows = SystemMonitor._rows(
        {
            "CPU {}%": 23.4,
            "BAT {}v {}%": (4.02, 74.5),
            "System Time {}": "2026-10-18 02:55:00",
            "Interface {}": None,
        }
    )
    assert rows[0][:2] == ("CPU", "23.4%")
    assert rows[1][:2] == ("BAT", "4.02v 74.5%")
    assert rows[2][:2] == ("System Time", "2026-10-18 02:55:00")
    assert rows[3][:2] == ("Interface", "--")


def test_monitor_follows_menu_theme():
    from chandelier.ui.themes import set_active_theme

    set_active_theme("ubuntu")
    monitor = _monitor()
    assert monitor._theme.id == "ubuntu"
    assert monitor._compose_frame().any()


def test_monitor_renders_translated_labels():
    from chandelier.i18n import translator

    translator.set_language("ru")
    frame = _monitor()._compose_frame()
    assert frame.any()


def test_monitor_value_styling_uses_theme_colors():
    monitor = _monitor()
    theme = monitor._theme
    assert monitor._value_color("CPU {}%", "95%") == theme.danger
    assert monitor._value_color("CPU {}%", "70%") == theme.warning
    assert monitor._value_color("CPU {}%", "20%") == theme.success
    assert monitor._value_color("BAT {}v {}%", "4.0v 10%") == theme.danger
    assert monitor._value_color("Internet {}", "connected") == theme.success
    assert monitor._bar_percent("BAT {}v {}%", "4.0v 42%") == 42.0
    assert monitor._bar_percent("System Time {}", "12:00") is None


@pytest.mark.parametrize("theme_id", tuple(THEMES))
def test_monitor_renders_in_every_menu_theme(theme_id):
    set_active_theme(theme_id)
    monitor = _monitor()
    frame = monitor._compose_frame()
    assert frame.shape == (240, 320, 3)
    assert frame.any()


def test_monitor_renders_dense_page():
    class DenseStats:
        def hardware_report(self):
            return {
                f"Metric {index} {{}}": f"{index * 8}%"
                for index in range(12)
            }

        def power_report(self):
            return {}

        def network_report(self):
            return {}

    monitor = SystemMonitor(DenseStats(), load_config())
    frame = monitor._compose_frame()
    assert frame.shape == (240, 320, 3)
    assert frame.any()


def test_dashboard_renderer_handles_empty_rows():
    set_active_theme("dark_pro")
    renderer = DashboardRenderer(
        get_theme("dark_pro"), 320, 240, get_renderer()
    )
    frame = renderer.render(
        title="SYSTEM",
        page=0,
        pages=1,
        rows=[],
        hints="UP/DOWN PAGE",
    )
    assert frame.shape == (240, 320, 3)
    assert frame.any()
