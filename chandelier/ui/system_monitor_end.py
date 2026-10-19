"""System monitor screen.

Four pages (hardware, power, network, service) are rendered by
:class:`DashboardRenderer` in the active menu theme, so the monitor follows
the modern, classic and retro menu styles.  Reports are refreshed twice per
second and guarded so a missing sensor or command can never take the screen
down.
"""

from __future__ import annotations

import logging
import re
import threading
import time

import numpy as np

from chandelier.config import AppConfig
from chandelier.constants import EndId
from chandelier.i18n import tr
from chandelier.services.system_stats import SystemStats
from chandelier.ui.controlled_end import ControlledEnd
from chandelier.ui.dashboard_render import DashboardRenderer, DashboardRow
from chandelier.ui.text_render import get_renderer
from chandelier.ui.themes import get_active_theme

logger = logging.getLogger(__name__)

_PERCENT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")
_NUMBER_RE = re.compile(r"(-?\d+(?:\.\d+)?)")
_DBM_RE = re.compile(r"(-?\d+)\s*dBm")


class _PageCursor:
    """Minimal page state shared with the input handlers."""

    def __init__(self, total: int):
        self.total_pages = max(1, int(total))
        self._index = 0

    @property
    def current_page(self) -> int:
        return self._index

    def next_page(self) -> None:
        self._index = min(self._index + 1, self.total_pages - 1)

    def previous_page(self) -> None:
        self._index = max(self._index - 1, 0)

    def set_page(self, page: int) -> None:
        if not 0 <= page < self.total_pages:
            raise IndexError(
                f"page index out of range [0, {self.total_pages - 1}]"
            )
        self._index = page


class SystemMonitor(ControlledEnd):
    """Four-page dashboard rendered in the active menu theme.

    Reports come from :class:`SystemStats`; each page is rebuilt on a
    fixed cadence, and a failing probe becomes an "unavailable" row
    instead of raising into the frame loop.
    """

    PAGE_TITLES = ("HARDWARE", "POWER", "NETWORK", "SERVICE")
    REFRESH_SECONDS = 0.5

    def __init__(
        self,
        stats: SystemStats,
        config: AppConfig,
        _id: str = EndId.SYSTEM_MONITOR.value,
        preview_server=None,
        localizer=None,
        camera=None,
    ):
        """Store dependencies and initialise the exit-event handshake."""
        super().__init__(_id)
        self._stats = stats
        self._preview_server = preview_server
        self._localizer = localizer
        self._camera = camera
        self._width = config.screen.width
        self._height = config.screen.height
        self._renderer = get_renderer()
        self._exit_event = threading.Event()
        self._decorator = _PageCursor(len(self.PAGE_TITLES))
        # Track when the localizer batch generation last changed, so the
        # UI shows the age of the newest DOA result rather than the age of
        # the most recent poll.
        self._last_generation = -1
        self._last_generation_at = 0.0

    @property
    def _theme(self):
        # Follow the menu theme, including live theme changes.
        return get_active_theme()

    # ------------------------------------------------------------------
    # Report pages
    # ------------------------------------------------------------------
    def _hardware_report(self):
        """Return the CPU, memory, disk and thermal report."""
        return self._report(self._stats.hardware_report, "Hardware")

    def _power_report(self):
        """Return the battery, shunt and throttle report."""
        return self._report(self._stats.power_report, "Power")

    def _network_report(self):
        """Return the network report plus the preview-server status."""
        report = self._report(self._stats.network_report, "Network")
        if self._preview_server is not None:
            server = self._preview_server
            if server.enabled:
                value = f"{tr('ON')} :{server.port}"
                clients = server.client_count
                if clients:
                    value += f" {clients}cl"
            else:
                value = tr("OFF")
            report = {**report, "Web {}": value}
        return report

    @staticmethod
    def _report(report, label: str):
        """Call a stats probe, replacing failures with a placeholder row."""
        try:
            data = report()
        except Exception:
            logger.exception("%s report failed", label.lower())
            data = None
        if not data:
            return {f"{label} {{}}": "unavailable"}
        return data

    def _service_report(self) -> dict:
        """Compose the localizer, preview-server and camera health page."""
        report: dict[str, object] = {}

        localizer = self._localizer
        if localizer is None:
            report["Localizer {}"] = "N/A"
        else:
            process = getattr(localizer, "process", None)
            alive = False
            if process is not None:
                try:
                    alive = bool(process.is_alive())
                except Exception:
                    alive = False
            report["Localizer {}"] = "RUNNING" if alive else "STOPPED"

            latest = getattr(localizer, "latest", None)
            batch = None
            if latest is not None:
                try:
                    batch = latest.read()
                except Exception:
                    batch = None
            if batch is None:
                report["Sources {}"] = "--"
                report["Confidence {}"] = "--"
                report["Last DOA {}"] = "--"
            else:
                now = time.monotonic()
                # Only a new generation restarts the age counter; otherwise
                # polling every frame would always report ~0 s, hiding a
                # stalled localizer.
                if batch.generation != self._last_generation:
                    self._last_generation = batch.generation
                    self._last_generation_at = now
                report["Sources {}"] = len(batch.sources)
                report["Confidence {}"] = f"{batch.confidence:.2f}"
                age = now - self._last_generation_at
                report["Last DOA {}"] = f"{age:.1f}s"

        server = self._preview_server
        if server is None:
            report["Web {}"] = "N/A"
        else:
            report["Web {}"] = (
                f"{tr('ON')} :{server.port}"
                if server.enabled
                else tr("OFF")
            )
            report["Clients {}"] = server.client_count
            if server.enabled:
                ip = self._ip_address()
                report["URL {}"] = (
                    f"http://{ip}:{server.port}"
                    if ip
                    else f":{server.port}"
                )

        if self._camera is not None:
            report["Camera {}"] = (
                f"{self._camera.frame_per_second:.0f}fps"
            )
        return report

    def _ip_address(self) -> str | None:
        """Return the first IP row from the (cached) network report."""
        try:
            report = self._stats.network_report()
        except Exception:
            return None
        for key, value in report.items():
            if str(key).startswith("IP"):
                return str(value) if value else None
        return None

    def _current_report(self) -> dict:
        """Return the report for the selected page index."""
        page = self._decorator.current_page
        if page == 0:
            return self._hardware_report()
        if page == 1:
            return self._power_report()
        if page == 2:
            return self._network_report()
        return self._service_report()

    @staticmethod
    def _rows(report: dict) -> list[tuple[str, str, str]]:
        """Flatten report entries into ``(label, value, key)`` rows.

        A ``{}`` in the key is the menu convention for embedding the
        value; malformed formats degrade to ``label value`` so one bad
        probe cannot drop an entire page.
        """
        rows: list[tuple[str, str, str]] = []
        for key, value in report.items():
            if "{}" not in key:
                label = str(key).strip()
                rendered = str(value)
            else:
                label = key.split("{}", 1)[0].strip()
                try:
                    if isinstance(value, (tuple, list)):
                        rendered = key.format(*value)
                    else:
                        rendered = key.format(value)
                except (IndexError, KeyError, TypeError, ValueError):
                    rendered = f"{label} {value}"

            if rendered.startswith(label):
                value_text = rendered[len(label):].strip()
            else:
                value_text = rendered.strip()
            value_text = value_text.replace("None", "--") or "--"
            rows.append((label or key.strip(), value_text, key))
        return rows

    # ------------------------------------------------------------------
    # Value styling
    # ------------------------------------------------------------------
    @staticmethod
    def _percent(value_text: str) -> float | None:
        """Return the last percentage in a value string, if any."""
        matches = _PERCENT_RE.findall(value_text)
        return float(matches[-1]) if matches else None

    @staticmethod
    def _number(value_text: str) -> float | None:
        """Return the first number in a value string, if any."""
        match = _NUMBER_RE.search(value_text)
        return float(match.group(1)) if match else None

    def _value_color(self, key: str, value_text: str):
        """Pick a semantic colour for a row from its label and value.

        Thresholds follow the device's green/amber/red convention; they
        are duplicated here because reports carry preformatted strings,
        not structured readings.
        """
        theme = self._theme
        label = key.split("{}", 1)[0].strip().lower()

        if label in {"cpu", "mem", "disk", "swap"}:
            percent = self._percent(value_text)
            if percent is None:
                return theme.text
            if percent >= 85:
                return theme.danger
            if percent >= 60:
                return theme.warning
            return theme.success

        if label == "bat":
            # Battery is inverted: low values are the dangerous ones.
            percent = self._percent(value_text)
            if percent is None:
                return theme.text
            if percent < 15:
                return theme.danger
            if percent < 30:
                return theme.warning
            return theme.success

        if "temp" in label:
            temperature = self._number(value_text)
            if temperature is None:
                return theme.text
            if temperature >= 75:
                return theme.danger
            if temperature >= 60:
                return theme.warning
            return theme.success

        if label.startswith("internet"):
            lowered = value_text.lower()
            if lowered.startswith("connected"):
                return theme.success
            if lowered.startswith("checking"):
                return theme.warning
            return theme.danger

        if label.startswith("signal"):
            match = _DBM_RE.search(value_text)
            if match is None:
                return theme.text
            dbm = int(match.group(1))
            if dbm >= -60:
                return theme.success
            if dbm >= -75:
                return theme.warning
            return theme.danger

        if label.startswith("throttle"):
            lowered = value_text.lower()
            if lowered.startswith("ok"):
                return theme.warning if "!" in lowered else theme.success
            if lowered.startswith("n/a"):
                return theme.text_dim
            return theme.danger

        if label.startswith("localizer"):
            upper = value_text.upper()
            if upper.startswith("RUNNING"):
                return theme.success
            if upper.startswith("STOPPED"):
                return theme.danger
            return theme.text_dim

        if label.startswith("confidence"):
            confidence = self._number(value_text)
            if confidence is None:
                return theme.text
            if confidence >= 0.5:
                return theme.success
            if confidence >= 0.2:
                return theme.warning
            return theme.danger

        if label.startswith("last doa"):
            age = self._number(value_text)
            if age is None:
                return theme.text
            if age <= 5:
                return theme.success
            if age <= 30:
                return theme.warning
            return theme.danger

        if label.startswith("web") and self._preview_server is not None:
            return (
                theme.success
                if self._preview_server.enabled
                else theme.danger
            )

        return theme.text

    def _bar_percent(self, key: str, value_text: str) -> float | None:
        """Return a fill percentage for rows that represent usage."""
        label = key.split("{}", 1)[0].strip().lower()
        if label in {"cpu", "mem", "disk", "swap", "bat"}:
            return self._percent(value_text)
        return None

    # ------------------------------------------------------------------
    # Frame composition
    # ------------------------------------------------------------------
    def _compose_frame(self) -> np.ndarray:
        """Build one dashboard frame from the current report and theme."""
        page = self._decorator.current_page
        if page < len(self.PAGE_TITLES):
            page_title = self.PAGE_TITLES[page]
        else:
            page_title = "SYSTEM"
        rows = [
            DashboardRow(
                label=label,
                value=value_text,
                color=self._value_color(key, value_text),
                percent=self._bar_percent(key, value_text),
            )
            for label, value_text, key in self._rows(
                self._current_report()
            )
        ]
        # Rebuild per frame so a live theme change takes effect on the next
        # refresh; the renderer only stores geometry, so this is cheap.
        renderer = DashboardRenderer(
            self._theme,
            self._width,
            self._height,
            self._renderer,
        )
        return renderer.render(
            title=f"{tr('SYSTEM')} / {tr(page_title)}",
            page=page,
            pages=self._decorator.total_pages,
            rows=rows,
            hints=f"{tr('UP/DOWN PAGE')}   {tr('CROSS EXIT')}",
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def up_press_action(self) -> None:
        self._decorator.previous_page()

    def down_press_action(self) -> None:
        self._decorator.next_page()

    def cross_press_action(self) -> None:
        if self._irq is not None:
            self._irq(EndId.CAMERA.value)

    def rotary_encoder_clockwise(self) -> None:
        self._decorator.previous_page()

    def rotary_encoder_counter_clockwise(self) -> None:
        self._decorator.next_page()

    def rotary_encoder_select(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def current_page(self) -> int:
        return self._decorator.current_page

    @property
    def total_pages(self) -> int:
        return self._decorator.total_pages

    def on_enter(self, last_id: str | None) -> None:
        """Clear the exit handshake when the end becomes active."""
        self._exit_event.clear()

    def on_exit(self) -> None:
        """Signal main_loop to yield its final frame and return."""
        self._exit_event.set()

    def main_loop(self):
        """Yield a dashboard frame every ``REFRESH_SECONDS`` until exit.

        Report probes run on this thread; :class:`SystemStats` caches the
        slow network scan, so redrawing at 2 Hz stays cheap.
        """
        while True:
            if self._exit_event.is_set():
                # A switched-away end must yield one frame so the controller
                # can observe the pending switch and reset its flag.
                yield np.zeros(
                    (self._height, self._width, 3), dtype=np.uint8
                )
                return

            yield self._compose_frame()

            # wait() doubles as the refresh delay and wakes immediately on
            # exit, so a switch request never waits for the full period.
            if self._exit_event.wait(self.REFRESH_SECONDS):
                continue
