"""System monitor screen.

Three pages (hardware, power, network) are drawn as a dark dashboard with a
header, alternating rows, right-aligned values, percentage bars and a page
indicator.  Reports are refreshed twice per second and guarded so a missing
sensor or command can never take the screen down.
"""

from __future__ import annotations

import logging
import re
import threading

import cv2
import numpy as np

from chandelier.config import AppConfig
from chandelier.constants import EndId
from chandelier.i18n import tr
from chandelier.services.system_stats import SystemStats
from chandelier.ui.controlled_end import ControlledEnd
from chandelier.ui.text_render import get_renderer
from chandelier.ui.themes import get_theme

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
    PAGE_TITLES = ("HARDWARE", "POWER", "NETWORK")
    HEADER_HEIGHT = 28
    FOOTER_HEIGHT = 16
    REFRESH_SECONDS = 0.5

    def __init__(
        self,
        stats: SystemStats,
        config: AppConfig,
        _id: str = EndId.SYSTEM_MONITOR.value,
    ):
        super().__init__(_id)
        self._stats = stats
        self._width = config.screen.width
        self._height = config.screen.height
        self._theme = get_theme("dark_pro")
        self._renderer = get_renderer()
        self._exit_event = threading.Event()
        self._decorator = _PageCursor(len(self.PAGE_TITLES))
        self._header_height_px = 11
        self._label_height = 9
        self._value_height = 10
        self._footer_height_px = 8

    # ------------------------------------------------------------------
    # Report pages
    # ------------------------------------------------------------------
    def _hardware_report(self):
        return self._report(self._stats.hardware_report, "Hardware")

    def _power_report(self):
        return self._report(self._stats.power_report, "Power")

    def _network_report(self):
        return self._report(self._stats.network_report, "Network")

    @staticmethod
    def _report(report, label: str):
        try:
            data = report()
        except Exception:
            logger.exception("%s report failed", label.lower())
            data = None
        if not data:
            return {f"{label} {{}}": "unavailable"}
        return data

    def _current_report(self) -> dict:
        page = self._decorator.current_page
        if page == 0:
            return self._hardware_report()
        if page == 1:
            return self._power_report()
        return self._network_report()

    @staticmethod
    def _rows(report: dict) -> list[tuple[str, str, str]]:
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
        matches = _PERCENT_RE.findall(value_text)
        return float(matches[-1]) if matches else None

    @staticmethod
    def _number(value_text: str) -> float | None:
        match = _NUMBER_RE.search(value_text)
        return float(match.group(1)) if match else None

    def _value_color(self, key: str, value_text: str):
        theme = self._theme
        label = key.split("{}", 1)[0].strip().lower()

        if label in {"cpu", "mem", "disk"}:
            percent = self._percent(value_text)
            if percent is None:
                return theme.text
            if percent >= 85:
                return theme.danger
            if percent >= 60:
                return theme.warning
            return theme.success

        if label == "bat":
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

        return theme.text

    def _bar_percent(self, key: str, value_text: str) -> float | None:
        label = key.split("{}", 1)[0].strip().lower()
        if label in {"cpu", "mem", "disk", "bat"}:
            return self._percent(value_text)
        return None

    # ------------------------------------------------------------------
    # Frame composition
    # ------------------------------------------------------------------
    def _measure_width(self, text: str, height: int) -> int:
        return self._renderer.measure(tr(text), height)[0]

    def _put_text(
        self,
        frame: np.ndarray,
        text: str,
        origin: tuple[int, int],
        height: int,
        color,
        thickness: int = 1,
    ) -> None:
        self._renderer.draw(
            frame,
            tr(text),
            (int(origin[0]), int(origin[1])),
            height,
            tuple(color),
            thickness,
        )

    def _fit_text(
        self, text: str, max_width: int, height: int
    ) -> str:
        text = tr(text)
        if not text or max_width <= 0:
            return ""
        if self._measure_width(text, height) <= max_width:
            return text
        suffix = ".."
        while text:
            text = text[:-1]
            if self._measure_width(text + suffix, height) <= max_width:
                return text + suffix
        return ""

    def _draw_header(self, frame: np.ndarray) -> None:
        theme = self._theme
        page = self._decorator.current_page
        cv2.rectangle(
            frame,
            (0, 0),
            (self._width, self.HEADER_HEIGHT),
            theme.header_bg,
            -1,
        )
        cv2.rectangle(frame, (0, 0), (3, self.HEADER_HEIGHT), theme.accent, -1)
        self._put_text(
            frame,
            "SYSTEM",
            (10, 19),
            self._header_height_px,
            theme.accent,
        )
        if page < len(self.PAGE_TITLES):
            title = self.PAGE_TITLES[page]
        else:
            title = "SYSTEM"
        self._put_text(
            frame,
            title,
            (68, 19),
            self._header_height_px,
            theme.text,
        )

        page_text = f"{page + 1}/{self._decorator.total_pages}"
        page_width = self._measure_width(page_text, self._label_height)
        self._put_text(
            frame,
            page_text,
            (self._width - 10 - page_width, 18),
            self._label_height,
            theme.text_dim,
        )
        cv2.line(
            frame,
            (0, self.HEADER_HEIGHT - 1),
            (self._width, self.HEADER_HEIGHT - 1),
            theme.accent,
            1,
            cv2.LINE_AA,
        )

    def _draw_row(
        self,
        frame: np.ndarray,
        y: int,
        row_height: int,
        index: int,
        label: str,
        value_text: str,
        key: str,
    ) -> None:
        theme = self._theme
        left, right = 10, self._width - 10
        row_bg = theme.surface if index % 2 == 0 else theme.background
        cv2.rectangle(
            frame, (0, y), (self._width, y + row_height), row_bg, -1
        )

        baseline = y + row_height // 2 + 4
        label = self._fit_text(
            label, int((right - left) * 0.45), self._label_height
        )
        label_width = self._measure_width(label, self._label_height)
        value = self._fit_text(
            value_text,
            max(1, right - left - label_width - 18),
            self._value_height,
        )
        value_width = self._measure_width(value, self._value_height)
        value_x = right - value_width
        color = self._value_color(key, value_text)

        percent = self._bar_percent(key, value_text)
        if percent is not None:
            bar_left = left + label_width + 10
            bar_right = value_x - 10
            if bar_right - bar_left >= 18:
                bar_y = y + row_height // 2 - 2
                cv2.rectangle(
                    frame,
                    (bar_left, bar_y),
                    (bar_right, bar_y + 5),
                    (48, 48, 48),
                    -1,
                )
                fill = int(
                    (bar_right - bar_left)
                    * max(0.0, min(100.0, percent))
                    / 100.0
                )
                if fill > 0:
                    cv2.rectangle(
                        frame,
                        (bar_left, bar_y),
                        (bar_left + fill, bar_y + 5),
                        color,
                        -1,
                    )

        self._put_text(
            frame,
            label,
            (left, baseline),
            self._label_height,
            theme.text_dim,
        )
        self._put_text(
            frame,
            value,
            (value_x, baseline),
            self._value_height,
            color,
        )
        cv2.line(
            frame,
            (left, y + row_height - 1),
            (right, y + row_height - 1),
            theme.border,
            1,
            cv2.LINE_AA,
        )

    def _draw_rows(self, frame: np.ndarray, rows: list[tuple]) -> None:
        content_top = self.HEADER_HEIGHT + 4
        content_bottom = self._height - self.FOOTER_HEIGHT - 4
        available = max(1, content_bottom - content_top)
        if not rows:
            text = "NO DATA"
            text_width = self._measure_width(text, self._label_height)
            self._put_text(
                frame,
                text,
                ((self._width - text_width) // 2, content_top + available // 2),
                self._label_height,
                self._theme.text_dim,
            )
            return

        row_height = min(28, max(15, available // len(rows)))
        block_height = row_height * len(rows)
        y = content_top + max(0, (available - block_height) // 2)
        for index, (label, value_text, key) in enumerate(rows):
            self._draw_row(
                frame, y, row_height, index, label, value_text, key
            )
            y += row_height

    def _draw_footer(self, frame: np.ndarray) -> None:
        theme = self._theme
        y0 = self._height - self.FOOTER_HEIGHT
        cv2.rectangle(
            frame, (0, y0), (self._width, self._height), theme.header_bg, -1
        )
        cv2.line(
            frame,
            (0, y0),
            (self._width, y0),
            theme.border,
            1,
            cv2.LINE_AA,
        )
        self._put_text(
            frame,
            "UP/DOWN PAGE",
            (8, self._height - 5),
            self._footer_height_px,
            theme.text_dim,
        )
        exit_text = "CROSS EXIT"
        exit_width = self._measure_width(
            exit_text, self._footer_height_px
        )
        self._put_text(
            frame,
            exit_text,
            (self._width - 8 - exit_width, self._height - 5),
            self._footer_height_px,
            theme.text_dim,
        )

        page = self._decorator.current_page
        total = self._decorator.total_pages
        spacing = 14
        center_x = self._width // 2
        center_y = y0 + self.FOOTER_HEIGHT // 2
        start_x = center_x - (total - 1) * spacing // 2
        for index in range(total):
            x = start_x + index * spacing
            if index == page:
                cv2.circle(
                    frame, (x, center_y), 3, theme.accent, -1, cv2.LINE_AA
                )
            else:
                cv2.circle(
                    frame,
                    (x, center_y),
                    2,
                    theme.text_dim,
                    1,
                    cv2.LINE_AA,
                )

    def _compose_frame(self) -> np.ndarray:
        frame = np.empty((self._height, self._width, 3), dtype=np.uint8)
        frame[:] = self._theme.background
        self._draw_header(frame)
        self._draw_rows(frame, self._rows(self._current_report()))
        self._draw_footer(frame)
        return frame

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def up_release_action(self) -> None:
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
        self._exit_event.clear()

    def on_exit(self) -> None:
        self._exit_event.set()

    def main_loop(self):
        while True:
            if self._exit_event.is_set():
                # A switched-away end must yield one frame so the controller
                # can observe the pending switch and reset its flag.
                yield np.zeros(
                    (self._height, self._width, 3), dtype=np.uint8
                )
                return

            yield self._compose_frame()

            if self._exit_event.wait(self.REFRESH_SECONDS):
                continue
