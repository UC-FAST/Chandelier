"""Theme-aware rendering for the system-monitor dashboard.

The menu owns the visual language of the device: a modern card list for the
``dark_pro`` family, the original packed list for ``classic`` and the
Windows/Aero chrome styles for the retro themes.  The system monitor used to
paint its own generic dashboard, so its colours followed the theme but its
layout did not.  :class:`DashboardRenderer` draws the same header, rows,
value badges and footer that the menu uses, while keeping the monitor's data
(percentage bars, status colours and icons) intact.

All colours are OpenCV BGR triples, exactly like :mod:`themes`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from chandelier.i18n import tr
from chandelier.ui.decorators.icons import draw_icon
from chandelier.ui.text_render import TextRenderer
from chandelier.ui.themes import MenuTheme


@dataclass(frozen=True)
class DashboardRow:
    """One label/value line on the monitor.

    ``percent`` (0-100) adds a usage bar, and ``icon`` overrides the
    icon inferred from the label.
    """

    label: str
    value: str
    color: tuple[int, int, int]
    percent: float | None = None
    icon: str | None = None


def icon_for_label(label: str) -> str:
    """Pick a menu-style icon for a monitor row label."""

    # First match wins, so specific keywords precede generic ones: a
    # "CPU_FREQ" row should get the speed icon, not the CPU icon.
    text = label.strip().lower()
    if "freq" in text:
        return "speed"
    if "cpu" in text or "core" in text:
        return "cpu"
    if "mem" in text or "swap" in text:
        return "memory"
    if "disk" in text or "storage" in text:
        return "disk"
    if "temp" in text:
        return "temperature"
    if "bat" in text:
        return "battery"
    if "throttle" in text:
        return "warning"
    if "load" in text:
        return "speed"
    if "uptime" in text or "time" in text:
        return "clock"
    if (
        "signal" in text
        or "quality" in text
        or "frequency" in text
        or "internet" in text
        or "essid" in text
    ):
        return "wifi"
    if (
        "ip" in text
        or "mac" in text
        or "interface" in text
        or "rx/tx" in text
        or "web" in text
        or "client" in text
        or "url" in text
    ):
        return "network"
    if (
        "bus" in text
        or "shunt" in text
        or "current" in text
        or "power" in text
    ):
        return "power"
    if "localizer" in text or "service" in text:
        return "service"
    if "source" in text or "confidence" in text or "doa" in text:
        return "signal"
    if "camera" in text:
        return "camera"
    return "sliders"


class DashboardRenderer:
    """Paint a data dashboard in the active menu theme's visual style.

    :meth:`render` dispatches on the theme's ``render_style``.  The
    modern, classic and retro families each have a header/rows/footer
    triplet but share the text, shape and colour helpers below.
    """

    def __init__(
        self,
        theme: MenuTheme,
        width: int,
        height: int,
        renderer: TextRenderer,
        *,
        padding: int = 5,
        header_height: int = 20,
        footer_height: int = 16,
        font_height: int = 14,
        small_height: int = 10,
    ):
        """Store the theme and the fixed pixel metrics for every layout."""
        self._theme = theme
        self._width = int(width)
        self._height = int(height)
        self._renderer = renderer
        self._padding = int(padding)
        self._header_height = int(header_height)
        self._footer_height = int(footer_height)
        self._font_height = int(font_height)
        self._small_height = int(small_height)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def render(
        self,
        *,
        title: str,
        page: int,
        pages: int,
        rows: Sequence[DashboardRow],
        hints: str,
    ) -> np.ndarray:
        """Render one dashboard frame for the given page and rows."""
        # Allocate a fresh background rather than recycling a buffer:
        # consumers (display, preview server) may still reference the
        # previous frame while the next one is rendered.
        frame = np.full(
            (self._height, self._width, 3),
            self._theme.background,
            dtype=np.uint8,
        )
        pages = max(1, int(pages))
        page = min(max(0, int(page)), pages - 1)
        style = self._theme.render_style

        if style == "classic":
            self._render_classic(frame, title, page, pages, rows, hints)
        elif style in {"win2000", "win7", "aero"}:
            self._render_retro(
                frame, style, title, page, pages, rows, hints
            )
        else:
            self._render_modern(frame, title, page, pages, rows, hints)
        return frame

    # ------------------------------------------------------------------
    # Text helpers (same renderer as the menu)
    # ------------------------------------------------------------------
    def _text(
        self,
        frame: np.ndarray,
        text: str,
        origin: tuple[int, int],
        height: int,
        color: tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        """Draw localized text, clamping size and colour for OpenCV."""
        value = tr(str(text))
        if not value:
            return
        self._renderer.draw(
            frame,
            value,
            (int(origin[0]), int(origin[1])),
            max(6, int(height)),
            tuple(int(channel) for channel in color),
            max(1, int(thickness)),
        )

    def _measure_raw(self, text: str, height: int) -> int:
        """Measure already-localized text so callers can compare widths."""
        if not text:
            return 0
        return self._renderer.measure(str(text), max(6, int(height)))[0]

    def _fit(self, text: str, max_width: int, height: int) -> str:
        """Localize and truncate ``text`` with a ``..`` suffix to fit."""
        value = tr(str(text))
        if max_width <= 0 or not value:
            return ""
        if self._measure_raw(value, height) <= max_width:
            return value
        suffix = ".."
        while value and self._measure_raw(value + suffix, height) > max_width:
            value = value[:-1]
        return (value + suffix) if value else ""

    # ------------------------------------------------------------------
    # Shared primitives (kept in step with MenuControlledEnd)
    # ------------------------------------------------------------------
    @staticmethod
    def _blend(
        first: tuple[int, int, int],
        second: tuple[int, int, int],
        fraction: float,
    ) -> tuple[int, int, int]:
        """Interpolate two BGR colours, clamping the fraction to 0..1."""
        fraction = max(0.0, min(1.0, float(fraction)))
        return tuple(
            int(first[index] + (second[index] - first[index]) * fraction)
            for index in range(3)
        )

    @staticmethod
    def _rounded_rect(
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: tuple[int, int, int],
        radius: int,
        thickness: int = -1,
    ) -> None:
        """Draw a filled or outlined rounded rectangle."""
        # OpenCV has no rounded-rect primitive: fill the centre cross and
        # four corner discs, or draw straight edges plus quarter arcs.
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)
        radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
        if radius == 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            return
        if thickness < 0:
            cv2.rectangle(
                frame, (x1 + radius, y1), (x2 - radius, y2), color, -1
            )
            cv2.rectangle(
                frame, (x1, y1 + radius), (x2, y2 - radius), color, -1
            )
            for cx, cy in (
                (x1 + radius, y1 + radius),
                (x2 - radius, y1 + radius),
                (x1 + radius, y2 - radius),
                (x2 - radius, y2 - radius),
            ):
                cv2.circle(frame, (cx, cy), radius, color, -1, cv2.LINE_AA)
            return

        cv2.line(frame, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(frame, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(frame, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(frame, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        for center, start, end in (
            ((x1 + radius, y1 + radius), 180, 270),
            ((x2 - radius, y1 + radius), 270, 360),
            ((x1 + radius, y2 - radius), 90, 180),
            ((x2 - radius, y2 - radius), 0, 90),
        ):
            cv2.ellipse(
                frame,
                center,
                (radius, radius),
                0,
                start,
                end,
                color,
                thickness,
                cv2.LINE_AA,
            )

    @staticmethod
    def _bevel_rect(
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        fill: tuple[int, int, int],
        raised: bool = True,
    ) -> None:
        """Draw a flat rectangle with a one-pixel 3D bevel border."""
        cv2.rectangle(frame, (x1, y1), (x2, y2), fill, -1)
        light = (255, 255, 255)
        dark = (128, 128, 128)
        top_left, bottom_right = (
            (light, dark) if raised else (dark, light)
        )
        cv2.line(frame, (x1, y1), (x2, y1), top_left, 1)
        cv2.line(frame, (x1, y1), (x1, y2), top_left, 1)
        cv2.line(frame, (x1, y2), (x2, y2), bottom_right, 1)
        cv2.line(frame, (x2, y1), (x2, y2), bottom_right, 1)

    def _glass_rect(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        radius: int,
        top: tuple[int, int, int],
        bottom: tuple[int, int, int],
        border: tuple[int, int, int] | None = None,
        gloss: bool = True,
    ) -> None:
        """Fill a rounded rectangle with a vertical gradient and gloss."""
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
        self._rounded_rect(frame, x1, y1, x2, y2, bottom, radius)
        height = max(1, y2 - y1)
        # OpenCV cannot draw gradients, so paint one scanline per row and
        # inset the ends to follow the rounded corners.
        for y in range(y1, y2 + 1):
            fraction = (y - y1) / height
            color = self._blend(top, bottom, fraction)
            inset = 0
            if radius and y < y1 + radius:
                dy = (y1 + radius - y) / radius
                inset = int(
                    round(
                        radius
                        - radius * math.sqrt(max(0.0, 1.0 - dy * dy))
                    )
                )
            elif radius and y > y2 - radius:
                dy = (y - (y2 - radius)) / radius
                inset = int(
                    round(
                        radius
                        - radius * math.sqrt(max(0.0, 1.0 - dy * dy))
                    )
                )
            cv2.line(
                frame, (x1 + inset, y), (x2 - inset, y), color, 1
            )
        if gloss and y2 - y1 >= 6:
            highlight = self._blend(top, (255, 255, 255), 0.65)
            cv2.line(
                frame,
                (x1 + radius + 2, y1 + 2),
                (x2 - radius - 2, y1 + 2),
                highlight,
                1,
                cv2.LINE_AA,
            )
        if border is not None:
            self._rounded_rect(
                frame, x1, y1, x2, y2, border, radius, 1
            )

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------
    def _row_geometry(
        self, count: int
    ) -> tuple[int, int, int, int]:
        """Return first-row y, row height and content bounds.

        Row height is clamped to 12..32 px and the block is vertically
        centred between the header and footer.
        """
        top = self._header_height + 2
        bottom = self._height - self._footer_height - 2
        available = max(1, bottom - top)
        count = max(1, int(count))
        row_height = max(12, min(32, available // count))
        block_height = row_height * count
        y = top + max(0, (available - block_height) // 2)
        return y, row_height, top, bottom

    def _draw_no_data(self, frame: np.ndarray) -> None:
        """Centre the localized NO DATA placeholder in the content area."""
        top = self._header_height + 2
        bottom = self._height - self._footer_height - 2
        text = tr("NO DATA")
        width = self._measure_raw(text, 12)
        self._text(
            frame,
            text,
            ((self._width - width) // 2, (top + bottom) // 2),
            12,
            self._theme.text_dim,
        )

    @staticmethod
    def _percent(value: float | None) -> float:
        """Clamp a bar percentage into 0..100 (missing becomes zero)."""
        if value is None:
            return 0.0
        return max(0.0, min(100.0, float(value)))

    # ------------------------------------------------------------------
    # Modern themes: dark_pro, high_contrast, industrial
    # ------------------------------------------------------------------
    def _render_modern(
        self,
        frame: np.ndarray,
        title: str,
        page: int,
        pages: int,
        rows: Sequence[DashboardRow],
        hints: str,
    ) -> None:
        """Render using the card-list layout of the modern theme family."""
        self._modern_header(frame, title, page, pages)
        if rows:
            self._modern_rows(frame, rows)
        else:
            self._draw_no_data(frame)
        self._modern_footer(frame, hints)

    def _modern_header(
        self, frame: np.ndarray, title: str, page: int, pages: int
    ) -> None:
        """Draw the header band, accent stripe and page pill."""
        theme = self._theme
        cv2.rectangle(
            frame,
            (0, 0),
            (self._width, self._header_height),
            theme.header_bg,
            -1,
        )
        cv2.rectangle(
            frame, (0, 0), (3, self._header_height), theme.accent, -1
        )
        page_text = f"{page + 1}/{pages}"
        page_width = self._measure_raw(page_text, self._small_height)
        pill_width = page_width + 12
        x2 = self._width - self._padding
        x1 = x2 - pill_width
        y1 = (self._header_height - 14) // 2
        self._rounded_rect(
            frame, x1, y1, x2, y1 + 14, theme.surface_alt, 7
        )
        self._text(
            frame,
            page_text,
            (x1 + 6, y1 + 11),
            self._small_height,
            theme.accent,
        )
        title_text = self._fit(
            title, x1 - (self._padding + 4) - 8, self._font_height
        )
        self._text(
            frame,
            title_text,
            (self._padding + 4, self._header_height - 5),
            self._font_height,
            theme.text,
        )

    def _modern_rows(
        self, frame: np.ndarray, rows: Sequence[DashboardRow]
    ) -> None:
        """Draw icon/label rows with optional bars and value badges."""
        theme = self._theme
        y, row_height, _, _ = self._row_geometry(len(rows))
        x1 = self._padding
        x2 = self._width - self._padding
        for index, row in enumerate(rows):
            # Zebra striping keeps long value lists readable.
            if index % 2 == 0:
                cv2.rectangle(
                    frame, (x1, y), (x2, y + row_height - 1),
                    theme.surface, -1,
                )
            cv2.line(
                frame,
                (x1, y + row_height - 1),
                (x2, y + row_height - 1),
                theme.border,
                1,
                cv2.LINE_AA,
            )
            label_height = min(
                self._font_height, max(9, row_height - 5)
            )
            value_height = min(18, max(14, row_height - 6))
            value_font = min(
                self._small_height, max(8, value_height - 4)
            )
            icon_size = min(14, max(10, row_height - 4))
            icon_x = x1 + 5
            icon_y = y + (row_height - icon_size) // 2
            draw_icon(
                frame,
                row.icon or icon_for_label(row.label),
                icon_x,
                icon_y,
                icon_size,
                theme.text_dim,
            )
            label_x = icon_x + icon_size + 5
            baseline = y + (row_height + label_height) // 2

            value_text = tr(row.value)
            value_width = self._measure_raw(value_text, value_font)
            # Cap the badge at 45% of the row so labels keep room.
            badge_width = min(
                int((x2 - x1) * 0.45), max(34, value_width + 12)
            )
            badge_right = x2 - 4
            badge_left = badge_right - badge_width
            label = self._fit(
                row.label, badge_left - label_x - 4, label_height
            )
            self._text(
                frame,
                label,
                (label_x, baseline),
                label_height,
                theme.text,
            )

            if row.percent is not None:
                bar_left = (
                    label_x + self._measure_raw(label, label_height) + 8
                )
                bar_right = badge_left - 8
                # Skip the bar when label and badge leave no useful width.
                if bar_right - bar_left >= 18:
                    bar_y = y + row_height // 2 - 2
                    cv2.rectangle(
                        frame,
                        (bar_left, bar_y),
                        (bar_right, bar_y + 4),
                        theme.surface_alt,
                        -1,
                    )
                    fill = int(
                        (bar_right - bar_left)
                        * self._percent(row.percent)
                        / 100.0
                    )
                    if fill > 0:
                        cv2.rectangle(
                            frame,
                            (bar_left, bar_y),
                            (bar_left + fill, bar_y + 4),
                            row.color,
                            -1,
                        )

            badge_top = y + (row_height - value_height) // 2
            badge_bottom = badge_top + value_height
            self._rounded_rect(
                frame,
                badge_left,
                badge_top,
                badge_right,
                badge_bottom,
                theme.surface_alt,
                value_height // 2,
            )
            self._rounded_rect(
                frame,
                badge_left,
                badge_top,
                badge_right,
                badge_bottom,
                theme.border,
                value_height // 2,
                1,
            )
            fitted = self._fit(
                row.value, badge_width - 12, value_font
            )
            self._text(
                frame,
                fitted,
                (badge_left + 6, badge_bottom - 4),
                value_font,
                row.color,
            )
            y += row_height

    def _modern_footer(self, frame: np.ndarray, hints: str) -> None:
        """Draw the footer band with the localized control hints."""
        theme = self._theme
        y0 = self._height - self._footer_height
        cv2.rectangle(
            frame, (0, y0), (self._width, self._height),
            theme.footer_bg, -1,
        )
        self._text(
            frame,
            hints,
            (self._padding, self._height - 4),
            self._small_height,
            theme.text_dim,
        )

    # ------------------------------------------------------------------
    # Classic theme: the original packed black list
    # ------------------------------------------------------------------
    def _render_classic(
        self,
        frame: np.ndarray,
        title: str,
        page: int,
        pages: int,
        rows: Sequence[DashboardRow],
        hints: str,
    ) -> None:
        """Render the original packed black list used by the classic theme."""
        theme = self._theme
        title_y = self._padding + self._font_height
        self._text(
            frame,
            title,
            (self._padding, title_y),
            self._font_height,
            theme.text,
        )
        # Scale chrome from the shorter edge so small and large screens
        # keep the same visual proportions.
        reference = min(self._width, self._height)
        bar_width = max(10, reference // 21)
        bar_x = self._width - self._padding - bar_width
        content_right = bar_x - 4
        page_text = f"{page + 1}/{pages}"
        page_width = self._measure_raw(page_text, self._small_height)
        self._text(
            frame,
            page_text,
            (
                content_right - page_width - 8,
                title_y,
            ),
            self._small_height,
            theme.text_dim,
        )

        top = title_y + 6
        bottom = self._height - self._padding - 10
        if rows:
            count = len(rows)
            row_height = max(
                12, min(30, max(1, bottom - top) // count)
            )
            block_height = row_height * count
            y = top + max(0, (bottom - top - block_height) // 2)
            for row in rows:
                label_height = min(
                    self._font_height, max(9, row_height - 3)
                )
                baseline = y + (row_height + label_height) // 2
                value_height = min(18, max(12, row_height - 4))
                value_font = min(
                    self._small_height, max(8, value_height - 4)
                )
                value_text = tr(row.value)
                value_width = self._measure_raw(value_text, value_font)
                block_width = min(
                    int((content_right - self._padding) * 0.5),
                    max(48, value_width + 12),
                )
                block_left = content_right - block_width
                label = self._fit(
                    row.label,
                    block_left - self._padding - 6,
                    label_height,
                )
                self._text(
                    frame,
                    label,
                    (self._padding, baseline),
                    label_height,
                    theme.text,
                )

                bar_color = (
                    row.color
                    if tuple(row.color) != tuple(theme.text)
                    else theme.numeral
                )
                if row.percent is not None:
                    bar_left = (
                        self._padding
                        + self._measure_raw(label, label_height)
                        + 6
                    )
                    bar_right = block_left - 6
                    if bar_right - bar_left >= 16:
                        bar_y = y + row_height // 2 - 2
                        cv2.rectangle(
                            frame,
                            (bar_left, bar_y),
                            (bar_right, bar_y + 4),
                            theme.surface_alt,
                            -1,
                        )
                        fill = int(
                            (bar_right - bar_left)
                            * self._percent(row.percent)
                            / 100.0
                        )
                        if fill > 0:
                            cv2.rectangle(
                                frame,
                                (bar_left, bar_y),
                                (bar_left + fill, bar_y + 4),
                                bar_color,
                                -1,
                            )

                block_top = y + (row_height - value_height) // 2
                block_bottom = block_top + value_height
                cv2.rectangle(
                    frame,
                    (block_left, block_top),
                    (content_right, block_bottom),
                    bar_color,
                    -1,
                )
                fitted = self._fit(
                    value_text, block_width - 10, value_font
                )
                self._text(
                    frame,
                    fitted,
                    (block_left + 5, block_bottom - 4),
                    value_font,
                    theme.text,
                )
                y += row_height
        else:
            self._draw_no_data(frame)

        self._classic_slide_bar(frame, page, pages, bar_x, bar_width)
        self._text(
            frame,
            hints,
            (self._padding, self._height - 4),
            self._small_height,
            theme.text_dim,
        )

    def _classic_slide_bar(
        self,
        frame: np.ndarray,
        page: int,
        pages: int,
        bar_x: int,
        bar_width: int,
    ) -> None:
        """Draw the classic right-edge scrollbar and page thumb."""
        theme = self._theme
        reference = min(self._width, self._height)
        arrow_height = max(5, reference // 21)
        gap = max(2, reference // 42)
        top = self._padding
        bottom = self._height - self._padding
        color = theme.accent

        cv2.rectangle(
            frame,
            (bar_x, top),
            (bar_x + bar_width, bottom),
            theme.background,
            -1,
        )
        top_arrow = np.array(
            [
                (bar_x + bar_width, top + arrow_height),
                (bar_x, top + arrow_height),
                (bar_x + bar_width // 2, top),
            ],
            dtype=np.int32,
        )
        bottom_arrow = np.array(
            [
                (bar_x + bar_width, bottom - arrow_height),
                (bar_x, bottom - arrow_height),
                (bar_x + bar_width // 2, bottom),
            ],
            dtype=np.int32,
        )
        # A filled arrow means another page exists in that direction; an
        # outline marks the first or last page.
        if page > 0:
            cv2.fillConvexPoly(frame, top_arrow, color)
        else:
            cv2.polylines(frame, [top_arrow], True, color, 1)
        if page < pages - 1:
            cv2.fillConvexPoly(frame, bottom_arrow, color)
        else:
            cv2.polylines(frame, [bottom_arrow], True, color, 1)

        track_top = top + arrow_height + gap
        track_bottom = bottom - arrow_height - gap
        thumb_height = max(
            4, (track_bottom - track_top) // max(1, pages)
        )
        thumb_top = track_top + page * thumb_height
        cv2.rectangle(
            frame,
            (bar_x, thumb_top),
            (bar_x + bar_width, thumb_top + thumb_height),
            color,
            -1,
        )

    # ------------------------------------------------------------------
    # Retro themes: XP, 2000, 7, Aero, 10
    # ------------------------------------------------------------------
    def _render_retro(
        self,
        frame: np.ndarray,
        style: str,
        title: str,
        page: int,
        pages: int,
        rows: Sequence[DashboardRow],
        hints: str,
    ) -> None:
        """Render the win2000, win7 or aero chrome for the given style."""
        self._retro_header(frame, style, title, page, pages)
        if rows:
            self._retro_rows(frame, style, rows)
        else:
            self._draw_no_data(frame)
        self._retro_footer(frame, hints, style == "aero")

    def _retro_header(
        self,
        frame: np.ndarray,
        style: str,
        title: str,
        page: int,
        pages: int,
    ) -> None:
        """Draw style-specific window chrome, the page pill and title."""
        theme = self._theme
        height = self._header_height
        page_text = f"{page + 1}/{pages}"
        page_width = self._measure_raw(page_text, self._small_height)

        if style == "win2000":
            cv2.rectangle(
                frame, (0, 0), (self._width, height),
                theme.header_bg, -1,
            )
            cv2.line(
                frame,
                (0, height - 1),
                (self._width, height - 1),
                (255, 255, 255),
                1,
            )
            controls = 3
            box = 11
            gap = 2
            x = self._width - self._padding - controls * box - gap
            page_x = x - page_width - 6
            self._text(
                frame,
                page_text,
                (page_x, height - 5),
                self._small_height,
                theme.accent_text,
            )
            # Three line-art boxes mimic minimise/maximise/close.
            for index in range(controls):
                bx = x + index * (box + gap)
                self._bevel_rect(
                    frame, bx, 4, bx + box, 4 + box,
                    theme.surface_alt, raised=True,
                )
                if index == 0:
                    cv2.line(
                        frame,
                        (bx + 3, 4 + box - 4),
                        (bx + box - 3, 4 + box - 4),
                        (0, 0, 0),
                        1,
                    )
                elif index == 1:
                    cv2.rectangle(
                        frame,
                        (bx + 3, 7),
                        (bx + box - 3, 4 + box - 3),
                        (0, 0, 0),
                        1,
                    )
                else:
                    cv2.line(
                        frame,
                        (bx + 3, 7),
                        (bx + box - 3, 4 + box - 3),
                        (0, 0, 0),
                        1,
                    )
                    cv2.line(
                        frame,
                        (bx + box - 3, 7),
                        (bx + 3, 4 + box - 3),
                        (0, 0, 0),
                        1,
                    )
            title_color = theme.accent_text
            title_right = page_x - 6
        elif style == "win7":
            # Blend a light blue into the theme colour for the header band.
            top = (210, 160, 100)
            for y in range(height):
                color = self._blend(
                    top, theme.header_bg, y / max(1, height - 1)
                )
                cv2.line(
                    frame, (0, y), (self._width, y), color, 1
                )
            cv2.line(
                frame,
                (0, height - 1),
                (self._width, height - 1),
                theme.accent,
                1,
            )
            pill_width = page_width + 12
            x2 = self._width - self._padding
            x1 = x2 - pill_width
            y1 = (height - 14) // 2
            self._rounded_rect(
                frame, x1, y1, x2, y1 + 14, theme.surface_alt, 7
            )
            self._text(
                frame,
                page_text,
                (x1 + 6, y1 + 11),
                self._small_height,
                theme.accent,
            )
            title_color = theme.header_text or theme.text
            title_right = x1 - 8
        else:  # aero
            self._glass_rect(
                frame,
                0,
                0,
                self._width,
                height - 1,
                0,
                self._blend(theme.header_bg, (255, 255, 255), 0.65),
                theme.header_bg,
            )
            cv2.line(
                frame, (0, 0), (self._width, 0), (255, 255, 255), 1
            )
            cv2.line(
                frame,
                (0, height - 1),
                (self._width, height - 1),
                self._blend(theme.header_bg, (0, 0, 0), 0.35),
                1,
            )
            pill_width = page_width + 12
            x2 = self._width - self._padding
            x1 = x2 - pill_width
            y1 = (height - 14) // 2
            self._glass_rect(
                frame,
                x1,
                y1,
                x2,
                y1 + 14,
                7,
                self._blend(theme.header_bg, (255, 255, 255), 0.55),
                self._blend(theme.header_bg, (255, 255, 255), 0.2),
                border=(255, 255, 255),
                gloss=False,
            )
            self._text(
                frame,
                page_text,
                (x1 + 6, y1 + 11),
                self._small_height,
                theme.header_text or theme.text,
            )
            title_color = theme.header_text or theme.text
            title_right = x1 - 8

        title_text = self._fit(
            title,
            title_right - (self._padding + 4) - 6,
            self._font_height,
        )
        if style == "aero":
            self._text(
                frame,
                title_text,
                (self._padding + 5, height - 4),
                self._font_height,
                self._blend(theme.header_bg, (0, 0, 0), 0.5),
            )
        self._text(
            frame,
            title_text,
            (self._padding + 4, height - 5),
            self._font_height,
            title_color,
        )
    def _retro_rows(
        self,
        frame: np.ndarray,
        style: str,
        rows: Sequence[DashboardRow],
    ) -> None:
        """Draw rows using each retro style's surface treatment."""
        theme = self._theme
        y, row_height, _, _ = self._row_geometry(len(rows))
        x1 = self._padding
        x2 = self._width - self._padding
        for index, row in enumerate(rows):
            # Sunken bevels match the Windows 2000 list-control look.
            if style == "win2000":
                self._bevel_rect(
                    frame,
                    x1,
                    y,
                    x2,
                    y + row_height - 1,
                    theme.surface,
                    raised=False,
                )
                foreground = theme.text
                icon_color = theme.text
            elif style == "win7":
                # Zebra stripes plus a hairline replace the bevels.
                row_bg = (
                    theme.surface if index % 2 == 0
                    else theme.surface_alt
                )
                cv2.rectangle(
                    frame, (x1, y), (x2, y + row_height - 1),
                    row_bg, -1,
                )
                cv2.line(
                    frame,
                    (x1, y + row_height - 1),
                    (x2, y + row_height - 1),
                    theme.border,
                    1,
                )
                foreground = theme.text
                icon_color = theme.accent
            else:  # aero
                # Aero rows are translucent glass panels with a border.
                row_bg = (
                    theme.surface if index % 2 == 0
                    else theme.surface_alt
                )
                self._glass_rect(
                    frame,
                    x1 + 2,
                    y + 1,
                    x2 - 2,
                    y + row_height - 2,
                    4,
                    self._blend(row_bg, (255, 255, 255), 0.5),
                    row_bg,
                    border=theme.border,
                )
                foreground = theme.text
                icon_color = theme.accent

            label_height = min(
                self._font_height, max(9, row_height - 5)
            )
            value_height = min(18, max(14, row_height - 6))
            value_font = min(
                self._small_height, max(8, value_height - 4)
            )
            icon_size = min(14, max(10, row_height - 4))
            icon_x = x1 + 5
            icon_y = y + (row_height - icon_size) // 2
            draw_icon(
                frame,
                row.icon or icon_for_label(row.label),
                icon_x,
                icon_y,
                icon_size,
                icon_color,
            )
            label_x = icon_x + icon_size + 5
            baseline = y + (row_height + label_height) // 2

            value_text = tr(row.value)
            value_width = self._measure_raw(value_text, value_font)
            badge_width = min(
                int((x2 - x1) * 0.45), max(34, value_width + 12)
            )
            badge_right = x2 - 4
            badge_left = badge_right - badge_width
            label = self._fit(
                row.label, badge_left - label_x - 4, label_height
            )
            self._text(
                frame,
                label,
                (label_x, baseline),
                label_height,
                foreground,
            )

            if row.percent is not None:
                bar_left = (
                    label_x + self._measure_raw(label, label_height) + 8
                )
                bar_right = badge_left - 8
                if bar_right - bar_left >= 18:
                    bar_y = y + row_height // 2 - 2
                    if style == "win2000":
                        self._bevel_rect(
                            frame,
                            bar_left,
                            bar_y,
                            bar_right,
                            bar_y + 4,
                            theme.surface_alt,
                            raised=False,
                        )
                    elif style == "aero":
                        self._rounded_rect(
                            frame,
                            bar_left,
                            bar_y,
                            bar_right,
                            bar_y + 4,
                            theme.surface_alt,
                            2,
                        )
                    else:
                        cv2.rectangle(
                            frame,
                            (bar_left, bar_y),
                            (bar_right, bar_y + 4),
                            theme.surface_alt,
                            -1,
                        )
                    fill = int(
                        (bar_right - bar_left)
                        * self._percent(row.percent)
                        / 100.0
                    )
                    if fill > 0:
                        cv2.rectangle(
                            frame,
                            (bar_left, bar_y),
                            (bar_left + fill, bar_y + 4),
                            row.color,
                            -1,
                        )

            badge_top = y + (row_height - value_height) // 2
            badge_bottom = badge_top + value_height
            if tuple(row.color) == tuple(theme.text):
                badge_foreground = (
                    theme.accent
                    if style in {"win7", "aero"}
                    else theme.text
                )
            else:
                badge_foreground = row.color

            if style == "win2000":
                self._bevel_rect(
                    frame,
                    badge_left,
                    badge_top,
                    badge_right,
                    badge_bottom,
                    theme.surface_alt,
                    raised=True,
                )
            elif style == "win7":
                self._rounded_rect(
                    frame,
                    badge_left,
                    badge_top,
                    badge_right,
                    badge_bottom,
                    theme.surface,
                    4,
                )
                cv2.rectangle(
                    frame,
                    (badge_left, badge_top),
                    (badge_right, badge_bottom),
                    theme.accent,
                    1,
                    cv2.LINE_AA,
                )
            else:  # aero
                self._glass_rect(
                    frame,
                    badge_left,
                    badge_top,
                    badge_right,
                    badge_bottom,
                    value_height // 2,
                    self._blend(theme.surface, (255, 255, 255), 0.5),
                    theme.surface,
                    border=theme.border,
                    gloss=False,
                )

            fitted = self._fit(
                row.value, badge_width - 12, value_font
            )
            text_width = self._measure_raw(fitted, value_font)
            self._text(
                frame,
                fitted,
                (
                    badge_left + (badge_width - text_width) // 2,
                    badge_bottom - 5,
                ),
                value_font,
                badge_foreground,
            )
            y += row_height

    def _retro_footer(
        self, frame: np.ndarray, hints: str, aero: bool
    ) -> None:
        """Draw the footer bar, using a gradient in the Aero style."""
        theme = self._theme
        y0 = self._height - self._footer_height
        if aero:
            # Aero fades into the border tone instead of a flat bar.
            top = theme.footer_bg
            bottom = self._blend(theme.footer_bg, theme.border, 0.35)
            span = max(1, self._height - y0 - 1)
            for y in range(y0, self._height):
                cv2.line(
                    frame,
                    (0, y),
                    (self._width, y),
                    self._blend(top, bottom, (y - y0) / span),
                    1,
                )
            cv2.line(
                frame, (0, y0), (self._width, y0), (255, 255, 255), 1
            )
        else:
            cv2.rectangle(
                frame, (0, y0), (self._width, self._height),
                theme.footer_bg, -1,
            )
        self._text(
            frame,
            hints,
            (self._padding, self._height - 4),
            self._small_height,
            theme.text_dim,
        )
