"""Compact camera status strip.

The strip is split into evenly spaced items with thin separators, a
translucent panel and an accent edge.  Items may be plain strings,
``(text, role)`` tuples, or :class:`StatusItem` instances that keep a label
and its value visually separate.  ``role`` selects a colour from the active
menu theme (``text``, ``dim``, ``accent``, ``success``, ``warning``,
``danger``).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from chandelier.ui.text_render import get_renderer
from chandelier.ui.themes import get_theme

_ROLE_ATTRS = {
    "text": "text",
    "dim": "text_dim",
    "accent": "accent",
    "success": "success",
    "warning": "warning",
    "danger": "danger",
}


@dataclass(frozen=True)
class StatusItem:
    """A label and value rendered as one item with a small gap."""

    label: str
    value: str
    label_role: str = "dim"
    value_role: str = "text"
    gap: int = 4


@dataclass(frozen=True)
class _Part:
    text: str
    role: str = "text"


@dataclass(frozen=True)
class _Item:
    parts: tuple[_Part, ...]
    gap: int = 4

    @property
    def text(self) -> str:
        return " ".join(part.text for part in self.parts)


class StatusBar:
    """Evenly spaced status strip with theme-coloured items.

    ``set_items`` accepts plain strings, ``(text, role)`` tuples, four- or
    five-element ``(label, value, label_role, value_role[, gap])`` tuples,
    or :class:`StatusItem` instances; anything else is ignored.  Drawing
    blends a translucent panel over the frame, so the overlay covers live
    preview content without hiding it completely.  All theme colours are
    BGR.
    """

    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        font_height: int = 10,
        position: str = "top",
        alpha: float = 0.78,
    ):
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if font_height <= 0:
            raise ValueError("font_height must be positive")

        self._width = width
        self._height = height
        self._font_height = font_height
        self._renderer = get_renderer()
        self._position = "top"
        self.set_position(position)
        self._alpha = max(0.0, min(1.0, float(alpha)))
        self._items: list[_Item] = []

    @property
    def items(self) -> list[str]:
        return [item.text for item in self._items]

    @property
    def bar_height(self) -> int:
        """Height of the horizontal top/bottom strip in pixels."""

        return max(20, self._font_height + 12)

    def set_position(self, position: str) -> None:
        """Select ``top``, ``bottom``, ``sides`` or ``top_bottom``.

        Unknown values fall back to ``top`` instead of raising, because the
        position may come from an older menu configuration.
        """

        self._position = (
            position
            if position in {"top", "bottom", "sides", "top_bottom"}
            else "top"
        )

    def set_items(self, items: list) -> None:
        """Parse and store the item list; re-parsing is cheap and rare."""

        parsed: list[_Item] = []
        for item in items:
            if isinstance(item, StatusItem):
                parts = []
                label = str(item.label).strip()
                value = str(item.value).strip()
                if label:
                    parts.append(_Part(label, str(item.label_role)))
                if value:
                    parts.append(_Part(value, str(item.value_role)))
                if parts:
                    parsed.append(
                        _Item(tuple(parts), max(0, int(item.gap)))
                    )
                continue

            if isinstance(item, str):
                item = (item,)
            if not isinstance(item, (tuple, list)) or not item:
                continue

            if len(item) >= 4:
                label, value = item[0], item[1]
                label_role = item[2]
                value_role = item[3]
                gap = item[4] if len(item) > 4 else 4
                label = str(label).strip() if label is not None else ""
                value = str(value).strip() if value is not None else ""
                parts = []
                if label:
                    parts.append(_Part(label, str(label_role)))
                if value:
                    parts.append(_Part(value, str(value_role)))
                if parts:
                    parsed.append(_Item(tuple(parts), max(0, int(gap))))
                continue

            text = item[0]
            role = item[1] if len(item) > 1 else "text"
            text = str(text).strip() if text is not None else ""
            if text:
                parsed.append(_Item((_Part(text, str(role)),)))
        self._items = parsed

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------
    def _text_width(self, text: str) -> int:
        return self._renderer.measure(text, self._font_height)[0]

    def _fit(self, text: str, max_width: int | None = None) -> str:
        """Truncate ``text`` with a ``..`` suffix until it fits.

        Removal is one character at a time because the renderer's target
        font may be proportional; estimating from character count would
        leave some strings too wide.
        """

        if max_width is None:
            max_width = self._width - 12
        if self._text_width(text) <= max_width:
            return text
        suffix = ".."
        while text:
            text = text[:-1]
            if self._text_width(text + suffix) <= max_width:
                return text + suffix
        return ""

    def _item_width(self, item: _Item) -> int:
        if not item.parts:
            return 0
        return sum(
            self._text_width(part.text) for part in item.parts
        ) + item.gap * (len(item.parts) - 1)

    @staticmethod
    def _item_line_groups(
        item: _Item,
    ) -> list[tuple[_Part, ...]]:
        """Return the item's display lines.

        Items with a label and value are stacked as two lines in the side
        layout; plain items keep a single line.
        """

        if len(item.parts) <= 1:
            return [item.parts]
        return [(item.parts[0],), tuple(item.parts[1:])]

    def _parts_width(self, parts: tuple[_Part, ...], gap: int) -> int:
        if not parts:
            return 0
        return sum(
            self._text_width(part.text) for part in parts
        ) + gap * (len(parts) - 1)

    def _block_width(self, item: _Item) -> int:
        return max(
            (
                self._parts_width(parts, item.gap)
                for parts in self._item_line_groups(item)
            ),
            default=0,
        )

    def _fit_item_lines(self, item: _Item, max_width: int) -> _Item:
        """Fit a stacked label/value block to ``max_width``."""

        if self._block_width(item) <= max_width:
            return item

        parts = list(item.parts)
        while (
            parts
            and self._block_width(_Item(tuple(parts), item.gap)) > max_width
        ):
            last = parts[-1]
            if len(last.text) > 1:
                parts[-1] = _Part(last.text[:-1], last.role)
            elif len(parts) > 1:
                parts.pop()
            else:
                parts[-1] = _Part(
                    self._fit(last.text, max_width), last.role
                )
                break
        return _Item(tuple(parts), item.gap)

    def _fit_item(
        self,
        item: _Item,
        max_width: int | None = None,
    ) -> _Item:
        """Trim parts from the end until the single-line item fits.

        The trailing value is shortened first because the label is usually
        the more meaningful half; parts are dropped only when even a
        one-character value is too wide.
        """

        if max_width is None:
            max_width = self._width - 12
        if self._item_width(item) <= max_width:
            return item

        parts = list(item.parts)
        while (
            parts
            and self._item_width(_Item(tuple(parts), item.gap)) > max_width
        ):
            last = parts[-1]
            if len(last.text) > 1:
                parts[-1] = _Part(last.text[:-1], last.role)
            elif len(parts) > 1:
                parts.pop()
            else:
                parts[-1] = _Part(
                    self._fit(last.text, max_width), last.role
                )
                break
        return _Item(tuple(parts), item.gap)

    def _layout(
        self,
        width: int,
        items: list[_Item] | None = None,
    ) -> list[tuple[_Item, int]]:
        """Return the items that fit and their drawn widths."""

        items = list(self._items if items is None else items)
        widths = [self._item_width(item) for item in items]
        available = max(1, width - 14)
        gap = 6

        while (
            len(items) > 1
            and sum(widths) + gap * (len(items) - 1) > available
        ):
            items.pop()
            widths.pop()

        if not items:
            return []

        if widths[0] > available:
            items[0] = self._fit_item(items[0], available)
            widths[0] = self._item_width(items[0])
        return list(zip(items, widths))

    def _split_for_sides(self) -> tuple[list[_Item], list[_Item]]:
        """Split items evenly for the stacked side columns.

        The side layout stacks each label above its value, so vertical space
        matters more than text width.  An even item count keeps both columns
        the same height.
        """

        items = list(self._items)
        if len(items) <= 1:
            return items, []
        split = (len(items) + 1) // 2
        return items[:split], items[split:]

    def _split_for_two_bars(self) -> tuple[list[_Item], list[_Item]]:
        """Split items into two width-balanced groups.

        Splitting by item count can leave one row much wider than the other
        (for example ``1/100`` versus ``ISO1600``).  Choosing the prefix whose
        width is closest to half the total keeps both rows as short as
        possible.
        """

        items = list(self._items)
        if len(items) <= 1:
            return items, []

        widths = [self._item_width(item) for item in items]
        total = sum(widths)
        best_index = 1
        best_diff = None
        cumulative = 0
        for index in range(1, len(items)):
            cumulative += widths[index - 1]
            difference = abs(total - 2 * cumulative)
            if best_diff is None or difference < best_diff:
                best_diff = difference
                best_index = index
        return items[:best_index], items[best_index:]

    def _role_color(self, theme, role: str):
        """Resolve a role name to a BGR colour for the given theme.

        Themes may override header colours (Aero and Windows 2000 use a
        coloured title bar); ``accent`` deliberately prefers the dim
        header text because the accent itself is the panel background on
        those themes.
        """

        if role == "text" and theme.header_text is not None:
            return theme.header_text
        if role == "dim" and theme.header_text_dim is not None:
            return theme.header_text_dim
        if role == "accent" and theme.header_text is not None:
            return theme.header_text_dim or theme.header_text
        return getattr(theme, _ROLE_ATTRS.get(role, "text"), theme.text)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def _decorate_sides(self, frame, theme) -> None:
        """Draw two mirrored columns with label/value blocks.

        Each item occupies one line when it has no separate value, and two
        lines when it does: the label is drawn above its value.  Both panels
        use the same width and height, and the left column is right-aligned
        while the right column is left-aligned so the two sides mirror each
        other around the centre of the frame.
        """

        frame_height, frame_width = frame.shape[:2]
        width = min(self._width, frame_width)
        font_height = self._font_height
        line_gap = max(2, font_height // 3)
        block_gap = max(8, line_gap + 5)
        padding = 4
        max_line_width = max(20, width // 2 - 4 - 2 * padding)

        left, right = self._split_for_sides()
        columns = [
            [self._fit_item_lines(item, max_line_width) for item in group]
            for group in (left, right)
        ]

        def item_height(item):
            lines = len(self._item_line_groups(item))
            return font_height + (lines - 1) * (font_height + line_gap)

        def column_height(column):
            if not column:
                return 0
            return (
                sum(item_height(item) for item in column)
                + block_gap * (len(column) - 1)
            )

        while (
            max(column_height(columns[0]), column_height(columns[1]))
            + 2 * padding
            > frame_height
        ):
            longer = (
                0
                if column_height(columns[0]) >= column_height(columns[1])
                else 1
            )
            if not columns[longer]:
                break
            columns[longer].pop()

        content_widths = [
            max((self._block_width(item) for item in column), default=0)
            for column in columns
        ]
        content_heights = [column_height(column) for column in columns]
        panel_width = max(content_widths) + 2 * padding
        panel_width = min(width // 2 - 4, panel_width)
        panel_width = max(panel_width, font_height + 2 * padding)
        panel_height = min(
            frame_height,
            max(content_heights) + 2 * padding,
        )
        y0 = max(0, (frame_height - panel_height) // 2)
        y1 = min(frame_height, y0 + panel_height)

        for side, column in enumerate(columns):
            if not column:
                continue
            if side == 0:
                x0, x1 = 0, panel_width
            else:
                x0, x1 = width - panel_width, width

            # Fill a scratch surface with the header colour and blend it
            # into the frame region; BGRA frames need an explicit alpha
            # byte because theme colours are only three channels.
            region = frame[y0:y1, x0:x1]
            surface = np.empty_like(region)
            color = theme.header_bg
            if region.shape[2] == 4:
                color = (*color, 255)
            surface[:] = color
            cv2.addWeighted(
                surface, self._alpha, region, 1.0 - self._alpha, 0, region
            )

            accent_x = x1 - 1 if side == 0 else x0
            cv2.line(
                frame,
                (accent_x, y0),
                (accent_x, y1 - 1),
                theme.accent,
                2,
                cv2.LINE_AA,
            )

            block_y = (
                y0 + (panel_height - content_heights[side]) // 2
            )
            divider_color = theme.text_dim or theme.border
            for index, item in enumerate(column):
                lines = self._item_line_groups(item)
                for line_index, parts in enumerate(lines):
                    line_item = _Item(tuple(parts), item.gap)
                    line_width = self._parts_width(parts, item.gap)
                    line_x = (
                        x1 - padding - line_width
                        if side == 0
                        else x0 + padding
                    )
                    baseline = (
                        block_y
                        + font_height
                        + line_index * (font_height + line_gap)
                    )
                    self._draw_item(
                        frame,
                        line_item,
                        line_x,
                        baseline,
                        font_height,
                        theme,
                    )
                if index < len(column) - 1:
                    divider_y = (
                        block_y
                        + item_height(item)
                        + block_gap // 2
                    )
                    cv2.line(
                        frame,
                        (x0 + padding, divider_y),
                        (x1 - padding - 1, divider_y),
                        divider_color,
                        1,
                        cv2.LINE_AA,
                    )
                block_y += item_height(item) + block_gap

    def _draw_item(
        self,
        frame,
        item: _Item,
        x: int,
        baseline: int,
        font_height: int,
        theme,
    ) -> None:
        """Draw the item parts left to right at a shared baseline."""

        # Advance by measured width (not a fixed pitch) so proportional
        # fonts and mixed label/value roles line up correctly.
        cursor = x
        for index, part in enumerate(item.parts):
            self._renderer.draw(
                frame,
                part.text,
                (cursor, baseline),
                font_height,
                self._role_color(theme, part.role),
                1,
            )
            cursor += self._text_width(part.text)
            if index < len(item.parts) - 1:
                cursor += item.gap

    def _decorate_horizontal(
        self,
        frame,
        theme,
        items: list[_Item],
        y0: int,
        edge: str,
    ) -> None:
        """Draw a translucent strip at ``y0`` with centred items."""

        if not items:
            return
        frame_height, frame_width = frame.shape[:2]
        width = min(self._width, frame_width)
        bar_height = self.bar_height
        y1 = min(frame_height, y0 + bar_height)
        if y1 <= y0:
            return

        # Translucent header surface, blended in place over the region.
        region = frame[y0:y1, :width]
        surface = np.empty_like(region)
        color = theme.header_bg
        if region.shape[2] == 4:
            color = (*color, 255)
        surface[:] = color
        cv2.addWeighted(
            surface, self._alpha, region, 1.0 - self._alpha, 0, region
        )

        accent_y = y1 - 2 if edge == "top" else y0
        cv2.line(
            frame,
            (0, accent_y),
            (width - 1, accent_y),
            theme.accent,
            2,
            cv2.LINE_AA,
        )

        layout = self._layout(width, items)
        if not layout:
            return

        # Distribute the free width between items, with a floor so labels
        # never touch and a cap so empty space stays next to the edges.
        total_width = sum(item_width for _, item_width in layout)
        gap = 8
        if len(layout) > 1:
            remaining = width - 14 - total_width
            gap = max(
                4, min(18, remaining // (len(layout) - 1))
            )
        # Centre the resulting block, but keep it off the frame edge.
        block_width = total_width + gap * (len(layout) - 1)
        x = max(7, (width - block_width) // 2)
        baseline = y0 + (bar_height + self._font_height) // 2 - 1

        for index, (item, item_width) in enumerate(layout):
            self._draw_item(
                frame,
                item,
                x,
                baseline,
                self._font_height,
                theme,
            )
            if index < len(layout) - 1:
                separator_x = x + item_width + gap // 2
                cv2.line(
                    frame,
                    (separator_x, y0 + 5),
                    (separator_x, y1 - 6),
                    theme.border,
                    1,
                    cv2.LINE_AA,
                )
            x += item_width + gap

    def _decorate_top_bottom(self, frame, theme) -> None:
        """Draw one balanced row at the top and one at the bottom."""

        frame_height = frame.shape[0]
        bar_height = self.bar_height
        top_items, bottom_items = self._split_for_two_bars()
        self._decorate_horizontal(frame, theme, top_items, 0, "top")
        self._decorate_horizontal(
            frame,
            theme,
            bottom_items,
            max(0, frame_height - bar_height),
            "bottom",
        )

    def decorate(self, frame, theme_id: str = "dark_pro") -> None:
        """Draw the configured layout in place using ``theme_id``.

        Frames must be colour (three or four channels); gray frames are
        ignored because theme colours need channel data.
        """

        if not self._items or frame is None or frame.size == 0:
            return
        if frame.ndim != 3 or frame.shape[2] < 3:
            return

        theme = get_theme(theme_id)
        frame_height = frame.shape[0]
        if self._position == "sides":
            self._decorate_sides(frame, theme)
            return
        if self._position == "top_bottom":
            self._decorate_top_bottom(frame, theme)
            return

        bar_height = self.bar_height
        if self._position == "bottom":
            y0 = max(0, frame_height - bar_height)
            edge = "bottom"
        else:
            y0 = 0
            edge = "top"
        self._decorate_horizontal(frame, theme, self._items, y0, edge)
