"""Menu colour themes.

All colours are OpenCV BGR triples.  A theme is selected from the menu
(``camera.display.theme``), stored in ``menus.json`` and read by
``MenuControlledEnd`` on startup, so the choice survives a restart.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2

logger = logging.getLogger(__name__)

DEFAULT_THEME_ID = "dark_pro"


@dataclass(frozen=True)
class MenuTheme:
    id: str
    label: str
    background: tuple[int, int, int]
    header_bg: tuple[int, int, int]
    surface: tuple[int, int, int]
    surface_alt: tuple[int, int, int]
    border: tuple[int, int, int]
    footer_bg: tuple[int, int, int]
    text: tuple[int, int, int]
    text_dim: tuple[int, int, int]
    accent: tuple[int, int, int]
    accent_text: tuple[int, int, int]
    success: tuple[int, int, int]
    warning: tuple[int, int, int]
    danger: tuple[int, int, int]
    numeral: tuple[int, int, int]
    msg: tuple[int, int, int]
    option: tuple[int, int, int]
    irq: tuple[int, int, int]
    cursor_style: str = "fill"  # "fill" | "bar" | "outline"
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX
    render_style: str = "modern"  # "modern" | "classic"
    disabled_cursor: tuple[int, int, int] | None = None

    def color_for_type(self, item_type: str) -> tuple[int, int, int]:
        return {
            "numeral": self.numeral,
            "msg": self.msg,
            "option": self.option,
            "irq": self.irq,
            "bool": self.text,
            "menu": self.accent,
        }.get(item_type, self.text)

    def legacy_dict(self) -> dict[str, Any]:
        """Mapping for code paths that have not been migrated yet."""

        return {
            "background": self.background,
            "cursor": self.accent,
            "text": self.text,
            "boolTrue": self.success,
            "boolFalse": self.danger,
            "numeral": self.numeral,
            "msg": self.msg,
            "option": self.option,
            "irq": self.irq,
            "cursorDisable": self.disabled_cursor or self.text_dim,
            "textDisable": self.text_dim,
        }


THEMES: dict[str, MenuTheme] = {
    "dark_pro": MenuTheme(
        id="dark_pro",
        label="Dark Pro",
        background=(16, 16, 16),
        header_bg=(24, 24, 24),
        surface=(32, 32, 32),
        surface_alt=(45, 45, 45),
        border=(62, 62, 62),
        footer_bg=(22, 22, 22),
        text=(235, 235, 235),
        text_dim=(135, 135, 135),
        accent=(0, 190, 255),
        accent_text=(15, 15, 15),
        success=(90, 190, 90),
        warning=(0, 180, 255),
        danger=(70, 70, 220),
        numeral=(0, 170, 255),
        msg=(150, 150, 150),
        option=(200, 120, 220),
        irq=(230, 90, 200),
        cursor_style="fill",
    ),
    "high_contrast": MenuTheme(
        id="high_contrast",
        label="High Contrast",
        background=(0, 0, 0),
        header_bg=(0, 0, 0),
        surface=(18, 18, 18),
        surface_alt=(30, 30, 30),
        border=(255, 255, 255),
        footer_bg=(0, 0, 0),
        text=(255, 255, 255),
        text_dim=(160, 160, 160),
        accent=(0, 255, 255),
        accent_text=(0, 0, 0),
        success=(0, 255, 0),
        warning=(0, 255, 255),
        danger=(0, 0, 255),
        numeral=(0, 255, 255),
        msg=(200, 200, 200),
        option=(255, 0, 255),
        irq=(255, 0, 255),
        cursor_style="outline",
    ),
    "classic": MenuTheme(
        id="classic",
        label="Classic",
        # The original pre-0.3 menu palette and layout.  This theme is drawn
        # by the legacy renderer in MenuControlledEnd.
        background=(0, 0, 0),
        header_bg=(0, 0, 0),
        surface=(0, 0, 0),
        surface_alt=(40, 40, 40),
        border=(90, 90, 90),
        footer_bg=(0, 0, 0),
        text=(255, 255, 255),
        text_dim=(128, 128, 128),
        accent=(180, 130, 70),       # steelblue
        accent_text=(0, 0, 0),
        success=(47, 107, 85),       # darkolivegreen
        warning=(11, 134, 184),      # darkgoldenrod
        danger=(0, 0, 139),          # darkred
        numeral=(11, 134, 184),      # darkgoldenrod
        msg=(128, 128, 128),         # gray
        option=(147, 112, 219),      # palevioletred
        irq=(204, 50, 153),          # darkorchid
        cursor_style="fill",
        render_style="classic",
        disabled_cursor=(169, 169, 169),  # darkgray
    ),
    "industrial": MenuTheme(
        id="industrial",
        label="Industrial",
        background=(30, 20, 14),
        header_bg=(45, 30, 20),
        surface=(55, 38, 25),
        surface_alt=(75, 50, 30),
        border=(110, 80, 45),
        footer_bg=(38, 26, 17),
        text=(235, 220, 180),
        text_dim=(150, 130, 95),
        accent=(0, 170, 255),
        accent_text=(20, 15, 10),
        success=(80, 170, 90),
        warning=(0, 190, 255),
        danger=(70, 70, 210),
        numeral=(0, 190, 255),
        msg=(160, 160, 120),
        option=(120, 150, 220),
        irq=(90, 90, 230),
        cursor_style="bar",
    ),
}


def get_theme(theme_id: str | None) -> MenuTheme:
    if theme_id in THEMES:
        return THEMES[theme_id]
    if theme_id is not None:
        logger.warning("unknown menu theme %r; falling back to %s",
                       theme_id, DEFAULT_THEME_ID)
    return THEMES[DEFAULT_THEME_ID]


def theme_id_from_options(tree: dict[str, Any]) -> str:
    """Find the ``theme`` option in a raw menu tree."""

    for sections in tree.values():
        if not isinstance(sections, dict):
            continue
        for section in sections.values():
            if not isinstance(section, dict):
                continue
            for option in section.get("options", []):
                if not isinstance(option, dict) or option.get("id") != "theme":
                    continue
                value = option.get("value")
                if isinstance(value, dict):
                    value = value.get("value")
                if isinstance(value, str) and value:
                    return value
    return DEFAULT_THEME_ID
