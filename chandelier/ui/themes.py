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
    # "modern" | "classic" | "aero" | "luna" | "win2000" | "win7" | "win10"
    render_style: str = "modern"
    disabled_cursor: tuple[int, int, int] | None = None
    header_text: tuple[int, int, int] | None = None
    header_text_dim: tuple[int, int, int] | None = None
    selection_text: tuple[int, int, int] | None = None

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
    "aero": MenuTheme(
        id="aero",
        label="Aero",
        # Vista/7 Aero glass: a sky-blue chrome above pale, translucent
        # content cards.  All colours are BGR.
        background=(246, 234, 217),      # #D9EAF6 desktop blue
        header_bg=(177, 127, 60),        # #3C7FB1 title bar
        surface=(255, 252, 248),         # bright glass card
        surface_alt=(252, 242, 230),     # pale blue tint
        border=(222, 192, 160),          # #A0C0DE glass edge
        footer_bg=(248, 236, 222),       # #DEECF8 status strip
        text=(42, 32, 24),
        text_dim=(140, 122, 104),
        accent=(190, 124, 54),           # #367CBE selection blue
        accent_text=(255, 255, 255),
        success=(70, 150, 45),
        warning=(0, 140, 210),
        danger=(45, 45, 195),
        numeral=(160, 95, 25),
        msg=(115, 100, 85),
        option=(140, 70, 90),
        irq=(150, 50, 140),
        cursor_style="fill",
        render_style="aero",
        header_text=(255, 255, 255),
        header_text_dim=(255, 238, 218),
        selection_text=(255, 255, 255),
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
    "xp_luna": MenuTheme(
        id="xp_luna",
        label="Windows XP",
        # Windows XP "Luna" blue: warm window grey, blue title bar,
        # navy selection and pale blue highlights (all BGR).
        background=(216, 233, 236),
        header_bg=(227, 84, 0),
        surface=(255, 255, 255),
        surface_alt=(255, 240, 224),
        border=(180, 180, 180),
        footer_bg=(240, 214, 200),
        text=(20, 20, 20),
        text_dim=(110, 110, 110),
        accent=(197, 106, 49),
        accent_text=(255, 255, 255),
        success=(80, 175, 76),
        warning=(0, 165, 255),
        danger=(40, 40, 200),
        numeral=(160, 90, 20),
        msg=(100, 100, 100),
        option=(140, 60, 20),
        irq=(150, 40, 140),
        cursor_style="fill",
        render_style="luna",
        header_text=(255, 255, 255),
        header_text_dim=(255, 235, 220),
    ),
    "win2000": MenuTheme(
        id="win2000",
        label="Windows 2000",
        # Classic grey 3D UI with a navy title bar and bevel edges (BGR).
        background=(192, 192, 192),
        header_bg=(128, 0, 0),
        surface=(192, 192, 192),
        surface_alt=(200, 208, 212),
        border=(128, 128, 128),
        footer_bg=(192, 192, 192),
        text=(0, 0, 0),
        text_dim=(64, 64, 64),
        accent=(128, 0, 0),
        accent_text=(255, 255, 255),
        success=(0, 128, 0),
        warning=(0, 128, 255),
        danger=(0, 0, 192),
        numeral=(128, 0, 0),
        msg=(80, 80, 80),
        option=(128, 0, 128),
        irq=(128, 0, 128),
        cursor_style="fill",
        render_style="win2000",
        header_text=(255, 255, 255),
        header_text_dim=(200, 200, 200),
    ),
    "win7": MenuTheme(
        id="win7",
        label="Windows 7",
        # Windows 7 Aero glass: soft blue title bar, white window with pale
        # blue selection (all BGR).
        background=(240, 240, 240),
        header_bg=(160, 100, 50),
        surface=(255, 255, 255),
        surface_alt=(250, 235, 220),
        border=(190, 190, 190),
        footer_bg=(232, 232, 232),
        text=(20, 20, 20),
        text_dim=(100, 100, 100),
        accent=(255, 153, 51),
        accent_text=(255, 255, 255),
        success=(80, 160, 70),
        warning=(0, 150, 230),
        danger=(40, 40, 200),
        numeral=(200, 120, 20),
        msg=(90, 90, 90),
        option=(160, 80, 20),
        irq=(160, 50, 140),
        cursor_style="fill",
        render_style="win7",
        header_text=(255, 255, 255),
        header_text_dim=(225, 235, 245),
        selection_text=(20, 20, 20),
    ),
    "win10": MenuTheme(
        id="win10",
        label="Windows 10",
        # Windows 10 dark Fluent: near-black surfaces with the #0078D7
        # accent blue (BGR).
        background=(32, 32, 32),
        header_bg=(32, 32, 32),
        surface=(45, 45, 45),
        surface_alt=(58, 58, 58),
        border=(70, 70, 70),
        footer_bg=(32, 32, 32),
        text=(245, 245, 245),
        text_dim=(160, 160, 160),
        accent=(215, 120, 0),
        accent_text=(255, 255, 255),
        success=(70, 180, 70),
        warning=(0, 170, 255),
        danger=(60, 60, 220),
        numeral=(0, 170, 255),
        msg=(160, 160, 160),
        option=(200, 120, 220),
        irq=(230, 90, 200),
        cursor_style="fill",
        render_style="win10",
        header_text=(255, 255, 255),
        header_text_dim=(200, 200, 200),
        selection_text=(255, 255, 255),
    ),
}


_active_theme_id = DEFAULT_THEME_ID


def set_active_theme(theme_id: str | None) -> str:
    """Remember the menu theme for non-menu screens (system monitor, ...)."""

    global _active_theme_id
    _active_theme_id = theme_id if theme_id in THEMES else DEFAULT_THEME_ID
    return _active_theme_id


def active_theme_id() -> str:
    return _active_theme_id


def get_active_theme() -> "MenuTheme":
    return get_theme(_active_theme_id)


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
