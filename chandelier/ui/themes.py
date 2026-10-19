"""Backward-compatible import location for theme data.

The canonical theme registry lives in :mod:`chandelier.themes`; this module
re-exports it for imports that still use the old path.
"""

from chandelier.themes import (
    DEFAULT_THEME_ID,
    THEMES,
    THEME_CHOICES,
    MenuTheme,
    active_theme_id,
    get_active_theme,
    get_theme,
    set_active_theme,
    theme_id_from_options,
)

__all__ = [
    "DEFAULT_THEME_ID",
    "THEMES",
    "THEME_CHOICES",
    "MenuTheme",
    "active_theme_id",
    "get_active_theme",
    "get_theme",
    "set_active_theme",
    "theme_id_from_options",
]
