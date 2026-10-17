"""Default menu fragments added by newer releases.

``ensure_menu_defaults`` upgrades an existing ``menus.json`` in memory without
overwriting any user value: missing sections and options are added by id, and
the function is idempotent.  The caller decides whether to persist the result.
"""

from __future__ import annotations

import copy
from typing import Any

DEFAULT_SECTIONS: dict[str, dict[str, dict[str, Any]]] = {
    "camera": {
        "display": {
            "title": "Display",
            "options": [
                {
                    "id": "theme",
                    "content": "Theme",
                    "type": "option",
                    "icon": "theme",
                    "value": {"content": "Dark Pro", "value": "dark_pro"},
                    "options": [
                        {"content": "Dark Pro", "value": "dark_pro"},
                        {"content": "High Contrast", "value": "high_contrast"},
                        {"content": "Industrial", "value": "industrial"},
                        {"content": "Classic", "value": "classic"},
                    ],
                },
                {
                    "id": "show_grid",
                    "content": "Rule of Thirds",
                    "type": "bool",
                    "icon": "grid",
                    "value": False,
                },
                {
                    "id": "show_center",
                    "content": "Center Mark",
                    "type": "bool",
                    "icon": "focus",
                    "value": False,
                },
                {
                    "id": "show_zebra",
                    "content": "Zebra",
                    "type": "bool",
                    "icon": "zebra",
                    "value": False,
                },
                {
                    "id": "zebra_threshold",
                    "content": "Zebra Level",
                    "type": "numeral",
                    "icon": "zebra",
                    "value": 235,
                    "min": 180,
                    "max": 255,
                    "step": 5,
                },
                {
                    "id": "show_status_bar",
                    "content": "Status Bar",
                    "type": "bool",
                    "icon": "monitor",
                    "value": False,
                },
            ],
        },
        "capture": {
            "title": "Capture",
            "options": [
                {
                    "id": "capture_mode",
                    "content": "Mode",
                    "type": "option",
                    "icon": "timer",
                    "value": {"content": "Single", "value": 0},
                    "options": [
                        {"content": "Single", "value": 0},
                        {"content": "Burst", "value": 1},
                        {"content": "Interval", "value": 2},
                    ],
                },
                {
                    "id": "burst_count",
                    "content": "Burst Count",
                    "type": "numeral",
                    "icon": "burst",
                    "value": 3,
                    "min": 2,
                    "max": 20,
                    "step": 1,
                },
                {
                    "id": "burst_interval",
                    "content": "Burst Gap",
                    "type": "numeral",
                    "icon": "timer",
                    "value": 0.5,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "step_options": [0.1, 0.5, 1.0],
                    "format": "seconds",
                },
                {
                    "id": "interval_count",
                    "content": "Interval Count",
                    "type": "numeral",
                    "icon": "burst",
                    "value": 10,
                    "min": 1,
                    "max": 999,
                    "step": 1,
                },
                {
                    "id": "interval_seconds",
                    "content": "Interval",
                    "type": "numeral",
                    "icon": "timer",
                    "value": 5,
                    "min": 1,
                    "max": 3600,
                    "step": 1,
                    "format": "seconds",
                },
            ],
        },
        "manual": {
            "title": "Manual",
            "options": [
                {
                    "id": "exp_compensation",
                    "content": "EV Comp",
                    "type": "numeral",
                    "icon": "sun",
                    "value": 0.0,
                    "min": -4.0,
                    "max": 4.0,
                    "step": 0.5,
                    "step_options": [0.1, 0.5],
                    "format": "ev",
                },
                {
                    "id": "ae_lock",
                    "content": "AE Lock",
                    "type": "bool",
                    "icon": "sliders",
                    "value": False,
                },
                {
                    "id": "awb_lock",
                    "content": "AWB Lock",
                    "type": "bool",
                    "icon": "wb",
                    "value": False,
                },
                {
                    "id": "manual_focus",
                    "content": "Manual Focus",
                    "type": "bool",
                    "icon": "focus",
                    "value": False,
                    "set_enable": ["focus_position"],
                },
                {
                    "id": "focus_position",
                    "content": "Focus Pos",
                    "type": "numeral",
                    "icon": "focus",
                    "value": 500,
                    "min": 0,
                    "max": 1000,
                    "step": 10,
                    "enable": False,
                },
                {
                    "id": "saturation",
                    "content": "Saturation",
                    "type": "numeral",
                    "icon": "sliders",
                    "value": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                },
                {
                    "id": "contrast",
                    "content": "Contrast",
                    "type": "numeral",
                    "icon": "sliders",
                    "value": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                },
                {
                    "id": "sharpness",
                    "content": "Sharpness",
                    "type": "numeral",
                    "icon": "sliders",
                    "value": 1.0,
                    "min": 0.0,
                    "max": 4.0,
                    "step": 0.1,
                },
            ],
        },
        "record": {
            "title": "Record",
            "options": [
                {
                    "id": "record_mode",
                    "content": "Shutter Records",
                    "type": "bool",
                    "icon": "record",
                    "value": False,
                },
                {
                    "id": "record_resolution",
                    "content": "Video Size",
                    "type": "option",
                    "icon": "video",
                    "value": {"content": "1280x720", "value": [1280, 720]},
                    "options": [
                        {"content": "640x480", "value": [640, 480]},
                        {"content": "1280x720", "value": [1280, 720]},
                        {"content": "1920x1080", "value": [1920, 1080]},
                    ],
                },
            ],
        },
    },
    "gallery": {
        "player": {
            "title": "Player",
            "options": [
                {
                    "id": "show_progress",
                    "content": "Show Progress",
                    "type": "bool",
                    "icon": "play",
                    "value": True,
                },
                {
                    "id": "loop_playback",
                    "content": "Loop Video",
                    "type": "bool",
                    "icon": "loop",
                    "value": False,
                },
                {
                    "id": "seek_seconds",
                    "content": "Seek Step",
                    "type": "numeral",
                    "icon": "play",
                    "value": 5,
                    "min": 1,
                    "max": 60,
                    "step": 1,
                    "format": "seconds",
                },
            ],
        }
    },
}

DEFAULT_LINKS: dict[str, dict[str, list[dict[str, Any]]]] = {
    "camera": {
        "camera_settings": [
            {
                "id": "display",
                "content": "Display",
                "type": "menu",
                "icon": "display",
                "value": "display",
            },
            {
                "id": "capture",
                "content": "Capture",
                "type": "menu",
                "icon": "timer",
                "value": "capture",
            },
            {
                "id": "manual",
                "content": "Manual",
                "type": "menu",
                "icon": "sliders",
                "value": "manual",
            },
            {
                "id": "record",
                "content": "Record",
                "type": "menu",
                "icon": "video",
                "value": "record",
            },
        ]
    },
    "gallery": {
        "0": [
            {
                "id": "player",
                "content": "Player",
                "type": "menu",
                "icon": "play",
                "value": "player",
            }
        ]
    },
}


def _append_missing_options(
    section: dict[str, Any], defaults: list[dict[str, Any]]
) -> bool:
    options = section.get("options")
    if not isinstance(options, list):
        return False

    existing_ids = {
        option.get("id")
        for option in options
        if isinstance(option, dict) and isinstance(option.get("id"), str)
    }
    changed = False
    for default in defaults:
        if default["id"] in existing_ids:
            continue
        options.append(copy.deepcopy(default))
        existing_ids.add(default["id"])
        changed = True
    return changed


def _append_missing_theme_choices(tree: dict[str, Any]) -> bool:
    """Add newly introduced theme choices without touching the current one."""

    changed = False
    for group in tree.values():
        if not isinstance(group, dict):
            continue
        for section in group.values():
            if not isinstance(section, dict):
                continue
            for option in section.get("options", []):
                if not isinstance(option, dict) or option.get("id") != "theme":
                    continue
                choices = option.get("options")
                if not isinstance(choices, list):
                    continue
                values = {
                    choice.get("value")
                    for choice in choices
                    if isinstance(choice, dict)
                }
                for default in (
                    {"content": "Classic", "value": "classic"},
                ):
                    if default["value"] not in values:
                        choices.append(copy.deepcopy(default))
                        values.add(default["value"])
                        changed = True
    return changed


def ensure_menu_defaults(tree: dict[str, Any]) -> bool:
    """Add missing sections/options/links.  Returns True when modified."""

    changed = False
    for group_name, sections in DEFAULT_SECTIONS.items():
        group = tree.get(group_name)
        if not isinstance(group, dict):
            continue
        for section_name, default_section in sections.items():
            section = group.get(section_name)
            if not isinstance(section, dict):
                group[section_name] = copy.deepcopy(default_section)
                changed = True
            else:
                changed |= _append_missing_options(
                    section, default_section.get("options", [])
                )

    for group_name, links_by_section in DEFAULT_LINKS.items():
        group = tree.get(group_name)
        if not isinstance(group, dict):
            continue
        for section_name, links in links_by_section.items():
            section = group.get(section_name)
            # Do not inject links into a custom root menu that has no
            # camera_settings section; the user's layout is left untouched.
            if isinstance(section, dict):
                changed |= _append_missing_options(section, links)

    changed |= _append_missing_theme_choices(tree)

    return changed
