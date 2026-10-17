"""Default menu fragments added by newer releases.

``ensure_menu_defaults`` upgrades an existing ``menus.json`` in memory without
overwriting any user value: missing sections and options are added by id, and
the function is idempotent.  The caller decides whether to persist the result.
"""

from __future__ import annotations

import copy
from typing import Any

THEME_CHOICES = [
    {"content": "Dark Pro", "value": "dark_pro"},
    {"content": "High Contrast", "value": "high_contrast"},
    {"content": "Industrial", "value": "industrial"},
    {"content": "Classic", "value": "classic"},
]

DEFAULT_SECTIONS: dict[str, dict[str, dict[str, Any]]] = {
    "camera": {
        "display": {
            "title": "Display",
            "options": [
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
        "localizer": {
            "title": "Localizer",
            "options": [
                {
                    "id": "loc_n_sources",
                    "content": "Sources",
                    "type": "numeral",
                    "icon": "sliders",
                    "value": 3,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                },
                {
                    "id": "loc_confidence",
                    "content": "Confidence",
                    "type": "numeral",
                    "icon": "sliders",
                    "value": 0.03,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                },
                {
                    "id": "loc_max_chunks",
                    "content": "Max Chunks",
                    "type": "numeral",
                    "icon": "timer",
                    "value": 5,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                },
                {
                    "id": "loc_speech_threshold",
                    "content": "Speech Level",
                    "type": "numeral",
                    "icon": "sliders",
                    "value": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                },
                {
                    "id": "loc_mic_rotation",
                    "content": "Mic Rotation",
                    "type": "numeral",
                    "icon": "focus",
                    "value": 0,
                    "min": 0,
                    "max": 359,
                    "step": 1,
                    "format": "degrees",
                },
                {
                    "id": "loc_delay_max",
                    "content": "Delay Max",
                    "type": "numeral",
                    "icon": "timer",
                    "value": 0.0,
                    "min": 0.0,
                    "max": 16.0,
                    "step": 0.1,
                },
                {
                    "id": "loc_delay_bins",
                    "content": "Delay Bins",
                    "type": "numeral",
                    "icon": "sliders",
                    "value": 50,
                    "min": 8,
                    "max": 512,
                    "step": 1,
                },
            ],
        },
        "system": {
            "title": "System",
            "options": [
                {
                    "id": "language",
                    "content": "Language",
                    "type": "option",
                    "icon": "display",
                    "value": {"content": "English", "value": "en"},
                    "options": [
                        {"content": "English", "value": "en"},
                        {"content": "简体中文", "value": "zh_CN"},
                        {"content": "繁體中文", "value": "zh_TW"},
                        {"content": "Русский", "value": "ru"},
                    ],
                },
                {
                    "id": "theme",
                    "content": "Theme",
                    "type": "option",
                    "icon": "theme",
                    "value": {"content": "Dark Pro", "value": "dark_pro"},
                    "options": copy.deepcopy(THEME_CHOICES),
                },
                {
                    "id": "reboot",
                    "content": "Reboot System",
                    "type": "msg",
                    "icon": "power",
                    "badge": "REBOOT",
                    "receiver": "controller",
                    "value": "restart",
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

# Links that moved to a different section in a newer release.  They are
# removed from the old location so a migrated menu does not show duplicates.
MOVED_LINKS: dict[str, dict[str, dict[str, str]]] = {
    "camera": {
        "camera_settings": {
            "id": "localizer",
            "value": "localizer",
        },
    },
}

# Options that moved between sections.  The existing value is preserved.
MOVED_OPTIONS: dict[str, dict[str, dict[str, str]]] = {
    "camera": {
        "theme": {"from": "display", "to": "system"},
    },
}

# Canonical order by expected daily use: frequently changed controls first,
# navigation/feature entries next, diagnostics and one-off settings last,
# destructive actions at the very end.
PREFERRED_ORDER: dict[str, dict[str, list[str]]] = {
    "camera": {
        "0": [
            "camera_settings",
            "gallery",
            "localizer",
            "system_monitor",
            "system",
        ],
        "camera_settings": [
            "exposure",
            "focus",
            "white_balance",
            "hdr",
            "delay",
            "capture",
            "resolution",
            "display",
            "manual",
            "record",
            "pict_format",
            "save_metadata",
            "dng_enable",
            "watermark",
        ],
        "exposure": [
            "auto_expose",
            "show_hist",
            "exposure_time",
            "analogue_gain",
            "exposure_mode",
            "metering_mode",
            "constraint_mode",
            "flicker",
        ],
        "focus": ["mf_assist"],
        "white_balance": ["awb", "awb_mode", "color_gains"],
        "color_gains": ["red_gain", "blue_gain"],
        "flicker_menu": ["flicker_mode", "flicker_period"],
        "display": [
            "show_grid",
            "show_center",
            "show_zebra",
            "zebra_threshold",
            "show_status_bar",
        ],
        "capture": [
            "capture_mode",
            "burst_count",
            "burst_interval",
            "interval_count",
            "interval_seconds",
        ],
        "manual": [
            "exp_compensation",
            "ae_lock",
            "awb_lock",
            "manual_focus",
            "focus_position",
            "saturation",
            "contrast",
            "sharpness",
        ],
        "record": ["record_mode", "record_resolution"],
        "localizer": [
            "loc_n_sources",
            "loc_confidence",
            "loc_mic_rotation",
            "loc_speech_threshold",
            "loc_max_chunks",
            "loc_delay_max",
            "loc_delay_bins",
        ],
        "system": ["language", "theme", "reboot"],
    },
    "gallery": {
        "0": ["player", "show_hist", "delete"],
        "player": ["show_progress", "loop_playback", "seek_seconds"],
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
        ],
        # Localizer is a top-level entry next to Camera Settings/Gallery/
        # System Monitor so it is directly reachable from the root menu.
        "0": [
            {
                "id": "localizer",
                "content": "Localizer",
                "type": "menu",
                "icon": "sliders",
                "value": "localizer",
            },
            {
                "id": "system",
                "content": "System",
                "type": "menu",
                "icon": "display",
                "value": "system",
            },
        ],
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


def _ensure_theme_choices(tree: dict[str, Any]) -> bool:
    """Add newly introduced theme choices without overwriting an old value."""

    changed = False
    for group in tree.values():
        if not isinstance(group, dict):
            continue
        for section in group.values():
            if not isinstance(section, dict):
                continue
            for option in section.get("options", []):
                if not (
                    isinstance(option, dict)
                    and option.get("id") == "theme"
                    and isinstance(option.get("options"), list)
                ):
                    continue
                existing = {
                    choice.get("value")
                    for choice in option["options"]
                    if isinstance(choice, dict)
                }
                for choice in THEME_CHOICES:
                    if choice["value"] not in existing:
                        option["options"].append(copy.deepcopy(choice))
                        existing.add(choice["value"])
                        changed = True
    return changed


def _reorder_options(
    section: dict[str, Any], order: list[str]
) -> bool:
    options = section.get("options")
    if not isinstance(options, list):
        return False

    positions = {option_id: index for index, option_id in enumerate(order)}
    known = [
        option
        for option in options
        if isinstance(option, dict) and option.get("id") in positions
    ]
    unknown = [
        option
        for option in options
        if not (isinstance(option, dict) and option.get("id") in positions)
    ]
    known.sort(key=lambda option: positions[option["id"]])
    ordered = known + unknown

    if [id(option) for option in ordered] == [id(option) for option in options]:
        return False
    section["options"] = ordered
    return True


def ensure_menu_defaults(tree: dict[str, Any]) -> bool:
    """Add missing sections/options/links.  Returns True when modified."""

    changed = False

    # Move options that changed sections before adding defaults, so a user's
    # existing value wins over the newly added default.
    for group_name, moves in MOVED_OPTIONS.items():
        group = tree.get(group_name)
        if not isinstance(group, dict):
            continue
        for option_id, move in moves.items():
            source = group.get(move["from"])
            if not isinstance(source, dict):
                continue
            options = source.get("options")
            if not isinstance(options, list):
                continue
            moved = None
            kept = []
            for option in options:
                if isinstance(option, dict) and option.get("id") == option_id:
                    moved = option
                else:
                    kept.append(option)
            if moved is None:
                continue
            source["options"] = kept
            target = group.get(move["to"])
            if not isinstance(target, dict):
                defaults = DEFAULT_SECTIONS.get(group_name, {}).get(
                    move["to"], {}
                )
                target = {
                    "title": defaults.get("title", move["to"].title()),
                    "options": [],
                }
                group[move["to"]] = target
            target_options = target.setdefault("options", [])
            if not any(
                isinstance(option, dict)
                and option.get("id") == option_id
                for option in target_options
            ):
                target_options.append(moved)
            changed = True

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
            # Root links are only added to the standard menu shape; a custom
            # root without a camera_settings section is left untouched.
            if section_name == "0" and "camera_settings" not in group:
                continue
            section = group.get(section_name)
            if isinstance(section, dict):
                changed |= _append_missing_options(section, links)

    for group_name, sections in MOVED_LINKS.items():
        group = tree.get(group_name)
        if not isinstance(group, dict):
            continue
        for section_name, link in sections.items():
            section = group.get(section_name)
            if not isinstance(section, dict):
                continue
            options = section.get("options")
            if not isinstance(options, list):
                continue
            moved_id = link["id"]
            moved_value = link["value"]
            kept = [
                option
                for option in options
                if not (
                    isinstance(option, dict)
                    and option.get("id") == moved_id
                    and str(option.get("type", "")).lower() == "menu"
                    and option.get("value") == moved_value
                )
            ]
            if len(kept) != len(options):
                section["options"] = kept
                changed = True

    changed |= _ensure_theme_choices(tree)

    for group_name, sections in PREFERRED_ORDER.items():
        group = tree.get(group_name)
        if not isinstance(group, dict):
            continue
        for section_name, order in sections.items():
            section = group.get(section_name)
            if isinstance(section, dict):
                changed |= _reorder_options(section, order)

    return changed
