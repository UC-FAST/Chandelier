"""Default menu fragments added by newer releases.

``ensure_menu_defaults`` upgrades an existing ``menus.json`` in memory without
overwriting any user value: missing sections and options are added by id, and
the function is idempotent.  The caller decides whether to persist the result.
"""

from __future__ import annotations

import copy
from decimal import Decimal, InvalidOperation
from typing import Any

from chandelier.themes import DEFAULT_THEME_ID, THEME_CHOICES

# Schema fragments keyed by group -> section.  Option ``content`` strings are
# English translation keys; the UI translates them at render time.  The
# layout mirrors menus.json, and the same entry may be referenced by the
# migration helpers below, so keep values deep-copied when inserting.
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
                    "icon": "crosshair",
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
                    "id": "status_bar_position",
                    "content": "Status Bar",
                    "type": "option",
                    "icon": "monitor",
                    "value": {"content": "Off", "value": "off"},
                    "options": [
                        {"content": "Off", "value": "off"},
                        {"content": "Top + Bottom", "value": "top_bottom"},
                        {"content": "Sides", "value": "sides"},
                    ],
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
                    "enable_if": {"id": "auto_expose", "value": True},
                },
                {
                    "id": "ae_lock",
                    "content": "AE Lock",
                    "type": "bool",
                    "icon": "lock",
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
                    "id": "saturation",
                    "content": "Saturation",
                    "type": "numeral",
                    "icon": "droplet",
                    "value": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                },
                {
                    "id": "contrast",
                    "content": "Contrast",
                    "type": "numeral",
                    "icon": "contrast",
                    "value": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                },
                {
                    "id": "sharpness",
                    "content": "Sharpness",
                    "type": "numeral",
                    "icon": "sharp",
                    "value": 1.0,
                    "min": 0.0,
                    "max": 4.0,
                    "step": 0.1,
                },
            ],
        },
        "focus": {
            "title": "Focus",
            "options": [
                {
                    "id": "af_mode",
                    "content": "AF Mode",
                    "type": "option",
                    "icon": "focus",
                    "value": {"content": "Continuous AF", "value": 2},
                    "options": [
                        {"content": "Continuous AF", "value": 2},
                        {"content": "Single AF", "value": 1},
                        {"content": "Manual", "value": 0},
                    ],
                },
                {
                    "id": "af_trigger",
                    "content": "AF Trigger",
                    "type": "msg",
                    "icon": "focus",
                    "badge": "AF",
                    "receiver": "camera",
                    "value": "autofocus",
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
                    "step_options": [1, 10, 50],
                    "format": "lens",
                    "enable": False,
                    "enable_if": {"id": "af_mode", "value": 0},
                },
                {
                    "id": "af_speed",
                    "content": "AF Speed",
                    "type": "option",
                    "icon": "speed",
                    "value": {"content": "Normal", "value": 0},
                    "options": [
                        {"content": "Normal", "value": 0},
                        {"content": "Fast", "value": 1},
                    ],
                },
                {
                    "id": "af_range",
                    "content": "AF Range",
                    "type": "option",
                    "icon": "frame",
                    "value": {"content": "Normal", "value": 0},
                    "options": [
                        {"content": "Normal", "value": 0},
                        {"content": "Macro", "value": 1},
                        {"content": "Full", "value": 2},
                    ],
                },
                {
                    "id": "mf_assist",
                    "content": "MF Assist",
                    "type": "bool",
                    "icon": "focus",
                    "value": False,
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
                    "icon": "sound",
                    "value": 3,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                },
                {
                    "id": "loc_confidence",
                    "content": "Confidence",
                    "type": "numeral",
                    "icon": "signal",
                    "value": 0.03,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                },
                {
                    "id": "loc_max_chunks",
                    "content": "Max Chunks",
                    "type": "numeral",
                    "icon": "memory",
                    "value": 5,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                },
                {
                    "id": "loc_speech_threshold",
                    "content": "Speech Level",
                    "type": "numeral",
                    "icon": "sound",
                    "value": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                },
                {
                    "id": "loc_mic_rotation",
                    "content": "Mic Rotation",
                    "type": "numeral",
                    "icon": "angle",
                    "value": 0,
                    "min": 0,
                    "max": 359,
                    "step": 1,
                    "format": "degrees",
                },
                {
                    "id": "loc_marker_fov",
                    "content": "Camera FOV",
                    "type": "numeral",
                    "icon": "frame",
                    "value": 83,
                    "min": 1,
                    "max": 179,
                    "step": 1,
                    "step_options": [1, 5],
                    "format": "degrees",
                },
                {
                    "id": "loc_marker_azimuth",
                    "content": "Marker Yaw",
                    "type": "numeral",
                    "icon": "angle",
                    "value": 0,
                    "min": 0,
                    "max": 359,
                    "step": 1,
                    "step_options": [1, 10],
                    "format": "degrees",
                },
                {
                    "id": "loc_marker_elevation",
                    "content": "Marker Elev",
                    "type": "numeral",
                    "icon": "angle",
                    "value": 0,
                    "min": -90,
                    "max": 90,
                    "step": 1,
                    "step_options": [1, 5],
                    "format": "degrees",
                },
                {
                    "id": "loc_marker_x_gain",
                    "content": "X Gain",
                    "type": "numeral",
                    "icon": "gain",
                    "value": 0.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.1,
                },
                {
                    "id": "loc_marker_y_gain",
                    "content": "Y Gain",
                    "type": "numeral",
                    "icon": "gain",
                    "value": 0.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.1,
                },
                {
                    "id": "loc_marker_mic_x",
                    "content": "Mic X",
                    "type": "numeral",
                    "icon": "mic",
                    "value": 0.0,
                    "min": -30.0,
                    "max": 30.0,
                    "step": 0.1,
                    "step_options": [0.1, 0.5, 1.0],
                    "format": "cm",
                },
                {
                    "id": "loc_marker_mic_y",
                    "content": "Mic Y",
                    "type": "numeral",
                    "icon": "mic",
                    "value": 0.0,
                    "min": -30.0,
                    "max": 30.0,
                    "step": 0.1,
                    "step_options": [0.1, 0.5, 1.0],
                    "format": "cm",
                },
                {
                    "id": "loc_marker_mic_z",
                    "content": "Mic Z",
                    "type": "numeral",
                    "icon": "mic",
                    "value": 0.0,
                    "min": -30.0,
                    "max": 30.0,
                    "step": 0.1,
                    "step_options": [0.1, 0.5, 1.0],
                    "format": "cm",
                },
                {
                    "id": "loc_talk_distance",
                    "content": "Talk Distance",
                    "type": "numeral",
                    "icon": "ruler",
                    "value": 0.0,
                    "min": 0.0,
                    "max": 500.0,
                    "step": 1,
                    "format": "distance",
                },
                {
                    "id": "loc_delay_max",
                    "content": "Delay Max",
                    "type": "numeral",
                    "icon": "clock",
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
                    "icon": "globe",
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
                    "id": "web_server",
                    "content": "Web Server",
                    "type": "bool",
                    "icon": "server",
                    "receiver": "camera",
                    "value": True,
                },
                {
                    "id": "restart_service",
                    "content": "Restart Service",
                    "type": "msg",
                    "icon": "service",
                    "badge": "RESTART",
                    "receiver": "controller",
                    "value": "restart_service",
                },
                {
                    "id": "power",
                    "content": "Power Action",
                    "type": "option",
                    "icon": "power",
                    "receiver": "controller",
                    "value": {"content": "Reboot", "value": "reboot"},
                    "options": [
                        {"content": "Reboot", "value": "reboot"},
                        {"content": "Shutdown", "value": "shutdown"},
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
                    "icon": "seek",
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
        "focus_position": {"from": "manual", "to": "focus"},
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
        "focus": [
            "af_mode",
            "af_trigger",
            "focus_position",
            "af_speed",
            "af_range",
            "mf_assist",
        ],
        "white_balance": ["awb", "awb_mode", "color_gains"],
        "color_gains": ["red_gain", "blue_gain"],
        "flicker_menu": ["flicker_mode", "flicker_period"],
        "display": [
            "show_grid",
            "show_center",
            "show_zebra",
            "zebra_threshold",
            "status_bar_position",
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
            "saturation",
            "contrast",
            "sharpness",
        ],
        "record": ["record_mode", "record_resolution"],
        "localizer": [
            "loc_n_sources",
            "loc_confidence",
            "loc_mic_rotation",
            "loc_marker_fov",
            "loc_marker_azimuth",
            "loc_marker_elevation",
            "loc_marker_x_gain",
            "loc_marker_y_gain",
            "loc_marker_mic_x",
            "loc_marker_mic_y",
            "loc_marker_mic_z",
            "loc_talk_distance",
            "loc_speech_threshold",
            "loc_max_chunks",
            "loc_delay_max",
            "loc_delay_bins",
        ],
        "system": [
            "language",
            "theme",
            "web_server",
            "restart_service",
            "power",
        ],
    },
    "gallery": {
        "0": ["player", "show_hist", "delete"],
        "player": ["show_progress", "loop_playback", "seek_seconds"],
    },
}

# Navigation entries added to existing menus.  Section key ``"0"`` denotes the
# group's root option list rather than a named section.
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
    """Append defaults whose id is absent, leaving existing options alone.

    Defaults are deep-copied so the module-level schema never aliases live
    menu state.  New options land at the end; ``PREFERRED_ORDER`` sorts them
    on the next pass.  Returns True when at least one option was added.
    """

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


def _migrate_focus_mode(tree: dict[str, Any]) -> bool:
    """Convert the legacy ``manual_focus`` bool into the new ``af_mode``."""

    group = tree.get("camera")
    if not isinstance(group, dict):
        return False
    manual = group.get("manual")
    if not isinstance(manual, dict):
        return False
    options = manual.get("options")
    if not isinstance(options, list):
        return False

    legacy = None
    kept = []
    for option in options:
        if isinstance(option, dict) and option.get("id") == "manual_focus":
            legacy = option
        else:
            kept.append(option)
    if legacy is None:
        return False
    manual["options"] = kept

    focus = group.get("focus")
    if not isinstance(focus, dict):
        focus = copy.deepcopy(DEFAULT_SECTIONS["camera"]["focus"])
        group["focus"] = focus

    focus_options = focus.setdefault("options", [])
    af_mode = next(
        (
            option
            for option in focus_options
            if isinstance(option, dict) and option.get("id") == "af_mode"
        ),
        None,
    )
    if af_mode is None:
        default = next(
            option
            for option in DEFAULT_SECTIONS["camera"]["focus"]["options"]
            if option["id"] == "af_mode"
        )
        af_mode = copy.deepcopy(default)
        focus_options.append(af_mode)

    if bool(legacy.get("value")):
        manual_choice = next(
            choice
            for choice in af_mode.get("options", [])
            if choice.get("value") == 0
        )
        af_mode["value"] = copy.deepcopy(manual_choice)
    return True


def _merge_option_defaults(
    section: dict[str, Any], defaults: list[dict[str, Any]]
) -> bool:
    """Add newly introduced metadata to existing options by id."""

    options = section.get("options")
    if not isinstance(options, list):
        return False
    by_id = {
        option.get("id"): option
        for option in options
        if isinstance(option, dict) and isinstance(option.get("id"), str)
    }
    changed = False
    for default in defaults:
        option = by_id.get(default.get("id"))
        if option is None:
            continue
        for key, value in default.items():
            # Never overwrite user state: ``value`` and the editable choice
            # list belong to the user, and existing metadata may have been
            # customized too.  Only fill genuinely missing keys.
            if key in {"value", "options"} or key in option:
                continue
            option[key] = copy.deepcopy(value)
            changed = True
    return changed


def _migrate_power_option(tree: dict[str, Any]) -> bool:
    """Replace the old reboot-only MSG item with a combined power option."""

    group = tree.get("camera")
    if not isinstance(group, dict):
        return False
    system = group.get("system")
    if not isinstance(system, dict):
        return False
    options = system.get("options")
    if not isinstance(options, list):
        return False

    changed = False
    kept = []
    has_power = False
    for option in options:
        if not isinstance(option, dict):
            kept.append(option)
            continue
        if option.get("id") == "reboot":
            changed = True
            continue
        if option.get("id") == "power":
            has_power = True
        kept.append(option)
    system["options"] = kept

    if not has_power:
        default = next(
            option
            for option in DEFAULT_SECTIONS["camera"]["system"]["options"]
            if option["id"] == "power"
        )
        kept.append(copy.deepcopy(default))
        changed = True
    return changed


def _status_bar_default() -> dict[str, Any]:
    return next(
        option
        for option in DEFAULT_SECTIONS["camera"]["display"]["options"]
        if option["id"] == "status_bar_position"
    )


def _migrate_status_bar_option(tree: dict[str, Any]) -> bool:
    """Merge the legacy ``show_status_bar`` bool into the position option."""

    changed = False
    default = _status_bar_default()
    # The old single-row Top/Bottom modes are gone; keep old menus working
    # by moving them to the split Top + Bottom layout.
    legacy = {"top": "top_bottom", "bottom": "top_bottom"}
    valid = {"off", "top_bottom", "sides"}

    for group in tree.values():
        if not isinstance(group, dict):
            continue
        for section in group.values():
            if not isinstance(section, dict):
                continue
            options = section.get("options")
            if not isinstance(options, list):
                continue

            show = next(
                (
                    option
                    for option in options
                    if isinstance(option, dict)
                    and option.get("id") == "show_status_bar"
                ),
                None,
            )
            position = next(
                (
                    option
                    for option in options
                    if isinstance(option, dict)
                    and option.get("id") == "status_bar_position"
                ),
                None,
            )
            if show is None and position is None:
                continue

            before = copy.deepcopy(options)

            value = None
            if position is not None:
                raw = position.get("value")
                if isinstance(raw, dict):
                    raw = raw.get("value")
                if raw in legacy:
                    value = legacy[raw]
                elif raw in valid:
                    value = raw
            if show is not None:
                if not bool(show.get("value")):
                    value = "off"
                elif value in {None, "off"}:
                    value = "top_bottom"
            if value is None:
                value = "off"

            if position is None:
                position = copy.deepcopy(default)
                options.append(position)
            position["content"] = "Status Bar"
            position["type"] = "option"
            position["icon"] = "monitor"
            position["options"] = copy.deepcopy(default["options"])
            position.pop("enable_if", None)
            position.pop("enable", None)
            label = next(
                choice["content"]
                for choice in default["options"]
                if choice["value"] == value
            )
            position["value"] = {"content": label, "value": value}

            if show is not None:
                options.remove(show)

            if options != before:
                changed = True
    return changed


def _ensure_theme_choices(tree: dict[str, Any]) -> bool:
    """Replace the theme list with the current registry.

    Removed themes disappear, the order and labels match ``THEME_CHOICES``,
    and a stale selection falls back to ``DEFAULT_THEME_ID``.
    """

    changed = False
    canonical = [copy.deepcopy(choice) for choice in THEME_CHOICES]
    valid = {choice["value"]: choice for choice in canonical}
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
                if option["options"] != canonical:
                    option["options"] = [
                        copy.deepcopy(choice) for choice in canonical
                    ]
                    changed = True

                value = option.get("value")
                selected = (
                    value.get("value")
                    if isinstance(value, dict)
                    else value
                )
                choice = valid.get(selected, valid[DEFAULT_THEME_ID])
                if value != choice:
                    option["value"] = copy.deepcopy(choice)
                    changed = True
    return changed


def _reorder_options(
    section: dict[str, Any], order: list[str]
) -> bool:
    """Sort known option ids into canonical order, unknown ones last.

    Unknown ids (user additions or entries from a newer menu) keep their
    relative order after the known block.  Identity comparison avoids a deep
    equality pass over option dictionaries and ensures the caller only sees
    a change when the list actually moved.
    """

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


def _migrate_camera_control_ranges(tree: dict[str, Any]) -> bool:
    """Update camera control metadata that was too coarse in older menus."""

    camera = tree.get("camera")
    if not isinstance(camera, dict):
        return False

    updates = {
        "analogue_gain": {
            "step": 0.5,
            "step_options": [0.1, 0.5, 1],
        },
        "red_gain": {
            "min": 0.1,
            "step": 0.1,
            "step_options": [0.1, 0.5, 1],
        },
        "blue_gain": {
            "min": 0.1,
            "step": 0.1,
            "step_options": [0.1, 0.5, 1],
        },
        "exp_compensation": {
            "enable_if": {"id": "auto_expose", "value": True},
        },
        "hdr": {
            "enable_if": {"id": "auto_expose", "value": True},
        },
    }

    changed = False
    by_id = {}
    for section in camera.values():
        if not isinstance(section, dict):
            continue
        options = section.get("options")
        if not isinstance(options, list):
            continue
        for option in options:
            if not isinstance(option, dict):
                continue
            option_id = option.get("id")
            by_id[option_id] = option
            for key, value in updates.get(option_id, {}).items():
                if option.get(key) != value:
                    option[key] = copy.deepcopy(value)
                    changed = True
            if option_id in {"red_gain", "blue_gain"}:
                value = option.get("value")
                # Older menus allowed zero/negative gains, which produce a
                # black channel or fail camera validation; reset to unity.
                # ``bool`` is excluded because it is an ``int`` subclass.
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and value <= 0
                ):
                    option["value"] = 1.0
                    changed = True

    # Manual exposure wins over HDR when an old menu contains both.
    auto = by_id.get("auto_expose")
    hdr = by_id.get("hdr")
    if isinstance(auto, dict) and isinstance(hdr, dict):
        auto_value = auto.get("value")
        if isinstance(auto_value, dict):
            auto_value = auto_value.get("value")
        hdr_value = hdr.get("value")
        if isinstance(hdr_value, dict):
            hdr_value = hdr_value.get("value")
        if not bool(auto_value) and bool(hdr_value):
            off = next(
                (
                    choice
                    for choice in hdr.get("options", [])
                    if isinstance(choice, dict)
                    and choice.get("value") == 0
                ),
                None,
            )
            if off is not None and hdr.get("value") != off:
                hdr["value"] = copy.deepcopy(off)
                changed = True
    return changed


# (label, value) pairs where the value is the sensor's integer exposure time
# in microseconds.  The labels are the shutter speeds shown in the menu.
_EXPOSURE_PRESETS: tuple[tuple[str, int], ...] = (
    ("1/8000", 125),
    ("1/4000", 250),
    ("1/2000", 500),
    ("1/1000", 1_000),
    ("1/500", 2_000),
    ("1/250", 4_000),
    ("1/125", 8_000),
    ("1/60", 16_667),
    ("1/30", 33_333),
    ("1/25", 40_000),
    ("1/20", 50_000),
    ("1/15", 66_666),
    ("1/10 (Still)", 100_000),
    ("1/8 (Still)", 125_000),
    ("1/4 (Still)", 250_000),
    ("1/2 (Still)", 500_000),
    ("1s (Still)", 1_000_000),
    ("2s (Still)", 2_000_000),
    ("4s (Still)", 4_000_000),
    ("8s (Still)", 8_000_000),
    ("15s (Still)", 15_000_000),
    ("30s (Still)", 30_000_000),
)


def _migrate_exposure_presets(tree: dict[str, Any]) -> bool:
    """Replace the free-running exposure numeral with shutter presets."""

    camera = tree.get("camera")
    if not isinstance(camera, dict):
        return False

    exposure = None
    for section in camera.values():
        if not isinstance(section, dict):
            continue
        for option in section.get("options", []):
            if (
                isinstance(option, dict)
                and option.get("id") == "exposure_time"
            ):
                exposure = option
                break
        if exposure is not None:
            break
    if exposure is None:
        return False

    current = exposure.get("value")
    if isinstance(current, dict):
        current = current.get("value")
    try:
        current_number = int(current)
    except (TypeError, ValueError):
        current_number = None
    if current_number is None:
        choice = _EXPOSURE_PRESETS[0]
    else:
        # Snap to the nearest preset so a migrated menu keeps the user's
        # exposure as closely as the fixed list allows.
        choice = min(
            _EXPOSURE_PRESETS,
            key=lambda item: abs(item[1] - current_number),
        )

    expected_options = [
        {"content": content, "value": value}
        for content, value in _EXPOSURE_PRESETS
    ]
    expected_value = {
        "content": choice[0],
        "value": choice[1],
    }

    changed = False
    if exposure.get("type") != "option":
        exposure["type"] = "option"
        changed = True
    if exposure.get("options") != expected_options:
        exposure["options"] = copy.deepcopy(expected_options)
        changed = True
    if exposure.get("value") != expected_value:
        exposure["value"] = copy.deepcopy(expected_value)
        changed = True
    for key in ("min", "max", "step", "step_options"):
        if key in exposure:
            exposure.pop(key)
            changed = True
    return changed


def _round_numeric_value(value, step):
    """Round ``value`` to the decimal precision implied by ``step``.

    ``Decimal(str(step))`` is used only to read the exponent, avoiding
    binary-float artifacts such as 0.1 carrying many digits.  The value is
    not snapped to multiples of the step.  Unparsable or non-numeric input
    is returned unchanged; integer values stay integers when possible.
    """

    try:
        exponent = Decimal(str(step)).as_tuple().exponent
    except (InvalidOperation, TypeError, ValueError):
        return value
    decimals = max(0, -exponent)
    try:
        rounded = round(value, decimals)
    except TypeError:
        return value
    if isinstance(value, int) and decimals == 0:
        return int(rounded)
    return rounded


def _normalise_numeric_values(tree: dict[str, Any]) -> bool:
    """Round numeral values to their step precision and clamp to min/max."""

    changed = False
    for group in tree.values():
        if not isinstance(group, dict):
            continue
        for section in group.values():
            if not isinstance(section, dict):
                continue
            options = section.get("options")
            if not isinstance(options, list):
                continue
            for option in options:
                if not isinstance(option, dict):
                    continue
                if str(option.get("type", "")).lower() != "numeral":
                    continue
                value = option.get("value")
                step = option.get("step")
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not isinstance(step, (int, float))
                    or isinstance(step, bool)
                ):
                    continue
                rounded = _round_numeric_value(value, step)
                minimum = option.get("min")
                maximum = option.get("max")
                if (
                    isinstance(minimum, (int, float))
                    and not isinstance(minimum, bool)
                ):
                    rounded = max(minimum, rounded)
                if (
                    isinstance(maximum, (int, float))
                    and not isinstance(maximum, bool)
                ):
                    rounded = min(maximum, rounded)
                if rounded != value:
                    option["value"] = rounded
                    changed = True
    return changed


def ensure_menu_defaults(tree: dict[str, Any]) -> bool:
    """Add missing sections/options/links.  Returns True when modified.

    Idempotent: each rule compares against the current tree and only writes
    when a value differs, so calling it repeatedly is safe.  Ordering is
    significant -- moved options are relocated before defaults are appended
    (so a user's value wins over the default), new options are merged before
    normalization (so new bounds apply), and reordering runs last.
    """

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

            # Give a moved option any new metadata (format, dependency,
            # icon, ...) without touching the user's value.
            destination_defaults = DEFAULT_SECTIONS.get(
                group_name, {}
            ).get(move["to"], {})
            default_option = next(
                (
                    option
                    for option in destination_defaults.get("options", [])
                    if option.get("id") == option_id
                ),
                None,
            )
            if default_option is not None:
                for key, default_value in default_option.items():
                    if key == "value":
                        continue
                    moved.setdefault(key, copy.deepcopy(default_value))

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

    changed |= _migrate_focus_mode(tree)
    changed |= _migrate_power_option(tree)
    changed |= _migrate_status_bar_option(tree)

    # Create missing groups/sections wholesale, but merge existing ones
    # option-by-option so user values and custom entries survive upgrades.
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
                default_options = default_section.get("options", [])
                changed |= _append_missing_options(
                    section, default_options
                )
                changed |= _merge_option_defaults(
                    section, default_options
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

    # Theme ids and labels come from the code registry, not menus.json, so
    # they are replaced wholesale instead of merged like normal options.
    changed |= _ensure_theme_choices(tree)

    for group_name, sections in PREFERRED_ORDER.items():
        group = tree.get(group_name)
        if not isinstance(group, dict):
            continue
        for section_name, order in sections.items():
            section = group.get(section_name)
            if isinstance(section, dict):
                changed |= _reorder_options(section, order)

    # These run after defaults exist, so they also apply to options that
    # were just inserted; normalization comes last to use the updated bounds.
    changed |= _migrate_camera_control_ranges(tree)
    changed |= _migrate_exposure_presets(tree)
    changed |= _normalise_numeric_values(tree)

    return changed
