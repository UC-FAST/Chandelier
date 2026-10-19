import json

import pytest

from chandelier.config import (
    ConfigError,
    MenuConfigError,
    PROJECT_ROOT,
    clear_config_cache,
    load_config,
    load_menu_config,
    save_menu_config,
    validate_menu_config,
)


def test_load_config_resolves_paths_against_project_root(config_file):
    clear_config_cache()
    config = load_config(config_file)

    assert config.screen.width == 320
    assert config.pin.rotary_encoder_1a == 17
    assert config.camera.path == (PROJECT_ROOT / "pict").resolve()
    assert config.camera.video_path == (PROJECT_ROOT / "pict/video").resolve()
    assert config.camera.tuning_file is None
    assert (config.camera.preview_width, config.camera.preview_height) == (
        640,
        480,
    )
    assert config.localizer.ignored_channels == (0, 5)
    assert config.marker.mode == "single"
    assert config.marker.max_sources == 3
    assert config.marker.trail == 3
    assert config.preview.auth_file == (
        PROJECT_ROOT / "config/preview_auth.json"
    ).resolve()
    assert config.menus_file == (PROJECT_ROOT / "config/menus.json").resolve()


def test_config_is_independent_of_working_directory(config_file, tmp_path, monkeypatch):
    clear_config_cache()
    monkeypatch.chdir(tmp_path)
    config = load_config(config_file)
    assert config.screen.height == 240


def test_preview_auth_file_is_configurable(config_file, tmp_path):
    raw = json.loads(config_file.read_text(encoding="utf-8"))
    raw["preview"] = {"auth_file": "config/custom_auth.json"}
    target = tmp_path / "custom_preview.json"
    target.write_text(json.dumps(raw), encoding="utf-8")
    clear_config_cache()

    config = load_config(target)
    assert config.preview.auth_file == (
        PROJECT_ROOT / "config/custom_auth.json"
    ).resolve()


def test_preview_section_must_be_an_object(config_file, tmp_path):
    raw = json.loads(config_file.read_text(encoding="utf-8"))
    raw["preview"] = "not an object"
    target = tmp_path / "bad_preview.json"
    target.write_text(json.dumps(raw), encoding="utf-8")
    clear_config_cache()

    with pytest.raises(ConfigError, match="preview"):
        load_config(target)


def test_legacy_pin_keys_are_rejected(config_file, tmp_path):
    raw = json.loads(config_file.read_text(encoding="utf-8"))
    raw["pin"]["rotaryEncoder1A"] = raw["pin"].pop("rotary_encoder_1a")
    bad = tmp_path / "legacy.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    clear_config_cache()

    with pytest.raises(ConfigError, match="legacy pin keys"):
        load_config(bad)


def test_missing_file_raises_config_error(tmp_path):
    clear_config_cache()
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.json")


def test_invalid_marker_mode_is_rejected(config_file, tmp_path):
    raw = json.loads(config_file.read_text(encoding="utf-8"))
    raw["marker"]["mode"] = "all"
    bad = tmp_path / "bad_marker.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    clear_config_cache()

    with pytest.raises(ConfigError, match="marker.mode"):
        load_config(bad)


def test_multiple_marker_mode_is_accepted(config_file, tmp_path):
    raw = json.loads(config_file.read_text(encoding="utf-8"))
    raw["marker"]["mode"] = "multiple"
    raw["marker"]["max_sources"] = 2
    target = tmp_path / "multiple_marker.json"
    target.write_text(json.dumps(raw), encoding="utf-8")
    clear_config_cache()

    config = load_config(target)
    assert config.marker.mode == "multiple"
    assert config.marker.max_sources == 2


@pytest.mark.parametrize("fov", [0, 180, -5])
def test_invalid_marker_fov_is_rejected(config_file, tmp_path, fov):
    raw = json.loads(config_file.read_text(encoding="utf-8"))
    raw["localizer"]["marker_fov"] = fov
    bad = tmp_path / "bad_fov.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    clear_config_cache()

    with pytest.raises(ConfigError, match="marker_fov"):
        load_config(bad)


def test_marker_projection_settings_are_parsed(config_file, tmp_path):
    raw = json.loads(config_file.read_text(encoding="utf-8"))
    raw["localizer"]["marker_fov"] = 70.5
    raw["localizer"]["marker_azimuth_offset"] = 370
    raw["localizer"]["marker_elevation_offset"] = -18.0
    raw["localizer"]["marker_x_gain"] = 0.5
    raw["localizer"]["marker_y_gain"] = 2.5
    raw["localizer"]["marker_mic_offset_x"] = 0.03
    raw["localizer"]["marker_mic_offset_y"] = -0.05
    raw["localizer"]["marker_mic_offset_z"] = 0.01
    raw["localizer"]["marker_source_distance"] = 0.45
    target = tmp_path / "projection.json"
    target.write_text(json.dumps(raw), encoding="utf-8")
    clear_config_cache()

    config = load_config(target)
    assert config.localizer.marker_fov == pytest.approx(70.5)
    assert config.localizer.marker_azimuth_offset == pytest.approx(10.0)
    assert config.localizer.marker_elevation_offset == pytest.approx(-18.0)
    assert config.localizer.marker_x_gain == pytest.approx(0.5)
    assert config.localizer.marker_y_gain == pytest.approx(2.5)
    assert config.localizer.marker_mic_offset_x == pytest.approx(0.03)
    assert config.localizer.marker_mic_offset_y == pytest.approx(-0.05)
    assert config.localizer.marker_mic_offset_z == pytest.approx(0.01)
    assert config.localizer.marker_source_distance == pytest.approx(0.45)


def test_negative_source_distance_is_rejected(config_file, tmp_path):
    raw = json.loads(config_file.read_text(encoding="utf-8"))
    raw["localizer"]["marker_source_distance"] = -0.1
    bad = tmp_path / "bad_distance.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    clear_config_cache()

    with pytest.raises(ConfigError, match="marker_source_distance"):
        load_config(bad)


def test_shipped_menu_config_is_valid():
    tree = load_menu_config()
    assert "camera" in tree
    assert "gallery" in tree
    assert tree["camera"]["0"]["options"][0]["type"] == "menu"


def test_shipped_menu_camera_options_are_consistent():
    tree = load_menu_config()
    camera_settings = {
        option["id"]: option
        for option in tree["camera"]["camera_settings"]["options"]
    }
    assert camera_settings["resolution"]["value"]["value"] == [4608, 2592]
    assert camera_settings["pict_format"]["value"]["value"] == "jpeg"
    assert all(
        option["type"] == option["type"].lower()
        for sections in tree.values()
        for section in sections.values()
        for option in section["options"]
    )


def test_menu_type_is_normalized(tmp_path):
    tree = {
        "camera": {
            "0": {
                "title": "Camera",
                "options": [
                    {
                        "id": "sub",
                        "content": "Sub",
                        "type": "Menu",
                        "value": "sub",
                    }
                ],
            },
            "sub": {"title": "Sub", "options": []},
        },
        "gallery": {"0": {"title": "Gallery", "options": []}},
    }
    path = tmp_path / "menus.json"
    path.write_text(json.dumps(tree), encoding="utf-8")

    loaded = load_menu_config(path)
    assert loaded["camera"]["0"]["options"][0]["type"] == "menu"


def test_menu_validation_reports_missing_target(tmp_path):
    tree = {
        "camera": {
            "0": {
                "title": "Camera",
                "options": [
                    {
                        "id": "sub",
                        "content": "Sub",
                        "type": "menu",
                        "value": "nope",
                    }
                ],
            }
        },
        "gallery": {"0": {"title": "Gallery", "options": []}},
    }
    path = tmp_path / "menus.json"
    path.write_text(json.dumps(tree), encoding="utf-8")

    with pytest.raises(MenuConfigError, match="missing menu"):
        load_menu_config(path)


def test_menu_validation_rejects_duplicate_ids():
    tree = {
        "camera": {
            "0": {
                "title": "Camera",
                "options": [
                    {
                        "id": "x",
                        "content": "A",
                        "type": "bool",
                        "value": True,
                    },
                    {
                        "id": "x",
                        "content": "B",
                        "type": "bool",
                        "value": False,
                    },
                ],
            }
        },
        "gallery": {"0": {"title": "Gallery", "options": []}},
    }
    errors = validate_menu_config(tree)
    assert any("duplicate" in error for error in errors)


def test_save_menu_config_round_trip(tmp_path):
    tree = load_menu_config()
    target = tmp_path / "menus.json"
    save_menu_config(tree, target)

    saved = json.loads(target.read_text(encoding="utf-8"))
    assert saved == tree
    assert target.read_text(encoding="utf-8").endswith("\n")


def test_shipped_menu_has_pro_sections():
    tree = load_menu_config()
    camera = tree["camera"]
    for section in ("display", "capture", "manual", "record"):
        assert section in camera
    assert "player" in tree["gallery"]

    link_ids = {
        option.get("id")
        for option in camera["camera_settings"]["options"]
    }
    assert {"display", "capture", "manual", "record"}.issubset(link_ids)
    gallery_ids = {
        option.get("id") for option in tree["gallery"]["0"]["options"]
    }
    assert "player" in gallery_ids


def test_ensure_menu_defaults_migrates_old_tree(tmp_path):
    from chandelier.menu_defaults import ensure_menu_defaults

    tree = {
        "camera": {
            "0": {"title": "Root", "options": []},
            "camera_settings": {"title": "Camera", "options": []},
        },
        "gallery": {"0": {"title": "Gallery", "options": []}},
    }
    assert ensure_menu_defaults(tree) is True
    assert "display" in tree["camera"]
    assert "capture" in tree["camera"]
    assert "manual" in tree["camera"]
    assert "record" in tree["camera"]
    assert "player" in tree["gallery"]

    ids = {
        option.get("id")
        for option in tree["camera"]["camera_settings"]["options"]
    }
    assert {"display", "capture", "manual", "record"}.issubset(ids)
    assert ensure_menu_defaults(tree) is False


def test_ensure_menu_defaults_adds_localizer_section():
    from chandelier.menu_defaults import ensure_menu_defaults

    tree = {
        "camera": {
            "0": {"title": "Root", "options": []},
            "camera_settings": {"title": "Camera", "options": []},
        },
        "gallery": {"0": {"title": "Gallery", "options": []}},
    }
    assert ensure_menu_defaults(tree) is True
    section = tree["camera"]["localizer"]
    ids = [option["id"] for option in section["options"]]
    assert ids == [
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
    ]

    root_links = {
        option.get("id") for option in tree["camera"]["0"]["options"]
    }
    settings_links = {
        option.get("id")
        for option in tree["camera"]["camera_settings"]["options"]
    }
    assert "localizer" in root_links
    assert "localizer" not in settings_links
    assert ensure_menu_defaults(tree) is False


def test_shipped_menu_has_web_server_before_power():
    tree = load_menu_config()
    options = tree["camera"]["system"]["options"]
    ids = [option["id"] for option in options]
    assert "web_server" in ids
    assert "restart_service" in ids
    assert (
        ids.index("web_server")
        < ids.index("restart_service")
        < ids.index("power")
    )

    option = options[ids.index("web_server")]
    assert option["type"] == "bool"
    assert option["receiver"] == "camera"
    assert isinstance(option["value"], bool)


def test_ensure_menu_defaults_adds_web_server_before_power():
    from chandelier.menu_defaults import ensure_menu_defaults

    tree = {
        "camera": {
            "0": {"title": "Root", "options": []},
            "system": {
                "title": "System",
                "options": [
                    {
                        "id": "language",
                        "content": "Language",
                        "type": "option",
                        "value": {"content": "English", "value": "en"},
                        "options": [
                            {"content": "English", "value": "en"}
                        ],
                    },
                    {
                        "id": "theme",
                        "content": "Theme",
                        "type": "option",
                        "value": {"content": "Dark Pro", "value": "dark_pro"},
                        "options": [
                            {"content": "Dark Pro", "value": "dark_pro"}
                        ],
                    },
                    {
                        "id": "power",
                        "content": "Power Action",
                        "type": "option",
                        "value": {"content": "Reboot", "value": "reboot"},
                        "options": [
                            {"content": "Reboot", "value": "reboot"}
                        ],
                    },
                ],
            },
        },
        "gallery": {"0": {"title": "Gallery", "options": []}},
    }
    assert ensure_menu_defaults(tree) is True
    ids = [option["id"] for option in tree["camera"]["system"]["options"]]
    assert ids == [
        "language",
        "theme",
        "web_server",
        "restart_service",
        "power",
    ]
    assert ensure_menu_defaults(tree) is False


@pytest.mark.parametrize(
    ("show", "position", "expected"),
    [
        (True, "bottom", "top_bottom"),
        (True, "top", "top_bottom"),
        (True, "top_bottom", "top_bottom"),
        (True, "sides", "sides"),
        (False, "top", "off"),
        (True, None, "top_bottom"),
        (False, None, "off"),
    ],
)
def test_ensure_menu_defaults_merges_status_bar_options(
    show, position, expected
):
    from chandelier.menu_defaults import ensure_menu_defaults

    options = [
        {
            "id": "show_status_bar",
            "content": "Status Bar",
            "type": "bool",
            "icon": "monitor",
            "value": show,
        }
    ]
    if position is not None:
        options.append(
            {
                "id": "status_bar_position",
                "content": "Bar Position",
                "type": "option",
                "icon": "monitor",
                "value": {
                    "content": position.title(),
                    "value": position,
                },
                "options": [
                    {"content": "Top", "value": "top"},
                    {"content": "Bottom", "value": "bottom"},
                    {"content": "Sides", "value": "sides"},
                ],
                "enable_if": {"id": "show_status_bar", "value": True},
                "enable": True,
            }
        )

    tree = {
        "camera": {
            "0": {"title": "Root", "options": []},
            "display": {"title": "Display", "options": options},
        },
        "gallery": {"0": {"title": "Gallery", "options": []}},
    }
    assert ensure_menu_defaults(tree) is True

    merged_options = tree["camera"]["display"]["options"]
    ids = [option["id"] for option in merged_options]
    assert "show_status_bar" not in ids

    merged = next(
        option
        for option in merged_options
        if option["id"] == "status_bar_position"
    )
    assert merged["value"]["value"] == expected
    assert merged["content"] == "Status Bar"
    assert "enable_if" not in merged
    merged_values = {
        choice["value"] for choice in merged["options"]
    }
    assert {"off", "top_bottom", "sides"} <= merged_values
    assert "top" not in merged_values
    assert "bottom" not in merged_values
    assert ensure_menu_defaults(tree) is False


def test_ensure_menu_defaults_updates_camera_control_steps():
    from chandelier.menu_defaults import ensure_menu_defaults

    tree = {
        "camera": {
            "0": {"title": "Root", "options": []},
            "exposure": {
                "title": "Exposure",
                "options": [
                    {
                        "id": "exposure_time",
                        "type": "numeral",
                        "value": 40001,
                        "min": 1,
                        "max": 66666,
                        "step": 10000,
                        "step_options": [100, 1000, 10000],
                    },
                    {
                        "id": "analogue_gain",
                        "type": "numeral",
                        "value": 16,
                        "min": 1,
                        "max": 16,
                        "step": 1,
                        "step_options": [1, 10],
                    },
                ],
            },
            "color_gains": {
                "title": "Color Gains",
                "options": [
                    {
                        "id": "red_gain",
                        "type": "numeral",
                        "value": 0,
                        "min": 0,
                        "max": 32,
                        "step": 1,
                        "step_options": [0.1, 1],
                    },
                    {
                        "id": "blue_gain",
                        "type": "numeral",
                        "value": 0,
                        "min": 0,
                        "max": 32,
                        "step": 1,
                        "step_options": [0.1, 1],
                    },
                ],
            },
            "manual": {
                "title": "Manual",
                "options": [
                    {
                        "id": "exp_compensation",
                        "type": "numeral",
                        "value": 0.0,
                        "min": -4.0,
                        "max": 4.0,
                        "step": 0.5,
                        "step_options": [0.1, 0.5],
                    },
                    {
                        "id": "contrast",
                        "type": "numeral",
                        "value": 0.9999999999999999,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.1,
                    },
                ],
            },
        },
        "gallery": {"0": {"title": "Gallery", "options": []}},
    }

    assert ensure_menu_defaults(tree) is True

    options = {}
    for section in tree["camera"].values():
        for option in section.get("options", []):
            options[option.get("id")] = option

    assert options["exposure_time"]["type"] == "option"
    assert options["exposure_time"]["value"] == {
        "content": "1/25",
        "value": 40000,
    }
    assert any(
        choice["value"] == 30_000_000
        for choice in options["exposure_time"]["options"]
    )
    assert "step" not in options["exposure_time"]
    assert options["analogue_gain"]["step"] == 0.5
    assert options["analogue_gain"]["step_options"] == [0.1, 0.5, 1]
    for gain_id in ("red_gain", "blue_gain"):
        assert options[gain_id]["min"] == 0.1
        assert options[gain_id]["step"] == 0.1
        assert options[gain_id]["value"] == 1.0
    assert options["exp_compensation"]["enable_if"] == {
        "id": "auto_expose",
        "value": True,
    }
    assert options["contrast"]["value"] == 1.0
    assert ensure_menu_defaults(tree) is False


def test_ensure_menu_defaults_clears_hdr_without_auto_exposure():
    from chandelier.menu_defaults import ensure_menu_defaults

    tree = {
        "camera": {
            "0": {"title": "Root", "options": []},
            "exposure": {
                "title": "Exposure",
                "options": [
                    {
                        "id": "auto_expose",
                        "type": "bool",
                        "value": False,
                    }
                ],
            },
            "camera_settings": {
                "title": "Camera Settings",
                "options": [
                    {
                        "id": "hdr",
                        "type": "option",
                        "value": {
                            "content": "MultiExposure",
                            "value": 2,
                        },
                        "options": [
                            {"content": "Off", "value": 0},
                            {"content": "MultiExposure", "value": 2},
                        ],
                    }
                ],
            },
        },
        "gallery": {"0": {"title": "Gallery", "options": []}},
    }

    assert ensure_menu_defaults(tree) is True

    hdr = next(
        option
        for section in tree["camera"].values()
        for option in section.get("options", [])
        if option.get("id") == "hdr"
    )
    assert hdr["value"]["value"] == 0
    assert hdr["enable_if"] == {"id": "auto_expose", "value": True}


def test_shipped_focus_menu_has_af_controls():
    tree = load_menu_config()
    focus_ids = [option["id"] for option in tree["camera"]["focus"]["options"]]
    assert focus_ids == [
        "af_mode",
        "af_trigger",
        "focus_position",
        "af_speed",
        "af_range",
        "mf_assist",
    ]
    manual_ids = [
        option["id"] for option in tree["camera"]["manual"]["options"]
    ]
    assert "manual_focus" not in manual_ids
    assert "focus_position" not in manual_ids


def test_ensure_menu_defaults_migrates_legacy_focus():
    from chandelier.menu_defaults import ensure_menu_defaults

    tree = {
        "camera": {
            "0": {"title": "Root", "options": []},
            "manual": {
                "title": "Manual",
                "options": [
                    {
                        "id": "manual_focus",
                        "content": "Manual Focus",
                        "type": "bool",
                        "value": True,
                    },
                    {
                        "id": "focus_position",
                        "content": "Focus Pos",
                        "type": "numeral",
                        "value": 250,
                        "min": 0,
                        "max": 1000,
                        "step": 10,
                    },
                ],
            },
        },
        "gallery": {"0": {"title": "Gallery", "options": []}},
    }
    ensure_menu_defaults(tree)

    focus_options = {
        option["id"]: option
        for option in tree["camera"]["focus"]["options"]
    }
    assert focus_options["af_mode"]["value"]["value"] == 0
    assert focus_options["focus_position"]["value"] == 250
    assert focus_options["focus_position"]["enable_if"] == {
        "id": "af_mode",
        "value": 0,
    }
    manual_ids = {
        option["id"] for option in tree["camera"]["manual"]["options"]
    }
    assert "manual_focus" not in manual_ids


def test_ensure_menu_defaults_reorders_options_by_frequency():
    from chandelier.menu_defaults import ensure_menu_defaults

    tree = {
        "camera": {
            "0": {
                "title": "Root",
                "options": [
                    {"id": "system", "type": "menu", "value": "system"},
                    {"id": "gallery", "type": "irq", "value": "gallery"},
                    {
                        "id": "camera_settings",
                        "type": "menu",
                        "value": "camera_settings",
                    },
                    {
                        "id": "system_monitor",
                        "type": "irq",
                        "value": "system_monitor",
                    },
                ],
            },
            "camera_settings": {
                "title": "Camera",
                "options": [
                    {"id": "watermark", "type": "bool", "value": False},
                    {"id": "exposure", "type": "menu", "value": "exposure"},
                ],
            },
        },
        "gallery": {"0": {"title": "Gallery", "options": []}},
    }
    ensure_menu_defaults(tree)

    root_ids = [option["id"] for option in tree["camera"]["0"]["options"]]
    assert root_ids[:5] == [
        "camera_settings",
        "gallery",
        "localizer",
        "system_monitor",
        "system",
    ]
    settings_ids = [
        option["id"]
        for option in tree["camera"]["camera_settings"]["options"]
    ]
    assert settings_ids[0] == "exposure"
    assert settings_ids[-1] == "watermark"


def test_ensure_menu_defaults_moves_localizer_link_to_root():
    from chandelier.menu_defaults import ensure_menu_defaults

    tree = {
        "camera": {
            "0": {"title": "Root", "options": []},
            "camera_settings": {
                "title": "Camera",
                "options": [
                    {
                        "id": "localizer",
                        "content": "Localizer",
                        "type": "menu",
                        "value": "localizer",
                    }
                ],
            },
        },
        "gallery": {"0": {"title": "Gallery", "options": []}},
    }
    assert ensure_menu_defaults(tree) is True
    root_ids = {option["id"] for option in tree["camera"]["0"]["options"]}
    settings_ids = {
        option["id"]
        for option in tree["camera"]["camera_settings"]["options"]
    }
    assert "localizer" in root_ids
    assert "localizer" not in settings_ids
    assert ensure_menu_defaults(tree) is False


def test_migration_preserves_existing_theme_value():
    from chandelier.menu_defaults import ensure_menu_defaults

    tree = {
        "camera": {
            "0": {"title": "Root", "options": []},
            "display": {
                "title": "My Display",
                "options": [
                    {
                        "id": "theme",
                        "content": "Theme",
                        "type": "option",
                        "value": {
                            "content": "Industrial",
                            "value": "industrial",
                        },
                        "options": [
                            {"content": "Industrial", "value": "industrial"}
                        ],
                    }
                ],
            },
        },
        "gallery": {"0": {"title": "Gallery", "options": []}},
    }
    ensure_menu_defaults(tree)
    theme = next(
        option
        for option in tree["camera"]["system"]["options"]
        if option["id"] == "theme"
    )
    assert theme["value"]["value"] == "industrial"
    assert any(
        choice["value"] == "classic" for choice in theme["options"]
    )
    assert ensure_menu_defaults(tree) is False
    assert any(
        option.get("id") == "show_grid"
        for option in tree["camera"]["display"]["options"]
    )


def test_migration_removes_retired_themes_and_orders_choices():
    from chandelier.menu_defaults import ensure_menu_defaults
    from chandelier.ui.themes import THEME_CHOICES

    tree = {
        "camera": {
            "0": {"title": "Root", "options": []},
            "system": {
                "title": "System",
                "options": [
                    {
                        "id": "theme",
                        "content": "Theme",
                        "type": "option",
                        "value": {
                            "content": "Windows XP",
                            "value": "xp_luna",
                        },
                        "options": [
                            {"content": "Windows XP", "value": "xp_luna"},
                            {"content": "Windows 10", "value": "win10"},
                            {"content": "Dark Pro", "value": "dark_pro"},
                        ],
                    }
                ],
            },
        },
        "gallery": {"0": {"title": "Gallery", "options": []}},
    }

    assert ensure_menu_defaults(tree) is True
    theme = next(
        option
        for option in tree["camera"]["system"]["options"]
        if option["id"] == "theme"
    )
    assert theme["options"] == THEME_CHOICES
    assert theme["value"] == {
        "content": "Dark Pro",
        "value": "dark_pro",
    }
    assert ensure_menu_defaults(tree) is False


def test_migration_refreshes_stale_theme_label():
    from chandelier.menu_defaults import ensure_menu_defaults

    tree = {
        "camera": {
            "0": {"title": "Root", "options": []},
            "system": {
                "title": "System",
                "options": [
                    {
                        "id": "theme",
                        "content": "Theme",
                        "type": "option",
                        "value": {
                            "content": "Ubuntu (old)",
                            "value": "ubuntu",
                        },
                        "options": [
                            {"content": "Ubuntu (old)", "value": "ubuntu"}
                        ],
                    }
                ],
            },
        },
        "gallery": {"0": {"title": "Gallery", "options": []}},
    }

    assert ensure_menu_defaults(tree) is True
    theme = next(
        option
        for option in tree["camera"]["system"]["options"]
        if option["id"] == "theme"
    )
    assert theme["value"] == {"content": "Ubuntu", "value": "ubuntu"}
    assert ensure_menu_defaults(tree) is False


def test_validation_rejects_bad_option_and_numeral():
    tree = {
        "camera": {
            "0": {
                "title": "Root",
                "options": [
                    {
                        "id": "bad_option",
                        "content": "Bad",
                        "type": "option",
                        "value": {"content": "A", "value": "a"},
                    },
                    {
                        "id": "bad_numeral",
                        "content": "Bad Num",
                        "type": "numeral",
                        "value": 99,
                        "min": 0,
                        "max": 10,
                        "step": 1,
                    },
                ],
            }
        },
        "gallery": {"0": {"title": "Gallery", "options": []}},
    }
    errors = validate_menu_config(tree)
    assert any("options list" in error for error in errors)
    assert any("within" in error for error in errors)
