import json
import shutil
import time

import numpy as np

from chandelier.config import load_menu_config
from chandelier.ui.menu_end import MenuControlledEnd
from chandelier.ui.themes import THEMES


def _tree():
    return {
        "camera": {
            "0": {
                "title": "Root",
                "options": [
                    {
                        "id": "num",
                        "content": "Num",
                        "type": "numeral",
                        "value": 5,
                        "min": 0,
                        "max": 10,
                        "step": 1,
                    },
                    {
                        "id": "flag",
                        "content": "Flag",
                        "type": "bool",
                        "value": False,
                    },
                ],
            }
        },
        "gallery": {"0": {"title": "Gallery", "options": []}},
    }


def _make_menu(tmp_path, tree=None):
    path = tmp_path / "menus.json"
    path.write_text(json.dumps(tree or _tree()), encoding="utf-8")
    menu = MenuControlledEnd(
        path=path,
        width=320,
        height=240,
        row_count=5,
        show_preview=True,
        show_index=True,
        font_height=14,
        padding=(5, 5, 5, 5),
    )
    irqs = []
    messages = []
    menu._irq = irqs.append
    menu._msg_sender = lambda sender, receiver, payload: messages.append(
        (sender, receiver, payload)
    )
    menu.msg_receiver("camera", "camera")
    menu.on_enter(None)
    return menu, path, irqs, messages


def test_initial_state(tmp_path):
    menu, _, _, _ = _make_menu(tmp_path)
    assert menu.options["0"]["title"] == "Root"
    assert len(menu._options) == 2


def test_navigation_wraps_within_bounds(tmp_path):
    menu, _, _, _ = _make_menu(tmp_path)
    menu.down_action()
    assert menu._current_index == 1
    menu.down_action()
    assert menu._current_index == 1
    menu.up_action()
    assert menu._current_index == 0
    menu.up_action()
    assert menu._current_index == 0


def test_numerical_editing(tmp_path):
    menu, path, _, _ = _make_menu(tmp_path)
    menu.select()
    assert menu._select_index == 0
    menu._value_plus()
    menu.unselect()

    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["camera"]["0"]["options"][0]["value"] == 6


def test_bool_toggle_is_persisted(tmp_path):
    menu, path, irqs, messages = _make_menu(tmp_path)
    menu.down_action()
    menu.select()

    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["camera"]["0"]["options"][1]["value"] is True
    assert messages


def test_submenu_jump_and_back(tmp_path):
    tree = _tree()
    tree["camera"]["sub"] = {
        "title": "Sub",
        "options": [
            {"id": "x", "content": "X", "type": "bool", "value": False}
        ],
    }
    tree["camera"]["0"]["options"].append(
        {"id": "sub", "content": "Sub", "type": "menu", "value": "sub"}
    )
    menu, _, _, _ = _make_menu(tmp_path, tree)

    menu.down_action()
    menu.down_action()
    menu.select()
    assert menu._title == "Sub"

    menu.cross_press_action()
    assert menu._title == "Root"


def test_main_loop_yields_rendered_frame(tmp_path):
    menu, _, _, _ = _make_menu(tmp_path)
    menu.decorate()
    frame = next(menu.main_loop())
    assert frame.shape == (240, 320, 3)
    assert frame.dtype == np.uint8


def test_touch_taps_menu_rows(tmp_path):
    menu, _, _, _ = _make_menu(tmp_path)
    x = 0.5
    y = (
        menu._content_top + menu._row_height * 1.5
    ) / menu._height

    assert menu.handle_touch(x, y) is True
    assert menu._current_index == 1
    assert menu._cursor_anim_start is not None

    # A second tap on the selected row activates it.
    assert menu.handle_touch(x, y) is True
    assert menu._current_options[1]["value"] is True


def test_touch_ignores_header_footer_and_disabled_rows(tmp_path):
    tree = _tree()
    tree["camera"]["0"]["options"][1]["enable"] = False
    menu, _, _, _ = _make_menu(tmp_path, tree)
    x = 0.5
    y_row = (
        menu._content_top + menu._row_height * 1.5
    ) / menu._height

    assert menu.handle_touch(x, 0.01) is False
    assert menu.handle_touch(x, y_row) is False
    assert menu._current_index == 0
    assert menu.handle_touch(x, 0.99) is False


def test_touch_selects_option_editor_choices(tmp_path):
    tree = {
        "camera": {
            "0": {
                "title": "Root",
                "options": [
                    {
                        "id": "choice",
                        "content": "Choice",
                        "type": "option",
                        "value": {"content": "A", "value": "a"},
                        "options": [
                            {"content": "A", "value": "a"},
                            {"content": "B", "value": "b"},
                            {"content": "C", "value": "c"},
                        ],
                    }
                ],
            }
        },
        "gallery": {"0": {"title": "Gallery", "options": []}},
    }
    menu, path, _, _ = _make_menu(tmp_path, tree)
    menu.select()
    assert menu._select_index == 0

    _, _, row_height, block_top = menu._option_list_geometry(0, 3)
    y = (block_top + row_height * 1.5) / menu._height

    assert menu.handle_touch(0.5, y) is True
    assert menu._value_temp == {"content": "B", "value": "b"}

    # A second tap on the highlighted choice confirms it.
    assert menu.handle_touch(0.5, y) is True
    assert menu._select_index is None
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert (
        reloaded["camera"]["0"]["options"][0]["value"]["value"] == "b"
    )


def test_option_list_animates_between_choices(tmp_path):
    tree = {
        "camera": {
            "0": {
                "title": "Root",
                "options": [
                    {
                        "id": "choice",
                        "content": "Choice",
                        "type": "option",
                        "value": {"content": "A", "value": "a"},
                        "options": [
                            {"content": "A", "value": "a"},
                            {"content": "B", "value": "b"},
                            {"content": "C", "value": "c"},
                        ],
                    }
                ],
            }
        },
        "gallery": {"0": {"title": "Gallery", "options": []}},
    }
    menu, _, _, _ = _make_menu(tmp_path, tree)
    menu.select()
    menu._option_down()
    assert menu._option_anim_start is not None
    assert 0.0 <= menu._option_position(1) < 0.5

    menu.decorate()
    frames = []
    generator = menu.main_loop()
    deadline = time.monotonic() + 1.0
    while (
        menu._option_anim_start is not None
        and time.monotonic() < deadline
    ):
        frames.append(next(generator))

    assert menu._option_anim_start is None
    assert menu._option_position(1) == 1.0
    assert frames


def test_theme_preview_keeps_option_animation(tmp_path):
    tree = {
        "camera": {
            "0": {
                "title": "Root",
                "options": [
                    {
                        "id": "theme",
                        "content": "Theme",
                        "type": "option",
                        "value": {
                            "content": "Dark Pro",
                            "value": "dark_pro",
                        },
                        "options": [
                            {"content": "Dark Pro", "value": "dark_pro"},
                            {"content": "Ubuntu", "value": "ubuntu"},
                        ],
                    }
                ],
            }
        },
        "gallery": {"0": {"title": "Gallery", "options": []}},
    }
    menu, _, _, _ = _make_menu(tmp_path, tree)
    menu._apply_theme("dark_pro")
    menu.select()
    assert menu._theme_preview_active

    menu._option_down()
    assert menu._option_anim_start is not None
    assert menu._menu_theme.id == "high_contrast"

    menu._cancel_theme_preview()
    assert menu._menu_theme.id == "dark_pro"
    menu._cancel_option_animation()


def test_numeric_slider_animates_on_value_change(tmp_path):
    menu, _, _, _ = _make_menu(tmp_path)
    menu.select()
    menu._value_plus()
    assert menu._value_anim_start is not None
    assert 5.0 <= menu._numeric_position(6) < 6.0

    menu.decorate()
    frames = []
    generator = menu.main_loop()
    deadline = time.monotonic() + 1.0
    while (
        menu._value_anim_start is not None
        and time.monotonic() < deadline
    ):
        frames.append(next(generator))

    assert menu._value_anim_start is None
    assert menu._numeric_position(6) == 6.0
    assert frames


def test_shipped_menu_loads(tmp_path, project_root):
    path = tmp_path / "menus.json"
    shutil.copy(project_root / "config" / "menus.json", path)
    menu = MenuControlledEnd(path=path)
    assert menu._menu_options["camera"]["0"]["title"] == "Settings"
    assert load_menu_config(path)["camera"]["0"]["options"][0]["type"] == "menu"


def test_shipped_menu_bool_toggle_round_trips(tmp_path, project_root):
    path = tmp_path / "menus.json"
    shutil.copy(project_root / "config" / "menus.json", path)
    menu = MenuControlledEnd(path=path)
    menu._msg_sender = lambda *args: None
    menu.msg_receiver("camera", "camera")
    menu.on_enter(None)

    menu.select()  # camera -> camera_settings
    assert menu._title == "Camera Settings"
    index = next(
        position
        for position, option in enumerate(menu._options)
        if option["id"] == "save_metadata"
    )
    menu._current_page = index // menu._row_count
    page_start = menu._current_page * menu._row_count
    menu._current_options = menu._options[
        page_start:page_start + menu._row_count
    ]
    menu._current_index = index - page_start
    menu.select()

    reloaded = load_menu_config(path)
    options = reloaded["camera"]["camera_settings"]["options"]
    save_metadata = next(
        option for option in options if option["id"] == "save_metadata"
    )
    assert save_metadata["value"] is True


def _make_shipped_menu(tmp_path, project_root):
    path = tmp_path / "menus.json"
    shutil.copy(project_root / "config" / "menus.json", path)
    menu = MenuControlledEnd(
        path=path,
        width=320,
        height=240,
        row_count=5,
        show_preview=True,
        show_index=True,
        font_height=14,
        padding=(5, 5, 5, 5),
    )
    menu._irq = lambda target: None
    menu._msg_sender = lambda *args: None
    menu.msg_receiver("camera", "camera")
    menu.on_enter(None)
    return menu, path


def test_power_option_confirms_and_sends_selected_action(
    tmp_path, project_root
):
    path = tmp_path / "menus.json"
    shutil.copy(project_root / "config" / "menus.json", path)
    menu = MenuControlledEnd(path=path)
    messages = []
    menu._msg_sender = lambda *args: messages.append(args)
    menu._irq = lambda target: None
    menu.msg_receiver("camera", "camera")
    menu.on_enter(None)

    system_index = next(
        position
        for position, option in enumerate(menu._options)
        if option["id"] == "system"
    )
    menu._current_page = system_index // menu._row_count
    page_start = menu._current_page * menu._row_count
    menu._current_options = menu._options[
        page_start:page_start + menu._row_count
    ]
    menu._current_index = system_index - page_start
    menu.select()  # enter System

    power_index = next(
        position
        for position, option in enumerate(menu._options)
        if option["id"] == "power"
    )
    menu._current_page = power_index // menu._row_count
    page_start = menu._current_page * menu._row_count
    menu._current_options = menu._options[
        page_start:page_start + menu._row_count
    ]
    menu._current_index = power_index - page_start
    menu.select()  # edit the power action
    menu._value_temp = {"content": "Shutdown", "value": "shutdown"}

    messages.clear()
    menu.unselect()
    assert not any(
        receiver == "controller" for _, receiver, _ in messages
    )

    menu.center_press_action()  # confirmation
    controller_messages = [
        payload
        for _, receiver, payload in messages
        if receiver == "controller"
    ]
    assert controller_messages
    assert controller_messages[-1]["id"] == "power"
    assert controller_messages[-1]["value"]["value"] == "shutdown"


def test_cursor_animation_slides_between_rows(tmp_path, project_root):
    menu, _ = _make_shipped_menu(tmp_path, project_root)
    menu._apply_theme("liquid_glass")
    menu._drain_frames()

    menu.down_one_step()
    assert menu._cursor_anim_start is not None
    assert 0.0 <= menu._cursor_position() < 0.2

    menu.decorate()
    frames = []
    generator = menu.main_loop()
    deadline = time.monotonic() + 1.0
    while (
        menu._cursor_anim_start is not None
        and time.monotonic() < deadline
    ):
        frames.append(next(generator))

    assert menu._cursor_anim_start is None
    assert menu._cursor_position() == 1.0
    assert frames
    assert all(frame.shape == (240, 320, 3) for frame in frames)


def test_cursor_animation_cancels_on_page_change(tmp_path, project_root):
    menu, _ = _make_shipped_menu(tmp_path, project_root)
    menu._apply_theme("liquid_glass")
    menu._drain_frames()

    menu.down_one_step()
    assert menu._cursor_anim_start is not None
    menu._page_down()
    assert menu._cursor_anim_start is None


def test_restart_service_confirms_and_sends(tmp_path, project_root):
    path = tmp_path / "menus.json"
    shutil.copy(project_root / "config" / "menus.json", path)
    menu = MenuControlledEnd(path=path)
    messages = []
    menu._msg_sender = lambda *args: messages.append(args)
    menu._irq = lambda target: None
    menu.msg_receiver("camera", "camera")
    menu.on_enter(None)

    system_index = next(
        position
        for position, option in enumerate(menu._options)
        if option["id"] == "system"
    )
    menu._current_page = system_index // menu._row_count
    page_start = menu._current_page * menu._row_count
    menu._current_options = menu._options[
        page_start:page_start + menu._row_count
    ]
    menu._current_index = system_index - page_start
    menu.select()  # enter System

    restart_index = next(
        position
        for position, option in enumerate(menu._options)
        if option["id"] == "restart_service"
    )
    menu._current_page = restart_index // menu._row_count
    page_start = menu._current_page * menu._row_count
    menu._current_options = menu._options[
        page_start:page_start + menu._row_count
    ]
    menu._current_index = restart_index - page_start

    messages.clear()
    menu.select()  # arm the confirmation, do not send yet
    assert not any(
        receiver == "controller" for _, receiver, _ in messages
    )

    menu.center_press_action()  # confirmation
    controller_messages = [
        payload
        for _, receiver, payload in messages
        if receiver == "controller"
    ]
    assert controller_messages
    assert controller_messages[-1][0] == "restart_service"


def test_theme_change_applies_live_and_persists(tmp_path, project_root):
    menu, path = _make_shipped_menu(tmp_path, project_root)

    system_index = next(
        position
        for position, option in enumerate(menu._options)
        if option["id"] == "system"
    )
    menu._current_page = system_index // menu._row_count
    page_start = menu._current_page * menu._row_count
    menu._current_options = menu._options[
        page_start:page_start + menu._row_count
    ]
    menu._current_index = system_index - page_start
    menu.select()  # camera -> system
    assert menu._title == "System"

    theme_index = next(
        position
        for position, option in enumerate(menu._options)
        if option["id"] == "theme"
    )
    menu._current_page = theme_index // menu._row_count
    page_start = menu._current_page * menu._row_count
    menu._current_options = menu._options[
        page_start:page_start + menu._row_count
    ]
    menu._current_index = theme_index - page_start
    menu.select()  # theme option

    option = menu._current_options[menu._select_index]
    choices = option["options"]
    current = choices.index(option["value"])
    target = choices[(current + 2) % len(choices)]
    menu._option_down()
    menu._option_down()
    menu.unselect()
    assert menu._menu_theme.id == target["value"]

    reloaded = MenuControlledEnd(path=path)
    reloaded_theme = next(
        option
        for option in reloaded._menu_options["camera"]["system"]["options"]
        if option["id"] == "theme"
    )
    assert reloaded_theme["value"]["value"] == target["value"]


def test_theme_preview_and_cancel(tmp_path, project_root):
    menu, path = _make_shipped_menu(tmp_path, project_root)

    system_index = next(
        position
        for position, option in enumerate(menu._options)
        if option["id"] == "system"
    )
    menu._current_page = system_index // menu._row_count
    page_start = menu._current_page * menu._row_count
    menu._current_options = menu._options[
        page_start:page_start + menu._row_count
    ]
    menu._current_index = system_index - page_start
    menu.select()

    theme_index = next(
        position
        for position, option in enumerate(menu._options)
        if option["id"] == "theme"
    )
    menu._current_page = theme_index // menu._row_count
    page_start = menu._current_page * menu._row_count
    menu._current_options = menu._options[
        page_start:page_start + menu._row_count
    ]
    menu._current_index = theme_index - page_start
    original = menu._menu_theme.id
    menu.select()
    menu._option_down()
    assert menu._menu_theme.id != original

    menu.cross_press_action()  # cancel preview
    assert menu._menu_theme.id == original

    menu.select()
    menu._option_down()
    previewed = menu._menu_theme.id
    menu.unselect()  # confirm preview
    assert menu._menu_theme.id == previewed

    reloaded = MenuControlledEnd(path=path)
    assert reloaded._menu_theme.id == previewed


def test_option_edit_window_shows_more_choices():
    for selected in range(8):
        start, visible = MenuControlledEnd._option_window(selected, 8)
        assert visible == 5
        assert start <= selected < start + visible

    assert MenuControlledEnd._option_window(0, 3) == (0, 3)
    assert MenuControlledEnd._option_window(2, 3) == (0, 3)


def test_enable_if_dependency_updates_live(tmp_path):
    tree = {
        "camera": {
            "0": {
                "title": "Root",
                "options": [
                    {"id": "focus", "content": "Focus", "type": "menu",
                     "value": "focus"},
                ],
            },
            "focus": {
                "title": "Focus",
                "options": [
                    {
                        "id": "af_mode",
                        "content": "AF Mode",
                        "type": "option",
                        "value": {"content": "Continuous", "value": 2},
                        "options": [
                            {"content": "Continuous", "value": 2},
                            {"content": "Manual", "value": 0},
                        ],
                    },
                    {
                        "id": "focus_position",
                        "content": "Focus Pos",
                        "type": "numeral",
                        "value": 500,
                        "min": 0,
                        "max": 1000,
                        "step": 10,
                        "enable": False,
                        "enable_if": {"id": "af_mode", "value": 0},
                    },
                ],
            },
        },
        "gallery": {"0": {"title": "Gallery", "options": []}},
    }
    path = tmp_path / "menus.json"
    path.write_text(json.dumps(tree), encoding="utf-8")
    menu = MenuControlledEnd(path=path, row_count=5)
    menu._msg_sender = lambda *args: None
    menu._irq = lambda target: None
    menu.msg_receiver("camera", "camera")
    menu.on_enter(None)

    menu.select()  # enter Focus
    position = next(
        option for option in menu._options if option["id"] == "focus_position"
    )
    assert position["enable"] is False

    menu._current_index = 0
    menu.select()  # edit af_mode
    menu._value_temp = {"content": "Manual", "value": 0}
    menu.unselect()
    position = next(
        option for option in menu._options if option["id"] == "focus_position"
    )
    assert position["enable"] is True


def test_enable_if_can_reference_another_section(tmp_path):
    tree = {
        "camera": {
            "0": {
                "title": "Root",
                "options": [
                    {
                        "id": "exposure",
                        "content": "Exposure",
                        "type": "menu",
                        "value": "exposure",
                    },
                    {
                        "id": "manual",
                        "content": "Manual",
                        "type": "menu",
                        "value": "manual",
                    },
                ],
            },
            "exposure": {
                "title": "Exposure",
                "options": [
                    {
                        "id": "auto_expose",
                        "content": "Auto Expose",
                        "type": "bool",
                        "value": True,
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
                        "value": 0.0,
                        "min": -4.0,
                        "max": 4.0,
                        "step": 0.5,
                        "enable_if": {
                            "id": "auto_expose",
                            "value": True,
                        },
                    },
                ],
            },
        },
        "gallery": {"0": {"title": "Gallery", "options": []}},
    }
    path = tmp_path / "menus.json"
    path.write_text(json.dumps(tree), encoding="utf-8")
    menu = MenuControlledEnd(path=path, row_count=5)
    menu._msg_sender = lambda *args: None
    menu._irq = lambda target: None
    menu.msg_receiver("camera", "camera")
    menu.on_enter(None)

    menu._current_index = 1  # Manual
    menu.select()
    compensation = next(
        option
        for option in menu._options
        if option["id"] == "exp_compensation"
    )
    assert compensation["enable"] is True

    menu._jump_by_id("0")
    menu._current_index = 0  # Exposure
    menu.select()
    menu._current_index = 0  # Auto Expose
    menu.select()  # toggle off

    menu._jump_by_id("0")
    menu._current_index = 1  # Manual
    menu.select()
    compensation = next(
        option
        for option in menu._options
        if option["id"] == "exp_compensation"
    )
    assert compensation["enable"] is False


def test_numeric_step_rounding_removes_float_drift():
    assert MenuControlledEnd._round_to_step(0.7 + 0.1, 0.1) == 0.8
    assert MenuControlledEnd._round_to_step(
        0.9999999999999999, 0.1
    ) == 1.0
    assert MenuControlledEnd._round_to_step(10 + 5, 5) == 15


def test_hdr_requires_auto_exposure(tmp_path):
    tree = {
        "camera": {
            "0": {
                "title": "Root",
                "options": [
                    {
                        "id": "hdr",
                        "content": "HDR",
                        "type": "option",
                        "value": {"content": "Off", "value": 0},
                        "options": [
                            {"content": "Off", "value": 0},
                            {"content": "MultiExposure", "value": 2},
                        ],
                    },
                    {
                        "id": "exposure",
                        "content": "Exposure",
                        "type": "menu",
                        "value": "exposure",
                    },
                ],
            },
            "exposure": {
                "title": "Exposure",
                "options": [
                    {
                        "id": "auto_expose",
                        "content": "Auto Expose",
                        "type": "bool",
                        "value": False,
                    },
                    {
                        "id": "ae_lock",
                        "content": "AE Lock",
                        "type": "bool",
                        "value": True,
                    },
                ],
            },
        },
        "gallery": {"0": {"title": "Gallery", "options": []}},
    }
    path = tmp_path / "menus.json"
    path.write_text(json.dumps(tree), encoding="utf-8")
    menu = MenuControlledEnd(path=path, row_count=5)
    menu._msg_sender = lambda *args: None
    menu._irq = lambda target: None
    menu.msg_receiver("camera", "camera")
    menu.on_enter(None)

    def options_by_id():
        return {
            option["id"]: option
            for section in menu._option_list.values()
            for option in section.get("options", [])
        }

    menu._current_index = 0  # HDR
    menu.select()
    menu._value_temp = {"content": "MultiExposure", "value": 2}
    menu.unselect()

    options = options_by_id()
    assert options["hdr"]["value"]["value"] == 2
    assert options["auto_expose"]["value"] is True
    assert options["ae_lock"]["value"] is False

    menu._jump_by_id("exposure")
    menu._current_index = 0  # Auto Expose
    menu.select()  # toggle off

    options = options_by_id()
    assert options["auto_expose"]["value"] is False
    assert options["hdr"]["value"]["value"] == 0


def test_all_themes_and_edit_views_render(tmp_path):
    tree = {
        "camera": {
            "0": {
                "title": "Root",
                "options": [
                    {
                        "id": "num",
                        "content": "Num",
                        "type": "numeral",
                        "value": 5,
                        "min": 0,
                        "max": 10,
                        "step": 1,
                    },
                    {
                        "id": "choice",
                        "content": "Choice",
                        "type": "option",
                        "value": {"content": "A", "value": "a"},
                        "options": [
                            {"content": "A", "value": "a"},
                            {"content": "B", "value": "b"},
                        ],
                    },
                ],
            }
        },
        "gallery": {"0": {"title": "Gallery", "options": []}},
    }
    path = tmp_path / "menus.json"
    path.write_text(json.dumps(tree), encoding="utf-8")
    menu = MenuControlledEnd(
        path=path, width=320, height=240, row_count=5,
        show_preview=True, show_index=True, font_height=14,
        padding=(5, 5, 5, 5),
    )
    menu._irq = lambda target: None
    menu._msg_sender = lambda *args: None
    menu.msg_receiver("camera", "camera")
    menu.on_enter(None)

    for theme_id in THEMES:
        menu._apply_theme(theme_id)
        menu._current_index = 0
        menu._select_index = None
        menu.decorate()
        assert next(menu.main_loop()).shape == (240, 320, 3)

    menu._current_index = 0
    menu.select()
    menu.decorate()
    assert next(menu.main_loop()).shape == (240, 320, 3)
    menu.unselect()

    menu._current_index = 1
    menu.select()
    menu.decorate()
    assert next(menu.main_loop()).shape == (240, 320, 3)
    menu.unselect()


def test_every_menu_option_has_a_drawable_icon():
    from chandelier.config import iter_options
    from chandelier.ui.decorators.icons import draw_icon

    fallback = np.zeros((24, 24, 3), dtype=np.uint8)
    draw_icon(fallback, "__missing__", 0, 0, 24, (255, 255, 255))

    for option in iter_options(load_menu_config()):
        name = option.get("icon")
        assert name, option.get("id")
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        draw_icon(frame, name, 0, 0, 24, (255, 255, 255))
        # An unknown name silently draws the fallback square.
        assert not np.array_equal(frame, fallback), (
            option.get("id"),
            name,
        )
