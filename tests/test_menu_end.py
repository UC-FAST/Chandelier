import json
import shutil

import numpy as np

from chandelier.config import load_menu_config
from chandelier.ui.menu_end import MenuControlledEnd


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

    for theme_id in (
        "dark_pro",
        "high_contrast",
        "industrial",
        "classic",
        "xp_luna",
        "win2000",
        "win7",
        "win10",
    ):
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
