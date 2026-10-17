"""Tests for the language layer and Unicode menu rendering."""

import shutil

import numpy as np
import pytest

from chandelier.config import load_menu_config
from chandelier.i18n import (
    language_from_options,
    normalize_language,
    tr,
    translator,
)
from chandelier.ui.menu_end import MenuControlledEnd
from chandelier.ui.text_render import get_renderer


@pytest.fixture(autouse=True)
def reset_language():
    translator.set_language("en")
    yield
    translator.set_language("en")


def test_language_normalization():
    assert normalize_language("en") == "en"
    assert normalize_language("zh-CN") == "zh_CN"
    assert normalize_language("zh_hans") == "zh_CN"
    assert normalize_language("zh-TW") == "zh_TW"
    assert normalize_language("ru") == "ru"
    assert normalize_language("unknown") == "en"


def test_translations_cover_main_menus():
    translator.set_language("zh_CN")
    assert tr("Camera Settings") == "相机设置"
    assert tr("Show Histogram") == "显示直方图"
    translator.set_language("zh_TW")
    assert tr("Camera Settings") == "相機設定"
    translator.set_language("ru")
    assert tr("Camera Settings") == "Настройки камеры"
    assert tr("Reboot System") == "Перезагрузить систему"


def test_language_from_options():
    tree = {
        "camera": {
            "system": {
                "options": [
                    {
                        "id": "language",
                        "value": {"content": "Русский", "value": "ru"},
                    }
                ]
            }
        }
    }
    assert language_from_options(tree) == "ru"


def test_unicode_renderer_handles_mixed_scripts():
    translator.set_language("zh_CN")
    renderer = get_renderer()
    text = "CPU 频率 1 OK"
    width, height = renderer.measure(text, 14)
    assert width > 0 and height > 0

    frame = np.zeros((40, 240, 3), dtype=np.uint8)
    renderer.draw(frame, text, (4, 28), 14, (255, 255, 255), 1)
    assert frame.any()


def test_menu_language_change_is_applied_and_persisted(
    tmp_path, project_root
):
    path = tmp_path / "menus.json"
    shutil.copy(project_root / "config" / "menus.json", path)
    menu = MenuControlledEnd(path=path)
    menu._msg_sender = lambda *args: None
    menu.msg_receiver("camera", "camera")

    option = next(
        option
        for option in menu._menu_options["camera"]["system"]["options"]
        if option["id"] == "language"
    )
    menu._from = "camera"
    menu._current_menu_id = "system"
    menu._current_options = [option]
    menu._current_index = 0
    menu._select_index = 0
    menu._value_temp = {"content": "简体中文", "value": "zh_CN"}
    menu.unselect()

    assert translator.language == "zh_CN"
    reloaded = load_menu_config(path)
    language = next(
        option
        for option in reloaded["camera"]["system"]["options"]
        if option["id"] == "language"
    )
    assert language["value"]["value"] == "zh_CN"
