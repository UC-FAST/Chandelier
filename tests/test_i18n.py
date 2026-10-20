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


def test_new_settings_have_translations():
    expected = {
        "zh_CN": (
            "Web 服务器",
            "状态栏位置",
            "相机视场角",
            "标记偏航",
            "两侧",
        ),
        "zh_TW": (
            "Web 伺服器",
            "狀態列位置",
            "相機視場角",
            "標記偏航",
            "兩側",
        ),
        "ru": (
            "Веб-сервер",
            "Положение панели",
            "Угол обзора",
            "Поворот метки",
            "По бокам",
        ),
    }
    keys = ("Web Server", "Bar Position", "Camera FOV", "Marker Yaw", "Sides")
    for language, values in expected.items():
        translator.set_language(language)
        assert tuple(tr(key) for key in keys) == values


def test_status_bar_layout_translations():
    for language, expected in (
        ("zh_CN", "上下"),
        ("zh_TW", "上下"),
        ("ru", "Сверху и снизу"),
    ):
        translator.set_language(language)
        assert tr("Top + Bottom") == expected


def test_monitor_web_status_translations():
    translator.set_language("zh_CN")
    assert tr("Web") == "网页"
    assert tr("ON") == "开"
    assert tr("OFF") == "关"
    assert tr("SERVICE") == "服务"
    assert tr("Throttle") == "降频"
    assert tr("Last DOA") == "最近定位"


def test_web_page_follows_the_active_language():
    from chandelier.services.preview_server import _gallery_html, _index_html

    translator.set_language("zh_CN")
    html = _index_html().decode("utf-8")
    assert 'lang="zh_CN"' in html
    assert ">相机<" in html
    assert ">菜单<" in html
    assert 'alt="实时画面"' in html
    assert "快照" in html
    assert "图库" in html
    assert "图库预览" in html

    gallery = _gallery_html().decode("utf-8")
    assert 'lang="zh_CN"' in gallery
    assert "图库" in gallery
    assert "暂无媒体" in gallery
    assert "下载" in gallery


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


def test_new_camera_controls_have_translations():
    keys = (
        "Touch Focus",
        "Exposure Priority",
        "Shutter Priority",
        "Gain Priority",
        "Noise Reduction",
        "High Quality",
        "AF Pause",
        "Preview FPS",
        "Colour Temperature",
        "ISP Stats",
    )
    expected = {
        "zh_CN": (
            "触摸对焦",
            "曝光优先",
            "快门优先",
            "增益优先",
            "降噪",
            "高质量",
            "对焦暂停",
            "预览帧率",
            "色温",
            "ISP 统计",
        ),
        "zh_TW": (
            "觸控對焦",
            "曝光優先",
            "快門優先",
            "增益優先",
            "降噪",
            "高品質",
            "對焦暫停",
            "預覽幀率",
            "色溫",
            "ISP 統計",
        ),
        "ru": (
            "Сенсорная фокусировка",
            "Приоритет экспозиции",
            "Приоритет выдержки",
            "Приоритет усиления",
            "Шумоподавление",
            "Высокое качество",
            "Пауза автофокуса",
            "Частота кадров",
            "Цветовая температура",
            "Статистика ISP",
        ),
    }
    for language, values in expected.items():
        translator.set_language(language)
        assert tuple(tr(key) for key in keys) == values


def test_virtual_button_labels_have_translations():
    keys = ("Overlay", "Rotate left", "Rotate right", "Select")
    expected = {
        "zh_CN": ("叠加", "左旋", "右旋", "选择"),
        "zh_TW": ("疊加", "左旋", "右旋", "選取"),
        "ru": ("Оверлей", "Влево", "Вправо", "Выбор"),
    }
    for language, values in expected.items():
        translator.set_language(language)
        assert tuple(tr(key) for key in keys) == values
