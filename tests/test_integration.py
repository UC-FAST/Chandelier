"""Headless integration test: real ends, controller and message bus."""

import shutil
import threading
import time

import cv2
import numpy as np

from chandelier.config import load_config
from chandelier.constants import EndId
from chandelier.controller import Controller
from chandelier.events import Action
from chandelier.media.library import MediaBrowser
from chandelier.ui.camera_end import CameraControlledEnd
from chandelier.ui.gallery_end import GalleryControlledEnd
from chandelier.ui.menu_end import MenuControlledEnd
from chandelier.ui.system_monitor_end import SystemMonitor
from tests.fakes import FakeCamera, FakeDisplay, FakeLeds, FakeStats


def _wait_for(predicate, timeout=3.0):
    deadline = time.time() + timeout
    while not predicate() and time.time() < deadline:
        time.sleep(0.01)
    return predicate()


def test_full_application_switches_screens(tmp_path, project_root):
    config = load_config()

    image_dir = tmp_path / "pict"
    image_dir.mkdir()
    cv2.imwrite(
        str(image_dir / "a.png"),
        np.full((60, 80, 3), 120, dtype=np.uint8),
    )

    menus_path = tmp_path / "menus.json"
    shutil.copy(project_root / "config" / "menus.json", menus_path)

    display = FakeDisplay(config.screen.width, config.screen.height)
    camera = FakeCamera(config.screen.width, config.screen.height)
    leds = FakeLeds()
    library = MediaBrowser(
        str(image_dir), config.screen.width, config.screen.height
    )

    ends = [
        CameraControlledEnd(camera, leds, config),
        SystemMonitor(FakeStats(), config),
        MenuControlledEnd(
            path=menus_path,
            width=config.screen.width,
            height=config.screen.height,
            show_preview=True,
            row_count=5,
            show_index=True,
            font_height=14,
            padding=(5, 5, 5, 5),
        ),
        GalleryControlledEnd(library, config),
    ]
    controller = Controller(display, ends, config)

    thread = threading.Thread(
        target=controller.main_loop, kwargs={"localization": None}, daemon=True
    )
    thread.start()

    assert _wait_for(lambda: len(display.frames) > 0)
    assert camera.calls  # camera on_enter received the menu tree

    controller.switch_to(EndId.MENU.value)
    assert _wait_for(
        lambda: controller.current_end.id == EndId.MENU.value
    )
    time.sleep(0.1)
    assert controller.handle_action(Action.CENTER)

    controller.switch_to(EndId.GALLERY.value)
    assert _wait_for(
        lambda: controller.current_end.id == EndId.GALLERY.value
    )

    controller.switch_to(EndId.SYSTEM_MONITOR.value)
    assert _wait_for(
        lambda: controller.current_end.id == EndId.SYSTEM_MONITOR.value
    )

    controller.switch_to(EndId.CAMERA.value)
    assert _wait_for(
        lambda: controller.current_end.id == EndId.CAMERA.value
    )

    controller.request_stop()
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert display.backlight_state is False
