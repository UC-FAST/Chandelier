import os

import cv2
import numpy as np

from chandelier.config import load_config
from chandelier.media.library import MediaBrowser
from chandelier.ui.gallery_end import GalleryControlledEnd


def _write_image(directory, name, value):
    image = np.full((60, 80, 3), value, dtype=np.uint8)
    cv2.imwrite(str(directory / name), image)


def _make_gallery(tmp_path):
    _write_image(tmp_path, "a.png", 10)
    _write_image(tmp_path, "b.png", 200)
    config = load_config()
    library = MediaBrowser(str(tmp_path), config.screen.width, config.screen.height)
    end = GalleryControlledEnd(library, config)
    irqs = []
    messages = []
    end._irq = irqs.append
    end._msg_sender = lambda sender, receiver, payload: messages.append(
        (sender, receiver, payload)
    )
    return end, library, irqs, messages


def test_enter_refreshes_and_yields_frame(tmp_path):
    end, library, _, messages = _make_gallery(tmp_path)
    end.on_enter("menu")

    assert library.total_media() == 2
    assert messages
    frame = next(end.main_loop())
    assert frame.shape == (240, 320, 3)


def test_navigation_actions(tmp_path):
    end, library, _, _ = _make_gallery(tmp_path)
    end.on_enter("menu")
    next(end.main_loop())

    first = library.current_position()
    end.right_press_action()
    assert library.current_position() != first
    frame = next(end.main_loop())
    assert frame.shape == (240, 320, 3)

    end.left_press_action()
    assert library.current_position() == first


def test_circle_toggles_text_overlay(tmp_path):
    end, _, _, _ = _make_gallery(tmp_path)
    end.on_enter("menu")
    next(end.main_loop())
    end.circle_press_action()
    assert end._simple_text_enabled is True
    assert next(end.main_loop()).shape == (240, 320, 3)


def test_empty_gallery_returns_to_camera(tmp_path):
    config = load_config()
    library = MediaBrowser(str(tmp_path), config.screen.width, config.screen.height)
    end = GalleryControlledEnd(library, config)
    irqs = []
    end._irq = irqs.append
    end._msg_sender = lambda *args: None

    end.on_enter("menu")
    assert end._empty is True
    assert next(end.main_loop()).shape == (128, 128, 3)

    end.cross_press_action()
    assert end._empty is False
    assert irqs[-1] == "menu"


def test_delete_message_removes_file(tmp_path):
    end, library, _, _ = _make_gallery(tmp_path)
    end.on_enter("menu")
    path = library.get_current_media_info()["path"]

    tree = {"0": {"title": "Gallery", "options": []}}
    end.msg_receiver("menu", ("delete", tree))

    assert not os.path.exists(path)
    assert library.total_media() == 1


def test_on_exit_emits_warning_frame(tmp_path):
    end, _, _, _ = _make_gallery(tmp_path)
    end.on_enter("menu")
    next(end.main_loop())  # consume the frame queued by on_enter
    end.on_exit()
    frame = next(end.main_loop())
    assert frame.shape == (128, 128, 3)


def _write_video(path, frames=30, fps=10, size=(64, 48)):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size
    )
    if not writer.isOpened():
        import pytest
        pytest.skip("MJPG VideoWriter unavailable")
    for index in range(frames):
        writer.write(
            np.full((size[1], size[0], 3), index * 5 % 255, dtype=np.uint8)
        )
    writer.release()


def _make_video_gallery(tmp_path):
    _write_video(tmp_path / "clip.avi")
    config = load_config()
    library = MediaBrowser(
        str(tmp_path), config.screen.width, config.screen.height
    )
    end = GalleryControlledEnd(library, config)
    end._irq = lambda target: None
    end._msg_sender = lambda *args: None
    end.on_enter("menu")
    return end, library


def test_video_center_starts_and_pauses_playback(tmp_path):
    end, library = _make_video_gallery(tmp_path)
    assert next(end.main_loop()).shape == (240, 320, 3)
    assert library.is_playing() is False

    end.center_press_action()
    assert library.is_playing() is True
    assert next(end.main_loop()).shape == (240, 320, 3)
    assert end._hud.visible is True

    end.center_press_action()
    assert library.is_playing() is False


def test_video_timed_pump_advances_frames(tmp_path):
    end, library = _make_video_gallery(tmp_path)
    next(end.main_loop())
    end.center_press_action()
    next(end.main_loop())  # consume the queued current frame

    start = library.get_video_time()[0]
    next(end.main_loop())  # timed pump produces the next frame
    assert library.get_video_time()[0] > start


def test_video_step_seek_and_loop(tmp_path):
    end, library = _make_video_gallery(tmp_path)
    next(end.main_loop())

    end.right_press_action()
    assert next(end.main_loop()).shape == (240, 320, 3)
    position = library.get_video_time()[0]

    end.rotary_encoder_clockwise()
    assert next(end.main_loop()).shape == (240, 320, 3)
    assert library.get_video_time()[0] >= position

    end.shutter_press_action()
    assert library.is_looping() is True
    end.shutter_press_action()
    assert library.is_looping() is False
