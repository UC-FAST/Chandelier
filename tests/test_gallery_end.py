import os
import time

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


def test_navigation_loads_an_image_that_was_not_prefetched(
    tmp_path, monkeypatch
):
    from chandelier.media import library as library_module

    _write_image(tmp_path, "a.png", 10)
    _write_image(tmp_path, "b.png", 200)
    monkeypatch.setattr(library_module, "_PREFETCH_MAX_BYTES", 0)

    config = load_config()
    library = MediaBrowser(
        str(tmp_path), config.screen.width, config.screen.height
    )
    end = GalleryControlledEnd(library, config)
    end._irq = lambda *args: None
    end._msg_sender = lambda *args: None
    end.on_enter("menu")

    generator = end.main_loop()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        next(generator)
        if library.is_current_frame_ready():
            break
    assert library.is_current_frame_ready()

    end.right_press_action()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        frame = next(generator)
        if library.is_current_frame_ready():
            break
    assert library.is_current_frame_ready()
    assert frame.shape == (240, 320, 3)


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

    # Loop playback is configured from the menu now.
    end._menu_tree = {
        "player": {
            "options": [{"id": "loop_playback", "value": True}]
        }
    }
    end._read_player_settings()
    assert library.is_looping() is True
    end._menu_tree["player"]["options"][0]["value"] = False
    end._read_player_settings()
    assert library.is_looping() is False

    # Shutter toggles the thumbnail grid.
    end.shutter_press_action()
    assert end._grid_mode is True
    assert next(end.main_loop()).shape == (240, 320, 3)
    end.shutter_press_action()
    assert end._grid_mode is False
    assert next(end.main_loop()).shape == (240, 320, 3)


def test_grid_view_navigation_and_open(tmp_path):
    end, library, _, _ = _make_gallery(tmp_path)
    end.on_enter("menu")
    next(end.main_loop())

    end.shutter_press_action()
    assert end._grid_mode is True
    first = next(end.main_loop())
    assert first.shape == (240, 320, 3)

    end.right_press_action()
    assert end._grid_selection == 1
    assert next(end.main_loop()).shape == (240, 320, 3)

    end.center_press_action()
    assert end._grid_mode is False
    assert library.current_position() == 1
    assert next(end.main_loop()).shape == (240, 320, 3)


def test_grid_view_pages_with_rotary(tmp_path):
    for index in range(12):
        _write_image(tmp_path, f"{index:02d}.png", index * 10)
    config = load_config()
    library = MediaBrowser(
        str(tmp_path), config.screen.width, config.screen.height
    )
    end = GalleryControlledEnd(library, config)
    end._irq = lambda target: None
    end._msg_sender = lambda *args: None
    end.on_enter("menu")
    next(end.main_loop())

    end.shutter_press_action()
    next(end.main_loop())
    end.rotary_encoder_clockwise()
    assert end._grid_selection == 9
    assert next(end.main_loop()).shape == (240, 320, 3)
    end.rotary_encoder_counter_clockwise()
    assert end._grid_selection == 0


def test_image_is_letterboxed_not_stretched(tmp_path):
    image = np.full((30, 80, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "wide.png"), image)
    config = load_config()
    library = MediaBrowser(
        str(tmp_path), config.screen.width, config.screen.height
    )
    frame = library.get_current_frame()
    assert frame.shape == (240, 320, 3)
    # A 80x30 source scaled to 320 wide is only 120 high: black bars remain.
    assert int(frame[0].max()) == 0
    assert int(frame[-1].max()) == 0
    assert int(frame[120].max()) > 0


def test_info_line_includes_position_and_details(tmp_path):
    end, _, _, _ = _make_gallery(tmp_path)
    end.on_enter("menu")
    text = next(iter(end._filename_page().values()))
    assert "1/2" in text
    assert ".png" in text
    assert "80x60" in text


def test_thumbnail_generation(tmp_path):
    end, library, _, _ = _make_gallery(tmp_path)
    end.on_enter("menu")
    deadline = time.monotonic() + 10.0
    thumbnail = library.get_thumbnail(0, 40, 30)
    while thumbnail is None and time.monotonic() < deadline:
        time.sleep(0.02)
        thumbnail = library.get_thumbnail(0, 40, 30)
    assert thumbnail is not None
    assert thumbnail.shape == (30, 40, 3)
    details = library.get_media_details(0)
    assert details["type"] == "image"
    assert details["width"] == 80
    assert details["height"] == 60
    assert details["size"] > 0


def test_rotary_select_toggles_the_grid(tmp_path):
    end, library, _, _ = _make_gallery(tmp_path)
    end.on_enter("menu")
    next(end.main_loop())

    end.rotary_encoder_select()
    assert end._grid_mode is True
    assert next(end.main_loop()).shape == (240, 320, 3)

    first = library.current_position()
    end.rotary_encoder_select()
    assert end._grid_mode is False
    assert library.current_position() == first
    assert next(end.main_loop()).shape == (240, 320, 3)


def test_loading_indicator_uses_red_square(tmp_path):
    end, _, _, _ = _make_gallery(tmp_path)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    end._draw_loading_indicator(frame)
    assert tuple(int(v) for v in frame[9, 310]) == (0, 0, 255)


def test_compose_shows_loading_indicator_until_ready(tmp_path, monkeypatch):
    end, library, _, _ = _make_gallery(tmp_path)
    end.on_enter("menu")

    monkeypatch.setattr(library, "is_current_frame_ready", lambda: False)
    frame = end._compose(np.zeros((240, 320, 3), dtype=np.uint8))
    assert tuple(int(v) for v in frame[9, 310]) == (0, 0, 255)

    monkeypatch.setattr(library, "is_current_frame_ready", lambda: True)
    frame = end._compose(np.zeros((240, 320, 3), dtype=np.uint8))
    assert tuple(int(v) for v in frame[9, 310]) == (0, 0, 0)


def test_grid_shows_loading_indicator_while_pending(tmp_path, monkeypatch):
    end, library, _, _ = _make_gallery(tmp_path)
    end.on_enter("menu")
    monkeypatch.setattr(
        type(library),
        "has_pending_thumbnails",
        property(lambda self: True),
    )
    frame = end._render_grid()
    assert tuple(int(v) for v in frame[9, 310]) == (0, 0, 255)
