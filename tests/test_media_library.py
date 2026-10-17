import os

import cv2
import numpy as np
import pytest

from chandelier.media.library import MediaBrowser


def _image(path, value=10):
    cv2.imwrite(str(path), np.full((60, 80, 3), value, dtype=np.uint8))


def test_discovery_and_frame_loading(tmp_path):
    _image(tmp_path / "a.png", 10)
    _image(tmp_path / "b.png", 200)
    (tmp_path / "notes.txt").write_text("ignored")

    browser = MediaBrowser(str(tmp_path), 320, 240)
    assert browser.total_media() == 2
    frame = browser.get_current_frame()
    assert frame.shape == (240, 320, 3)
    assert browser.get_current_file_name() in {"a.png", "b.png"}


def test_navigation_and_position(tmp_path):
    _image(tmp_path / "a.png", 10)
    _image(tmp_path / "b.png", 200)
    browser = MediaBrowser(str(tmp_path), 320, 240)
    start = browser.current_position()

    browser.next()
    assert browser.current_position() != start
    browser.previous()
    assert browser.current_position() == start


def test_refresh_keeps_cached_current_frame(tmp_path):
    _image(tmp_path / "a.png", 10)
    browser = MediaBrowser(str(tmp_path), 320, 240)
    frame = browser.get_current_frame()
    cached = browser._media_cache[1]
    assert cached is not None

    browser.refresh_media_list()
    assert browser._media_cache[1] is cached
    assert browser.get_current_frame() is frame


def test_delete_current(tmp_path):
    _image(tmp_path / "a.png", 10)
    browser = MediaBrowser(str(tmp_path), 320, 240)
    path = browser.get_current_media_info()["path"]

    assert browser.delete_current() is True
    assert not os.path.exists(path)
    assert browser.total_media() == 0


def test_empty_directory_raises(tmp_path):
    browser = MediaBrowser(str(tmp_path), 320, 240)
    assert browser.total_media() == 0
    with pytest.raises(FileExistsError):
        browser.get_current_frame()


def test_jump_to_index(tmp_path):
    for index in range(3):
        _image(tmp_path / f"{index}.png", index * 20)
    browser = MediaBrowser(str(tmp_path), 320, 240)
    browser.jump_to_index(2)
    assert browser.current_position() == 2


def test_toggle_play_pause_on_image_is_noop(tmp_path):
    _image(tmp_path / "a.png", 10)
    browser = MediaBrowser(str(tmp_path), 320, 240)
    browser.toggle_play_pause()
    assert browser.is_playing() is False


def _write_video(path, frames=20, fps=10, size=(64, 48)):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size
    )
    if not writer.isOpened():
        pytest.skip("MJPG VideoWriter unavailable")
    for index in range(frames):
        writer.write(
            np.full((size[1], size[0], 3), index * 5 % 255, dtype=np.uint8)
        )
    writer.release()


def test_video_playback_advance_pause_and_eof(tmp_path):
    path = tmp_path / "clip.avi"
    _write_video(path)
    browser = MediaBrowser(str(tmp_path), 320, 240)

    assert browser.is_video() is True
    assert browser.get_current_frame().shape == (240, 320, 3)
    assert browser.video_fps == pytest.approx(10.0, rel=0.2)
    duration = browser.get_video_time()[1]
    assert duration > 0

    assert browser.play() is True
    assert browser.is_playing() is True
    frame = browser.advance_video_frame()
    assert frame is not None
    assert browser.get_video_time()[0] > 0

    browser.pause()
    assert browser.is_playing() is False

    # Advance to the end without looping.
    browser.play()
    for _ in range(200):
        if browser.advance_video_frame() is None:
            break
    assert browser.is_playing() is False


def test_video_loop_step_and_seek(tmp_path):
    path = tmp_path / "clip.avi"
    _write_video(path)
    browser = MediaBrowser(str(tmp_path), 320, 240)
    browser.get_current_frame()

    browser.set_loop(True)
    assert browser.is_looping() is True
    browser.play()
    for _ in range(25):
        browser.advance_video_frame()
    assert browser.is_playing() is True

    browser.pause()
    frame = browser.step_frame(1)
    assert frame is not None
    browser.step_frame(-1)
    assert browser.get_video_position() >= 0.0
    browser.seek_video(0.5)
    assert browser.get_video_position() == pytest.approx(0.5, abs=0.08)
    assert browser.is_playing() is False

    frame = browser.seek_relative(0.1)
    assert frame is not None
    frame = browser.seek_relative(1000)
    assert frame is not None
    assert browser.get_video_position() >= 0.9


def test_toggle_play_pause_returns_state(tmp_path):
    path = tmp_path / "clip.avi"
    _write_video(path)
    browser = MediaBrowser(str(tmp_path), 320, 240)
    assert browser.toggle_play_pause() is True
    assert browser.toggle_play_pause() is False
