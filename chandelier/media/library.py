"""Lazy image/video browser with a three-slot LRU-style cache.

The cache holds ``[previous, current, next]``.  Images are loaded eagerly for
the current item and asynchronously for neighbours; videos are opened lazily.
"""

from __future__ import annotations

import os
import queue
import threading
from typing import Optional

import cv2
import numpy as np

from chandelier.i18n import tr

IMAGE_EXTENSIONS = {"jpeg", "jpg", "png", "bmp", "tiff", "tif", "webp", "gif"}
VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "flv", "wmv"}


class MediaBrowser:
    def __init__(
        self,
        media_path: str = "./media/",
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):
        self._media_path = os.path.abspath(media_path)
        os.makedirs(self._media_path, exist_ok=True)

        self._target_width = width
        self._target_height = height

        self._cache_size = 3
        self._media_cache: list[Optional[dict]] = [None] * self._cache_size
        self._cache_lock = threading.RLock()

        self._current_index = 0
        self._media_files: list[dict] = []
        self._load_generation = 0
        self._load_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._load_worker: threading.Thread | None = None

        self._video_capture: Optional[cv2.VideoCapture] = None
        self._is_playing = False
        self._video_position = 0
        self._video_fps = 0.0
        self._video_total_frames = 0
        self._video_ready = False
        self._video_frame_cache: Optional[np.ndarray] = None
        self._video_cache_position = -1
        self._current_video_path: Optional[str] = None
        self._video_loop = False
        self._video_lock = threading.RLock()
        self._loading_animation = 0

        self.refresh_media_list()

    # ------------------------------------------------------------------
    # Discovery and cache
    # ------------------------------------------------------------------
    def refresh_media_list(self) -> None:
        # Invalidate any in-flight background decode before the file list and
        # cache are rebuilt.
        self._load_generation += 1
        previous_files = self._media_files
        previous_info = self.get_current_media_info()
        with self._cache_lock:
            previous_cache = self._media_cache[1]

        entries: list[tuple[float, str, str]] = []
        with os.scandir(self._media_path) as directory:
            for entry in directory:
                try:
                    if not entry.is_file():
                        continue
                    extension = os.path.splitext(entry.name)[1][1:].lower()
                    if extension in IMAGE_EXTENSIONS:
                        media_type = "image"
                    elif extension in VIDEO_EXTENSIONS:
                        media_type = "video"
                    else:
                        continue
                    entries.append(
                        (entry.stat().st_mtime, entry.path, media_type)
                    )
                except OSError:
                    continue

        entries.sort(key=lambda item: item[0], reverse=True)
        self._media_files = [
            {"path": path, "type": media_type}
            for _, path, media_type in entries
        ]

        if not self._media_files:
            with self._cache_lock:
                self._media_cache = [None] * self._cache_size
            return
        self._current_index = min(
            self._current_index, len(self._media_files) - 1
        )

        # Re-entering the gallery used to clear the cache and decode the
        # current photo again even when the directory had not changed.  Keep
        # the cached current item when the sorted list and index still match.
        same_files = [item["path"] for item in previous_files] == [
            item["path"] for item in self._media_files
        ]
        current_path = self._media_files[self._current_index]["path"]
        keep_current = bool(
            same_files
            and previous_cache is not None
            and previous_cache.get("index") == self._current_index
            and previous_info is not None
            and previous_info["path"] == current_path
        )

        with self._cache_lock:
            if keep_current:
                self._media_cache[0] = None
                self._media_cache[2] = None
            else:
                self._media_cache = [None] * self._cache_size

        if not keep_current:
            self._load_media(self._current_index, cache_pos=1)
        if self._current_index < len(self._media_files) - 1:
            self._load_media(
                self._current_index + 1, cache_pos=2, async_load=True
            )

    def _load_media(
        self, index: int, cache_pos: int, async_load: bool = False
    ) -> None:
        if index < 0 or index >= len(self._media_files):
            return

        # A monotonically increasing generation token makes older background
        # loads abandon their result without ever joining their thread from
        # the UI thread (which used to stall navigation for up to 100 ms).
        self._load_generation += 1
        generation = self._load_generation

        if async_load:
            self._load_queue.put((index, cache_pos, generation))
            if (
                self._load_worker is None
                or not self._load_worker.is_alive()
            ):
                self._load_worker = threading.Thread(
                    target=self._load_worker_loop,
                    daemon=True,
                    name="media-loader",
                )
                self._load_worker.start()
        else:
            self._load_task(index, cache_pos, generation)

    def _load_worker_loop(self) -> None:
        # One decoder thread only: stale requests are skipped by generation,
        # so rapid navigation cannot pile up concurrent image decodes.
        while True:
            index, cache_pos, generation = self._load_queue.get()
            if generation != self._load_generation:
                continue
            self._load_task(index, cache_pos, generation)

    def _decode_image(self, file_path: str) -> np.ndarray | None:
        """Decode an image near the display size instead of full resolution.

        Camera photos are commonly 12 MP while the LCD is only 320x240.
        ``IMREAD_REDUCED_*`` lets libjpeg/libpng scale during decoding, which
        avoids both the decode time and the large temporary buffer.
        """

        image_data = np.fromfile(file_path, dtype=np.uint8)
        if not self._target_width or not self._target_height:
            return cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        for flag in (
            cv2.IMREAD_REDUCED_COLOR_4,
            cv2.IMREAD_REDUCED_COLOR_2,
        ):
            image = cv2.imdecode(image_data, flag)
            if image is None:
                continue
            height, width = image.shape[:2]
            if (
                width >= self._target_width
                and height >= self._target_height
            ):
                return image
        return cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    def _load_task(
        self, index: int, cache_pos: int, generation: int
    ) -> None:
        if generation != self._load_generation:
            return
        if index < 0 or index >= len(self._media_files):
            return

        media_info = self._media_files[index]
        media_type = media_info["type"]
        file_path = media_info["path"]
        result = None

        try:
            if media_type == "image":
                image = self._decode_image(file_path)
                if image is not None:
                    if (
                        self._target_width
                        and self._target_height
                        and (
                            image.shape[1] != self._target_width
                            or image.shape[0] != self._target_height
                        )
                    ):
                        image = cv2.resize(
                            image,
                            (self._target_width, self._target_height),
                        )
                    result = {
                        "index": index,
                        "type": "image",
                        "data": image,
                    }
            elif media_type == "video":
                result = {
                    "index": index,
                    "type": "video",
                    "path": file_path,
                }
        except Exception:
            return

        if result is not None and generation == self._load_generation:
            with self._cache_lock:
                self._media_cache[cache_pos] = result

    # ------------------------------------------------------------------
    # Video playback
    # ------------------------------------------------------------------
    def _open_video(self, file_path: str) -> bool:
        with self._video_lock:
            self._close_video()
            self._video_capture = cv2.VideoCapture(file_path)
            if not self._video_capture.isOpened():
                self._video_capture = None
                return False

            self._video_fps = float(
                self._video_capture.get(cv2.CAP_PROP_FPS) or 0.0
            )
            self._video_total_frames = int(
                self._video_capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            )
            self._video_position = 0
            self._video_ready = False
            self._video_frame_cache = None
            self._video_cache_position = -1
            self._is_playing = False
            self._current_video_path = file_path
            self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return True

    def _close_video(self) -> None:
        with self._video_lock:
            if self._video_capture is not None:
                self._video_capture.release()
                self._video_capture = None
            self._video_ready = False
            self._video_frame_cache = None
            self._video_cache_position = -1
            self._is_playing = False
            self._current_video_path = None

    def _ensure_video_open_locked(self, media: dict | None = None) -> bool:
        media = media or self.get_current_media_info()
        if media is None or media.get("type") != "video":
            return False
        if (
            self._video_capture is None
            or self._current_video_path != media["path"]
        ):
            return self._open_video(media["path"])
        return True

    def _decode_at_locked(self, position: int) -> np.ndarray | None:
        if self._video_capture is None:
            return None
        if self._video_total_frames > 0:
            position = max(0, min(position, self._video_total_frames - 1))
        else:
            position = max(0, position)
        self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, position)
        ok, frame = self._video_capture.read()
        if not ok:
            self._video_ready = False
            return None
        self._video_position = position
        self._video_ready = True
        result = self._resize(frame)
        self._video_frame_cache = result.copy()
        self._video_cache_position = position
        return result

    def _read_next_locked(self) -> np.ndarray | None:
        """Decode the next frame sequentially without an expensive seek."""

        if self._video_capture is None:
            return None
        ok, frame = self._video_capture.read()
        if not ok:
            self._video_ready = False
            return None
        if self._video_ready:
            self._video_position += 1
        self._video_ready = True
        result = self._resize(frame)
        self._video_frame_cache = result.copy()
        self._video_cache_position = self._video_position
        return result

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def next(self) -> "MediaBrowser":
        if self._current_index >= len(self._media_files) - 1:
            return self

        self._close_video()
        self._current_index += 1
        with self._cache_lock:
            self._media_cache[0] = self._media_cache[1]
            self._media_cache[1] = self._media_cache[2]
            self._media_cache[2] = None

        if self._current_index < len(self._media_files) - 1:
            self._load_media(
                self._current_index + 1, cache_pos=2, async_load=True
            )
        return self

    def previous(self) -> "MediaBrowser":
        if self._current_index <= 0:
            return self

        self._close_video()
        self._current_index -= 1
        with self._cache_lock:
            self._media_cache[2] = self._media_cache[1]
            self._media_cache[1] = self._media_cache[0]
            self._media_cache[0] = None

        if self._current_index > 0:
            self._load_media(
                self._current_index - 1, cache_pos=0, async_load=True
            )
        return self

    def jump_to_index(self, index: int) -> None:
        if not 0 <= index < len(self._media_files):
            return

        self._close_video()
        self._current_index = index
        with self._cache_lock:
            self._media_cache = [None] * self._cache_size

        self._load_media(index, cache_pos=1)
        if index > 0:
            self._load_media(index - 1, cache_pos=0, async_load=True)
        if index < len(self._media_files) - 1:
            self._load_media(index + 1, cache_pos=2, async_load=True)

    def delete_current(self) -> bool:
        if not self._media_files:
            return False

        self._close_video()
        file_path = self._media_files[self._current_index]["path"]
        try:
            os.remove(file_path)
        except OSError:
            return False

        self._media_files.pop(self._current_index)
        if self._current_index >= len(self._media_files):
            self._current_index = max(0, len(self._media_files) - 1)

        with self._cache_lock:
            self._media_cache = [None] * self._cache_size
        if self._media_files:
            self._load_media(self._current_index, cache_pos=1)
            if self._current_index < len(self._media_files) - 1:
                self._load_media(
                    self._current_index + 1, cache_pos=2, async_load=True
                )
        return True

    # ------------------------------------------------------------------
    # Playback state
    # ------------------------------------------------------------------
    @property
    def video_fps(self) -> float:
        return self._video_fps if self._video_fps > 0 else 30.0

    def is_video(self) -> bool:
        media = self.get_current_media_info()
        return bool(media and media.get("type") == "video")

    def set_loop(self, enable: bool) -> None:
        self._video_loop = bool(enable)

    def is_looping(self) -> bool:
        return self._video_loop

    def play(self) -> bool:
        with self._video_lock:
            media = self.get_current_media_info()
            if media is None or media.get("type") != "video":
                return False
            if not self._ensure_video_open_locked(media):
                return False
            if (
                self._video_total_frames > 0
                and self._video_position >= self._video_total_frames - 1
            ):
                self._video_position = 0
                self._video_ready = False
                self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._is_playing = True
            return True

    def pause(self) -> None:
        with self._video_lock:
            self._is_playing = False

    def toggle_play_pause(self) -> bool:
        with self._video_lock:
            if self._is_playing:
                self._is_playing = False
                return False
            return self.play()

    def advance_video_frame(self) -> np.ndarray | None:
        with self._video_lock:
            if not self._is_playing:
                return None
            media = self.get_current_media_info()
            if not self._ensure_video_open_locked(media):
                self._is_playing = False
                return None

            next_position = (
                self._video_position + 1 if self._video_ready else 0
            )
            if (
                self._video_total_frames > 0
                and next_position >= self._video_total_frames
            ):
                if self._video_loop:
                    next_position = 0
                    self._video_position = 0
                    self._video_ready = False
                    self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    self._is_playing = False
                    return None

            # Sequential read is much cheaper than setting
            # CAP_PROP_POS_FRAMES for every frame (which performs a seek).
            frame = self._read_next_locked()
            if frame is None:
                self._is_playing = False
                return None
            return frame

    def step_frame(self, delta: int) -> np.ndarray | None:
        with self._video_lock:
            media = self.get_current_media_info()
            if not self._ensure_video_open_locked(media):
                return None
            self._is_playing = False
            return self._decode_at_locked(self._video_position + int(delta))

    def seek_relative(self, seconds: float) -> np.ndarray | None:
        with self._video_lock:
            media = self.get_current_media_info()
            if not self._ensure_video_open_locked(media):
                return None
            delta = int(round(float(seconds) * self.video_fps))
            return self._decode_at_locked(self._video_position + delta)

    def seek_video(self, position: float) -> None:
        with self._video_lock:
            media = self.get_current_media_info()
            if not self._ensure_video_open_locked(media):
                return
            if self._video_total_frames > 0:
                frame = int(position * self._video_total_frames)
                frame = max(0, min(frame, self._video_total_frames - 1))
            else:
                frame = 0
            self._is_playing = False
            self._decode_at_locked(frame)

    def is_playing(self) -> bool:
        return self._is_playing

    def get_video_time(self) -> tuple[float, float]:
        with self._video_lock:
            if self._video_capture is None or self._video_fps <= 0:
                return 0.0, 0.0
            return (
                self._video_position / self._video_fps,
                self._video_total_frames / self._video_fps,
            )

    def get_video_progress(self) -> tuple[float, float]:
        return self.get_video_time()

    def get_video_position(self) -> float:
        with self._video_lock:
            if self._video_capture is None or self._video_total_frames == 0:
                return 0.0
            return self._video_position / self._video_total_frames

    # ------------------------------------------------------------------
    # Current media
    # ------------------------------------------------------------------
    def get_current_media_info(self) -> Optional[dict]:
        if (
            not self._media_files
            or self._current_index >= len(self._media_files)
        ):
            return None
        return self._media_files[self._current_index]

    def get_current_file_name(self, full_path: bool = False) -> Optional[str]:
        media = self.get_current_media_info()
        if media is None:
            return None
        return media["path"] if full_path else os.path.basename(media["path"])

    def total_media(self) -> int:
        return len(self._media_files)

    def current_position(self) -> int:
        return self._current_index

    def get_current_frame(self) -> Optional[np.ndarray]:
        media = self.get_current_media_info()
        if media is None:
            raise FileExistsError("no media available")

        if media["type"] == "video":
            return self._video_frame(media)
        return self._image_frame()

    def _video_frame(self, media: dict) -> np.ndarray:
        with self._video_lock:
            if not self._ensure_video_open_locked(media):
                return self._create_loading_screen()
            if (
                self._video_frame_cache is not None
                and self._video_cache_position == self._video_position
            ):
                # Return a copy: the gallery draws overlays in place and must
                # not contaminate the cached frame.
                return self._video_frame_cache.copy()
            frame = self._decode_at_locked(self._video_position)
        return frame if frame is not None else self._create_loading_screen()

    def _image_frame(self) -> np.ndarray:
        image = None
        with self._cache_lock:
            cached = self._media_cache[1]
            if cached and cached["index"] == self._current_index:
                image = cached.get("data")

        if image is None:
            self._load_media(self._current_index, cache_pos=1)
            with self._cache_lock:
                cached = self._media_cache[1]
                if cached:
                    image = cached.get("data")

        if image is None:
            return self._create_loading_screen()
        return image

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        if (
            self._target_width
            and self._target_height
            and (
                frame.shape[1] != self._target_width
                or frame.shape[0] != self._target_height
            )
        ):
            return cv2.resize(
                frame, (self._target_width, self._target_height)
            )
        return frame

    def _create_loading_screen(self) -> np.ndarray:
        height = self._target_height or 480
        width = self._target_width or 640
        screen = np.zeros((height, width, 3), dtype=np.uint8)

        current = self.get_current_file_name() or ""
        text = (
            f"{tr('Loading')}: "
            f"{current[:20] + '...' if len(current) > 20 else current}"
        )
        # Imported lazily: media must remain importable before the UI package.
        from chandelier.ui.text_render import get_renderer

        get_renderer().draw(
            screen,
            text,
            (20, height // 2),
            14,
            (255, 255, 255),
            1,
        )

        self._loading_animation = (self._loading_animation + 8) % 360
        radius = min(height, width) // 10
        center = (width - 40, height - 40)
        for offset, color in (
            (0, (0, 200, 0)),
            (120, (200, 150, 0)),
            (240, (200, 0, 0)),
        ):
            cv2.ellipse(
                screen,
                center,
                (radius, radius),
                self._loading_animation + offset,
                0,
                60,
                color,
                -1,
            )
        return screen
