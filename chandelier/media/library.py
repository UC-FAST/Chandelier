"""Lazy image/video browser with a three-slot LRU-style cache.

The cache holds ``[previous, current, next]``.  Images are loaded eagerly for
the current item and asynchronously for neighbours; videos are opened lazily.
"""

from __future__ import annotations

import hashlib
import os
import queue
import shutil
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2
import numpy as np


IMAGE_EXTENSIONS = {"jpeg", "jpg", "png", "bmp", "tiff", "tif", "webp", "gif"}
VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "flv", "wmv"}
# Read files in chunks rather than one np.fromfile(), so load progress can be
# reported and an outdated read can be abandoned within one chunk.
_READ_CHUNK = 1024 * 1024
# Huge PNGs take seconds to decode.  Prefetching one blocks the single loader
# thread and makes the current image wait with no progress; load it on demand.
_PREFETCH_MAX_BYTES = 24 * 1024 * 1024


def default_cache_root() -> str:
    """Location shared by the device gallery and the web gallery cleanup."""

    return os.path.join(
        os.path.expanduser("~"), ".cache", "chandelier", "gallery"
    )


def source_cache_dir(
    path: str, cache_root: str | os.PathLike[str] | None = None
) -> str:
    """Return the per-source directory holding rendered cache files.

    Cache files for one photo (different display and thumbnail sizes, and
    different mtimes after an edit) all live under the same directory.  That
    makes deleting a photo a single recursive removal instead of trying to
    reverse the hashed cache filenames.
    """

    root = os.fspath(cache_root) if cache_root else default_cache_root()
    digest = hashlib.sha1(
        os.path.realpath(path).encode("utf-8")
    ).hexdigest()
    return os.path.join(root, "sources", digest[:2], digest)


def discard_media_cache(
    path: str, cache_root: str | os.PathLike[str] | None = None
) -> None:
    """Remove every rendered frame/thumbnail cached for ``path``."""

    directory = source_cache_dir(path, cache_root)
    shutil.rmtree(directory, ignore_errors=True)
    # The two-character shard is shared with other photos; rmdir only
    # succeeds once it is empty.
    try:
        os.rmdir(os.path.dirname(directory))
    except OSError:
        pass

_THUMBNAIL_EXECUTOR: ThreadPoolExecutor | None = None
_THUMBNAIL_EXECUTOR_LOCK = threading.Lock()
_THUMBNAIL_WORKERS = 3


def _thumbnail_executor() -> ThreadPoolExecutor:
    """Return the lazy process-wide pool used for thumbnail decoding.

    Creation is serialized by ``_THUMBNAIL_EXECUTOR_LOCK`` so racing callers
    do not leak a second pool.
    """

    global _THUMBNAIL_EXECUTOR
    with _THUMBNAIL_EXECUTOR_LOCK:
        if _THUMBNAIL_EXECUTOR is None:
            _THUMBNAIL_EXECUTOR = ThreadPoolExecutor(
                max_workers=_THUMBNAIL_WORKERS,
                thread_name_prefix="gallery-thumb",
            )
        return _THUMBNAIL_EXECUTOR


class MediaBrowser:
    """Browse the media files in one directory with a three-slot cache.

    Navigation only moves the cache window and queues background loads, so
    it is safe to call from the UI thread.  Playback state is guarded by
    ``_video_lock``; the image cache and thumbnail structures each have
    their own re-entrant lock.
    """

    def __init__(
        self,
        media_path: str = "./media/",
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):
        """Scan ``media_path`` and initialise caches, locks, and playback."""

        self._media_path = os.path.abspath(media_path)
        os.makedirs(self._media_path, exist_ok=True)

        self._target_width = width
        self._target_height = height

        self._cache_size = 3
        self._media_cache: list[Optional[dict]] = [None] * self._cache_size
        self._cache_lock = threading.RLock()
        self._progress_lock = threading.Lock()
        self._load_progress: Optional[dict] = None
        self._failed_indices: set[int] = set()

        self._current_index = 0
        self._media_files: list[dict] = []
        self._thumbnail_cache: OrderedDict = OrderedDict()
        self._details_cache: OrderedDict = OrderedDict()
        self._thumbnail_lock = threading.RLock()
        self._thumbnail_cache_size = 64
        self._details_cache_size = 128
        self._thumbnail_requests: set[tuple[str, int, int]] = set()
        self._thumbnail_generation = 0
        self._cache_root = default_cache_root()
        try:
            os.makedirs(self._cache_root, exist_ok=True)
            self._prepare_cache_root()
        except OSError:
            self._cache_root = None
        self._load_generation = 0
        self._load_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._load_worker: threading.Thread | None = None
        self._current_request_index = -1

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

        self.refresh_media_list()

    # ------------------------------------------------------------------
    # Discovery and cache
    # ------------------------------------------------------------------
    def refresh_media_list(self) -> None:
        """Rescan the directory, newest file first, and rebuild the list."""

        # Invalidate any in-flight background decode before the file list and
        # cache are rebuilt.
        self._load_generation += 1
        with self._progress_lock:
            self._failed_indices.clear()
            self._load_progress = None
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
            self._queue_display_batch()
        elif (
            self._current_index < len(self._media_files) - 1
            and self._can_prefetch(self._current_index + 1)
        ):
            self._load_media(
                self._current_index + 1, cache_pos=2, async_load=True
            )

    def _can_prefetch(self, index: int) -> bool:
        """Whether ``index`` is cheap enough to load in the background.

        Large images are excluded because decoding one occupies the single
        loader thread while the visible item waits.
        """

        info = self.get_media_info(index)
        if info is None:
            return False
        if info["type"] != "image":
            # Video prefetch only records the path; no decode happens.
            return True
        try:
            return os.path.getsize(info["path"]) <= _PREFETCH_MAX_BYTES
        except OSError:
            return False

    def _ensure_loader(self) -> None:
        """Start the background loader thread if it is not running."""

        if self._load_worker is None or not self._load_worker.is_alive():
            self._load_worker = threading.Thread(
                target=self._load_worker_loop,
                daemon=True,
                name="media-loader",
            )
            self._load_worker.start()

    def _queue_display_batch(self) -> None:
        """Queue the current image first and the next one as a prefetch."""

        if not self._media_files:
            return
        if self._is_failed(self._current_index):
            return
        with self._cache_lock:
            cached = self._media_cache[1]
            current_cached = bool(
                cached is not None
                and cached.get("index") == self._current_index
            )
        if current_cached:
            if (
                self._current_index < len(self._media_files) - 1
                and self._can_prefetch(self._current_index + 1)
            ):
                self._load_media(
                    self._current_index + 1,
                    cache_pos=2,
                    async_load=True,
                )
            return

        self._load_generation += 1
        generation = self._load_generation
        self._current_request_index = self._current_index
        self._load_queue.put((self._current_index, 1, generation))
        if (
            self._current_index < len(self._media_files) - 1
            and self._can_prefetch(self._current_index + 1)
        ):
            self._load_queue.put(
                (self._current_index + 1, 2, generation)
            )
        self._ensure_loader()

    def is_current_frame_ready(self) -> bool:
        """Whether the current item has something to draw right now.

        Videos count as ready as soon as they are selected; images need
        their cache slot filled.
        """

        media = self.get_current_media_info()
        if media is None:
            return False
        if media.get("type") == "video":
            return True
        with self._cache_lock:
            cached = self._media_cache[1]
            return bool(
                cached is not None
                and cached.get("index") == self._current_index
            )

    def request_current_load(self) -> None:
        """Start loading the current item if it is not ready yet.

        Navigation only shifts the three-slot cache.  If the new item was not
        prefetched (large files are skipped), the UI must explicitly ask the
        loader to start; otherwise the gallery shows the previous frame with
        a loading indicator forever.
        """

        if not self.is_current_frame_ready():
            self._load_current_async()

    def _load_current_async(self) -> None:
        """Queue a decode for the current item unless one is pending."""

        if self.is_current_frame_ready():
            return
        # A failed file stays failed until the list is refreshed or it is
        # deleted; otherwise every 0.1 s UI tick would retry the decode.
        if self._is_failed(self._current_index):
            return
        with self._cache_lock:
            # One outstanding request per item: the loader thread is already
            # working on it, so do not invalidate it with a new generation.
            if self._current_request_index == self._current_index:
                return
            self._current_request_index = self._current_index
        self._load_generation += 1
        generation = self._load_generation
        self._load_queue.put((self._current_index, 1, generation))
        self._ensure_loader()

    def _load_media(
        self, index: int, cache_pos: int, async_load: bool = False
    ) -> None:
        """Load ``index`` into cache slot ``cache_pos``.

        ``cache_pos`` is 0, 1, or 2 for the previous/current/next window
        slot.  Asynchronous requests are handed to the single loader thread.
        """

        if index < 0 or index >= len(self._media_files):
            return

        # A monotonically increasing generation token makes older background
        # loads abandon their result without ever joining their thread from
        # the UI thread (which used to stall navigation for up to 100 ms).
        self._load_generation += 1
        generation = self._load_generation

        if async_load:
            self._load_queue.put((index, cache_pos, generation))
            self._ensure_loader()
        else:
            self._load_task(index, cache_pos, generation)

    def _load_worker_loop(self) -> None:
        """Consume queued load requests until the process exits."""

        # One decoder thread only: stale requests are skipped by generation,
        # so rapid navigation cannot pile up concurrent image decodes.
        while True:
            index, cache_pos, generation = self._load_queue.get()
            if generation != self._load_generation:
                if cache_pos == 1:
                    with self._cache_lock:
                        if self._current_request_index == index:
                            self._current_request_index = -1
                continue
            try:
                self._load_task(index, cache_pos, generation)
            finally:
                if cache_pos == 1:
                    with self._cache_lock:
                        if self._current_request_index == index:
                            self._current_request_index = -1

    # ------------------------------------------------------------------
    # Load progress
    # ------------------------------------------------------------------
    # One slot is enough because the loader thread decodes one image at a
    # time.  ``index`` lets readers ignore a prefetch that is not on screen,
    # and ``stage`` is one of "read", "decode" or "error".
    def _begin_load(self, index: int, path: str, total: int) -> None:
        """Publish a fresh read-stage progress record for ``index``."""

        with self._progress_lock:
            self._load_progress = {
                "index": index,
                "path": path,
                "stage": "read",
                "read": 0,
                "total": total,
                "speed": 0.0,
                "elapsed": 0.0,
                "eta": None,
                "started": time.monotonic(),
            }

    def _update_load_progress(
        self,
        index: int,
        stage: str,
        read: int,
        total: int,
        started: float,
    ) -> None:
        """Update progress for ``index`` if it is still the active record."""

        now = time.monotonic()
        elapsed = max(0.001, now - started)
        # Cumulative average; it is stable enough for a 320 px UI and avoids
        # the jitter of an instantaneous byte-rate measurement.
        speed = read / elapsed if read > 0 else 0.0
        eta = None
        if stage == "read" and speed > 0.0 and total > read:
            # Decode time is not predictable here; the UI shows elapsed time
            # once the read finishes.
            eta = (total - read) / speed
        with self._progress_lock:
            progress = self._load_progress
            if progress is None or progress["index"] != index:
                return
            progress.update(
                {
                    "stage": stage,
                    "read": read,
                    "total": total,
                    "speed": speed,
                    "elapsed": elapsed,
                    "eta": eta,
                }
            )

    def _fail_load(self, index: int, path: str) -> None:
        """Remember a failed item and publish an error progress record."""

        # Remember the failure so _load_current_async() does not retry a
        # corrupt/unsupported file on every UI tick.  refresh_media_list()
        # clears the set and gives the file another chance.
        with self._progress_lock:
            self._failed_indices.add(index)
            self._load_progress = {
                "index": index,
                "path": path,
                "stage": "error",
                "read": 0,
                "total": 0,
                "speed": 0.0,
                "elapsed": 0.0,
                "eta": None,
                "started": time.monotonic(),
            }

    def _finish_load(self, index: int) -> None:
        """Clear failure/progress state after a successful load."""

        with self._progress_lock:
            self._failed_indices.discard(index)
            progress = self._load_progress
            if progress is not None and progress["index"] == index:
                self._load_progress = None

    def _is_failed(self, index: int) -> bool:
        with self._progress_lock:
            return index in self._failed_indices

    def get_load_progress(self) -> Optional[dict]:
        """Return progress for the currently displayed item, if any.

        In-flight read/decode progress is also returned for a different
        index: the single loader may still be finishing the previous image,
        and showing its speed beats a bare "loading" label.  ``for_current``
        tells the caller whether the numbers describe the visible item and
        whether the ETA is meaningful.
        """

        with self._progress_lock:
            progress = self._load_progress
            if progress is not None:
                result = dict(progress)
                result["for_current"] = (
                    progress["index"] == self._current_index
                )
                if result["for_current"] or progress["stage"] in {
                    "read",
                    "decode",
                }:
                    return result
            if self._current_index in self._failed_indices:
                return {
                    "index": self._current_index,
                    "path": "",
                    "stage": "error",
                    "read": 0,
                    "total": 0,
                    "speed": 0.0,
                    "elapsed": 0.0,
                    "eta": None,
                    "started": time.monotonic(),
                    "for_current": True,
                }
            return None

    def _decode_image(
        self,
        file_path: str,
        index: int | None = None,
        generation: int | None = None,
    ) -> np.ndarray | None:
        """Decode an image near the display size instead of full resolution.

        Camera photos are commonly 12 MP while the LCD is only 320x240.
        ``IMREAD_REDUCED_*`` lets libjpeg/libpng scale during decoding, which
        avoids both the decode time and the large temporary buffer.

        The file is read in chunks so progress can be reported and a request
        superseded by navigation can be abandoned before the expensive
        ``cv2.imdecode`` call.
        """

        try:
            total = os.path.getsize(file_path)
        except OSError:
            if index is not None:
                self._fail_load(index, file_path)
            return None

        # Navigation bumped the generation while this request was queued.
        if generation is not None and generation != self._load_generation:
            return None
        if index is not None:
            self._begin_load(index, file_path, total)
        started = time.monotonic()
        try:
            image_data = np.empty(total, dtype=np.uint8)
            view = memoryview(image_data)
            read = 0
            with open(file_path, "rb") as handle:
                while read < total:
                    chunk = handle.readinto(
                        view[read : read + _READ_CHUNK]
                    )
                    if not chunk:
                        break
                    read += chunk
                    if index is not None:
                        self._update_load_progress(
                            index, "read", read, total, started
                        )
                    if (
                        generation is not None
                        and generation != self._load_generation
                    ):
                        # Stale request: stop within one 1 MiB chunk rather
                        # than reading/decoding a file the user left behind.
                        if index is not None:
                            self._finish_load(index)
                        return None
        except (OSError, MemoryError):
            if index is not None:
                self._fail_load(index, file_path)
            return None

        if read < total:
            # The file was truncated while being read; decode what we got.
            image_data = image_data[:read]
        if generation is not None and generation != self._load_generation:
            if index is not None:
                self._finish_load(index)
            return None
        if index is not None:
            self._update_load_progress(
                index, "decode", read, total, started
            )

        try:
            image = self._decode_buffer(image_data)
        except (cv2.error, MemoryError):
            image = None
        if image is None:
            if index is not None:
                self._fail_load(index, file_path)
            return None
        if index is not None:
            self._finish_load(index)
        return image

    def _decode_buffer(
        self, image_data: np.ndarray
    ) -> np.ndarray | None:
        """Decode raw file bytes, preferring a reduced-scale decode."""

        if not self._target_width or not self._target_height:
            return cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        # Try the most aggressive reduction first; fall back until the
        # decoded image is still at least as large as the display.
        reduced_flags = [
            getattr(cv2, "IMREAD_REDUCED_COLOR_8", None),
            cv2.IMREAD_REDUCED_COLOR_4,
            cv2.IMREAD_REDUCED_COLOR_2,
        ]
        for flag in (value for value in reduced_flags if value is not None):
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
        """Decode one item and publish it into the cache.

        Runs on the loader thread, or inline for synchronous loads.  The
        result is discarded when navigation advanced the generation while
        the decode was in progress.
        """

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
                image = None
                cache_path = self._cache_file(
                    file_path, self._target_width, self._target_height
                )
                if cache_path:
                    # Cache hit: no decode and no progress to report.
                    image = self._read_cache_file(
                        cache_path,
                        self._target_width,
                        self._target_height,
                    )
                    if image is not None:
                        self._finish_load(index)
                if image is None:
                    # Miss: decode at display size, then persist the fitted
                    # frame so later visits avoid the expensive source decode.
                    image = self._decode_image(
                        file_path,
                        index=index,
                        generation=generation,
                    )
                    if image is not None:
                        image = self._fit_frame(image)
                        if cache_path:
                            self._write_cache_file(cache_path, image)
                if image is not None:
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
        """Open ``file_path`` and reset playback state for the new video.

        ``_video_lock`` is an ``RLock``, so this is safe to call from the
        locked playback helpers.  Returns False when OpenCV cannot open it.
        """

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
        """Release the capture and clear all per-video state."""

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
        """Open the selected video if it is not the one already open.

        Caller must hold ``_video_lock``.  Returns False when the item is
        missing, is not a video, or cannot be opened.
        """

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
        """Seek to a frame index, decode it, and update the frame cache.

        Caller must hold ``_video_lock``.  Positions are clamped to the
        known frame count, or to zero when the container reports none.  The
        returned frame is fitted to the display size.
        """

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
        """Move to the newer item and slide the cache window."""

        if self._current_index >= len(self._media_files) - 1:
            return self

        self._close_video()
        self._current_index += 1
        with self._cache_lock:
            self._media_cache[0] = self._media_cache[1]
            self._media_cache[1] = self._media_cache[2]
            self._media_cache[2] = None

        if (
            self._current_index < len(self._media_files) - 1
            and self._can_prefetch(self._current_index + 1)
        ):
            self._load_media(
                self._current_index + 1, cache_pos=2, async_load=True
            )
        return self

    def previous(self) -> "MediaBrowser":
        """Move to the older item and slide the cache window."""

        if self._current_index <= 0:
            return self

        self._close_video()
        self._current_index -= 1
        with self._cache_lock:
            self._media_cache[2] = self._media_cache[1]
            self._media_cache[1] = self._media_cache[0]
            self._media_cache[0] = None

        if (
            self._current_index > 0
            and self._can_prefetch(self._current_index - 1)
        ):
            self._load_media(
                self._current_index - 1, cache_pos=0, async_load=True
            )
        return self

    def jump_to_index(self, index: int) -> None:
        """Select ``index``, dropping the cache and queuing a fresh load."""

        if not 0 <= index < len(self._media_files):
            return

        self._close_video()
        self._current_index = index
        with self._cache_lock:
            self._media_cache = [None] * self._cache_size
        self._queue_display_batch()

    def delete_current(self) -> bool:
        """Delete the current file and every cached rendering of it."""

        if not self._media_files:
            return False

        self._close_video()
        file_path = self._media_files[self._current_index]["path"]
        try:
            os.remove(file_path)
        except OSError:
            return False

        # Cache keys include mtime/size, so orphaned files would never be
        # matched again; remove the whole per-source cache directory.
        self._discard_disk_cache(file_path)
        # Indices shifted, so any pending failure/progress is no longer valid.
        with self._progress_lock:
            self._failed_indices.clear()
            self._load_progress = None

        with self._thumbnail_lock:
            for key in [
                key
                for key in self._thumbnail_cache
                if key[0] == file_path
            ]:
                del self._thumbnail_cache[key]
            for key in [
                key
                for key in self._details_cache
                if key[0] == file_path
            ]:
                del self._details_cache[key]

        self._media_files.pop(self._current_index)
        if self._current_index >= len(self._media_files):
            self._current_index = max(0, len(self._media_files) - 1)

        with self._cache_lock:
            self._media_cache = [None] * self._cache_size
        if self._media_files:
            self._queue_display_batch()
        return True

    # ------------------------------------------------------------------
    # Playback state
    # ------------------------------------------------------------------
    @property
    def video_fps(self) -> float:
        # Some containers report an unknown/zero frame rate; assume 30 fps so
        # time-based seeking and playback pacing still work.
        return self._video_fps if self._video_fps > 0 else 30.0

    def is_video(self) -> bool:
        media = self.get_current_media_info()
        return bool(media and media.get("type") == "video")

    def set_loop(self, enable: bool) -> None:
        self._video_loop = bool(enable)

    def is_looping(self) -> bool:
        return self._video_loop

    def play(self) -> bool:
        """Start or resume playback, rewinding first if at the end."""

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
        """Toggle playback and return the new playing state."""

        with self._video_lock:
            if self._is_playing:
                self._is_playing = False
                return False
            return self.play()

    def advance_video_frame(self) -> np.ndarray | None:
        """Return the next playback frame, or ``None`` when stopped/ended."""

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
        """Pause and decode the frame ``delta`` frames from the current one."""

        with self._video_lock:
            media = self.get_current_media_info()
            if not self._ensure_video_open_locked(media):
                return None
            self._is_playing = False
            return self._decode_at_locked(self._video_position + int(delta))

    def seek_relative(self, seconds: float) -> np.ndarray | None:
        """Decode the frame ``seconds`` away from the current position."""

        with self._video_lock:
            media = self.get_current_media_info()
            if not self._ensure_video_open_locked(media):
                return None
            delta = int(round(float(seconds) * self.video_fps))
            return self._decode_at_locked(self._video_position + delta)

    def seek_video(self, position: float) -> None:
        """Seek to ``position`` (a duration fraction) and pause."""

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
        """Return ``(current, total)`` playback time in seconds."""

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
        """Return the playback position as a fraction of the total frames."""

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

    def get_current_frame(self, wait: bool = True) -> Optional[np.ndarray]:
        """Return the frame to draw for the current item.

        Images may wait briefly for the background decode when ``wait`` is
        true; videos decode on demand.  Raises ``FileExistsError`` when the
        media directory is empty.
        """

        media = self.get_current_media_info()
        if media is None:
            raise FileExistsError("no media available")

        if media["type"] == "video":
            return self._video_frame(media)
        return self._image_frame(wait=wait)

    def _video_frame(self, media: dict) -> np.ndarray:
        """Return the frame at the current position, or a loading screen."""

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

    def _image_frame(self, wait: bool = True) -> np.ndarray:
        """Return the cached image, waiting up to ten seconds if asked."""

        image = None
        with self._cache_lock:
            cached = self._media_cache[1]
            if cached and cached["index"] == self._current_index:
                image = cached.get("data")

        if image is None:
            self._load_current_async()
            if wait:
                # Bound the wait so a stuck decode cannot freeze the UI; on
                # timeout the caller still gets a loading screen.
                deadline = time.monotonic() + 10.0
                while time.monotonic() < deadline:
                    with self._cache_lock:
                        cached = self._media_cache[1]
                        if (
                            cached
                            and cached["index"] == self._current_index
                        ):
                            image = cached.get("data")
                            break
                    time.sleep(0.01)

        if image is None:
            return self._create_loading_screen()
        return image

    def _fit_frame(
        self,
        frame: np.ndarray,
        width: int | None = None,
        height: int | None = None,
    ) -> np.ndarray:
        """Letterbox ``frame`` instead of stretching it to the screen.

        Photos keep their aspect ratio; the unused area is filled with black.
        """

        target_width = int(width or self._target_width or 0)
        target_height = int(height or self._target_height or 0)
        if target_width <= 0 or target_height <= 0:
            return frame
        if frame.ndim != 3 or frame.shape[0] <= 0 or frame.shape[1] <= 0:
            return np.zeros((target_height, target_width, 3), np.uint8)

        source_height, source_width = frame.shape[:2]
        scale = min(
            target_width / source_width, target_height / source_height
        )
        new_width = max(1, int(round(source_width * scale)))
        new_height = max(1, int(round(source_height * scale)))
        if new_width != source_width or new_height != source_height:
            interpolation = (
                cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            )
            frame = cv2.resize(
                frame, (new_width, new_height), interpolation=interpolation
            )
        if new_width == target_width and new_height == target_height:
            return frame

        canvas = np.zeros((target_height, target_width, 3), np.uint8)
        offset_x = (target_width - new_width) // 2
        offset_y = (target_height - new_height) // 2
        canvas[
            offset_y : offset_y + new_height,
            offset_x : offset_x + new_width,
        ] = frame
        return canvas

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        return self._fit_frame(frame)

    # ------------------------------------------------------------------
    # Thumbnails and metadata (used by the gallery grid view)
    # ------------------------------------------------------------------
    def get_media_info(self, index: int) -> Optional[dict]:
        """Return a copy of one item's metadata, or ``None`` if out of range.

        Callers get a shallow copy so they cannot mutate the internal list.
        """

        if not 0 <= index < len(self._media_files):
            return None
        return dict(self._media_files[index])

    def _prepare_cache_root(self) -> None:
        """Drop caches written by the old flat layout once.

        Cache files are grouped by source now, so deleting a photo can remove
        every rendered copy with a single directory removal.
        """

        if not self._cache_root:
            return
        marker = os.path.join(self._cache_root, ".layout-v3")
        if os.path.exists(marker):
            return
        try:
            with os.scandir(self._cache_root) as entries:
                for entry in entries:
                    if entry.name in {".layout-v3", "sources"}:
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        shutil.rmtree(entry.path, ignore_errors=True)
                    else:
                        try:
                            os.remove(entry.path)
                        except OSError:
                            pass
            with open(marker, "w", encoding="utf-8") as handle:
                handle.write("3\n")
        except OSError:
            pass

    def _source_cache_dir(self, path: str) -> Optional[str]:
        """Return the per-source cache directory, or ``None`` if disabled."""

        if not self._cache_root:
            return None
        return source_cache_dir(path, self._cache_root)

    def _discard_disk_cache(self, path: str) -> None:
        """Remove persisted frames for ``path`` when disk caching is active."""

        if self._cache_root:
            discard_media_cache(path, self._cache_root)

    def _cache_file(
        self, path: str, width: int | None, height: int | None
    ) -> Optional[str]:
        """Return the cache filename for one source at one target size.

        The key hashes the source mtime and size (so an edited photo never
        reuses stale pixels), the target dimensions, and a layout version
        (so decode/letterbox changes can retire old entries).  Returns
        ``None`` when caching is disabled, the target size is unknown, or
        the source cannot be stat'ed.
        """

        directory = self._source_cache_dir(path)
        if directory is None or not width or not height:
            return None
        try:
            stat = os.stat(path)
        except OSError:
            return None
        raw = (
            f"{stat.st_mtime_ns}|{stat.st_size}"
            f"|{int(width)}x{int(height)}|v3"
        )
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
        return os.path.join(directory, digest + ".jpg")

    @staticmethod
    def _read_cache_file(
        cache_path: Optional[str], width: int | None, height: int | None
    ) -> Optional[np.ndarray]:
        """Load a cached JPEG, rejecting entries of the wrong size."""

        if not cache_path or not width or not height:
            return None
        if not os.path.isfile(cache_path):
            return None
        image = cv2.imread(cache_path, cv2.IMREAD_COLOR)
        # A size mismatch means a stale or partial entry; treat it as a miss
        # so the caller re-decodes instead of drawing the wrong geometry.
        if image is None or image.shape[:2] != (int(height), int(width)):
            return None
        return image

    @staticmethod
    def _write_cache_file(
        cache_path: Optional[str], image: np.ndarray
    ) -> None:
        """Store a JPEG cache entry atomically, ignoring write failures."""

        if not cache_path:
            return
        try:
            directory = os.path.dirname(cache_path)
            os.makedirs(directory, exist_ok=True)
            # Unique temp name plus os.replace(): readers never observe a
            # partially written JPEG, even with several thumbnail workers.
            temporary = (
                f"{cache_path}.{os.getpid()}.{threading.get_ident()}.tmp.jpg"
            )
            if cv2.imwrite(
                temporary,
                image,
                [cv2.IMWRITE_JPEG_QUALITY, 85],
            ):
                os.replace(temporary, cache_path)
            else:
                try:
                    os.remove(temporary)
                except OSError:
                    pass
        except (OSError, cv2.error):
            pass

    def get_thumbnail(
        self, index: int, width: int, height: int
    ) -> Optional[np.ndarray]:
        """Return a cached thumbnail, scheduling a background decode on miss.

        Decoding a 60 MP PNG takes seconds, so the thumbnail grid renders
        placeholders first and fills in as the workers finish.  Thumbnails are
        also persisted under ``~/.cache/chandelier/gallery`` so later visits
        are instant.
        """

        media = self.get_media_info(index)
        if media is None or media.get("type") != "image":
            return None
        width = max(1, int(width))
        height = max(1, int(height))
        key = (media["path"], width, height)

        with self._thumbnail_lock:
            cached = self._thumbnail_cache.get(key)
            if cached is not None:
                self._thumbnail_cache.move_to_end(key)
                return cached

        cache_path = self._cache_file(media["path"], width, height)
        image = self._read_cache_file(cache_path, width, height)
        if image is not None:
            with self._thumbnail_lock:
                self._thumbnail_cache[key] = image
                self._thumbnail_cache.move_to_end(key)
                while (
                    len(self._thumbnail_cache) > self._thumbnail_cache_size
                ):
                    self._thumbnail_cache.popitem(last=False)
            return image

        with self._thumbnail_lock:
            # The grid may ask again on every repaint; only one worker per
            # key is queued at a time.
            if key in self._thumbnail_requests:
                return None
            self._thumbnail_requests.add(key)
        _thumbnail_executor().submit(
            self._generate_thumbnail,
            media["path"],
            key,
            width,
            height,
            cache_path,
        )
        return None

    def _generate_thumbnail(
        self,
        path: str,
        key: tuple[str, int, int],
        width: int,
        height: int,
        cache_path: Optional[str],
    ) -> None:
        """Decode and persist a thumbnail, then notify waiting grid views."""

        try:
            image = self._read_cache_file(cache_path, width, height)
            if image is None:
                image = self._decode_image(path)
                if image is None:
                    return
                image = self._fit_frame(image, width, height)
                self._write_cache_file(cache_path, image)
            with self._thumbnail_lock:
                self._thumbnail_cache[key] = image
                self._thumbnail_cache.move_to_end(key)
                while (
                    len(self._thumbnail_cache) > self._thumbnail_cache_size
                ):
                    self._thumbnail_cache.popitem(last=False)
                self._thumbnail_generation += 1
        finally:
            with self._thumbnail_lock:
                self._thumbnail_requests.discard(key)

    @property
    def thumbnail_generation(self) -> int:
        """Bumped when a background thumbnail finishes, for UI repaints."""

        with self._thumbnail_lock:
            return self._thumbnail_generation

    @property
    def has_pending_thumbnails(self) -> bool:
        """Whether thumbnail jobs are still queued or running."""

        with self._thumbnail_lock:
            return bool(self._thumbnail_requests)

    def get_media_details(self, index: Optional[int] = None) -> Optional[dict]:
        """Return display metadata for one item (or the current item)."""

        index = self._current_index if index is None else index
        media = self.get_media_info(index)
        if media is None:
            return None
        path = media["path"]
        details = {
            "path": path,
            "name": os.path.basename(path),
            "type": media.get("type"),
            "size": 0,
            "mtime": 0.0,
            "width": None,
            "height": None,
            "duration": None,
        }
        try:
            stat = os.stat(path)
            details["size"] = stat.st_size
            details["mtime"] = stat.st_mtime
        except OSError:
            return details

        if media.get("type") == "image":
            # Including mtime and size invalidates cached dimensions when the
            # file is edited or replaced.
            cache_key = (path, details["mtime"], details["size"])
            with self._thumbnail_lock:
                cached = self._details_cache.get(cache_key)
                if cached is not None:
                    self._details_cache.move_to_end(cache_key)
                    return dict(cached)
            try:
                from PIL import Image

                with Image.open(path) as image:
                    details["width"], details["height"] = image.size
            except Exception:
                pass
            with self._thumbnail_lock:
                self._details_cache[cache_key] = dict(details)
                self._details_cache.move_to_end(cache_key)
                while len(self._details_cache) > self._details_cache_size:
                    self._details_cache.popitem(last=False)
        elif media.get("type") == "video":
            with self._video_lock:
                if self._current_video_path == path:
                    capture = self._video_capture
                    if capture is not None:
                        width = int(
                            capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0
                        )
                        height = int(
                            capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0
                        )
                        details["width"] = width or None
                        details["height"] = height or None
                        if self._video_fps > 0 and self._video_total_frames:
                            details["duration"] = (
                                self._video_total_frames / self._video_fps
                            )
        return details

    def _create_loading_screen(self) -> np.ndarray:
        # The gallery shows the previous frame with a red loading square;
        # this blank frame is only a safety fallback.
        height = self._target_height or 480
        width = self._target_width or 640
        return np.zeros((height, width, 3), dtype=np.uint8)
