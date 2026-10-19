"""Read-only media index and thumbnail generation for the web gallery.

The preview server exposes the same media directory as the on-device gallery,
but it deliberately keeps its own listing and thumbnail cache.  The device
gallery has a single current item and decodes frames for a 320x240 LCD; the
web gallery needs square, cover-cropped thumbnails for many items at once.
Sharing state between the two would couple the browser to the device's
navigation position, so this module only reuses the extension sets.
"""

from __future__ import annotations

import os
import stat
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np

from chandelier.media.library import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    discard_media_cache,
)

# Short enough that files copied in over USB/SMB appear promptly, long
# enough that a page full of thumbnail requests scans only once.
LISTING_TTL = 2.0
THUMBNAIL_QUALITY = 85
MAX_THUMBNAIL_EDGE = 1024
MIN_THUMBNAIL_EDGE = 32
DEFAULT_THUMBNAIL_SIZE = 320
# Snap requested sizes to a small ladder so clients cannot force an endless
# stream of distinct cache entries (and decodes) by varying w/h.
THUMBNAIL_EDGES = (32, 64, 128, 256, 512, 1024)
THUMBNAIL_CACHE_BYTES = 32 * 1024 * 1024
# Viewer renditions are capped well above any phone screen so zooming
# stays sharp without making encode time or memory unbounded on the Pi.
VIEW_MAX_EDGE = 2560
VIEW_QUALITY = 90
# Formats browsers can decode in an <img> directly.  TIFF (offered by the
# camera menu) is not one of them and needs a JPEG rendition for the viewer.
BROWSER_IMAGE_EXTENSIONS = {"jpeg", "jpg", "png", "bmp", "gif", "webp"}


@dataclass(frozen=True)
class GalleryItem:
    """One file in the media directory."""

    id: str
    name: str
    path: str
    media_type: str
    size: int
    mtime: float
    mtime_ns: int

    @property
    def extension(self) -> str:
        return os.path.splitext(self.name)[1][1:].lower()

    @property
    def version(self) -> str:
        """Short cache-busting token tied to the file's contents' stat."""

        return f"{self.mtime_ns:x}-{self.size:x}"

    def as_payload(self) -> dict:
        """Serialise the fields the gallery page needs (no path)."""

        return {
            "id": self.id,
            "name": self.name,
            "type": self.media_type,
            "size": self.size,
            "mtime": self.mtime,
            "version": self.version,
        }


class WebGallery:
    """Scan one or more media directories and produce web thumbnails.

    The first directory contributes bare filenames as item IDs.  Later
    directories are prefixed with their basename (for example
    ``video/<name>``) so recordings can be listed alongside photos without
    collisions.  Listings are cached for :data:`LISTING_TTL` seconds so
    rendering a page full of thumbnails does not rescan the directories once
    per image; the gallery page asks for a fresh scan when it loads or
    refreshes.
    """

    def __init__(
        self,
        media_path: (
            str
            | os.PathLike[str]
            | Iterable[str | os.PathLike[str]]
            | None
        ) = None,
        thumbnail_cache_size: int = 256,
    ):
        if media_path is None:
            roots = []
        elif isinstance(media_path, (str, bytes, os.PathLike)):
            roots = [media_path]
        else:
            roots = [path for path in media_path if path]

        normalised: list[str] = []
        for path in roots:
            root = os.path.abspath(os.fsdecode(os.fspath(path)))
            if root not in normalised:
                normalised.append(root)
        self._roots = normalised
        # Resolve roots once so containment checks compare against stable
        # canonical paths even if a root is later replaced by a symlink.
        self._root_reals = [os.path.realpath(root) for root in normalised]
        self._prefixes = self._make_prefixes(normalised)
        # RLock because find() and items() call each other while already
        # holding the lock.
        self._lock = threading.RLock()
        self._items: list[GalleryItem] = []
        self._scanned_at = 0.0
        # OrderedDict doubles as the LRU list: hits move to the end,
        # eviction pops from the front.
        self._thumbnail_cache: OrderedDict[tuple, bytes] = OrderedDict()
        # Entry-count cap; _remember() also enforces the byte budget.
        self._thumbnail_cache_size = max(1, int(thumbnail_cache_size))
        self._thumbnail_cache_bytes = 0
        # Decoding several 12 MP photos at once would starve the render loop.
        self._decode_slots = threading.Semaphore(2)

    def items(self, force: bool = False) -> list[GalleryItem]:
        """Return the current listing, newest first."""

        with self._lock:
            fresh = (
                self._scanned_at
                and time.monotonic() - self._scanned_at < LISTING_TTL
            )
            if fresh and not force:
                # Copy so callers cannot mutate the cached list outside
                # the lock.
                return list(self._items)
            self._items = self._scan()
            self._scanned_at = time.monotonic()
            return list(self._items)

    def find(self, item_id: str) -> GalleryItem | None:
        """Look up a listing item by its URL-safe ID.

        IDs come from the scan (``name`` for the photo directory and
        ``root/name`` for extra roots) and are matched against the listing,
        never joined into a filesystem path, so traversal attempts cannot
        resolve.
        """

        if not item_id or "\x00" in item_id:
            return None
        if item_id.startswith("/") or ".." in item_id.split("/"):
            return None

        with self._lock:
            fresh = (
                self._scanned_at
                and time.monotonic() - self._scanned_at < LISTING_TTL
            )
            if fresh:
                for item in self._items:
                    if item.id == item_id:
                        return item

        # The first thumbnail request can arrive before any listing was built;
        # items() scans on a cold cache and honours the short listing TTL.
        for item in self.items():
            if item.id == item_id:
                return item
        return None

    def delete_items(
        self, item_ids
    ) -> tuple[list[str], list[str]]:
        """Delete listed files and every rendered copy cached for them."""

        deleted: list[str] = []
        failed: list[str] = []
        seen: set[str] = set()
        for item_id in item_ids:
            if not isinstance(item_id, str) or item_id in seen:
                continue
            seen.add(item_id)
            # Resolve against the scanned listing; arbitrary paths are never
            # joined or opened, so traversal IDs simply fail here.
            item = self.find(item_id)
            if item is None:
                failed.append(item_id)
                continue
            try:
                os.remove(item.path)
            except OSError:
                failed.append(item_id)
                continue
            # The device-side 320x240 frame/thumbnail cache is keyed by path,
            # so remove it as well to avoid leaving deleted photos on disk.
            discard_media_cache(item.path)
            with self._lock:
                for key in [
                    key
                    for key in self._thumbnail_cache
                    if key[0] == item.path
                ]:
                    jpeg = self._thumbnail_cache.pop(key)
                    self._thumbnail_cache_bytes -= len(jpeg)
            deleted.append(item_id)

        if deleted:
            deleted_ids = set(deleted)
            with self._lock:
                # Drop them immediately and expire the listing so the next
                # /api/gallery request rescans the directories.
                self._items = [
                    item
                    for item in self._items
                    if item.id not in deleted_ids
                ]
                self._scanned_at = 0.0
        return deleted, failed

    def thumbnail(
        self, item_id: str, width: int, height: int
    ) -> bytes | None:
        """Return a cover-cropped JPEG thumbnail, or ``None``."""

        item = self.find(item_id)
        if item is None or item.media_type != "image":
            return None

        # Re-stat on every request: a cached entry must not keep serving a
        # deleted or replaced file.
        try:
            file_stat = os.stat(item.path, follow_symlinks=False)
        except OSError:
            return None
        if not stat.S_ISREG(file_stat.st_mode):
            return None

        width = self._clamp_edge(width)
        height = self._clamp_edge(height)
        key = (
            item.path,
            file_stat.st_mtime_ns,
            file_stat.st_size,
            width,
            height,
        )

        with self._lock:
            cached = self._thumbnail_cache.get(key)
            if cached is not None:
                self._thumbnail_cache.move_to_end(key)
                return cached

        try:
            # Decode outside the lock: a 12 MP decode must not stall
            # requests that could be answered from the cache.
            with self._decode_slots:
                image = self._decode(item.path, width, height)
                if image is None:
                    return None
                cropped = self._cover_crop(image, width, height)
                if cropped is None:
                    return None
                ok, buffer = cv2.imencode(
                    ".jpg",
                    cropped,
                    [int(cv2.IMWRITE_JPEG_QUALITY), THUMBNAIL_QUALITY],
                )
        except (cv2.error, MemoryError, OSError):
            return None
        if not ok:
            return None

        jpeg = buffer.tobytes()
        self._remember(key, jpeg)
        return jpeg

    def render_image(self, item_id: str) -> bytes | None:
        """Return a browser-friendly JPEG for the full-screen viewer.

        JPEG/PNG/GIF/WebP/BMP are served as-is by the handler.  Formats the
        browser cannot decode (notably TIFF from the camera menu) go through
        this path and are re-encoded without the square thumbnail crop.
        """

        item = self.find(item_id)
        if item is None or item.media_type != "image":
            return None
        try:
            file_stat = os.stat(item.path, follow_symlinks=False)
        except OSError:
            return None
        if not stat.S_ISREG(file_stat.st_mode):
            return None

        key = (
            item.path,
            file_stat.st_mtime_ns,
            file_stat.st_size,
            "view",
            VIEW_MAX_EDGE,
        )
        with self._lock:
            cached = self._thumbnail_cache.get(key)
            if cached is not None:
                self._thumbnail_cache.move_to_end(key)
                return cached

        try:
            with self._decode_slots:
                # Ask the decoder for a reduced scale where possible; the
                # final fit below never upscales, so memory stays bounded.
                image = self._decode(
                    item.path, VIEW_MAX_EDGE // 2, VIEW_MAX_EDGE // 2
                )
                if image is None:
                    return None
                image = self._fit_within(image, VIEW_MAX_EDGE)
                ok, buffer = cv2.imencode(
                    ".jpg",
                    image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), VIEW_QUALITY],
                )
        except (cv2.error, MemoryError, OSError):
            return None
        if not ok:
            return None

        jpeg = buffer.tobytes()
        self._remember(key, jpeg)
        return jpeg

    @staticmethod
    def _fit_within(image: np.ndarray, max_edge: int) -> np.ndarray:
        """Shrink to fit without ever upscaling."""

        height, width = image.shape[:2]
        largest = max(height, width)
        if largest <= max_edge:
            return image
        scale = max_edge / largest
        return cv2.resize(
            image,
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )

    def _remember(self, key: tuple, jpeg: bytes) -> None:
        """Insert or refresh a cache entry, evicting LRU entries."""

        # Bound by entry count *and* bytes: 1024x1024 renditions are large
        # enough that an entry-only limit could hold ~100 MB.
        with self._lock:
            previous = self._thumbnail_cache.pop(key, None)
            if previous is not None:
                self._thumbnail_cache_bytes -= len(previous)
            self._thumbnail_cache[key] = jpeg
            self._thumbnail_cache_bytes += len(jpeg)
            while self._thumbnail_cache and (
                len(self._thumbnail_cache) > self._thumbnail_cache_size
                or self._thumbnail_cache_bytes > THUMBNAIL_CACHE_BYTES
            ):
                # The front of an OrderedDict is the least recently used
                # entry.
                _, evicted = self._thumbnail_cache.popitem(last=False)
                self._thumbnail_cache_bytes -= len(evicted)

    def open_media(self, item_id: str):
        """Open a listed file without following a swapped-in symlink.

        ``O_NOFOLLOW`` protects the final path component between the listing
        scan and the read; directory components come from the configured
        media root.
        """

        item = self.find(item_id)
        if item is None:
            return None, None
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(item.path, flags)
        except OSError:
            return None, None
        try:
            return item, os.fdopen(descriptor, "rb")
        except OSError:
            os.close(descriptor)
            return None, None

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    @staticmethod
    def _make_prefixes(roots: list[str]) -> list[str]:
        """Build URL prefixes, uniquifying duplicate root basenames."""

        prefixes: list[str] = []
        used: set[str] = set()
        for index, root in enumerate(roots):
            if index == 0:
                # The primary photo directory keeps bare filenames.
                prefixes.append("")
                continue
            base = os.path.basename(root.rstrip(os.sep)) or f"root{index}"
            label = base
            counter = 2
            while label in used:
                label = f"{base}-{counter}"
                counter += 1
            used.add(label)
            prefixes.append(label + "/")
        return prefixes

    def _scan(self) -> list[GalleryItem]:
        """Scan every root, returning listable media newest first."""

        items: list[GalleryItem] = []
        for index, root in enumerate(self._roots):
            prefix = self._prefixes[index]
            root_real = self._root_reals[index]
            try:
                with os.scandir(root) as directory:
                    for entry in directory:
                        item = self._scan_entry(entry, root_real, prefix)
                        if item is not None:
                            items.append(item)
            except OSError:
                continue
        # Newest first: both the device gallery and the web grid show
        # recent captures at the top.
        items.sort(key=lambda item: item.mtime_ns, reverse=True)
        return items

    def _scan_entry(self, entry, root_real: str, prefix: str):
        """Turn one directory entry into an item, or ``None``."""

        try:
            # Directory entries are served with O_NOFOLLOW, so do not list
            # symlinks that could never be read.
            if not entry.is_file(follow_symlinks=False):
                return None
            # Names that are not valid UTF-8 cannot round-trip through URLs
            # or JSON; hide them from the web view.
            try:
                entry.name.encode("utf-8")
            except UnicodeEncodeError:
                return None
            extension = os.path.splitext(entry.name)[1][1:].lower()
            if extension in IMAGE_EXTENSIONS:
                media_type = "image"
            elif extension in VIDEO_EXTENSIONS:
                media_type = "video"
            else:
                return None
            if not self._inside_root(entry.path, root_real):
                return None
            file_stat = entry.stat(follow_symlinks=False)
        except OSError:
            return None
        return GalleryItem(
            id=prefix + entry.name,
            name=entry.name,
            path=entry.path,
            media_type=media_type,
            size=file_stat.st_size,
            mtime=file_stat.st_mtime,
            mtime_ns=file_stat.st_mtime_ns,
        )

    @staticmethod
    def _inside_root(path: str, root_real: str) -> bool:
        """Return whether ``path`` resolves to a location inside the root."""

        try:
            real = os.path.realpath(path)
            return os.path.commonpath([real, root_real]) == root_real
        except ValueError:
            return False

    # ------------------------------------------------------------------
    # Thumbnails
    # ------------------------------------------------------------------
    @staticmethod
    def _clamp_edge(value) -> int:
        """Snap a requested edge onto the small size ladder."""

        try:
            value = int(value)
        except (TypeError, ValueError):
            value = DEFAULT_THUMBNAIL_SIZE
        value = max(MIN_THUMBNAIL_EDGE, min(value, MAX_THUMBNAIL_EDGE))
        for edge in THUMBNAIL_EDGES:
            if value <= edge:
                return edge
        return MAX_THUMBNAIL_EDGE

    @staticmethod
    def _decode(
        path: str, width: int, height: int
    ) -> np.ndarray | None:
        """Read a file without following a final-component symlink."""

        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError:
            return None
        try:
            with os.fdopen(descriptor, "rb") as handle:
                data = np.fromfile(handle, dtype=np.uint8)
        except (OSError, ValueError, MemoryError):
            return None
        if data.size == 0:
            return None

        # Decode at a reduced scale when the source is much larger than the
        # tile.  A 12 MP JPEG decodes roughly 64x faster at 1/8 scale.
        minimum = min(width, height)
        for factor in (8, 4, 2):
            flag = getattr(cv2, f"IMREAD_REDUCED_COLOR_{factor}", None)
            if flag is None:
                continue
            image = cv2.imdecode(data, flag)
            if image is None:
                continue
            if min(image.shape[:2]) >= minimum:
                return image
        # The reduced decode may already be smaller than the requested
        # tile; fall back to full resolution rather than upscaling.
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    @staticmethod
    def _cover_crop(
        image: np.ndarray, width: int, height: int
    ) -> np.ndarray | None:
        """Center-crop to the tile aspect ratio, then resize."""

        if image.ndim != 3 or image.shape[0] <= 0 or image.shape[1] <= 0:
            return None

        source_height, source_width = image.shape[:2]
        target_ratio = width / height
        source_ratio = source_width / source_height

        # Crop the source to the target aspect ratio *before* resizing.  The
        # other order blows the intermediate up to source_width*scale by
        # source_height*scale, which is catastrophic for extreme panoramas.
        if source_ratio > target_ratio:
            crop_width = min(
                source_width,
                max(1, int(round(source_height * target_ratio))),
            )
            # Center the crop so the tile keeps the middle of the frame.
            offset_x = (source_width - crop_width) // 2
            region = image[:, offset_x : offset_x + crop_width]
        elif source_ratio < target_ratio:
            crop_height = min(
                source_height,
                max(1, int(round(source_width / target_ratio))),
            )
            offset_y = (source_height - crop_height) // 2
            region = image[offset_y : offset_y + crop_height, :]
        else:
            region = image

        if region.shape[1] == width and region.shape[0] == height:
            return region
        # INTER_AREA averages pixels while shrinking (avoids aliasing);
        # INTER_LINEAR is sufficient when enlarging.
        interpolation = (
            cv2.INTER_AREA
            if region.shape[1] > width or region.shape[0] > height
            else cv2.INTER_LINEAR
        )
        return cv2.resize(region, (width, height), interpolation=interpolation)
