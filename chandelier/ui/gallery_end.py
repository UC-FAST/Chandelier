"""Gallery screen with a thumbnail grid and a video player."""

from __future__ import annotations

import os
import queue
import time
from time import sleep

import cv2
import numpy as np

from chandelier.config import AppConfig
from chandelier.constants import EndId, MenuOptionId
from chandelier.i18n import tr
from chandelier.media.library import MediaBrowser
from chandelier.ui.controlled_end import ControlledEnd
from chandelier.ui.decorators import (
    Colors,
    Histogram,
    PlayerHud,
    SimpleText,
    Warning,
    corner_padding,
)
from chandelier.ui.text_render import get_renderer

# 3x3 thumbnail grid; margins, header and footer are reserved before the
# remaining area is split into equal tiles.
_GRID_COLUMNS = 3
_GRID_ROWS = 3
_GRID_MARGIN = 6
_GRID_GAP = 4
_GRID_HEADER = 18
_GRID_FOOTER = 16


class GalleryControlledEnd(ControlledEnd):
    """Media browser with a single-item player and a 3x3 thumbnail grid.

    Input handlers and lifecycle hooks enqueue frames on
    ``_frame_list``; :meth:`main_loop` is the only consumer, so all
    drawing happens on the controller thread.
    """

    def __init__(
        self,
        library: MediaBrowser,
        config: AppConfig,
        _id: str = EndId.GALLERY.value,
    ):
        """Create the decorators and the producer/consumer frame queue."""
        super().__init__(_id)
        self._library = library
        self._width = config.screen.width
        self._height = config.screen.height
        self._menu_tree: dict | None = None
        # Producers: input handlers, on_enter/on_exit and playback.  The
        # consumer is main_loop, which blocks here instead of polling.
        self._frame_list: queue.SimpleQueue = queue.SimpleQueue()
        self._rotate = 0
        self._from: str | None = None
        self._delete = False
        self._empty = False
        self._next_video_at = 0.0

        self._hist = Histogram(
            self._width,
            self._height,
            padding=corner_padding(
                self._width, self._height, corner="bottom-right"
            ),
        )
        self._hud = PlayerHud(self._width, self._height)
        self._decorator = SimpleText(
            [self._filename_page],
            height=self._height,
            padding=(10, 20, 0, 0),
            font_height=10,
            color=Colors.gold.value,
        )
        self._simple_text_enabled = False
        self._renderer = get_renderer()

        self._show_progress = True
        self._show_hist = False
        self._loop_playback = False
        self._seek_seconds = 5.0

        # Thumbnail grid and loading state.
        self._grid_mode = False
        self._grid_selection = 0
        self._grid_rendered_generation = -1
        self._display_loading = False
        self._last_raw: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Menu options
    # ------------------------------------------------------------------
    def _filename_page(self):
        """Build position/name/resolution/size/duration for the overlay."""
        details = self._library.get_media_details()
        if details is None:
            return {"{}": ""}

        total = self._library.total_media()
        position = self._library.current_position() + 1
        name = details.get("name") or ""
        if len(name) > 16:
            name = name[:13] + "..."

        parts = [f"{position}/{total}", name]
        width = details.get("width")
        height = details.get("height")
        if width and height:
            parts.append(f"{width}x{height}")
        size = details.get("size") or 0
        if size >= 1_000_000:
            parts.append(f"{size / 1_000_000:.1f}M")
        elif size >= 1000:
            parts.append(f"{size // 1000}K")
        duration = details.get("duration")
        if duration:
            seconds = int(duration)
            parts.append(f"{seconds // 60}:{seconds % 60:02d}")
        return {"{}": "  ".join(part for part in parts if part)}

    def _find_option_value(self, target: str):
        """Return a raw menu value, or ``None`` before the tree arrives."""
        if self._menu_tree is None:
            return None
        for section in self._menu_tree.values():
            for option in section.get("options", []):
                if option.get("id") == target:
                    return option.get("value")
        return None

    def _option_value_or(self, target: str, default=None):
        """Return a menu value, falling back when it is absent."""
        value = self._find_option_value(target)
        return default if value is None else value

    def _set_option_value(self, target: str, value) -> None:
        """Update a cached menu option in place."""
        if self._menu_tree is None:
            return
        for section in self._menu_tree.values():
            for option in section.get("options", []):
                if option.get("id") == target:
                    option["value"] = value
                    return

    def _read_player_settings(self) -> None:
        """Refresh the playback options from the menu tree."""
        self._show_progress = bool(
            self._option_value_or(MenuOptionId.SHOW_PROGRESS.value, True)
        )
        self._show_hist = bool(
            self._option_value_or(MenuOptionId.SHOW_HIST.value, False)
        )
        self._loop_playback = bool(
            self._option_value_or(MenuOptionId.LOOP_PLAYBACK.value, False)
        )
        try:
            self._seek_seconds = float(
                self._option_value_or(MenuOptionId.SEEK_SECONDS.value, 5)
            )
        except (TypeError, ValueError):
            self._seek_seconds = 5.0
        self._library.set_loop(self._loop_playback)

    # ------------------------------------------------------------------
    # Frame composition
    # ------------------------------------------------------------------
    def _draw_loading_indicator(self, frame: np.ndarray) -> None:
        """Small red square in the top-right corner while media is loading."""

        size = 9
        x = frame.shape[1] - size - 6
        y = 5
        cv2.rectangle(
            frame, (x - 1, y - 1), (x + size, y + size), (0, 0, 0), -1
        )
        cv2.rectangle(
            frame,
            (x, y),
            (x + size - 1, y + size - 1),
            (0, 0, 255),
            -1,
        )

    def _draw_load_progress(self, frame: np.ndarray, progress) -> None:
        """Draw read speed/ETA or decode elapsed time in the lower-left."""

        if progress is None:
            text = tr("Loading")
        elif progress["stage"] == "read":
            speed = progress["speed"] / 1_000_000.0
            eta = progress["eta"]
            if not progress.get("for_current", True):
                # The single loader is finishing another image first.
                text = f"{tr('Loading')}  {speed:.1f} MB/s"
            elif eta is None:
                text = f"{speed:.1f} MB/s"
            else:
                text = (
                    f"{speed:.1f} MB/s  "
                    f"{tr('remaining')} {eta:.0f}s"
                )
        else:
            text = f"{tr('Decoding')}  {progress['elapsed']:.0f}s"
        self._renderer.draw(
            frame,
            text,
            (10, self._height - 6),
            10,
            Colors.gold.value,
            1,
        )

    def _draw_load_error(self, frame: np.ndarray) -> None:
        """Show the localized failure label in the lower-left corner."""
        self._renderer.draw(
            frame,
            tr("Load failed"),
            (10, self._height - 6),
            10,
            Colors.red.value,
            1,
        )

    def _compose(self, raw: np.ndarray) -> np.ndarray:
        """Apply histogram, filename, HUD and loading overlays to a frame."""
        if raw.shape[0] != self._height or raw.shape[1] != self._width:
            return raw

        is_video = self._library.is_video()
        show_hud = self._show_progress and is_video
        progress = self._library.get_load_progress()
        # Only an error for the visible item suppresses the spinner; an
        # in-flight prefetch must still show "loading".
        load_failed = bool(
            progress is not None
            and progress["stage"] == "error"
            and progress.get("for_current", True)
        )
        ready = is_video or self._library.is_current_frame_ready()
        show_loading = not ready and not load_failed
        show_error = not ready and load_failed
        if not (
            self._show_hist
            or self._simple_text_enabled
            or show_hud
            or show_loading
            or show_error
        ):
            # No overlay needs to modify the image; avoid a full-frame copy.
            if self._rotate:
                # _rotate is clockwise degrees; np.rot90 counts
                # counter-clockwise quarter turns, hence the negation.
                return np.rot90(raw, -self._rotate // 90)
            return raw

        # Do not assume the queued frame is disposable: a stale still-image
        # frame can still be in the queue when the selection changes.  A small
        # copy is cheaper and safer than corrupting the image cache.
        frame = raw.copy()
        # While loading, raw is the previous photo or a blank frame; its
        # histogram would misrepresent the item being opened.
        if self._show_hist and not show_loading and not show_error:
            self._hist.decorate(frame)
        if self._simple_text_enabled:
            self._decorator.decorate(frame)
        if show_hud:
            position, duration = self._library.get_video_time()
            self._hud.set_state(
                position,
                duration,
                self._library.is_playing(),
                self._library.is_looping(),
            )
            self._hud.decorate(frame)
        if show_loading:
            self._draw_loading_indicator(frame)
            self._draw_load_progress(frame, progress)
        elif show_error:
            self._draw_load_error(frame)
        if self._rotate:
            return np.rot90(frame, -self._rotate // 90)
        return frame

    def _queue_current(self) -> None:
        """Queue the current item, requesting a decode when not ready."""
        if (
            not self._library.is_video()
            and not self._library.is_current_frame_ready()
        ):
            # Navigation may land on an item that was not prefetched (large
            # files are deliberately skipped), so explicitly request a load.
            self._library.request_current_load()
            # No dedicated loading screen: keep the previous frame and let
            # the blue-info/red-square overlay show that media is loading.
            frame = self._last_raw
            if frame is None or frame.shape != (
                self._height,
                self._width,
                3,
            ):
                frame = np.zeros(
                    (self._height, self._width, 3), dtype=np.uint8
                )
            self._frame_list.put(frame)
            return
        try:
            raw = self._library.get_current_frame(wait=False)
        except FileExistsError:
            self._empty = True
            self._irq(EndId.CAMERA.value)
            return
        self._last_raw = raw
        self._frame_list.put(raw)

    def _refresh_frame(self) -> None:
        """Re-queue the current item after navigation or settings changes."""
        self._queue_current()

    # ------------------------------------------------------------------
    # Thumbnail grid
    # ------------------------------------------------------------------
    def _render_grid(self) -> np.ndarray:
        """Render the visible 3x3 page of thumbnails and selection."""
        width, height = self._width, self._height
        frame = np.zeros((height, width, 3), np.uint8)
        frame[:] = (16, 18, 22)

        total = self._library.total_media()
        page_size = _GRID_COLUMNS * _GRID_ROWS
        pages = max(1, (total + page_size - 1) // page_size)
        # Derive the page from the highlighted cell so arrow navigation
        # scrolls naturally across page boundaries.
        page = min(
            pages - 1, max(0, self._grid_selection) // page_size
        )
        start = page * page_size

        tile_width = (
            width - 2 * _GRID_MARGIN - (_GRID_COLUMNS - 1) * _GRID_GAP
        ) // _GRID_COLUMNS
        tile_height = (
            height
            - _GRID_HEADER
            - _GRID_FOOTER
            - 2 * _GRID_MARGIN
            - (_GRID_ROWS - 1) * _GRID_GAP
        ) // _GRID_ROWS

        cv2.rectangle(
            frame, (0, 0), (width - 1, _GRID_HEADER - 1), (26, 30, 36), -1
        )
        cv2.rectangle(
            frame,
            (0, height - _GRID_FOOTER),
            (width - 1, height - 1),
            (26, 30, 36),
            -1,
        )
        self._renderer.draw(
            frame,
            f"{tr('Gallery')}  {page + 1}/{pages}",
            (_GRID_MARGIN, _GRID_HEADER - 5),
            12,
            Colors.gold.value,
            1,
        )

        for slot in range(page_size):
            index = start + slot
            x = _GRID_MARGIN + (slot % _GRID_COLUMNS) * (
                tile_width + _GRID_GAP
            )
            y = _GRID_HEADER + (slot // _GRID_COLUMNS) * (
                tile_height + _GRID_GAP
            )
            # Keep tile outlines so the last page still looks like a grid.
            if index >= total:
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + tile_width - 1, y + tile_height - 1),
                    (40, 44, 52),
                    1,
                )
                continue

            media = self._library.get_media_info(index) or {}
            # Videos get a play triangle instead of a decoded thumbnail.
            if media.get("type") == "video":
                cv2.rectangle(
                    frame,
                    (x + 1, y + 1),
                    (x + tile_width - 2, y + tile_height - 2),
                    (32, 38, 48),
                    -1,
                )
                center_x = x + tile_width // 2
                center_y = y + tile_height // 2
                triangle = np.array(
                    [
                        (center_x - 7, center_y - 10),
                        (center_x - 7, center_y + 10),
                        (center_x + 11, center_y),
                    ],
                    dtype=np.int32,
                )
                cv2.fillConvexPoly(
                    frame, triangle, (225, 225, 225), cv2.LINE_AA
                )
            else:
                thumbnail = self._library.get_thumbnail(
                    index, tile_width - 2, tile_height - 2
                )
                # A miss scheduled a worker decode; draw a placeholder
                # until the next refresh.
                if thumbnail is None:
                    cv2.rectangle(
                        frame,
                        (x + 1, y + 1),
                        (x + tile_width - 2, y + tile_height - 2),
                        (24, 26, 30),
                        -1,
                    )
                else:
                    frame[
                        y + 1 : y + 1 + thumbnail.shape[0],
                        x + 1 : x + 1 + thumbnail.shape[1],
                    ] = thumbnail

            selected = index == self._grid_selection
            cv2.rectangle(
                frame,
                (x, y),
                (x + tile_width - 1, y + tile_height - 1),
                (0, 215, 255) if selected else (70, 76, 86),
                2 if selected else 1,
            )

        if total:
            selected = min(total - 1, self._grid_selection)
            media = self._library.get_media_info(selected) or {}
            name = os.path.basename(media.get("path", ""))
            if len(name) > 20:
                name = name[:17] + "..."
            marker = "VIDEO  " if media.get("type") == "video" else ""
            self._renderer.draw(
                frame,
                f"{selected + 1}/{total}  {marker}{name}",
                (_GRID_MARGIN, height - 5),
                11,
                (230, 230, 230),
                1,
            )
        # Thumbnail workers are still busy; reuse the loading marker.
        if self._library.has_pending_thumbnails:
            self._draw_loading_indicator(frame)
        return frame

    def _queue_grid(self) -> None:
        """Queue a freshly rendered grid frame for the main loop."""
        self._frame_list.put(self._render_grid())

    def _drain_frames(self) -> None:
        """Drop queued frames so a stale selection cannot flash on screen."""
        while True:
            try:
                self._frame_list.get_nowait()
            except queue.Empty:
                break

    def _grid_move(self, delta: int) -> None:
        """Move the grid selection by ``delta`` cells and redraw."""
        total = self._library.total_media()
        if total <= 0:
            return
        self._grid_selection = max(
            0, min(total - 1, self._grid_selection + int(delta))
        )
        self._queue_grid()

    def _toggle_grid(self) -> None:
        """Enter the grid, or open the highlighted item if already in it."""
        if self._grid_mode:
            self._open_grid_selection()
            return
        self._library.pause()
        self._grid_mode = True
        self._grid_selection = self._library.current_position()
        self._drain_frames()
        self._queue_grid()

    def _open_grid_selection(self) -> None:
        """Jump to the highlighted item and return to single view."""
        if not self._grid_mode:
            return
        total = self._library.total_media()
        if total <= 0:
            self._grid_mode = False
            return
        self._library.jump_to_index(
            max(0, min(total - 1, self._grid_selection))
        )
        self._grid_mode = False
        self._drain_frames()
        self._refresh_frame()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _leave_if_empty(self) -> bool:
        """Return to the previous end once when the library is empty."""
        if not self._empty:
            return False
        self._empty = False
        self._irq(self._from or EndId.CAMERA.value)
        return True

    def center_press_action(self) -> None:
        """Open the grid cell or toggle playback for the current video."""
        if self._leave_if_empty():
            return
        if self._grid_mode:
            self._open_grid_selection()
            return
        if self._library.is_video():
            started = self._library.toggle_play_pause()
            if started:
                # Start immediately instead of waiting for the old
                # frame-pacing deadline.
                self._next_video_at = 0.0
            self._queue_current()
            return
        # Images have no additional center action.
        return

    def up_press_action(self) -> None:
        if self._leave_if_empty():
            return
        if self._grid_mode:
            self._grid_move(-_GRID_COLUMNS)
            return
        self._library.previous()
        self._refresh_frame()
        sleep(0.2)

    def down_press_action(self) -> None:
        if self._leave_if_empty():
            return
        if self._grid_mode:
            self._grid_move(_GRID_COLUMNS)
            return
        self._library.next()
        self._refresh_frame()
        sleep(0.2)

    def left_press_action(self) -> None:
        if self._leave_if_empty():
            return
        if self._grid_mode:
            self._grid_move(-1)
            return
        # In video mode the arrows step one frame instead of changing items.
        if self._library.is_video():
            frame = self._library.step_frame(-1)
            if frame is not None:
                self._frame_list.put(frame)
            sleep(0.1)
            return
        self._library.previous()
        self._refresh_frame()
        sleep(0.2)

    def right_press_action(self) -> None:
        if self._leave_if_empty():
            return
        if self._grid_mode:
            self._grid_move(1)
            return
        if self._library.is_video():
            frame = self._library.step_frame(1)
            if frame is not None:
                self._frame_list.put(frame)
            sleep(0.1)
            return
        self._library.next()
        self._refresh_frame()
        sleep(0.2)

    def circle_press_action(self) -> None:
        if self._leave_if_empty():
            return
        if self._grid_mode:
            return
        self._simple_text_enabled = not self._simple_text_enabled
        self._queue_current()

    def cross_press_action(self) -> None:
        if self._leave_if_empty():
            return
        if self._grid_mode:
            self._open_grid_selection()
            return
        self._irq(EndId.CAMERA.value)

    def square_press_action(self) -> None:
        if self._leave_if_empty():
            return
        self._irq(EndId.MENU.value)

    def shutter_press_action(self) -> None:
        if self._leave_if_empty():
            return
        self._toggle_grid()

    def rotary_encoder_clockwise(self) -> None:
        if self._leave_if_empty():
            return
        if self._grid_mode:
            self._grid_move(_GRID_COLUMNS * _GRID_ROWS)
            return
        if self._library.is_video():
            frame = self._library.seek_relative(self._seek_seconds)
            if frame is not None:
                self._frame_list.put(frame)
            return
        self._library.next()
        self._refresh_frame()

    def rotary_encoder_counter_clockwise(self) -> None:
        if self._leave_if_empty():
            return
        if self._grid_mode:
            self._grid_move(-_GRID_COLUMNS * _GRID_ROWS)
            return
        if self._library.is_video():
            frame = self._library.seek_relative(-self._seek_seconds)
            if frame is not None:
                self._frame_list.put(frame)
            return
        self._library.previous()
        self._refresh_frame()

    def rotary_encoder_select(self) -> None:
        # The rotary press mirrors the shutter key: toggle the 3x3 grid in
        # single view, and open the highlighted item while in the grid.
        self.shutter_press_action()

    # ------------------------------------------------------------------
    # Messages and lifecycle
    # ------------------------------------------------------------------
    def msg_receiver(self, sender: str, payload) -> None:
        """Handle menu tree updates plus delete/update actions."""
        if sender != EndId.MENU.value:
            return
        self._menu_tree = payload[1]
        self._read_player_settings()
        action = payload[0]
        if action == "delete":
            if self._grid_mode:
                self._library.jump_to_index(self._grid_selection)
            # Keep on_enter from re-scanning the folder after the delete.
            self._delete = True
            self._library.delete_current()
            total = self._library.total_media()
            if total <= 0:
                self._grid_mode = False
                self._empty = True
                self._irq(EndId.CAMERA.value)
                return
            self._grid_selection = min(self._grid_selection, total - 1)
            if self._grid_mode:
                self._queue_grid()
            else:
                self._refresh_frame()
        elif action == "update":
            self.update()

    def update(self) -> None:
        """Reserved hook; content updates arrive via :meth:`msg_receiver`."""
        pass

    def main_loop(self):
        """Yield gallery frames from the queue, grid or video decoder.

        Three modes share the loop: thumbnail-grid rendering, single-image
        loading (hold the previous frame until the decode lands) and video
        playback (paced to the clip's FPS).  Outside those modes the loop
        blocks on the frame queue rather than polling.
        """
        while True:
            # Thumbnail grid: render immediately, then poll while the
            # background workers decode the visible page.
            if self._grid_mode:
                generation = self._library.thumbnail_generation
                if self._library.has_pending_thumbnails:
                    try:
                        raw = self._frame_list.get(timeout=0.12)
                    except queue.Empty:
                        self._grid_rendered_generation = generation
                        yield self._render_grid()
                        continue
                    yield raw
                    continue
                # A worker finished a thumbnail; re-render only when the
                # generation actually changes.
                if generation != self._grid_rendered_generation:
                    self._grid_rendered_generation = generation
                    yield self._render_grid()
                    continue
                yield self._frame_list.get()
                continue

            # Single-image loading: keep the red indicator visible until the
            # background decode finishes.
            if (
                not self._library.is_video()
                and not self._library.is_current_frame_ready()
            ):
                self._display_loading = True
                try:
                    raw = self._frame_list.get(timeout=0.1)
                except queue.Empty:
                    self._queue_current()
                    continue
                yield self._compose(raw)
                continue
            if self._display_loading:
                self._display_loading = False
                self._queue_current()
                # Fall through so the freshly queued frame is yielded.

            try:
                raw = self._frame_list.get_nowait()
            except queue.Empty:
                raw = None

            if raw is not None:
                yield raw if self._grid_mode else self._compose(raw)
                continue

            if self._library.is_playing():
                now = time.monotonic()
                if now >= self._next_video_at:
                    frame = self._library.advance_video_frame()
                    # Pace advances to the clip's FPS, not the UI loop
                    # speed, so playback does not run fast or slow.
                    fps = self._library.video_fps or 30.0
                    self._next_video_at = now + 1.0 / max(1.0, fps)
                    if frame is not None:
                        yield self._compose(frame)
                        continue
                    if not self._library.is_playing():
                        # EOF without loop: redraw the held frame so the HUD
                        # switches to the paused icon.
                        try:
                            raw = self._library.get_current_frame()
                        except FileExistsError:
                            raw = None
                        if raw is not None:
                            yield self._compose(raw)
                            continue
                # Wait until the next frame is due (or an input frame
                # arrives) instead of polling every 5 ms.
                wait = min(0.05, max(0.001, self._next_video_at - now))
                try:
                    raw = self._frame_list.get(timeout=wait)
                except queue.Empty:
                    continue
                yield self._compose(raw)
                continue

            # No timed polling: input/playback/on_exit always enqueue a frame.
            yield self._compose(self._frame_list.get())

    def on_exit(self) -> None:
        """Queue a frame so a blocked main_loop can observe the switch."""
        self._frame_list.put(Warning().decorate("Empty"))

    def on_enter(self, last_id: str | None) -> None:
        """Rescan media (unless returning from a delete) and queue a frame."""
        try:
            if not self._delete:
                # Remember the exit target in case the library is empty.
                self._from = last_id
                self._library.refresh_media_list()
                if self._library.total_media() == 0:
                    # Reuse the empty-library handling below.
                    raise FileExistsError
            self._delete = False
            self.send_message(EndId.MENU.value, self.id)
            total = self._library.total_media()
            # After a delete the grid page may have shrunk, so clamp the
            # selection and redraw it instead of showing the single view.
            if self._grid_mode and total > 0:
                self._grid_selection = min(
                    self._grid_selection, total - 1
                )
                self._drain_frames()
                self._queue_grid()
            else:
                self._refresh_frame()
        except FileExistsError:
            self._grid_mode = False
            self._empty = True
            self._frame_list.put(Warning().decorate("Empty"))
