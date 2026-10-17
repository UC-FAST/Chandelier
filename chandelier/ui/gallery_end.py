"""Gallery screen with image browsing and a video player."""

from __future__ import annotations

import queue
import time
from time import sleep

import numpy as np

from chandelier.config import AppConfig
from chandelier.constants import EndId, MenuOptionId
from chandelier.media.library import MediaBrowser
from chandelier.ui.controlled_end import ControlledEnd
from chandelier.ui.decorators import (
    Colors,
    Histogram,
    PlayerHud,
    SimpleText,
    Warning,
)


class GalleryControlledEnd(ControlledEnd):
    def __init__(
        self,
        library: MediaBrowser,
        config: AppConfig,
        _id: str = EndId.GALLERY.value,
    ):
        super().__init__(_id)
        self._library = library
        self._width = config.screen.width
        self._height = config.screen.height
        self._menu_tree: dict | None = None
        self._frame_list: queue.SimpleQueue = queue.SimpleQueue()
        self._rotate = 0
        self._from: str | None = None
        self._delete = False
        self._empty = False
        self._next_video_at = 0.0

        self._hist = Histogram()
        self._hud = PlayerHud(self._width, self._height)
        self._decorator = SimpleText(
            [self._filename_page],
            height=self._height,
            padding=(10, 20, 0, 0),
            font_height=10,
            color=Colors.gold.value,
        )
        self._simple_text_enabled = False

        self._show_progress = True
        self._show_hist = False
        self._loop_playback = False
        self._seek_seconds = 5.0

    # ------------------------------------------------------------------
    # Menu options
    # ------------------------------------------------------------------
    def _filename_page(self):
        return {"{}": self._library.get_current_file_name()}

    def _find_option_value(self, target: str):
        if self._menu_tree is None:
            return None
        for section in self._menu_tree.values():
            for option in section.get("options", []):
                if option.get("id") == target:
                    return option.get("value")
        return None

    def _option_value_or(self, target: str, default=None):
        value = self._find_option_value(target)
        return default if value is None else value

    def _set_option_value(self, target: str, value) -> None:
        if self._menu_tree is None:
            return
        for section in self._menu_tree.values():
            for option in section.get("options", []):
                if option.get("id") == target:
                    option["value"] = value
                    return

    def _read_player_settings(self) -> None:
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
    def _compose(self, raw: np.ndarray) -> np.ndarray:
        if raw.shape[0] != self._height or raw.shape[1] != self._width:
            return raw

        is_video = self._library.is_video()
        show_hud = self._show_progress and is_video
        if not (self._show_hist or self._simple_text_enabled or show_hud):
            # No overlay needs to modify the image; avoid a full-frame copy.
            if self._rotate:
                return np.rot90(raw, -self._rotate // 90)
            return raw

        # Do not assume the queued frame is disposable: a stale still-image
        # frame can still be in the queue when the selection changes.  A small
        # copy is cheaper and safer than corrupting the image cache.
        frame = raw.copy()
        if self._show_hist:
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
        if self._rotate:
            return np.rot90(frame, -self._rotate // 90)
        return frame

    def _queue_current(self) -> None:
        try:
            raw = self._library.get_current_frame()
        except FileExistsError:
            self._empty = True
            self._irq(EndId.CAMERA.value)
            return
        self._frame_list.put(raw)

    def _refresh_frame(self) -> None:
        self._queue_current()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _leave_if_empty(self) -> bool:
        if not self._empty:
            return False
        self._empty = False
        self._irq(self._from or EndId.CAMERA.value)
        return True

    def center_press_action(self) -> None:
        if self._leave_if_empty():
            return
        if not self._library.is_video():
            return
        started = self._library.toggle_play_pause()
        if started:
            self._next_video_at = 0.0
        self._queue_current()

    def up_press_action(self) -> None:
        if self._leave_if_empty():
            return
        self._library.previous()
        self._refresh_frame()
        sleep(0.2)

    def down_press_action(self) -> None:
        if self._leave_if_empty():
            return
        self._library.next()
        self._refresh_frame()
        sleep(0.2)

    def left_press_action(self) -> None:
        if self._leave_if_empty():
            return
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
        self._simple_text_enabled = not self._simple_text_enabled
        self._queue_current()

    def cross_press_action(self) -> None:
        if self._leave_if_empty():
            return
        self._irq(EndId.CAMERA.value)

    def square_press_action(self) -> None:
        if self._leave_if_empty():
            return
        self._irq(EndId.MENU.value)

    def shutter_press_action(self) -> None:
        if self._leave_if_empty():
            return
        if not self._library.is_video():
            return
        self._loop_playback = not self._loop_playback
        self._library.set_loop(self._loop_playback)
        self._set_option_value(
            MenuOptionId.LOOP_PLAYBACK.value, self._loop_playback
        )
        self._queue_current()

    def rotary_encoder_clockwise(self) -> None:
        if self._leave_if_empty():
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
        if self._library.is_video():
            frame = self._library.seek_relative(-self._seek_seconds)
            if frame is not None:
                self._frame_list.put(frame)
            return
        self._library.previous()
        self._refresh_frame()

    def rotary_encoder_select(self) -> None:
        self.center_press_action()

    # ------------------------------------------------------------------
    # Messages and lifecycle
    # ------------------------------------------------------------------
    def msg_receiver(self, sender: str, payload) -> None:
        if sender != EndId.MENU.value:
            return
        self._menu_tree = payload[1]
        self._read_player_settings()
        action = payload[0]
        if action == "delete":
            self._delete = True
            self._library.delete_current()
            self._refresh_frame()
        elif action == "update":
            self.update()

    def update(self) -> None:
        pass

    def main_loop(self):
        while True:
            try:
                raw = self._frame_list.get_nowait()
            except queue.Empty:
                raw = None

            if raw is not None:
                yield self._compose(raw)
                continue

            if self._library.is_playing():
                now = time.monotonic()
                if now >= self._next_video_at:
                    frame = self._library.advance_video_frame()
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
                wait = min(
                    0.05, max(0.001, self._next_video_at - now)
                )
                try:
                    raw = self._frame_list.get(timeout=wait)
                except queue.Empty:
                    continue
                yield self._compose(raw)
                continue

            # No timed polling: input/playback/on_exit always enqueue a frame.
            yield self._compose(self._frame_list.get())

    def on_exit(self) -> None:
        self._frame_list.put(Warning().decorate("Empty"))

    def on_enter(self, last_id: str | None) -> None:
        try:
            if not self._delete:
                self._from = last_id
                self._library.refresh_media_list()
                if self._library.total_media() == 0:
                    raise FileExistsError
            self._delete = False
            self.send_message(EndId.MENU.value, self.id)
            self._refresh_frame()
        except FileExistsError:
            self._empty = True
            self._frame_list.put(Warning().decorate("Empty"))
