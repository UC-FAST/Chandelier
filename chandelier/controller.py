"""Input routing and the render loop.

Input callbacks run on GPIO threads, while the render loop may be blocked
inside an end's frame generator.  Switching ends therefore calls the old
end's ``on_exit`` (which pushes a sentinel frame) and sets a signal flag; the
render loop notices the flag after the next frame and re-enters the new end.
"""

from __future__ import annotations

import logging
import signal
import subprocess
import threading
import time
from typing import Any

import cv2

from chandelier.config import AppConfig
from chandelier.constants import EndId, end_id
from chandelier.events import Action, Phase, resolve_action
from chandelier.hardware.interfaces import Display
from chandelier.ui.controlled_end import ControlledEnd
from chandelier.ui.decorators import MarkerBox

logger = logging.getLogger(__name__)

_CONTROLLER_IDS = {
    EndId.CONTROLLER.value,
    "UniversalControl",  # accepted for backwards-compatible menu files
}


class Controller:
    def __init__(
        self,
        display: Display,
        ends: list[ControlledEnd],
        config: AppConfig,
        argv: list[str] | None = None,
    ):
        if not ends:
            raise ValueError("at least one controlled end is required")

        self._display = display
        self._ends = {end.id: end for end in ends}
        self._ordered_ends = list(ends)
        self._config = config
        self._current_index = 0
        self._last_end_id: str | None = None
        self._signal = False
        self._enabled = True
        self._stop_event = threading.Event()
        self._state_lock = threading.RLock()
        self._shutdown_done = False
        self._preview = bool(argv and "--preview" in argv)

        self._marker_box = MarkerBox(
            width=config.screen.width,
            height=config.screen.height,
            size=(config.marker.size, config.marker.size),
            cache=config.marker.smoothing,
            ttl=config.marker.ttl,
            trail=config.marker.trail,
        )
        self._last_marker_generation = -1

        for end in self._ordered_ends:
            end.irq(self.switch_to)
            end.msg_sender(self.route_message)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def current_end(self) -> ControlledEnd:
        with self._state_lock:
            return self._ordered_ends[self._current_index]

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = bool(value)

    # ------------------------------------------------------------------
    # End switching (called from GPIO callback threads)
    # ------------------------------------------------------------------
    def switch_to(self, target: str) -> None:
        target = end_id(target)
        with self._state_lock:
            try:
                index = next(
                    position
                    for position, end in enumerate(self._ordered_ends)
                    if end.id == target
                )
            except StopIteration as exc:
                raise LookupError(target) from exc

            previous = self._ordered_ends[self._current_index]
            self._last_end_id = previous.id
            self._current_index = index
            self._signal = True

        logger.info("switching from %s to %s", previous.id, target)
        # Wakes the old generator if it is blocked on its frame queue.
        previous.on_exit()

    def request_stop(self) -> None:
        self._stop_event.set()
        try:
            self.current_end.on_exit()
        except Exception:
            logger.debug("on_exit failed while stopping", exc_info=True)

    # ------------------------------------------------------------------
    # Messages between ends
    # ------------------------------------------------------------------
    def route_message(self, sender: str, receiver: str, payload: Any) -> None:
        receiver = end_id(receiver)
        sender = end_id(sender) if sender else self.current_end.id
        logger.debug("message %s -> %s", sender, receiver)

        if receiver in _CONTROLLER_IDS:
            self._handle_controller_message(payload)
            return

        try:
            end = self._ends[receiver]
        except KeyError as exc:
            raise LookupError(receiver) from exc
        end.msg_receiver(sender, payload)

    def _handle_controller_message(self, payload: Any) -> None:
        # Menu MSG options route ``(command, menu_tree)``; direct callers may
        # pass the command string itself.
        command = payload
        if isinstance(payload, (tuple, list)) and payload:
            command = payload[0]
        if command == "restart":
            self._reboot()

    @staticmethod
    def _reboot() -> None:
        logger.warning("restart requested; rebooting")
        commands = (
            ["sudo", "-n", "reboot"],
            ["systemctl", "reboot"],
        )
        for command in commands:
            try:
                result = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                )
            except (FileNotFoundError, OSError) as exc:
                logger.warning("reboot command %s unavailable: %s", command, exc)
                continue
            if result.returncode == 0:
                return
            logger.warning(
                "reboot command %s failed (%s): %s",
                command,
                result.returncode,
                (result.stderr or "").strip(),
            )
        logger.error("all reboot commands failed")

    # ------------------------------------------------------------------
    # Input dispatch
    # ------------------------------------------------------------------
    def handle_action(
        self, action: Action, phase: Phase = Phase.PRESS
    ) -> bool:
        spec = resolve_action(action, phase)
        if spec is None:
            return False
        if spec.requires_enable and not self._enabled:
            return False
        return self.current_end.handle_action(action, phase)

    # ------------------------------------------------------------------
    # Signal handling and render loop
    # ------------------------------------------------------------------
    def _install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        try:
            signal.signal(signal.SIGTERM, self._signal_handler)
        except (ValueError, OSError):
            logger.debug("SIGTERM handler not installed", exc_info=True)

    def _signal_handler(self, signum, frame) -> None:
        logger.info("received signal %s", signum)
        self.request_stop()

    def main_loop(self, localization=None) -> None:
        self._install_signal_handlers()
        try:
            while not self._stop_event.is_set():
                end = self.current_end
                end.on_enter(self._last_end_id)
                for frame in end.main_loop():
                    if self._stop_event.is_set():
                        break
                    with self._state_lock:
                        switched = self._signal
                        if switched:
                            self._signal = False
                    if switched:
                        break

                    # The marker maps sound directions onto the camera image,
                    # so it is only meaningful on the camera screen.
                    if (
                        localization is not None
                        and end.id == EndId.CAMERA.value
                    ):
                        snapshot = localization.read()
                        if (
                            snapshot is not None
                            and snapshot.confidence > 0
                            and snapshot.generation
                            != self._last_marker_generation
                        ):
                            self._last_marker_generation = snapshot.generation
                            sources = snapshot.sources
                            marker_config = self._config.marker
                            if marker_config.mode == "single":
                                sources = sources[:1]
                            else:
                                sources = sources[: marker_config.max_sources]

                            now = time.monotonic()
                            for index, source in enumerate(sources):
                                # Hershey fonts are ASCII-only.
                                if marker_config.mode == "single":
                                    text = (
                                        f"AZ {source.azimuth:.0f} "
                                        f"EL {source.elevation:.0f}"
                                    )
                                else:
                                    text = (
                                        f"S{index + 1} "
                                        f"{source.azimuth:.0f}/"
                                        f"{source.elevation:.0f}"
                                    )
                                self._marker_box.update(
                                    index,
                                    (source.screen_x, source.screen_y),
                                    text,
                                    now,
                                )
                        self._marker_box.decorate(frame)

                    if self._preview:
                        cv2.imshow(
                            "Monitor",
                            cv2.resize(
                                frame,
                                None,
                                fx=4,
                                fy=4,
                                interpolation=cv2.INTER_CUBIC,
                            ),
                        )
                        cv2.waitKey(5)

                    self._display.show(frame)
        except KeyboardInterrupt:
            logger.info("interrupted by user")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True
        self._stop_event.set()
        try:
            self._display.backlight(False)
        except Exception:
            logger.debug("failed to turn off backlight", exc_info=True)
        if self._preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                logger.debug("failed to destroy preview windows", exc_info=True)
        logger.info("controller stopped")
