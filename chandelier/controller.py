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
    """Own the render loop and route input between controlled ends.

    Ends advertise themselves through ``EndId`` values; switching happens on
    GPIO callback threads while frames are rendered on the main thread.
    """

    def __init__(
        self,
        display: Display,
        ends: list[ControlledEnd],
        config: AppConfig,
        argv: list[str] | None = None,
        preview_server=None,
    ):
        if not ends:
            raise ValueError("at least one controlled end is required")

        self._display = display
        self._ends = {end.id: end for end in ends}
        self._ordered_ends = list(ends)
        self._config = config
        self._preview_server = preview_server
        self._current_index = 0
        self._last_end_id: str | None = None
        self._signal = False
        self._enabled = True
        self._stop_event = threading.Event()
        # Guards the current index, the switch flag and the last end id; the
        # render loop and GPIO callbacks access them concurrently.
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
        """Schedule a switch to ``target`` from any thread.

        The render loop performs the actual switch after the current frame,
        because it may be blocked inside the old end's generator.
        """

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
        """Ask the render loop to exit; safe from signal/callback threads."""

        self._stop_event.set()
        try:
            self.current_end.on_exit()
        except Exception:
            logger.debug("on_exit failed while stopping", exc_info=True)

    # ------------------------------------------------------------------
    # Messages between ends
    # ------------------------------------------------------------------
    def route_message(self, sender: str, receiver: str, payload: Any) -> None:
        """Deliver a menu/camera/gallery message to its receiver.

        Payloads stay in their original ad-hoc shapes and are passed by
        reference: the menu relies on the camera seeing its live options.
        """

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
        # Menu options can route a command string, a ``(command, tree)``
        # tuple, or a complete option dict (the combined power selector).
        command = payload
        if isinstance(payload, dict):
            value = payload.get("value")
            if isinstance(value, dict):
                value = value.get("value")
            command = value
        elif isinstance(payload, (tuple, list)) and payload:
            command = payload[0]

        if command in {"restart", "reboot"}:
            self._reboot()
        elif command == "shutdown":
            self._shutdown()
        elif command == "restart_service":
            self._restart_service()

    @staticmethod
    def _run_power_command(commands, action: str) -> None:
        """Try each command variant in order until one succeeds.

        Passwordless sudo is tried before plain systemctl because the
        service normally runs unprivileged; a missing command or non-zero
        exit is logged and the next variant is attempted.
        """

        logger.warning("%s requested", action)
        for command in commands:
            try:
                result = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                )
            except (FileNotFoundError, OSError) as exc:
                logger.warning(
                    "%s command %s unavailable: %s",
                    action,
                    command,
                    exc,
                )
                continue
            if result.returncode == 0:
                return
            logger.warning(
                "%s command %s failed (%s): %s",
                action,
                command,
                result.returncode,
                (result.stderr or "").strip(),
            )
        logger.error("all %s commands failed", action)

    @staticmethod
    def _reboot() -> None:
        Controller._run_power_command(
            (
                ["sudo", "-n", "reboot"],
                ["systemctl", "reboot"],
            ),
            "reboot",
        )

    @staticmethod
    def _restart_service() -> None:
        Controller._run_power_command(
            (
                ["sudo", "-n", "systemctl", "restart", "cam.service"],
                ["systemctl", "restart", "cam.service"],
            ),
            "service restart",
        )

    @staticmethod
    def _shutdown() -> None:
        Controller._run_power_command(
            (
                ["sudo", "-n", "poweroff"],
                ["systemctl", "poweroff"],
            ),
            "shutdown",
        )

    # ------------------------------------------------------------------
    # Input dispatch
    # ------------------------------------------------------------------
    def handle_action(
        self, action: Action, phase: Phase = Phase.PRESS
    ) -> bool:
        """Dispatch an input event to the active end.

        Returns whether the end handled it; unknown (action, phase) pairs and
        gated events are ignored without touching the end.
        """

        spec = resolve_action(action, phase)
        if spec is None:
            return False
        if spec.requires_enable and not self._enabled:
            return False
        return self.current_end.handle_action(action, phase)

    def handle_touch(self, x: float, y: float) -> bool:
        """Forward a normalized tap to the active screen."""

        try:
            return bool(self.current_end.handle_touch(x, y))
        except Exception:
            logger.exception("touch handling failed")
            return False

    # ------------------------------------------------------------------
    # Signal handling and render loop
    # ------------------------------------------------------------------
    def _install_signal_handlers(self) -> None:
        # signal.signal only works in the main thread; tests and embedders
        # may drive the controller from a worker thread.
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
        """Render frames until stopped, re-entering the current end on switch.

        Ends yield frames from ``main_loop()``.  A switch request breaks out
        of the generator and the outer loop calls ``on_enter`` on the new
        end, passing the id of the end that was left.
        """

        self._install_signal_handlers()
        try:
            while not self._stop_event.is_set():
                end = self.current_end
                end.on_enter(self._last_end_id)
                for frame in end.main_loop():
                    if self._stop_event.is_set():
                        break
                    # Consume the switch flag under the lock so a concurrent
                    # switch_to() cannot be lost between check and clear.
                    with self._state_lock:
                        switched = self._signal
                        if switched:
                            self._signal = False
                    if switched:
                        break

                    # The marker maps sound directions onto the camera image,
                    # so it is only meaningful on the camera screen.  The
                    # generation check avoids re-adding the same snapshot.
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
                            # Single mode shows the first source only;
                            # multiple mode is capped by configuration.
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

                    # A preview failure must never take the display down.
                    if self._preview_server is not None:
                        try:
                            self._preview_server.publish(frame)
                        except Exception:
                            logger.debug(
                                "preview publish failed", exc_info=True
                            )

                    self._display.show(frame)
        except KeyboardInterrupt:
            logger.info("interrupted by user")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Release display/preview resources; safe to call more than once."""

        if self._shutdown_done:
            return
        self._shutdown_done = True
        self._stop_event.set()
        if self._preview_server is not None:
            try:
                self._preview_server.stop()
            except Exception:
                logger.debug(
                    "failed to stop preview server", exc_info=True
                )
        try:
            self._display.clear()
        except Exception:
            logger.debug("failed to clear display", exc_info=True)
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
