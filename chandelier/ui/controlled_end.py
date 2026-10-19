"""Base class for the four screens controlled by the application."""

from __future__ import annotations

import abc
from typing import Any, Callable, Iterator

import numpy as np

from chandelier.constants import end_id
from chandelier.events import Action, Phase, resolve_action


class ControlledEnd(abc.ABC):
    """A screen that consumes input events and yields frames.

    Action methods are intentionally no-ops by default; a screen overrides the
    ones it cares about.  The controller performs the per-action gating before
    calling :meth:`handle_action`.
    """

    def __init__(self, _id: str):
        self._id = _id
        # Injected by the controller: irq() switches the active end, while
        # msg_sender() delivers a payload to another end by id.
        self._irq: Callable[[str], None] | None = None
        self._msg_sender: Callable[[str, str, Any], None] | None = None

    # -- Multi-direction buttons --------------------------------------
    # Every input hook is a no-op by default so a screen only overrides the
    # controls it actually uses.
    def center_press_action(self) -> None: ...
    def center_release_action(self) -> None: ...
    def up_press_action(self) -> None: ...
    def up_release_action(self) -> None: ...
    def down_press_action(self) -> None: ...
    def down_release_action(self) -> None: ...
    def left_press_action(self) -> None: ...
    def left_release_action(self) -> None: ...
    def right_press_action(self) -> None: ...
    def right_release_action(self) -> None: ...

    # -- Touch input ---------------------------------------------------
    def handle_touch(self, x: float, y: float) -> bool:
        """Handle a normalized tap on the screen. Returns True if consumed."""

        return False

    # -- Function buttons ---------------------------------------------
    def circle_press_action(self) -> None: ...
    def square_press_action(self) -> None: ...
    def cross_press_action(self) -> None: ...
    def shutter_press_action(self) -> None: ...

    # -- Rotary encoder -----------------------------------------------
    def rotary_encoder_clockwise(self) -> None: ...
    def rotary_encoder_counter_clockwise(self) -> None: ...
    def rotary_encoder_select(self) -> None: ...

    # -- Communication -------------------------------------------------
    # Default no-op: screens that accept payloads override this hook.
    def msg_receiver(self, sender: str, payload: Any) -> None: ...

    def msg_sender(self, func: Callable[[str, str, Any], None]) -> None:
        self._msg_sender = func

    def irq(self, func: Callable[[str], None]) -> None:
        self._irq = func

    def send_message(
        self, receiver: str, payload: Any = None, sender: str | None = None
    ) -> None:
        """Send ``payload`` to ``receiver`` through the controller."""
        if self._msg_sender is None:
            raise RuntimeError(f"{self.id} has no message sender")
        # Default the sender to this end so the receiver can reply.
        self._msg_sender(sender or self.id, end_id(receiver), payload)

    # -- Dispatch ------------------------------------------------------
    def handle_action(
        self, action: Action, phase: Phase = Phase.PRESS
    ) -> bool:
        """Resolve an (action, phase) pair and invoke its handler."""
        # Unmapped combinations are ignored rather than raising, so input
        # handling stays forgiving while the action table evolves.
        spec = resolve_action(action, phase)
        if spec is None:
            return False
        handler = getattr(self, spec.method_name, None)
        if handler is None:
            return False
        handler()
        return True

    # -- Lifecycle -----------------------------------------------------
    # main_loop yields BGR frames until the controller switches away.
    @abc.abstractmethod
    def main_loop(self) -> Iterator[np.ndarray]: ...

    # Release resources here; called before another end becomes active.
    def on_exit(self) -> None: ...

    # last_id identifies the end we came from, or None on first start.
    def on_enter(self, last_id: str | None) -> None: ...

    @property
    def id(self) -> str:
        return end_id(self._id)
