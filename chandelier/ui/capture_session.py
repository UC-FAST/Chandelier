"""Pure capture-mode state machine.

The camera preview loop calls :meth:`CaptureSession.tick` with the current
monotonic time; the session calls the supplied capture callback when a shot is
due.  Keeping this independent of the camera makes it easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CaptureSettings:
    """Requested capture pattern.

    ``mode`` is ``"single"`` (``count`` frames as fast as ticks arrive),
    ``"burst"`` or ``"interval"`` (``count`` frames spaced by ``interval``
    seconds).  ``initial_delay`` postpones the first shot for a self-timer.
    """
    mode: str = "single"  # "single" | "burst" | "interval"
    count: int = 1
    interval: float = 0.0
    initial_delay: float = 0.0


class CaptureSession:
    """Drives one capture request from monotonic timestamps.

    The session stays active until all requested frames have fired or
    :meth:`cancel` is called.  It fires at most one shot per :meth:`tick`,
    keeping camera callbacks out of a tight loop even when the caller ticks
    faster than the requested interval.
    """

    def __init__(self, settings: CaptureSettings, now: float):
        self._settings = settings
        # Always request at least one frame; int() also absorbs stray float
        # counts coming from edited config.
        self._remaining = max(1, int(settings.count))
        self._captured = 0
        self._active = True
        # Burst/interval enforce a small floor so a zero or tiny interval
        # cannot spin the camera loop.  Single mode has no timer at all.
        self._interval = (
            max(0.05, float(settings.interval))
            if settings.mode in {"burst", "interval"}
            else 0.0
        )
        # initial_delay is a one-shot self-timer.
        self._next_shot = now + max(0.0, float(settings.initial_delay))

    # Read-only view used by the camera preview overlay.
    @property
    def active(self) -> bool:
        return self._active

    @property
    def done(self) -> bool:
        return not self._active

    @property
    def captured(self) -> int:
        return self._captured

    @property
    def remaining(self) -> int:
        return self._remaining

    def countdown_remaining(self, now: float) -> float:
        """Seconds until the first shot, or 0 once capture has started."""
        if not self._active or self._captured:
            return 0.0
        # The self-timer countdown only applies before the first frame;
        # burst/interval runs do not re-show it between shots.
        return max(0.0, self._next_shot - now)

    def tick(self, now: float, capture: Callable[[int], None]) -> int:
        """Capture at most one frame per call.  Returns 1 when a shot fired."""

        # Ignore ticks after completion or cancellation.
        if not self._active or self._remaining <= 0:
            return 0
        if now < self._next_shot:
            return 0

        # One frame per tick: even if several intervals elapsed while the
        # caller was busy, the remaining shots wait for later ticks.
        capture(self._captured)
        self._captured += 1
        self._remaining -= 1
        if self._remaining <= 0:
            # The final frame ends the session without scheduling another.
            self._active = False
            return 1

        # Schedule relative to the current tick, not the missed deadline,
        # so a slow caller cannot trigger a catch-up burst.
        self._next_shot = now + self._interval
        return 1

    def cancel(self) -> None:
        """Stop capturing without firing any further frames."""
        self._active = False
        # Clear the count too so countdown_remaining() reports zero.
        self._remaining = 0
