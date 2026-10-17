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
    mode: str = "single"  # "single" | "burst" | "interval"
    count: int = 1
    interval: float = 0.0
    initial_delay: float = 0.0


class CaptureSession:
    def __init__(self, settings: CaptureSettings, now: float):
        self._settings = settings
        self._remaining = max(1, int(settings.count))
        self._captured = 0
        self._active = True
        self._interval = (
            max(0.05, float(settings.interval))
            if settings.mode in {"burst", "interval"}
            else 0.0
        )
        self._next_shot = now + max(0.0, float(settings.initial_delay))

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
        if not self._active or self._captured:
            return 0.0
        return max(0.0, self._next_shot - now)

    def tick(self, now: float, capture: Callable[[int], None]) -> int:
        """Capture at most one frame per call.  Returns 1 when a shot fired."""

        if not self._active or self._remaining <= 0:
            return 0
        if now < self._next_shot:
            return 0

        capture(self._captured)
        self._captured += 1
        self._remaining -= 1
        if self._remaining <= 0:
            self._active = False
            return 1

        self._next_shot = now + self._interval
        return 1

    def cancel(self) -> None:
        self._active = False
        self._remaining = 0
