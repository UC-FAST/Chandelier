"""Numeric filters."""

from __future__ import annotations

from collections import deque


class SlidingWindowFilter:
    """Moving-average filter over a fixed-size window."""

    def __init__(self, max_size: int):
        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        self._max_size = max_size
        self._values: deque[float] = deque(maxlen=max_size)
        # Running total maintained incrementally so calc() stays O(1).
        self._sum = 0.0

    @property
    def max_size(self) -> int:
        return self._max_size

    def __len__(self) -> int:
        return len(self._values)

    def add_data(self, value: float) -> "SlidingWindowFilter":
        """Append ``value``, evicting the oldest sample when full.

        Returns ``self`` so calls can be chained.
        """

        if len(self._values) == self._max_size:
            self._sum -= self._values[0]
        self._values.append(value)
        self._sum += value
        return self

    def calc(self) -> float:
        """Return the mean of the current window; raise when empty."""

        if not self._values:
            raise ValueError("no data in the window")
        return self._sum / len(self._values)

    def clear(self) -> None:
        self._values.clear()
        self._sum = 0.0
