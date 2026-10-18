"""Protocols describing the hardware used by the application.

The UI and the controller depend on these structural types instead of concrete
drivers, which keeps them testable with the fakes in ``tests/fakes.py``.
"""

from __future__ import annotations

from typing import Iterator, Protocol

import numpy as np


class Display(Protocol):
    """A BGR frame sink."""

    width: int
    height: int

    def show(self, frame: np.ndarray) -> None: ...

    def clear(self) -> None: ...

    def backlight(self, state: bool) -> None: ...

    def close(self) -> None: ...


class Led(Protocol):
    def on(self) -> None: ...

    def off(self) -> None: ...

    def toggle(self) -> None: ...

    def close(self) -> None: ...


class InputPanel(Protocol):
    def close(self) -> None: ...


class Camera(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...

    def preview(self) -> Iterator[np.ndarray]: ...

    def zoom(self, zoom: float) -> None: ...

    def brightness(self, brightness: float) -> None: ...

    def save_frame(
        self,
        file_path: str,
        extension: str,
        width: int,
        height: int,
        save_metadata: bool = False,
        save_raw: bool = False,
    ) -> None: ...

    def start_recording(self, width: int, height: int, file_path: str) -> None: ...

    def stop_recording(self) -> None: ...


class Sensor(Protocol):
    def close(self) -> None: ...
