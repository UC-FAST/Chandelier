"""Protocols describing the hardware used by the application.

The UI and the controller depend on these structural types instead of concrete
drivers, which keeps them testable with the fakes in ``tests/fakes.py``.
"""

from __future__ import annotations

from typing import Iterator, Protocol

import numpy as np


class Display(Protocol):
    """A BGR frame sink.

    This mirrors only the surface the UI needs; concrete drivers may accept
    extra optional arguments (for example a ``clear`` colour).
    """

    width: int
    height: int

    def show(self, frame: np.ndarray) -> None: ...

    def clear(self) -> None: ...

    def backlight(self, state: bool) -> None: ...

    def close(self) -> None: ...


class Led(Protocol):
    """A binary indicator output."""

    def on(self) -> None: ...

    def off(self) -> None: ...

    def toggle(self) -> None: ...

    def close(self) -> None: ...


class InputPanel(Protocol):
    """A source of user actions from physical controls."""

    def close(self) -> None: ...


class Camera(Protocol):
    """Camera operations used by the preview, still and video code paths."""

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
    """Power/battery sensor with a releasable I2C handle."""

    def close(self) -> None: ...
