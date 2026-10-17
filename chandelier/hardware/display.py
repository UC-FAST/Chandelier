"""ST7735R 320x240 LCD driver over SPI.

Frames are BGR ``numpy.uint8`` arrays (the OpenCV convention).  The driver
converts to RGB565 before writing to the panel.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from chandelier.config import PinConfig

logger = logging.getLogger(__name__)


class St7735Display:
    """SPI display adapter.  Imports gpiozero/spidev lazily."""

    def __init__(
        self,
        width: int,
        height: int,
        pins: PinConfig,
        spi_freq: int = 80_000_000,
        spi_bus: int = 0,
        spi_device: int = 0,
    ):
        import gpiozero
        import spidev

        self.width = width
        self.height = height
        self._rst_pin = gpiozero.DigitalOutputDevice(
            pins.rst, active_high=True, initial_value=False
        )
        self._dc_pin = gpiozero.DigitalOutputDevice(
            pins.dc, active_high=True, initial_value=False
        )
        self._bl_pin = gpiozero.DigitalOutputDevice(
            pins.bl, active_high=True, initial_value=False
        )
        self._closed = False
        self._madctl_configured = False
        self.backlight(True)

        self._spi = spidev.SpiDev(spi_bus, spi_device)
        self._spi.max_speed_hz = spi_freq
        self._spi.mode = 0b00

    def backlight(self, state: bool) -> None:
        self._bl_pin.value = bool(state)

    def _command(self, command: int) -> None:
        self._dc_pin.value = False
        self._spi.writebytes([command])

    def _data(self, value: int) -> None:
        self._dc_pin.value = True
        self._spi.writebytes([value])

    def _data_bytes(self, values: bytes | bytearray) -> None:
        self._dc_pin.value = True
        self._spi.writebytes2(values)

    def _data_values(self, values: list[int]) -> None:
        self._dc_pin.value = True
        self._spi.writebytes(values)

    def reset(self) -> None:
        self._madctl_configured = False
        self._rst_pin.value = True
        time.sleep(0.01)
        self._rst_pin.value = False
        time.sleep(0.01)
        self._rst_pin.value = True
        time.sleep(0.01)

    def init(self) -> None:
        self.reset()
        for command, values in _INIT_SEQUENCE:
            self._command(command)
            for value in values:
                self._data(value)
        self._configure_madctl()

    def _configure_madctl(self) -> None:
        self._command(0x36)
        self._data(0x70)
        self._madctl_configured = True

    def _set_window(self, x0: int, y0: int, x1: int, y1: int) -> None:
        self._command(0x2A)
        self._data_values([x0 >> 8, x0 & 0xFF, x1 >> 8, x1 & 0xFF])
        self._command(0x2B)
        self._data_values([y0 >> 8, y0 & 0xFF, y1 >> 8, y1 & 0xFF])
        self._command(0x2C)

    def show(self, frame: np.ndarray) -> None:
        if frame is None:
            logger.error("cannot display a None frame")
            return
        if frame.ndim != 3 or frame.shape[2] != 3:
            logger.error("expected a BGR frame, got shape %s", frame.shape)
            return

        height, width = frame.shape[:2]
        if width != self.width or height != self.height:
            frame = cv2.resize(frame, (self.width, self.height))

        if not self._madctl_configured:
            self._configure_madctl()

        # OpenCV packs RGB565 in host (little-endian) byte pairs; the panel
        # expects the high byte first, so convert explicitly to big-endian.
        rgb565 = cv2.cvtColor(frame, cv2.COLOR_BGR2BGR565)
        payload = rgb565.view("<u2").astype(">u2").tobytes()

        self._set_window(0, 0, self.width - 1, self.height - 1)
        self._data_bytes(payload)

    def clear(self) -> None:
        self._set_window(0, 0, self.width - 1, self.height - 1)
        self._data_bytes(bytes([0xFF]) * (self.width * self.height * 2))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.backlight(False)
        try:
            self._spi.close()
        finally:
            for pin in (self._rst_pin, self._dc_pin, self._bl_pin):
                pin.close()


# (command, data bytes) initialisation sequence for the ST7735R panel.
_INIT_SEQUENCE: tuple[tuple[int, tuple[int, ...]], ...] = (
    (0x3A, (0x05,)),
    (0x21, ()),
    (0x2A, (0x00, 0x00, 0x01, 0x3F)),
    (0x2B, (0x00, 0x00, 0x00, 0xEF)),
    (0xB2, (0x0C, 0x0C, 0x00, 0x33, 0x33)),
    (0xB7, (0x35,)),
    (0xBB, (0x1F,)),
    (0xC0, (0x2C,)),
    (0xC2, (0x01,)),
    (0xC3, (0x12,)),
    (0xC4, (0x20,)),
    (0xC6, (0x0F,)),
    (0xD0, (0xA4, 0xA1)),
    (
        0xE0,
        (
            0xD0,
            0x08,
            0x11,
            0x08,
            0x0C,
            0x15,
            0x39,
            0x33,
            0x50,
            0x36,
            0x13,
            0x14,
            0x29,
            0x2D,
        ),
    ),
    (
        0xE1,
        (
            0xD0,
            0x08,
            0x10,
            0x08,
            0x06,
            0x06,
            0x39,
            0x44,
            0x51,
            0x0B,
            0x16,
            0x14,
            0x2F,
            0x31,
        ),
    ),
    (0x21, ()),
    (0x11, ()),
    (0x29, ()),
)
