"""ST7735R 320x240 LCD driver over SPI.

Frames are BGR ``numpy.uint8`` arrays (the OpenCV convention).  The driver
converts the changed bounding box to RGB565 and writes only that window;
identical frames are skipped entirely.  Menus and overlays therefore clock
out a few rows instead of the whole 320x240 framebuffer.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from chandelier.config import PinConfig

logger = logging.getLogger(__name__)


class St7735Display:
    """SPI display adapter for the ST7735R panel.

    gpiozero and spidev are imported lazily so this module can be imported
    on hosts without the hardware.  Frames are BGR (OpenCV order); RGB565
    conversion and the panel's byte order are handled internally.
    """

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
        # Kept so show() can diff frames and send only the changed rectangle.
        self._last_frame: np.ndarray | None = None
        # Cached address window; consecutive partial updates often reuse it.
        self._window: tuple[int, int, int, int] | None = None
        self.backlight(True)

        self._spi = spidev.SpiDev(spi_bus, spi_device)
        self._spi.max_speed_hz = spi_freq
        # Mode 0 (CPOL=0, CPHA=0) matches the panel's SPI timing.
        self._spi.mode = 0b00

    def backlight(self, state: bool) -> None:
        """Turn the panel backlight on or off (active-high pin)."""
        self._bl_pin.value = bool(state)

    # The D/C pin acts as a ninth data bit: low for a command byte, high for
    # parameter/data bytes.  These helpers set it before every SPI transfer.
    def _command(self, command: int) -> None:
        """Send one command byte (D/C low)."""
        self._dc_pin.value = False
        self._spi.writebytes([command])

    def _data(self, value: int) -> None:
        """Send one data byte (D/C high)."""
        self._dc_pin.value = True
        self._spi.writebytes([value])

    def _data_bytes(self, values: bytes | bytearray) -> None:
        """Send a buffer of display data, used for pixel payloads."""
        self._dc_pin.value = True
        self._spi.writebytes2(values)

    def _data_values(self, values: list[int]) -> None:
        """Send a list of data bytes in a single SPI transfer."""
        self._dc_pin.value = True
        self._spi.writebytes(values)

    def reset(self) -> None:
        """Hardware-reset the panel and invalidate cached frame state."""
        self._madctl_configured = False
        self._last_frame = None
        self._window = None
        # RST is active-low; 10 ms is far above the datasheet minimum and
        # also leaves time for the panel supply to settle.
        self._rst_pin.value = True
        time.sleep(0.01)
        self._rst_pin.value = False
        time.sleep(0.01)
        self._rst_pin.value = True
        time.sleep(0.01)

    def init(self) -> None:
        """Run the vendor init sequence and program the scan direction."""
        self.reset()
        for command, values in _INIT_SEQUENCE:
            self._command(command)
            for value in values:
                self._data(value)
        self._configure_madctl()

    def _configure_madctl(self) -> None:
        # 0x36 is Memory Access Control: 0x70 selects the panel's scan
        # direction.  It is lost on reset, hence the configured flag.
        # 0x70 swaps rows/columns and flips the address/refresh direction
        # to match the module's landscape orientation.
        self._command(0x36)
        self._data(0x70)
        self._madctl_configured = True

    def _set_window(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Select the GRAM address window and prepare a memory write."""
        # 0x2A/0x2B set the column/row address range.  The panel remembers
        # them, so only resend when the dirty rectangle moves.  0x2C must be
        # sent every time: it resets the GRAM write pointer to the window.
        window = (x0, y0, x1, y1)
        if self._window != window:
            self._command(0x2A)
            self._data_values([x0 >> 8, x0 & 0xFF, x1 >> 8, x1 & 0xFF])
            self._command(0x2B)
            self._data_values([y0 >> 8, y0 & 0xFF, y1 >> 8, y1 & 0xFF])
            self._window = window
        self._command(0x2C)

    def show(self, frame: np.ndarray) -> None:
        """Display a BGR frame, sending only the changed bounding box."""
        if frame is None:
            logger.error("cannot display a None frame")
            return
        if frame.ndim != 3 or frame.shape[2] != 3:
            logger.error("expected a BGR frame, got shape %s", frame.shape)
            return

        height, width = frame.shape[:2]
        # Resize only on a mismatch so the common path stays cheap.
        if width != self.width or height != self.height:
            frame = cv2.resize(frame, (self.width, self.height))

        if not self._madctl_configured:
            self._configure_madctl()

        # The first frame (or a size change) has no reference; send it all.
        full = (
            self._last_frame is None
            or self._last_frame.shape != frame.shape
        )
        if full:
            x0, y0 = 0, 0
            x1, y1 = self.width - 1, self.height - 1
        else:
            # Only the changed bounding box is converted and clocked out.
            # Static screens skip SPI entirely; menus and text overlays touch
            # a few rows instead of the whole 320x240 panel.
            # Collapse colour channels: a pixel changed if any channel did.
            changed = np.any(frame != self._last_frame, axis=2)
            if not changed.any():
                return
            # Rows/columns holding at least one changed pixel bound the
            # rectangle that has to be re-sent.
            rows = np.flatnonzero(changed.any(axis=1))
            columns = np.flatnonzero(changed.any(axis=0))
            y0, y1 = int(rows[0]), int(rows[-1])
            x0, x1 = int(columns[0]), int(columns[-1])

        region = frame[y0 : y1 + 1, x0 : x1 + 1]
        # OpenCV's RGB565 output is a (H, W, 2) uint8 array whose byte pairs
        # are little-endian.  Reinterpret the pairs as uint16 and cast to
        # big-endian; the resulting bytes are what the panel expects.
        rgb565 = cv2.cvtColor(region, cv2.COLOR_BGR2BGR565)
        payload = rgb565.view("<u2").astype(">u2").tobytes()

        self._set_window(x0, y0, x1, y1)
        self._data_bytes(payload)

        # Only the pixels just sent need to be remembered; the rest of the
        # panel still matches the previous frame.  On a full update every
        # pixel is written below, so an uninitialised buffer is sufficient.
        if full:
            self._last_frame = np.empty_like(frame)
        self._last_frame[y0 : y1 + 1, x0 : x1 + 1] = region

    def clear(self, color: int = 0x0000) -> None:
        """Fill the panel with ``color`` (RGB565, default black)."""

        # The GRAM no longer matches the cached frame; force a full repaint.
        self._last_frame = None
        # RGB565 is big-endian on the wire: high byte first.
        high = (int(color) >> 8) & 0xFF
        low = int(color) & 0xFF
        self._set_window(0, 0, self.width - 1, self.height - 1)
        self._data_bytes(
            bytes((high, low)) * (self.width * self.height)
        )

    def close(self) -> None:
        """Release SPI and GPIO resources; safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        # Kill the backlight first so the panel goes dark immediately.
        self.backlight(False)
        try:
            self._spi.close()
        finally:
            for pin in (self._rst_pin, self._dc_pin, self._bl_pin):
                pin.close()


# (command, data bytes) initialisation sequence for the ST7735R panel.
# Highlights: 0x3A sets 16-bit RGB565, 0x21 enables inversion, 0x11 leaves
# sleep mode, and 0x29 turns the display on.  0xB2-0xD0 configure porch,
# gate, VCOM and power rails; 0xE0/0xE1 hold the gamma curves.  Values are
# the vendor-recommended defaults for this 320x240 module.
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
