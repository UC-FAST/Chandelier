"""GPIO LEDs.  Devices are created only when :class:`Leds` is instantiated."""

from __future__ import annotations

from chandelier.config import LedConfig


class GpioLed:
    """Single GPIO-driven LED.

    ``active_high=False`` assumes the LED anode is tied to 3.3 V and the
    GPIO pin sinks the current.
    """

    def __init__(self, pin: int, active_high: bool = False):
        import gpiozero

        self._device = gpiozero.DigitalOutputDevice(
            pin=pin, active_high=active_high, initial_value=False
        )

    def on(self) -> None:
        self._device.on()

    def off(self) -> None:
        self._device.off()

    def toggle(self) -> None:
        self._device.toggle()

    @property
    def is_on(self) -> bool:
        return bool(self._device.value)

    def close(self) -> None:
        """Switch the LED off and release the GPIO line."""
        try:
            self._device.off()
        finally:
            # Release the pin even if switching it off fails.
            self._device.close()


class Leds:
    """The green and blue indicator LEDs."""

    def __init__(self, config: LedConfig):
        self.green = GpioLed(config.green)
        self.blue = GpioLed(config.blue)

    def close(self) -> None:
        """Close both indicator LEDs."""
        self.green.close()
        self.blue.close()
