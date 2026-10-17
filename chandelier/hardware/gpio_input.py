"""GPIO button/rotary-encoder adapter.

Callbacks run on gpiozero's own threads.  The controller is responsible for
thread-safe state changes and for waking its blocking render loop.
"""

from __future__ import annotations

from typing import Callable

from chandelier.config import PinConfig
from chandelier.events import Action, Phase

ActionCallback = Callable[[Action, Phase], None]


class GpioInputPanel:
    """Binds the physical controls to :class:`Action` events."""

    def __init__(self, pins: PinConfig, on_action: ActionCallback):
        import gpiozero

        self._on_action = on_action
        self._devices: list = []

        self._bind_press(Action.SHUTTER, pins.shutter)
        self._bind_press(Action.SQUARE, pins.square)
        self._bind_press(Action.CROSS, pins.cross)
        self._bind_press(Action.CIRCLE, pins.circle)

        self._bind_edges(Action.UP, pins.up)
        self._bind_edges(Action.DOWN, pins.down)
        self._bind_edges(Action.LEFT, pins.left)
        self._bind_edges(Action.RIGHT, pins.right)
        self._bind_edges(Action.CENTER, pins.center)

        self._bind_press(Action.ROTARY_SELECT, pins.rotary_encoder_1s)

        self._rotary = gpiozero.RotaryEncoder(
            pins.rotary_encoder_1a, pins.rotary_encoder_1b
        )
        self._rotary.when_rotated_clockwise = lambda: self._on_action(
            Action.ROTARY_CLOCKWISE, Phase.PRESS
        )
        self._rotary.when_rotated_counter_clockwise = lambda: self._on_action(
            Action.ROTARY_COUNTER_CLOCKWISE, Phase.PRESS
        )
        self._devices.append(self._rotary)

    def _bind_press(self, action: Action, pin: int) -> None:
        import gpiozero

        device = gpiozero.DigitalInputDevice(pin=pin)
        device.when_activated = lambda: self._on_action(action, Phase.PRESS)
        self._devices.append(device)

    def _bind_edges(self, action: Action, pin: int) -> None:
        """Create a device that emits both press and release actions."""

        import gpiozero

        device = gpiozero.DigitalInputDevice(pin=pin)
        device.when_activated = lambda: self._on_action(action, Phase.PRESS)
        device.when_deactivated = lambda: self._on_action(action, Phase.RELEASE)
        self._devices.append(device)

    def close(self) -> None:
        for device in self._devices:
            try:
                device.close()
            except Exception:
                pass
        self._devices.clear()
