"""Input actions and inter-screen messages."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Action(str, Enum):
    """Physical input events reported by the GPIO panel and web preview."""

    CENTER = "center"
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    CIRCLE = "circle"
    SQUARE = "square"
    CROSS = "cross"
    SHUTTER = "shutter"
    ROTARY_CLOCKWISE = "rotary_clockwise"
    ROTARY_COUNTER_CLOCKWISE = "rotary_counter_clockwise"
    ROTARY_SELECT = "rotary_select"


class Phase(str, Enum):
    """Whether a button event is a press or a release edge."""

    PRESS = "press"
    RELEASE = "release"


@dataclass(frozen=True)
class ActionSpec:
    """How one (action, phase) pair is dispatched to a controlled end."""

    method_name: str
    requires_enable: bool = False


# This table deliberately mirrors the original event routing, including its
# inconsistencies: the directional buttons are not gated by ``enabled`` while
# the function buttons and the rotary encoder are.
ACTION_SPECS: dict[tuple[Action, Phase], ActionSpec] = {
    (Action.CENTER, Phase.PRESS): ActionSpec("center_press_action", True),
    (Action.CENTER, Phase.RELEASE): ActionSpec("center_release_action", True),
    (Action.UP, Phase.PRESS): ActionSpec("up_press_action"),
    (Action.UP, Phase.RELEASE): ActionSpec("up_release_action"),
    (Action.DOWN, Phase.PRESS): ActionSpec("down_press_action"),
    (Action.DOWN, Phase.RELEASE): ActionSpec("down_release_action"),
    (Action.LEFT, Phase.PRESS): ActionSpec("left_press_action"),
    (Action.LEFT, Phase.RELEASE): ActionSpec("left_release_action"),
    (Action.RIGHT, Phase.PRESS): ActionSpec("right_press_action"),
    (Action.RIGHT, Phase.RELEASE): ActionSpec("right_release_action"),
    (Action.CIRCLE, Phase.PRESS): ActionSpec("circle_press_action", True),
    (Action.SQUARE, Phase.PRESS): ActionSpec("square_press_action", True),
    (Action.CROSS, Phase.PRESS): ActionSpec("cross_press_action", True),
    (Action.SHUTTER, Phase.PRESS): ActionSpec("shutter_press_action", True),
    (Action.ROTARY_CLOCKWISE, Phase.PRESS): ActionSpec(
        "rotary_encoder_clockwise", True
    ),
    (Action.ROTARY_COUNTER_CLOCKWISE, Phase.PRESS): ActionSpec(
        "rotary_encoder_counter_clockwise", True
    ),
    (Action.ROTARY_SELECT, Phase.PRESS): ActionSpec(
        "rotary_encoder_select", True
    ),
}


def resolve_action(
    action: Action, phase: Phase = Phase.PRESS
) -> ActionSpec | None:
    """Return the dispatch spec for an event, or ``None`` when unbound."""

    return ACTION_SPECS.get((action, phase))


@dataclass(frozen=True)
class Message:
    """A message between controlled ends.

    ``payload`` keeps the original ad-hoc shapes used by the menu/camera/
    gallery protocol.  The menu still passes its live dict to the camera, so
    the payload must not be copied.
    """

    sender: str
    receiver: str
    payload: Any = None
