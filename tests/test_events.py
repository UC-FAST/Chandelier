from chandelier.events import ACTION_SPECS, Action, Phase, resolve_action


def test_every_action_has_a_press_spec():
    actions = {action for action, phase in ACTION_SPECS if phase is Phase.PRESS}
    assert actions == set(Action)


def test_directional_actions_dispatch_press_and_release():
    for action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT):
        assert resolve_action(action, Phase.PRESS) is not None
        assert resolve_action(action, Phase.RELEASE) is not None


def test_function_buttons_are_gated():
    for action in (
        Action.CENTER,
        Action.CIRCLE,
        Action.SQUARE,
        Action.CROSS,
        Action.SHUTTER,
        Action.ROTARY_CLOCKWISE,
        Action.ROTARY_COUNTER_CLOCKWISE,
        Action.ROTARY_SELECT,
    ):
        spec = resolve_action(action, Phase.PRESS)
        assert spec is not None
        assert spec.requires_enable is True


def test_directional_buttons_are_not_gated():
    for action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT):
        spec = resolve_action(action, Phase.PRESS)
        assert spec is not None
        assert spec.requires_enable is False


def test_release_without_spec_returns_none():
    assert resolve_action(Action.SHUTTER, Phase.RELEASE) is None
