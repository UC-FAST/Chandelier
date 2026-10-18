"""Tests for the standalone Chandelier Breakout game."""

from __future__ import annotations

import numpy as np

from game.engine import (
    BALL_RADIUS,
    FIELD_LEFT,
    FIELD_RIGHT,
    PADDLE_WIDTH,
    PADDLE_Y,
    WIDE_PADDLE_WIDTH,
    BreakoutGame,
    Capsule,
    State,
)
from game.renderer import Renderer


def new_game() -> BreakoutGame:
    game = BreakoutGame(seed=7)
    game.primary_action()  # title -> ready
    return game


def playing_game() -> BreakoutGame:
    game = new_game()
    game.primary_action()  # ready -> playing
    return game


def test_new_game_has_bricks_and_a_ball() -> None:
    game = new_game()

    assert game.state is State.READY
    assert game.lives == 3
    assert game.level == 1
    assert len(game.bricks) == 45
    assert len(game.balls) == 1
    assert game.balls[0].dy < 0


def test_launch_moves_ball_up() -> None:
    game = playing_game()
    y = game.balls[0].y

    game.update(1 / 60)

    assert game.state is State.PLAYING
    assert game.balls[0].y < y


def test_paddle_is_clamped_to_the_field() -> None:
    game = new_game()

    game.nudge_paddle(-10_000)
    assert game.paddle_x == FIELD_LEFT

    game.nudge_paddle(10_000)
    assert game.paddle_x == FIELD_RIGHT - PADDLE_WIDTH


def test_brick_break_scores_and_starts_combo() -> None:
    game = playing_game()
    brick = game.bricks[0]
    ball = game.balls[0]
    ball.x = brick.x + brick.width / 2
    ball.y = brick.y - BALL_RADIUS - 0.5
    ball.dx = 0.0
    ball.dy = 1.0

    events = game.update(0.02)

    assert brick.hp == 0
    assert game.combo == 1
    assert game.score >= 50
    assert any(event.kind == "brick_break" for event in events)


def test_losing_the_ball_costs_a_life_and_returns_to_ready() -> None:
    game = playing_game()
    ball = game.balls[0]
    ball.x = 160
    ball.y = 260
    ball.dx = 0.0
    ball.dy = 1.0

    events = game.update(0.02)

    assert game.lives == 2
    assert game.state is State.READY
    assert len(game.balls) == 1
    assert any(event.kind == "life_lost" for event in events)


def test_three_lost_balls_end_the_game_and_record_a_high_score() -> None:
    game = playing_game()
    game.score = 1234

    for expected_lives in (2, 1, 0):
        ball = game.balls[0]
        ball.x = 160
        ball.y = 260
        ball.dx = 0.0
        ball.dy = 1.0
        events = game.update(0.02)
        assert game.lives == expected_lives
        if expected_lives:
            assert game.state is State.READY
            game.primary_action()  # launch again
        else:
            assert game.state is State.GAME_OVER
            assert any(event.kind == "game_over" for event in events)
            assert any(event.kind == "new_high_score" for event in events)

    assert game.high_score == game.score


def test_pause_freezes_the_world() -> None:
    game = playing_game()
    game.update(1 / 60)
    x, y = game.balls[0].x, game.balls[0].y

    game.toggle_pause()
    assert game.state is State.PAUSED
    game.update(1.0)
    assert (game.balls[0].x, game.balls[0].y) == (x, y)

    game.primary_action()
    assert game.state is State.PLAYING


def test_clearing_all_bricks_advances_the_level() -> None:
    game = playing_game()
    for brick in game.bricks:
        brick.hp = 0

    events = game.update(1 / 60)
    assert game.state is State.LEVEL_CLEAR
    assert any(event.kind == "level_clear" for event in events)
    assert game.score > 0

    for _ in range(40):
        game.update(0.05)
    assert game.state is State.READY
    assert game.level == 2
    assert len(game.bricks) == 45


def test_wide_capsule_widens_the_paddle() -> None:
    game = playing_game()
    game.capsules.append(
        Capsule(x=game.paddle_x + 5, y=PADDLE_Y - 2, kind="W")
    )

    events = game.update(0.01)

    assert game.wide_active
    assert game.paddle_width == WIDE_PADDLE_WIDTH
    assert any(event.kind == "powerup" for event in events)


def test_multiball_capsule_splits_the_ball() -> None:
    game = playing_game()
    game.capsules.append(
        Capsule(x=game.paddle_x + 5, y=PADDLE_Y - 2, kind="M")
    )

    game.update(0.01)

    assert len(game.balls) == 3
    assert len({(ball.dx, ball.dy) for ball in game.balls}) == 3


def test_rotary_nudge_is_clamped_and_ignored_on_title() -> None:
    game = BreakoutGame(seed=1)
    start = game.paddle_x
    game.nudge_paddle(9)
    assert game.paddle_x == start  # title screen does not move the paddle

    game.primary_action()
    game.nudge_paddle(9)
    assert game.paddle_x == start + 9


def test_renderer_handles_every_screen_state() -> None:
    renderer = Renderer(320, 240)
    camera = np.zeros((240, 320, 3), dtype=np.uint8)
    game = BreakoutGame(seed=3)

    frames = [renderer.render(game, None, 0.0)]
    game.primary_action()  # ready
    frames.append(renderer.render(game, camera, 0.5))
    game.primary_action()  # playing
    frames.append(renderer.render(game, camera, 1.0))
    game.toggle_pause()
    frames.append(renderer.render(game, None, 1.5, banner=("QUIT?", "X")))
    game.toggle_pause()

    for brick in game.bricks:
        brick.hp = 0
    game.update(1 / 60)
    frames.append(renderer.render(game, None, 2.0))

    finished = playing_game()
    finished.lives = 1
    finished.balls[0].y = 260
    finished.balls[0].dx = 0.0
    finished.balls[0].dy = 1.0
    finished.update(1 / 60)
    assert finished.state is State.GAME_OVER
    frames.append(renderer.render(finished, camera, 3.0))

    for frame in frames:
        assert frame.shape == (240, 320, 3)
        assert frame.dtype == np.uint8
