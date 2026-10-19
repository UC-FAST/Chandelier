"""Tests for the standalone game's entry-point helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("game.app")

from game.app import _parse_args, load_high_score, save_high_score


def test_high_score_round_trip() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "highscore.json"
        assert load_high_score(path) == 0

        save_high_score(4321, path)
        assert load_high_score(path) == 4321


def test_corrupt_high_score_is_ignored() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "highscore.json"
        path.write_text("not json", encoding="utf-8")
        assert load_high_score(path) == 0


def test_parse_args_accepts_a_script_path() -> None:
    args = _parse_args(["/home/pi/game/startup.py", "--window", "--no-camera"])

    assert args.window is True
    assert args.no_camera is True


def test_parse_args_defaults_to_hardware_mode() -> None:
    args = _parse_args([])

    assert args.window is False
    assert args.no_camera is False
    assert args.scale == 3
