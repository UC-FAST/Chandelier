import pytest

from chandelier.ui.capture_session import CaptureSession, CaptureSettings


def _run(session, times):
    shots = []
    for now in times:
        session.tick(now, shots.append)
    return shots


def test_single_with_delay():
    session = CaptureSession(
        CaptureSettings("single", count=1, initial_delay=2.0), now=10.0
    )
    assert session.tick(11.9, lambda index: None) == 0
    assert session.countdown_remaining(11.9) == pytest.approx(0.1)
    assert session.tick(12.0, lambda index: None) == 1
    assert session.done
    assert session.tick(13.0, lambda index: None) == 0


def test_burst_spacing_and_count():
    session = CaptureSession(
        CaptureSettings("burst", count=3, interval=0.5), now=0.0
    )
    shots = _run(session, [0.0, 0.2, 0.5, 0.8, 1.0, 1.5])
    assert shots == [0, 1, 2]
    assert session.done


def test_interval_spacing_and_count():
    session = CaptureSession(
        CaptureSettings("interval", count=3, interval=2.0), now=0.0
    )
    shots = _run(session, [0.0, 1.0, 2.0, 3.9, 4.0])
    assert shots == [0, 1, 2]


def test_duplicate_tick_does_not_duplicate_shot():
    session = CaptureSession(
        CaptureSettings("burst", count=2, interval=1.0), now=0.0
    )
    assert session.tick(0.0, lambda index: None) == 1
    assert session.tick(0.0, lambda index: None) == 0
    assert session.tick(1.0, lambda index: None) == 1


def test_cancel_stops_future_shots():
    session = CaptureSession(
        CaptureSettings("interval", count=10, interval=1.0), now=0.0
    )
    assert session.tick(0.0, lambda index: None) == 1
    session.cancel()
    assert session.tick(2.0, lambda index: None) == 0
    assert session.done


def test_zero_interval_is_clamped():
    session = CaptureSession(
        CaptureSettings("burst", count=2, interval=0.0), now=0.0
    )
    assert session.tick(0.0, lambda index: None) == 1
    assert session.tick(0.0, lambda index: None) == 0
