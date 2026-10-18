import multiprocessing as mp

import pytest

from chandelier.config import LocalizerConfig
from chandelier.localizer.worker import (
    LatestLocalization,
    LocalizerSettings,
    SharedLocalizerSettings,
    SourceSnapshot,
    _context_name,
)


def _source(azimuth, elevation, x, y, rank=0, confidence=0.25):
    return SourceSnapshot(
        confidence=confidence,
        azimuth=azimuth,
        elevation=elevation,
        screen_x=x,
        screen_y=y,
        rank=rank,
    )


def _child_write(latest):
    latest.write(
        0.25,
        [
            _source(12.0, 34.0, 5, 6, rank=0),
            _source(56.0, 78.0, 50, 60, rank=1),
        ],
    )


def test_latest_localization_round_trip():
    latest = LatestLocalization()
    assert latest.read() is None

    latest.write(0.5, [_source(90.0, 20.0, 160, 60)])
    batch = latest.read()
    assert batch is not None
    assert batch.confidence == 0.5
    assert len(batch.sources) == 1
    source = batch.strongest
    assert source is not None
    assert source.azimuth == 90.0
    assert source.elevation == 20.0
    assert (source.screen_x, source.screen_y) == (160, 60)
    assert batch.generation == 1


def test_multiple_sources_round_trip():
    latest = LatestLocalization()
    latest.write(
        0.5,
        [
            _source(10.0, 20.0, 100, 80, rank=0),
            _source(30.0, 40.0, 120, 90, rank=1),
            _source(50.0, 60.0, 140, 100, rank=2),
        ],
    )
    batch = latest.read()
    assert batch is not None
    assert [source.rank for source in batch.sources] == [0, 1, 2]
    assert batch.sources[1].azimuth == 30.0


def test_clear_only_bumps_generation():
    latest = LatestLocalization()
    latest.write(0.5, [_source(90.0, 20.0, 160, 60)])
    latest.clear()
    batch = latest.read()
    assert batch is not None
    assert batch.confidence == 0.0
    assert batch.sources == ()
    assert batch.generation == 2


def test_context_name_is_supported():
    assert _context_name() in {"fork", "spawn"}


def test_shared_localizer_settings_round_trip_and_clamp():
    config = LocalizerConfig(
        n_sources=3,
        confidence_threshold=0.03,
        max_to_cumulate=5,
        speech_threshold=0.0,
        mic_rotation=0.0,
        delay_max=0.0,
        n_delay_bins=50,
    )
    settings = SharedLocalizerSettings(config)
    assert settings.snapshot() == LocalizerSettings.from_config(config)
    assert settings.snapshot().marker_fov == pytest.approx(83.0)
    assert settings.snapshot().marker_azimuth_offset == pytest.approx(0.0)
    assert settings.snapshot().marker_elevation_offset == pytest.approx(0.0)
    assert settings.snapshot().marker_x_gain == pytest.approx(0.0)
    assert settings.snapshot().marker_y_gain == pytest.approx(0.0)
    assert settings.snapshot().marker_mic_offset_x == pytest.approx(0.0)
    assert settings.snapshot().marker_source_distance == pytest.approx(0.0)

    assert settings.update(
        n_sources=5,
        confidence_threshold=0.25,
        mic_rotation=370.0,
        marker_fov=90.5,
        marker_azimuth_offset=370.0,
        marker_elevation_offset=-18.0,
        marker_x_gain=0.5,
        marker_y_gain=2.5,
        marker_mic_offset_x=0.03,
        marker_mic_offset_y=-0.05,
        marker_mic_offset_z=0.01,
        marker_source_distance=0.45,
    )
    snapshot = settings.snapshot()
    assert snapshot.n_sources == 5
    assert snapshot.confidence_threshold == 0.25
    assert snapshot.mic_rotation == pytest.approx(10.0)
    assert snapshot.marker_fov == pytest.approx(90.5)
    assert snapshot.marker_azimuth_offset == pytest.approx(10.0)
    assert snapshot.marker_elevation_offset == pytest.approx(-18.0)
    assert snapshot.marker_x_gain == pytest.approx(0.5)
    assert snapshot.marker_y_gain == pytest.approx(2.5)
    assert snapshot.marker_mic_offset_x == pytest.approx(0.03)
    assert snapshot.marker_mic_offset_y == pytest.approx(-0.05)
    assert snapshot.marker_mic_offset_z == pytest.approx(0.01)
    assert snapshot.marker_source_distance == pytest.approx(0.45)

    # Integer fields are rounded and every value is clamped to the menu range.
    assert settings.update(
        n_sources=99,
        n_delay_bins=1,
        marker_fov=999,
        marker_mic_offset_x=99.0,
        marker_source_distance=-1.0,
        marker_x_gain=99.0,
        marker_y_gain=-99.0,
    )
    snapshot = settings.snapshot()
    assert snapshot.n_sources == 8
    assert snapshot.n_delay_bins == 2
    assert snapshot.marker_fov == pytest.approx(179.0)
    assert snapshot.marker_mic_offset_x == pytest.approx(10.0)
    assert snapshot.marker_source_distance == pytest.approx(0.0)
    assert snapshot.marker_x_gain == pytest.approx(3.0)
    assert snapshot.marker_y_gain == pytest.approx(0.0)

    assert settings.update(marker_fov=0)
    assert settings.snapshot().marker_fov == pytest.approx(1.0)

    with pytest.raises(KeyError):
        settings.update(unknown_setting=1)


def test_localizer_handle_forwards_updates():
    class FakeSettings:
        def __init__(self):
            self.updates = []

        def update(self, **values):
            self.updates.append(values)
            return True

    from chandelier.localizer.worker import LocalizerHandle

    handle = LocalizerHandle(
        process=None,
        latest=LatestLocalization(),
        stop_event=None,
        settings=FakeSettings(),
    )
    assert handle.update_settings(n_sources=4) is True
    assert handle.settings.updates == [{"n_sources": 4}]


def _child_update_settings(settings):
    settings.update(n_sources=7, mic_rotation=45.0, marker_fov=75.0)


def test_shared_settings_work_across_processes():
    context = mp.get_context("spawn")
    settings = SharedLocalizerSettings(LocalizerConfig(), context)
    process = context.Process(target=_child_update_settings, args=(settings,))
    process.start()
    process.join(timeout=10)

    assert process.exitcode == 0
    snapshot = settings.snapshot()
    assert snapshot.n_sources == 7
    assert snapshot.mic_rotation == pytest.approx(45.0)
    assert snapshot.marker_fov == pytest.approx(75.0)


def test_shared_state_works_across_processes():
    context = mp.get_context("spawn")
    latest = LatestLocalization(context)
    process = context.Process(target=_child_write, args=(latest,))
    process.start()
    process.join(timeout=10)

    assert process.exitcode == 0
    batch = latest.read()
    assert batch is not None
    assert len(batch.sources) == 2
    assert (batch.sources[1].screen_x, batch.sources[1].screen_y) == (50, 60)
