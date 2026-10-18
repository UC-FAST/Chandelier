"""Pytest configuration shared by the test suite."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def reset_ui_language():
    from chandelier.i18n import translator
    from chandelier.ui.themes import DEFAULT_THEME_ID, set_active_theme

    translator.set_language("en")
    set_active_theme(DEFAULT_THEME_ID)
    yield
    translator.set_language("en")
    set_active_theme(DEFAULT_THEME_ID)


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """A minimal valid config.json in a temporary directory."""

    path = tmp_path / "config.json"
    path.write_text(
        """
        {
            "pin": {
                "right": 6, "up": 0, "left": 22, "down": 1, "center": 5,
                "rotary_encoder_1a": 17, "rotary_encoder_1b": 24,
                "rotary_encoder_1s": 27, "shutter": 16, "cross": 23,
                "square": 25, "circle": 20, "rst": 19, "dc": 26, "bl": 21
            },
            "camera": {
                "path": "pict", "video_path": "pict/video",
                "video_bitrate": 1000000,
                "tuning_file": null,
                "preview_width": 640,
                "preview_height": 480
            },
            "screen": {"width": 320, "height": 240},
            "led": {"green": 12, "blue": 13},
            "sensor": {
                "bq32002": {"bus": 1},
                "ina230": {
                    "bus": 1, "shunt_resistance": 0.002,
                    "maximum_expected_current": 3.2
                },
                "max17048": {"bus": 1}
            },
            "localizer": {
                "device": 0, "rate": 16000, "chunk": 1600,
                "channels": 6, "ignored_channels": [0, 5],
                "n_sources": 3, "confidence_threshold": 0.03,
                "max_to_cumulate": 5, "speech_threshold": 0.0,
                "silero_model_dir": null, "delay_max": 0.0,
                "n_delay_bins": 50, "mic_rotation": 0.0
            },
            "marker": {
                "mode": "single", "max_sources": 3, "ttl": 3.0,
                "smoothing": 5, "trail": 3, "size": 24
            },
            "debug_level": 10,
            "internet_test_url": "http://example.invalid/connecttest.txt",
            "ntp_server": "ntp1.aliyun.com",
            "menus_file": "config/menus.json"
        }
        """,
        encoding="utf-8",
    )
    return path
