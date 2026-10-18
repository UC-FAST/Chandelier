import numpy as np
import pytest

from chandelier.ui.decorators import (
    BarChart,
    Histogram,
    MarkerBox,
    SimpleText,
    Toast,
    Warning,
)


def _frame():
    return np.zeros((240, 320, 3), dtype=np.uint8)


def test_bar_chart_tracks_data():
    chart = BarChart()
    chart.add_data(1).add_data(2).add_data(3)
    assert chart.data_list == [1, 2, 3]
    frame = _frame()
    chart.decorate(frame)
    assert frame.any()


def test_histogram_decorate():
    frame = _frame()
    frame[100:120, 100:120] = 200
    Histogram().decorate(frame)
    assert frame.shape == (240, 320, 3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scale": "log"},
        {"channel_mode": "luma"},
        {"scale": "log", "channel_mode": "luma"},
        {"alpha": 1.0, "background": None},
        {"smoothing": 0.5, "padding": (5, 5, 5, 5), "grid": False},
    ],
)
def test_histogram_modes_render_bgr_and_grayscale(kwargs):
    frame = _frame()
    frame[100:120, 100:120] = 200
    Histogram(**kwargs).decorate(frame)
    assert frame.shape == (240, 320, 3)

    gray = np.full((240, 320), 128, dtype=np.uint8)
    Histogram(**kwargs).decorate(gray)
    assert gray.shape == (240, 320)


def test_histogram_rotation_and_empty_frame():
    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    frame[100:120, 100:120] = 200
    Histogram().decorate(frame, rotate=90)
    assert frame.shape == (240, 240, 3)

    empty = _frame()
    Histogram().decorate(empty)
    assert empty.shape == (240, 320, 3)


def test_histogram_rejects_invalid_options():
    with pytest.raises(ValueError):
        Histogram(scale="cubic")
    with pytest.raises(ValueError):
        Histogram(channel_mode="xyz")
    with pytest.raises(ValueError):
        Histogram(padding=(-1, 0, 0, 0))


def test_marker_box_does_not_print(capsys):
    frame = _frame()
    MarkerBox(width=320, height=240, size=(20, 20), cache=3).decorate(
        frame, (160, 120), "50"
    )
    assert capsys.readouterr().out == ""


def test_marker_box_uses_median_and_expires():
    marker = MarkerBox(
        width=320, height=240, size=(20, 20), cache=3, ttl=2.0, trail=0
    )
    for x, timestamp in ((100, 0.0), (110, 0.1), (120, 0.2)):
        marker.update(0, (x, 100), "AZ 90 EL 45", timestamp=timestamp)

    frame = _frame()
    marker.decorate(frame, timestamp=0.2)
    # The crosshair is drawn at the median position (anti-aliased).
    blue, green, red = (int(value) for value in frame[100, 110])
    assert red > 200 and blue < 20 and green < 20

    stale = _frame()
    marker.decorate(stale, timestamp=3.0)
    assert not stale.any()


def test_marker_box_draws_edge_arrow_for_offscreen_source():
    marker = MarkerBox(width=320, height=240, size=(24, 24), ttl=2.0)
    frame = _frame()
    marker.decorate(frame, (-40, 120), "", timestamp=0.0)

    # Track 0 is red; the arrow points left from the left screen edge.
    blue, green, red = (int(value) for value in frame[120, 12])
    assert red > 150 and blue < 80 and green < 80
    # No box is drawn at the centre for an off-screen source.
    assert not frame[120, 160].any()


def test_marker_box_draws_arrow_at_frame_border():
    marker = MarkerBox(width=320, height=240, size=(24, 24), ttl=2.0)
    marker.update(0, (320, 120), "", timestamp=0.0)

    frame = _frame()
    marker.decorate(frame, timestamp=0.0)

    blue, green, red = (int(value) for value in frame[120, 307])
    assert red > 150 and blue < 80 and green < 80


def test_warning_does_not_accumulate_text():
    warning = Warning()
    first = warning.decorate("Empty")
    second = warning.decorate("Empty")
    assert first.shape == (128, 128, 3)
    assert np.array_equal(first, second)


def test_simple_text_decorate():
    frame = _frame()
    widget = SimpleText([lambda: {"Value {}": 3}], height=240)
    result = widget.decorate(frame)
    assert result.shape == frame.shape


def test_toast_is_cleared_after_draw():
    frame = _frame()
    toast = Toast()
    toast.set_text("hello")
    assert toast.is_update is True
    toast.decorate(frame)
    assert toast.is_update is False
    assert toast.visible is True
    assert frame.any()


def test_toast_expires_and_can_be_cleared():
    now = [0.0]
    toast = Toast(duration=1.0, clock=lambda: now[0])
    toast.set_text("saved")
    assert toast.visible is True

    now[0] = 0.9
    assert toast.visible is True
    now[0] = 1.1
    assert toast.visible is False

    frame = _frame()
    toast.decorate(frame)
    assert not frame.any()

    toast.set_text("again", duration=0)
    assert toast.visible is True
    toast.clear()
    assert toast.visible is False


def test_toast_same_text_refreshes_timer():
    now = [0.0]
    toast = Toast(duration=1.0, clock=lambda: now[0])
    toast.set_text("X")
    now[0] = 0.9
    toast.set_text("X")
    now[0] = 1.5
    assert toast.visible is True


def test_toast_fits_long_text_and_small_frames():
    toast = Toast(clock=lambda: 0.0)
    toast.set_text("A" * 200)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    toast.decorate(frame)
    assert frame.any()
    # The text is clipped to the frame rather than overflowing the array.
    assert frame[110:, :].any()


def test_toast_rotation_keeps_frame_shape():
    toast = Toast(clock=lambda: 0.0)
    toast.set_text("ROT")
    frame = _frame()
    toast.decorate(frame, rotate=90)
    assert frame.shape == (240, 320, 3)


def test_marker_box_draws_multiple_tracks():
    marker = MarkerBox(
        width=320, height=240, size=(20, 20), cache=3, ttl=2.0, trail=0
    )
    marker.update(0, (100, 100), "S1 90/45", timestamp=0.0)
    marker.update(1, (200, 100), "S2 180/20", timestamp=0.0)

    frame = _frame()
    marker.decorate(frame, timestamp=0.0)

    red_blue, red_green, red_red = (int(v) for v in frame[100, 100])
    green_blue, green_green, green_red = (int(v) for v in frame[100, 200])
    assert red_red > 200 and red_blue < 20 and red_green < 20
    assert green_green > 200 and green_blue < 20 and green_red < 20


def test_rule_of_thirds_and_center_mark():
    from chandelier.ui.decorators import CenterMark, RuleOfThirds

    frame = _frame()
    RuleOfThirds(320, 240).decorate(frame)
    assert frame[:, 320 // 3].any()
    assert frame[240 // 3, :].any()

    frame = _frame()
    CenterMark(320, 240).decorate(frame)
    assert frame[120, 160].any()


def test_zebra_threshold():
    from chandelier.ui.decorators import ZebraOverlay

    bright = np.full((240, 320, 3), 250, dtype=np.uint8)
    ZebraOverlay(threshold=235).decorate(bright)
    assert (bright[..., 2] == 255).any()

    dark = np.full((240, 320, 3), 100, dtype=np.uint8)
    ZebraOverlay(threshold=235).decorate(dark)
    assert not (dark[..., 2] == 255).any()


def test_status_bar():
    from chandelier.ui.decorators import StatusBar

    frame = _frame()
    bar = StatusBar(320, 240)
    bar.set_items(["SGL", "1/125", "G1.0"])
    bar.decorate(frame)
    assert frame[:18, :].any()


def test_status_bar_roles_and_bottom_position():
    from chandelier.ui.decorators import StatusBar

    frame = _frame()
    bar = StatusBar(320, 240, position="bottom")
    bar.set_items(
        [
            ("REC", "danger"),
            ("1/125", "text"),
            ("G1.0", "dim"),
            ("HDR2", "warning"),
        ]
    )
    bar.decorate(frame, theme_id="dark_pro")
    assert bar.items == ["REC", "1/125", "G1.0", "HDR2"]
    assert frame[240 - 24 :, :].any()


def test_status_bar_drops_overflowing_items():
    from chandelier.ui.decorators import StatusBar

    frame = _frame()
    bar = StatusBar(320, 240, font_height=12)
    bar.set_items([f"ITEM{index}-LONG" for index in range(12)])
    bar.decorate(frame)
    assert frame.any()


def test_countdown_and_recording_badge():
    from chandelier.ui.decorators import Countdown, RecordingBadge

    frame = _frame()
    Countdown(320, 240).decorate(frame, 3)
    assert frame.any()

    frame = _frame()
    RecordingBadge(320, 240).decorate(frame, 12.0)
    assert frame.any()


def test_player_hud():
    from chandelier.ui.decorators import PlayerHud, format_time

    assert format_time(0) == "0:00"
    assert format_time(65) == "1:05"
    assert format_time(3661) == "1:01:01"

    frame = _frame()
    hud = PlayerHud(320, 240)
    hud.set_state(30.0, 120.0, True, True)
    hud.decorate(frame)
    assert hud.visible is True
    assert frame[240 - 24 :, :].any()


def test_status_bar_keeps_typical_camera_items():
    from chandelier.ui.decorators import StatusBar

    bar = StatusBar(320, 240, font_height=10)
    bar.set_items(
        ["SGL", "AF-C", "1/100", "G1.5", "5000K", "CPU85%", "T65C",
         "30fps", "Z1.0"]
    )
    assert len(bar._layout(320)) == 9


def test_status_bar_sides_render_in_two_columns():
    from chandelier.ui.decorators import StatusBar

    frame = _frame()
    bar = StatusBar(320, 240, font_height=10)
    bar.set_position("sides")
    bar.set_items(
        [
            ("SGL", "success"),
            ("AF-C", "success"),
            ("1/100", "text"),
            ("G1.5", "text"),
            ("CPU23%", "success"),
            ("T45C", "success"),
            ("30fps", "text"),
            ("Z1.0", "text"),
        ]
    )
    bar.decorate(frame, theme_id="dark_pro")

    assert frame[:, :40].any()       # left column
    assert frame[:, -40:].any()      # right column
    assert not frame[120, 160].any()  # centre stays untouched


def test_status_bar_position_falls_back_to_top():
    from chandelier.ui.decorators import StatusBar

    bar = StatusBar(320, 240)
    bar.set_position("invalid")
    bar.set_items(["SGL"])
    frame = _frame()
    bar.decorate(frame, theme_id="dark_pro")
    assert frame[:18, :].any()
