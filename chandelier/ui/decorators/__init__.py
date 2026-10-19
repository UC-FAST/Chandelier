"""Frame overlay widgets."""

from chandelier.ui.decorators.bar_chart import BarChart
from chandelier.ui.decorators.busy import Busy
from chandelier.ui.decorators.colors import Colors, to_bgr, to_hex
from chandelier.ui.decorators.grid import CenterMark, RuleOfThirds
from chandelier.ui.decorators.histogram import Histogram, corner_padding
from chandelier.ui.decorators.icons import draw_icon
from chandelier.ui.decorators.marker_box import MarkerBox
from chandelier.ui.decorators.player_hud import PlayerHud, format_time
from chandelier.ui.decorators.recording_badge import Countdown, RecordingBadge
from chandelier.ui.decorators.simple_text import SimpleText
from chandelier.ui.decorators.status_bar import StatusBar, StatusItem
from chandelier.ui.decorators.toast import Toast
from chandelier.ui.decorators.warning import Warning
from chandelier.ui.decorators.watermark import Watermark
from chandelier.ui.decorators.zebra import ZebraOverlay

# Public decorator API re-exported for the controller and menu screens.
__all__ = [
    "BarChart",
    "Busy",
    "CenterMark",
    "Colors",
    "Countdown",
    "Histogram",
    "MarkerBox",
    "PlayerHud",
    "RecordingBadge",
    "RuleOfThirds",
    "SimpleText",
    "StatusBar",
    "StatusItem",
    "Toast",
    "Warning",
    "Watermark",
    "ZebraOverlay",
    "draw_icon",
    "format_time",
    "to_bgr",
    "to_hex",
    "corner_padding",
]
