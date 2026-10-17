"""Shared string constants used by the UI, menu configuration and controller."""

from enum import Enum


class EndId(str, Enum):
    """Identifiers for the controlled ends (screens) of the application."""

    CAMERA = "camera"
    GALLERY = "gallery"
    MENU = "menu"
    SYSTEM_MONITOR = "system_monitor"
    CONTROLLER = "controller"


class MenuItemType(str, Enum):
    """Menu option types.  Values match the ``type`` field in menus.json."""

    BOOL = "bool"
    MENU = "menu"
    IRQ = "irq"
    MSG = "msg"
    OPTION = "option"
    NUMERAL = "numeral"


class MenuOptionId(str, Enum):
    """Well-known option ids shared by the menu and the camera end."""

    CAMERA_SETTINGS = "camera_settings"
    GALLERY = "gallery"
    SYSTEM_MONITOR = "system_monitor"
    EXPOSURE = "exposure"
    HDR = "hdr"
    DELAY = "delay"
    RESOLUTION = "resolution"
    PICT_FORMAT = "pict_format"
    SAVE_METADATA = "save_metadata"
    DNG_ENABLE = "dng_enable"
    WATERMARK = "watermark"
    AUTO_EXPOSE = "auto_expose"
    EXPOSURE_MODE = "exposure_mode"
    CONSTRAINT_MODE = "constraint_mode"
    METERING_MODE = "metering_mode"
    SHOW_HIST = "show_hist"
    FLICKER = "flicker"
    EXPOSURE_TIME = "exposure_time"
    ANALOGUE_GAIN = "analogue_gain"
    FLICKER_MODE = "flicker_mode"
    FLICKER_PERIOD = "flicker_period"
    AWB = "awb"
    AWB_MODE = "awb_mode"
    COLOR_GAINS = "color_gains"
    RED_GAIN = "red_gain"
    BLUE_GAIN = "blue_gain"
    DELETE = "delete"

    # Display / menu theme
    DISPLAY = "display"
    THEME = "theme"
    SHOW_GRID = "show_grid"
    SHOW_CENTER = "show_center"
    SHOW_ZEBRA = "show_zebra"
    ZEBRA_THRESHOLD = "zebra_threshold"
    SHOW_STATUS_BAR = "show_status_bar"

    # Capture modes
    CAPTURE = "capture"
    CAPTURE_MODE = "capture_mode"
    BURST_COUNT = "burst_count"
    BURST_INTERVAL = "burst_interval"
    INTERVAL_COUNT = "interval_count"
    INTERVAL_SECONDS = "interval_seconds"

    # Focus
    FOCUS = "focus"
    AF_MODE = "af_mode"
    AF_TRIGGER = "af_trigger"
    AF_SPEED = "af_speed"
    AF_RANGE = "af_range"
    MANUAL_FOCUS = "manual_focus"
    FOCUS_POSITION = "focus_position"
    MF_ASSIST = "mf_assist"

    # Manual controls
    MANUAL = "manual"
    EXP_COMPENSATION = "exp_compensation"
    AE_LOCK = "ae_lock"
    AWB_LOCK = "awb_lock"
    SATURATION = "saturation"
    CONTRAST = "contrast"
    SHARPNESS = "sharpness"

    # Recording
    RECORD = "record"
    RECORD_MODE = "record_mode"
    RECORD_RESOLUTION = "record_resolution"

    # System
    SYSTEM = "system"
    LANGUAGE = "language"
    REBOOT = "reboot"

    # Sound localizer
    LOCALIZER = "localizer"
    LOC_N_SOURCES = "loc_n_sources"
    LOC_CONFIDENCE = "loc_confidence"
    LOC_MAX_CHUNKS = "loc_max_chunks"
    LOC_SPEECH_THRESHOLD = "loc_speech_threshold"
    LOC_MIC_ROTATION = "loc_mic_rotation"
    LOC_DELAY_MAX = "loc_delay_max"
    LOC_DELAY_BINS = "loc_delay_bins"

    # Gallery player
    PLAYER = "player"
    SHOW_PROGRESS = "show_progress"
    LOOP_PLAYBACK = "loop_playback"
    SEEK_SECONDS = "seek_seconds"


class MenuSectionId(str, Enum):
    """Well-known menu section keys in menus.json."""

    CAMERA_SETTINGS = "camera_settings"
    EXPOSURE = "exposure"
    FLICKER_MENU = "flicker_menu"
    FOCUS = "focus"
    WHITE_BALANCE = "white_balance"
    COLOR_GAINS = "color_gains"
    DISPLAY = "display"
    CAPTURE = "capture"
    MANUAL = "manual"
    RECORD = "record"
    LOCALIZER = "localizer"
    SYSTEM = "system"
    PLAYER = "player"


def end_id(value: EndId | str) -> str:
    """Return the plain string value of an :class:`EndId` (or any string)."""

    return value.value if isinstance(value, EndId) else str(value)
