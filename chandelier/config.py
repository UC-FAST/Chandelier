"""Application configuration.

Two kinds of configuration exist:

* ``config/config.json`` - hardware, screen, camera, sensor and localizer
  settings.  It is parsed into frozen dataclasses.
* ``config/menus.json`` - the menu tree.  It intentionally stays a raw dict:
  the menu edits it in place, compares option ``value`` dicts by equality and
  writes the same structure back to disk.

All relative paths are resolved against the project root, never the process
working directory.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from chandelier.constants import EndId, MenuItemType, MenuOptionId

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"
DEFAULT_MENUS_PATH = PROJECT_ROOT / "config" / "menus.json"

_KNOWN_END_IDS = {item.value for item in EndId}
_KNOWN_ITEM_TYPES = {item.value for item in MenuItemType}


class ConfigError(ValueError):
    """Raised when config.json is missing or malformed."""


class MenuConfigError(ValueError):
    """Raised when menus.json fails validation."""


@dataclass(frozen=True)
class PinConfig:
    """BCM pin numbers for buttons, encoder, shutter and display control."""

    right: int
    up: int
    left: int
    down: int
    center: int
    rotary_encoder_1a: int
    rotary_encoder_1b: int
    rotary_encoder_1s: int
    shutter: int
    cross: int
    square: int
    circle: int
    rst: int
    dc: int
    bl: int


@dataclass(frozen=True)
class CameraConfig:
    """Camera capture paths, bitrate, tuning file and preview size."""

    path: Path
    video_path: Path
    video_bitrate: int
    tuning_file: Path | None
    preview_width: int
    preview_height: int


@dataclass(frozen=True)
class ScreenConfig:
    """Display dimensions in pixels."""

    width: int
    height: int


@dataclass(frozen=True)
class LedConfig:
    """GPIO pins driving the green and blue status LEDs."""

    green: int
    blue: int


@dataclass(frozen=True)
class I2CBusConfig:
    """I2C bus number for a sensor."""

    bus: int


@dataclass(frozen=True)
class Ina230Config:
    """INA230 bus plus shunt and expected-current calibration values."""

    bus: int
    shunt_resistance: float
    maximum_expected_current: float


@dataclass(frozen=True)
class SensorConfig:
    """I2C configuration for the RTC, power monitor and fuel gauge."""

    bq32002: I2CBusConfig
    ina230: Ina230Config
    max17048: I2CBusConfig


@dataclass(frozen=True)
class LocalizerConfig:
    """Sound-localiser input and marker-mapping parameters.

    Defaults target the six-channel microphone array; the ``marker_*``
    fields project a detected direction onto the camera view.  Most values
    also have menu equivalents in ``menus.json``.
    """

    device: int | None = 0
    rate: int = 16000
    chunk: int = 1600
    channels: int = 6
    ignored_channels: tuple[int, ...] = (0, 5)
    n_sources: int = 3
    confidence_threshold: float = 0.03
    max_to_cumulate: int = 5
    speech_threshold: float = 0.0
    silero_model_dir: Path | None = None
    delay_max: float = 0.0
    n_delay_bins: int = 50
    mic_rotation: float = 0.0
    marker_fov: float = 83.0
    marker_azimuth_offset: float = 0.0
    marker_elevation_offset: float = 0.0
    # Integer screen-space gains; zero disables the correction.
    marker_x_gain: float = 0.0
    marker_y_gain: float = 0.0
    # Array centre relative to the camera (x right, y up, z forward), metres.
    marker_mic_offset_x: float = 0.0
    marker_mic_offset_y: float = 0.0
    marker_mic_offset_z: float = 0.0
    # Distance from the camera to the sound source, metres.  Zero disables
    # the parallax correction.
    marker_source_distance: float = 0.0


@dataclass(frozen=True)
class MarkerConfig:
    """Sound-source marker display options."""

    mode: str = "single"  # "single" or "multiple"
    max_sources: int = 3
    ttl: float = 3.0
    smoothing: int = 5
    trail: int = 3
    size: int = 24


@dataclass(frozen=True)
class PreviewConfig:
    """Web preview server settings.

    ``auth_file`` is an optional JSON file with ``username`` and ``password``
    keys.  It is deliberately separate from ``config.json`` so credentials do
    not have to be committed.
    """

    auth_file: Path


@dataclass(frozen=True)
class AppConfig:
    """Fully parsed ``config.json``; every path is already absolute."""

    pin: PinConfig
    camera: CameraConfig
    screen: ScreenConfig
    led: LedConfig
    sensors: SensorConfig
    localizer: LocalizerConfig
    marker: MarkerConfig
    preview: PreviewConfig
    debug_level: int
    internet_test_url: str
    ntp_server: str
    menus_file: Path

    @property
    def frame_size(self) -> tuple[int, int]:
        return self.screen.width, self.screen.height


def resolve_path(value: str | os.PathLike[str], base: Path = PROJECT_ROOT) -> Path:
    """Resolve a config path, joining relative values to ``base``.

    Resolving against the project root rather than the process working
    directory keeps behaviour identical for systemd and manual launches.
    """

    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def _as_int(mapping: dict[str, Any], key: str, where: str) -> int:
    """Read ``key`` as an int, adding the section name to the error."""

    try:
        return int(mapping[key])
    except KeyError as exc:
        raise ConfigError(f"missing '{key}' in {where}") from exc
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"'{key}' in {where} must be an integer") from exc


def _as_float(mapping: dict[str, Any], key: str, where: str) -> float:
    """Read ``key`` as a float, adding the section name to the error."""

    try:
        return float(mapping[key])
    except KeyError as exc:
        raise ConfigError(f"missing '{key}' in {where}") from exc
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"'{key}' in {where} must be a number") from exc


def _reject_legacy_pin_keys(pin: dict[str, Any]) -> None:
    """Fail loudly on the old camelCase encoder keys.

    Silently ignoring them would leave the encoder pins at zero and fail
    later in a less obvious place, so demand an explicit rename instead.
    """

    legacy = {"rotaryEncoder1A", "rotaryEncoder1B", "rotaryEncoder1S"}
    found = legacy.intersection(pin)
    if found:
        raise ConfigError(
            "config.json still uses legacy pin keys "
            f"{sorted(found)}; rename them to snake_case "
            "(rotary_encoder_1a, rotary_encoder_1b, rotary_encoder_1s)."
        )


def _parse_config(raw: dict[str, Any]) -> AppConfig:
    """Validate a raw ``config.json`` dict and build an :class:`AppConfig`.

    Raises :class:`ConfigError` with a human-readable message on bad types,
    missing sections or out-of-range values.
    """

    try:
        pin_raw = raw["pin"]
        camera_raw = raw["camera"]
        screen_raw = raw["screen"]
        led_raw = raw["led"]
        sensor_raw = raw["sensor"]
    except KeyError as exc:
        raise ConfigError(f"missing top-level section '{exc.args[0]}'") from exc

    _reject_legacy_pin_keys(pin_raw)
    pin = PinConfig(
        right=_as_int(pin_raw, "right", "pin"),
        up=_as_int(pin_raw, "up", "pin"),
        left=_as_int(pin_raw, "left", "pin"),
        down=_as_int(pin_raw, "down", "pin"),
        center=_as_int(pin_raw, "center", "pin"),
        rotary_encoder_1a=_as_int(pin_raw, "rotary_encoder_1a", "pin"),
        rotary_encoder_1b=_as_int(pin_raw, "rotary_encoder_1b", "pin"),
        rotary_encoder_1s=_as_int(pin_raw, "rotary_encoder_1s", "pin"),
        shutter=_as_int(pin_raw, "shutter", "pin"),
        cross=_as_int(pin_raw, "cross", "pin"),
        square=_as_int(pin_raw, "square", "pin"),
        circle=_as_int(pin_raw, "circle", "pin"),
        rst=_as_int(pin_raw, "rst", "pin"),
        dc=_as_int(pin_raw, "dc", "pin"),
        bl=_as_int(pin_raw, "bl", "pin"),
    )

    # Optional keys fall back to the layout shipped with the project; the
    # path defaults are relative and resolved against PROJECT_ROOT.
    tuning = camera_raw.get("tuning_file")
    camera = CameraConfig(
        path=resolve_path(camera_raw.get("path", "pict")),
        video_path=resolve_path(camera_raw.get("video_path", "pict/video")),
        video_bitrate=int(camera_raw.get("video_bitrate", 1_000_000)),
        tuning_file=resolve_path(tuning) if tuning else None,
        preview_width=int(camera_raw.get("preview_width", 640)),
        preview_height=int(camera_raw.get("preview_height", 480)),
    )
    screen = ScreenConfig(
        width=_as_int(screen_raw, "width", "screen"),
        height=_as_int(screen_raw, "height", "screen"),
    )
    led = LedConfig(
        green=_as_int(led_raw, "green", "led"),
        blue=_as_int(led_raw, "blue", "led"),
    )

    bq_raw = sensor_raw.get("bq32002", {})
    ina_raw = sensor_raw.get("ina230", {})
    max_raw = sensor_raw.get("max17048", {})
    sensors = SensorConfig(
        bq32002=I2CBusConfig(bus=int(bq_raw.get("bus", 1))),
        ina230=Ina230Config(
            bus=int(ina_raw.get("bus", 1)),
            shunt_resistance=float(ina_raw.get("shunt_resistance", 0.002)),
            maximum_expected_current=float(
                ina_raw.get("maximum_expected_current", 3.2)
            ),
        ),
        max17048=I2CBusConfig(bus=int(max_raw.get("bus", 1))),
    )

    localizer_raw = raw.get("localizer", {})
    silero_dir = localizer_raw.get("silero_model_dir")
    marker_fov = float(localizer_raw.get("marker_fov", 83.0))
    # 0 or 180 degrees would make the marker projection singular.
    if not 0.0 < marker_fov < 180.0:
        raise ConfigError(
            "localizer.marker_fov must be between 0 and 180 degrees, "
            f"got {localizer_raw.get('marker_fov')!r}"
        )
    # Wrap the azimuth so menu edits may use either -N or 360+N conventions.
    marker_azimuth_offset = float(
        localizer_raw.get("marker_azimuth_offset", 0.0)
    ) % 360.0
    marker_elevation_offset = float(
        localizer_raw.get("marker_elevation_offset", 0.0)
    )
    marker_x_gain = float(localizer_raw.get("marker_x_gain", 0.0))
    marker_y_gain = float(localizer_raw.get("marker_y_gain", 0.0))
    marker_mic_offset = tuple(
        float(localizer_raw.get(f"marker_mic_offset_{axis}", 0.0))
        for axis in ("x", "y", "z")
    )
    marker_source_distance = float(
        localizer_raw.get("marker_source_distance", 0.0)
    )
    if marker_source_distance < 0.0:
        raise ConfigError(
            "localizer.marker_source_distance must not be negative, "
            f"got {localizer_raw.get('marker_source_distance')!r}"
        )
    localizer = LocalizerConfig(
        device=localizer_raw.get("device", 0),
        rate=int(localizer_raw.get("rate", 16000)),
        chunk=int(localizer_raw.get("chunk", 1600)),
        channels=int(localizer_raw.get("channels", 6)),
        ignored_channels=tuple(
            int(channel) for channel in localizer_raw.get("ignored_channels", (0, 5))
        ),
        n_sources=int(localizer_raw.get("n_sources", 3)),
        confidence_threshold=float(
            localizer_raw.get("confidence_threshold", 0.03)
        ),
        max_to_cumulate=int(localizer_raw.get("max_to_cumulate", 5)),
        speech_threshold=float(localizer_raw.get("speech_threshold", 0.0)),
        silero_model_dir=resolve_path(silero_dir) if silero_dir else None,
        delay_max=float(localizer_raw.get("delay_max", 0.0)),
        n_delay_bins=int(localizer_raw.get("n_delay_bins", 50)),
        mic_rotation=float(localizer_raw.get("mic_rotation", 0.0)),
        marker_fov=marker_fov,
        marker_azimuth_offset=marker_azimuth_offset,
        marker_elevation_offset=marker_elevation_offset,
        marker_x_gain=marker_x_gain,
        marker_y_gain=marker_y_gain,
        marker_mic_offset_x=marker_mic_offset[0],
        marker_mic_offset_y=marker_mic_offset[1],
        marker_mic_offset_z=marker_mic_offset[2],
        marker_source_distance=marker_source_distance,
    )

    marker_raw = raw.get("marker", {})
    marker_mode = str(marker_raw.get("mode", "single")).lower()
    if marker_mode not in {"single", "multiple"}:
        raise ConfigError(
            "marker.mode must be 'single' or 'multiple', "
            f"got {marker_raw.get('mode')!r}"
        )
    # Marker tuning is clamped rather than rejected: menu-driven edits should
    # never make the screen unusable, even with a stale config file.
    marker = MarkerConfig(
        mode=marker_mode,
        max_sources=max(1, min(8, int(marker_raw.get("max_sources", 3)))),
        ttl=max(0.0, float(marker_raw.get("ttl", 3.0))),
        smoothing=max(1, int(marker_raw.get("smoothing", 5))),
        trail=max(0, int(marker_raw.get("trail", 3))),
        size=max(4, int(marker_raw.get("size", 24))),
    )

    preview_raw = raw.get("preview", {})
    if not isinstance(preview_raw, dict):
        raise ConfigError("'preview' must be an object")
    preview = PreviewConfig(
        auth_file=resolve_path(
            preview_raw.get("auth_file", "config/preview_auth.json")
        )
    )

    return AppConfig(
        pin=pin,
        camera=camera,
        screen=screen,
        led=led,
        sensors=sensors,
        localizer=localizer,
        marker=marker,
        preview=preview,
        debug_level=int(raw.get("debug_level", 10)),
        internet_test_url=str(
            raw.get(
                "internet_test_url",
                "http://www.msftconnecttest.com/connecttest.txt",
            )
        ),
        ntp_server=str(raw.get("ntp_server", "ntp1.aliyun.com")),
        menus_file=resolve_path(raw.get("menus_file", "config/menus.json")),
    )


@lru_cache(maxsize=16)
def _load_config_cached(path: str, _mtime_ns: int) -> AppConfig:
    """Parse once per (path, mtime) pair.

    ``mtime_ns`` is part of the cache key even though it is unused below:
    rewriting the file gives a new key, so the cache invalidates without
    callers having to remember to clear it.
    """

    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a JSON object")
    return _parse_config(raw)


def load_config(path: str | os.PathLike[str] | None = None) -> AppConfig:
    """Load ``config.json``.  Results are cached per resolved path+mtime.

    Relative ``path`` values resolve against the project root; the default
    is ``config/config.json`` next to the ``chandelier`` package.
    """

    resolved = resolve_path(path) if path is not None else DEFAULT_CONFIG_PATH
    try:
        mtime_ns = resolved.stat().st_mtime_ns
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration file not found: {resolved}") from exc
    return _load_config_cached(str(resolved), mtime_ns)


def clear_config_cache() -> None:
    """Drop cached configs, e.g. after a test rewrites the same file."""

    _load_config_cached.cache_clear()


def normalize_menu_tree(tree: dict[str, Any]) -> dict[str, Any]:
    """Lower-case option types in place and return the tree.

    The tree is nested as ``{end_id: {section_id: section}}``.  ``"type":
    "Menu"`` used to exist in the shipped file; callers lower-cased the value
    on every comparison.  Normalising once at load makes the rest of the code
    simpler and keeps the on-disk round-trip consistent.
    """

    if not isinstance(tree, dict):
        raise MenuConfigError("menu file must contain a JSON object")
    for end_name, sections in tree.items():
        if not isinstance(sections, dict):
            raise MenuConfigError(f"menu group '{end_name}' must be an object")
        for section_name, section in sections.items():
            if not isinstance(section, dict):
                raise MenuConfigError(
                    f"menu section '{end_name}.{section_name}' must be an object"
                )
            for option in section.get("options", []):
                if not isinstance(option, dict):
                    raise MenuConfigError(
                        f"option in section '{end_name}.{section_name}' "
                        "must be an object"
                    )
                if "type" in option:
                    option["type"] = str(option["type"]).lower()
    return tree


def _option_ids(section: dict[str, Any]) -> set[str]:
    return {
        option["id"]
        for option in section.get("options", [])
        if isinstance(option, dict) and isinstance(option.get("id"), str)
    }


def _is_number(value: Any) -> bool:
    # bool is an int subclass, but JSON ``true`` is not a menu number.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_menu_config(tree: dict[str, Any]) -> list[str]:
    """Return a list of human-readable validation errors (empty when valid).

    The expected shape is ``{end_id: {section_id: section}}``; every end tree
    must contain a root section ``"0"``.  Menu targets refer to section ids
    inside the same end tree; IRQ and message receivers refer to top-level
    end ids.
    """

    errors: list[str] = []
    if not isinstance(tree, dict):
        return ["menu configuration must be a JSON object"]

    # Camera and gallery are the minimum usable UI; the other end trees are
    # optional so a reduced menu file still validates.
    required_groups = {EndId.CAMERA.value, EndId.GALLERY.value}
    missing = required_groups.difference(tree)
    if missing:
        errors.append(f"missing menu group(s): {', '.join(sorted(missing))}")

    for end_name, sections in tree.items():
        where_group = end_name
        if end_name not in _KNOWN_END_IDS:
            errors.append(f"unknown menu group '{end_name}'")
        if not isinstance(sections, dict):
            errors.append(f"menu group '{where_group}' must be an object")
            continue
        if "0" not in sections:
            errors.append(f"menu group '{where_group}' has no root section '0'")
            continue

        for section_name, section in sections.items():
            where_section = f"{where_group}.{section_name}"
            if not isinstance(section, dict) or "options" not in section:
                errors.append(
                    f"section '{where_section}' has no options list"
                )
                continue

            options = section["options"]
            if not isinstance(options, list):
                errors.append(
                    f"section '{where_section}' options must be a list"
                )
                continue

            ids = [
                option.get("id")
                for option in options
                if isinstance(option, dict)
                and isinstance(option.get("id"), str)
            ]
            # Deduplicate so one repeated id produces one error, not N.
            duplicates = sorted({item for item in ids if ids.count(item) > 1})
            if duplicates:
                errors.append(
                    f"section '{where_section}' has duplicate option ids: "
                    f"{duplicates}"
                )
            known_ids = set(ids)

            for index, option in enumerate(options):
                where = f"{where_section}[{index}]"
                if not isinstance(option, dict):
                    errors.append(f"{where} must be an object")
                    continue

                if not isinstance(option.get("content"), str):
                    errors.append(f"{where} is missing a string content label")

                item_type = str(option.get("type", "")).lower()
                if item_type not in _KNOWN_ITEM_TYPES:
                    errors.append(
                        f"{where} has unknown type '{option.get('type')}'"
                    )
                    continue
                option["type"] = item_type

                if item_type == MenuItemType.OPTION.value:
                    choices = option.get("options")
                    value = option.get("value")
                    if not isinstance(choices, list) or not choices:
                        errors.append(
                            f"{where} option type needs a non-empty options list"
                        )
                    elif not (
                        isinstance(value, dict)
                        and "content" in value
                        and "value" in value
                    ):
                        errors.append(
                            f"{where} option value must be {{content, value}}"
                        )
                    else:
                        actual = value.get("value")
                        if not any(
                            isinstance(choice, dict)
                            and choice.get("value") == actual
                            for choice in choices
                        ):
                            errors.append(
                                f"{where} value {actual!r} is not one of the "
                                "declared options"
                            )
                elif item_type == MenuItemType.NUMERAL.value:
                    minimum = option.get("min")
                    maximum = option.get("max")
                    step = option.get("step")
                    value = option.get("value")
                    if not all(
                        _is_number(item) for item in (minimum, maximum, step)
                    ):
                        errors.append(
                            f"{where} numeral needs numeric min/max/step"
                        )
                    elif step <= 0:
                        errors.append(f"{where} numeral step must be positive")
                    elif minimum > maximum:
                        errors.append(
                            f"{where} numeral min must not exceed max"
                        )
                    elif not _is_number(value) or not minimum <= value <= maximum:
                        errors.append(
                            f"{where} numeral value must be within "
                            f"[{minimum}, {maximum}]"
                        )
                    step_options = option.get("step_options")
                    if step_options is not None and (
                        not isinstance(step_options, list)
                        or not step_options
                        or not all(_is_number(item) for item in step_options)
                    ):
                        errors.append(
                            f"{where}.step_options must be a non-empty "
                            "numeric list"
                        )

                receiver = option.get("receiver")
                if receiver is not None and receiver not in _KNOWN_END_IDS:
                    errors.append(
                        f"{where} has unknown receiver '{receiver}'"
                    )

                # Menu targets name a section in the same end tree; irq and
                # msg targets name top-level end ids instead.
                if item_type == MenuItemType.MENU.value:
                    target = option.get("value")
                    if target not in sections:
                        errors.append(
                            f"{where} points to missing menu '{target}'"
                        )
                elif item_type == MenuItemType.IRQ.value:
                    target = option.get("value")
                    if target not in _KNOWN_END_IDS:
                        errors.append(
                            f"{where} has unknown irq target '{target}'"
                        )
                elif item_type == MenuItemType.MSG.value:
                    if receiver is None:
                        errors.append(
                            f"{where} is a msg option without receiver"
                        )
                    elif receiver not in _KNOWN_END_IDS:
                        errors.append(
                            f"{where} has unknown receiver '{receiver}'"
                        )

                # Dependencies may only reference ids declared in this same
                # section, which is what the menu lookups assume.
                for dependency_key in (
                    "set_disable",
                    "set_enable",
                    "enable_with",
                ):
                    dependency_ids = option.get(dependency_key, [])
                    if not isinstance(dependency_ids, list):
                        errors.append(
                            f"{where}.{dependency_key} must be a list"
                        )
                        continue
                    for dependency_id in dependency_ids:
                        if dependency_id not in known_ids:
                            errors.append(
                                f"{where}.{dependency_key} references "
                                f"unknown option '{dependency_id}'"
                            )

                option_id = option.get("id")
                if option_id in {
                    MenuOptionId.RESOLUTION.value,
                    MenuOptionId.RECORD_RESOLUTION.value,
                }:
                    value = option.get("value", {})
                    actual = (
                        value.get("value") if isinstance(value, dict) else value
                    )
                    if not (
                        isinstance(actual, list)
                        and len(actual) == 2
                        and all(isinstance(item, int) for item in actual)
                    ):
                        errors.append(
                            f"{where} ({option_id}) must be [width, height] "
                            "integers"
                        )
                elif option_id == MenuOptionId.PICT_FORMAT.value:
                    value = option.get("value", {})
                    actual = (
                        value.get("value") if isinstance(value, dict) else value
                    )
                    if not isinstance(actual, str):
                        errors.append(
                            f"{where} (pict_format) must be a string"
                        )
                elif option_id == MenuOptionId.THEME.value:
                    value = option.get("value", {})
                    actual = (
                        value.get("value") if isinstance(value, dict) else value
                    )
                    if not isinstance(actual, str) or not actual:
                        errors.append(f"{where} (theme) must be a string")

    return errors


def load_menu_config(
    path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Load and validate ``menus.json``; return the raw mutable dict."""

    resolved = resolve_path(path) if path is not None else DEFAULT_MENUS_PATH
    try:
        with open(resolved, encoding="utf-8") as handle:
            tree = json.load(handle)
    except FileNotFoundError as exc:
        raise MenuConfigError(f"menu configuration not found: {resolved}") from exc
    except json.JSONDecodeError as exc:
        raise MenuConfigError(f"invalid JSON in {resolved}: {exc}") from exc

    tree = normalize_menu_tree(tree)
    errors = validate_menu_config(tree)
    if errors:
        raise MenuConfigError(
            f"invalid menu configuration {resolved}:\n- " + "\n- ".join(errors)
        )
    return tree


def save_menu_config(tree: dict[str, Any], path: str | os.PathLike[str]) -> None:
    """Atomically write the menu tree back to disk."""

    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    # Write a sibling temp file so os.replace is atomic on the same
    # filesystem; a crash cannot leave a truncated menus.json behind.
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(tree, handle, indent=4, ensure_ascii=False)
            handle.write("\n")
        os.replace(temp_name, resolved)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def iter_options(tree: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield every option dict in every end group and section."""

    for sections in tree.values():
        if not isinstance(sections, dict):
            continue
        for section in sections.values():
            if isinstance(section, dict):
                yield from section.get("options", [])
