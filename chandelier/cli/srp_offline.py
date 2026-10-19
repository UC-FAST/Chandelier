"""Run source localisation on a WAV file."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from chandelier.config import load_config
from chandelier.localizer.pipeline import detect_file


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for offline file localisation."""

    parser = argparse.ArgumentParser(description="Run SRP-PHAT on a WAV file")
    parser.add_argument("--wave", required=True, help="path to a WAV file")
    parser.add_argument(
        "--src", type=int, default=1, help="number of sources to detect"
    )
    parser.add_argument(
        "--ignored-channels",
        type=int,
        nargs="*",
        default=[],
        help="channel indices to discard",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run SRP-PHAT on one WAV file and print azimuth/elevation pairs.

    Microphone geometry, sample rate, and the other pipeline settings come
    from the normal config file; ``--src`` and ``--ignored-channels`` only
    override the localizer for this run.
    """

    args = build_parser().parse_args(argv)
    path = Path(args.wave)
    if not path.is_file():
        raise SystemExit(f"wave file not found: {path}")

    config = load_config()
    # ``replace`` copies the config object, so every setting not named here
    # keeps its persisted value.
    localizer = replace(
        config.localizer,
        n_sources=args.src,
        ignored_channels=tuple(args.ignored_channels),
    )
    detections = detect_file(path, localizer, n_sources=args.src)

    print(f"Find {len(detections)} available sources.")
    for detection in detections:
        print(
            f"azi: {detection.azimuth: 6.1f}, "
            f"ele: {detection.elevation: 6.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
