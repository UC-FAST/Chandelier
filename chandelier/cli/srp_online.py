"""Run source localisation on the live microphone array."""

from __future__ import annotations

import argparse
from dataclasses import replace

from chandelier.config import load_config
from chandelier.hardware.microphone import MicrophoneStream
from chandelier.localizer.pipeline import iter_detections


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for live microphone localisation.

    Every option defaults to ``None`` so an omitted flag keeps the value from
    the config file rather than overwriting it with argparse's zero value.
    """

    parser = argparse.ArgumentParser(
        description="Run SRP-PHAT on the live microphone array"
    )
    parser.add_argument(
        "--src", type=int, default=None, help="number of sources to detect"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--rate", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=None)
    parser.add_argument(
        "--ignored-channels", type=int, nargs="*", default=None
    )
    parser.add_argument("--max-to-cumulate", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Stream microphone audio, print detections until interrupted."""

    args = build_parser().parse_args(argv)
    config = load_config()
    localizer = config.localizer

    # Only explicitly supplied flags override the persisted config; ``None``
    # means "keep the configured value".
    overrides = {}
    if args.src is not None:
        overrides["n_sources"] = args.src
    if args.device is not None:
        overrides["device"] = args.device
    if args.channels is not None:
        overrides["channels"] = args.channels
    if args.rate is not None:
        overrides["rate"] = args.rate
    if args.chunk is not None:
        overrides["chunk"] = args.chunk
    if args.ignored_channels is not None:
        overrides["ignored_channels"] = tuple(args.ignored_channels)
    if args.max_to_cumulate is not None:
        overrides["max_to_cumulate"] = args.max_to_cumulate
    if args.threshold is not None:
        overrides["speech_threshold"] = args.threshold
    localizer = replace(localizer, **overrides)

    # Config stores ignored channels as a tuple; the stream adapter takes a
    # list and does the final device-side filtering.
    stream = MicrophoneStream(
        rate=localizer.rate,
        chunk=localizer.chunk,
        channels=localizer.channels,
        ignored_channels=list(localizer.ignored_channels),
        device=localizer.device,
    )

    try:
        for batch in iter_detections(stream, localizer):
            print(f"Find {len(batch.detections)} available sources.")
            for detection in batch.detections:
                print(
                    f"azi: {detection.azimuth: 6.1f}, "
                    f"ele: {detection.elevation: 6.1f}"
                )
            print("=" * 51)
    except KeyboardInterrupt:
        pass
    finally:
        # Always release the ALSA device, including on Ctrl-C, so the next
        # run (or the UI) can reopen the microphone.
        stream.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
