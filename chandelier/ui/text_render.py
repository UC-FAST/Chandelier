"""Unicode text rendering for the LCD UI.

OpenCV's Hershey fonts are ASCII-only, so translated labels (Chinese, Russian)
must go through a TrueType font.  Pillow renders each string into a small
anti-aliased RGBA patch which is alpha-composited onto the BGR frame.  Patches
and font objects are cached; menu screens are redrawn on input rather than at
camera frame rate, so the per-string overhead is not significant.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from math import ceil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


logger = logging.getLogger(__name__)

_CJK_FONTS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
)
_LATIN_FONTS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
)


def _first_existing(candidates: tuple[str, ...]) -> str | None:
    override = os.environ.get("CHANDELIER_FONT")
    if override and Path(override).is_file():
        return override
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    return None


def _is_cjk(character: str) -> bool:
    code = ord(character)
    return (
        0x3400 <= code <= 0x4DBF      # CJK extension A
        or 0x4E00 <= code <= 0x9FFF   # CJK unified ideographs
        or 0xF900 <= code <= 0xFAFF   # CJK compatibility ideographs
        or 0x3000 <= code <= 0x303F   # CJK punctuation
        or 0xFF00 <= code <= 0xFFEF   # full-width forms
    )


def _contains_cjk(text: str) -> bool:
    return any(_is_cjk(character) for character in text)


def _runs(text: str) -> list[tuple[str, bool]]:
    """Split text into ``(run, is_cjk)`` segments for font fallback."""

    runs: list[tuple[str, bool]] = []
    for character in text:
        cjk = _is_cjk(character)
        if runs and runs[-1][1] == cjk:
            runs[-1] = (runs[-1][0] + character, cjk)
        else:
            runs.append((character, cjk))
    return runs


class TextRenderer:
    def __init__(self):
        self._cjk_path = _first_existing(_CJK_FONTS)
        self._latin_path = _first_existing(_LATIN_FONTS)
        if self._cjk_path is None:
            logger.warning(
                "no CJK font found; Chinese/Russian text may not render"
            )

    # ------------------------------------------------------------------
    # Font selection
    # ------------------------------------------------------------------
    def _run_font_path(self, run: str) -> str | None:
        if _contains_cjk(run):
            return self._cjk_path or self._latin_path
        return self._latin_path or self._cjk_path

    @staticmethod
    @lru_cache(maxsize=64)
    def _load_font(path: str, size: int):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            logger.warning("failed to load font %s", path)
            return ImageFont.load_default()

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------
    @lru_cache(maxsize=2048)
    def _measure_cached(
        self, text: str, size: int, path: str | None
    ) -> tuple[int, int]:
        font = (
            self._load_font(path, max(6, int(size)))
            if path is not None
            else ImageFont.load_default()
        )
        advance = max(0, ceil(font.getlength(text)))
        bbox = font.getbbox(text, anchor="ls")
        height = max(0, ceil(bbox[3] - bbox[1]))
        return advance, height

    def measure(self, text: str, size: int) -> tuple[int, int]:
        """Return ``(width, height)`` for text drawn with ``size`` pixels."""

        text = str(text)
        if not text:
            return 0, 0
        width = 0
        height = 0
        for run in _runs(text):
            run_width, run_height = self._measure_cached(
                run[0], int(size), self._run_font_path(run[0])
            )
            width += run_width
            height = max(height, run_height)
        return width, height

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    @lru_cache(maxsize=1024)
    def _patch(
        self,
        text: str,
        size: int,
        color: tuple[int, int, int],
        thickness: int,
        path: str | None,
    ):
        font = (
            self._load_font(path, max(6, int(size)))
            if path is not None
            else ImageFont.load_default()
        )
        bbox = font.getbbox(text, anchor="ls", stroke_width=thickness - 1)
        width = ceil(bbox[2] - bbox[0])
        height = ceil(bbox[3] - bbox[1])
        if width <= 0 or height <= 0:
            return None, bbox

        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        blue, green, red = color
        draw.text(
            (-bbox[0], -bbox[1]),
            text,
            font=font,
            fill=(red, green, blue, 255),
            anchor="ls",
            stroke_width=max(0, thickness - 1),
            stroke_fill=(red, green, blue, 255),
        )
        return np.asarray(image), bbox

    def _composite(
        self, frame: np.ndarray, patch: np.ndarray, x: int, y: int
    ) -> None:
        patch_h, patch_w = patch.shape[:2]
        frame_h, frame_w = frame.shape[:2]

        src_x = max(0, -x)
        src_y = max(0, -y)
        dst_x = max(0, x)
        dst_y = max(0, y)
        width = min(patch_w - src_x, frame_w - dst_x)
        height = min(patch_h - src_y, frame_h - dst_y)
        if width <= 0 or height <= 0:
            return

        source = patch[src_y : src_y + height, src_x : src_x + width]
        region = frame[dst_y : dst_y + height, dst_x : dst_x + width]
        alpha = source[..., 3:4].astype(np.float32) / 255.0
        color_bgr = source[..., :3].astype(np.float32)[..., ::-1]
        region[:] = (
            color_bgr * alpha + region.astype(np.float32) * (1.0 - alpha)
        ).astype(np.uint8)

    def draw(
        self,
        frame: np.ndarray,
        text: str,
        origin: tuple[int, int],
        size: int,
        color: tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        """Draw ``text`` with its baseline at ``origin`` (OpenCV convention).

        Text is split into Latin/Cyrillic and CJK runs so a CJK fallback font
        without Latin glyphs (DroidSansFallback) can be combined with the
        regular Latin font.
        """

        text = str(text)
        if not text or frame is None or frame.size == 0:
            return
        color = tuple(int(value) for value in color)
        thickness = max(1, int(thickness))
        x = int(origin[0])
        baseline = int(origin[1])

        for run in _runs(text):
            path = self._run_font_path(run[0])
            font = (
                self._load_font(path, max(6, int(size)))
                if path is not None
                else ImageFont.load_default()
            )
            patch, bbox = self._patch(
                run[0], int(size), color, thickness, path
            )
            if patch is not None:
                self._composite(
                    frame,
                    patch,
                    x + int(bbox[0]),
                    baseline + int(bbox[1]),
                )
            x += int(round(font.getlength(run[0])))


_renderer: TextRenderer | None = None


def get_renderer() -> TextRenderer:
    global _renderer
    if _renderer is None:
        _renderer = TextRenderer()
    return _renderer
