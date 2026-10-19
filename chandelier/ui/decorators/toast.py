"""Bottom-centre transient message.

A toast is shown for a configurable duration instead of a single frame.  The
background is sized from the real text metrics (not ``len(text) * font``), the
text is truncated when it would not fit, and drawing is anti-aliased and
slightly translucent so it remains readable over live camera frames.
"""

from __future__ import annotations

import time
from typing import Callable

import cv2
import numpy as np

from chandelier.i18n import tr
from chandelier.ui.text_render import get_renderer

_TEXT_COLOR = (255, 255, 255)
_BACKGROUND = (40, 40, 40)
_ACCENT = (255, 0, 0)


class Toast:
    """Transient message anchored to the bottom centre of the frame.

    State is just the current text and an expiry timestamp; the clock is
    injectable so tests can advance time without sleeping.  Drawing goes to
    a scratch buffer and is composited through a non-zero mask, which keeps
    the black background transparent and allows the fade-out at the end of
    the duration.
    """

    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        font_height: int = 12,
        duration: float = 1.2,
        padding: tuple[int, int] = (8, 4),
        margin: int | None = None,
        alpha: float = 0.85,
        thickness: int = 1,
        clock: Callable[[], float] | None = None,
    ):
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if font_height <= 0:
            raise ValueError("font_height must be positive")

        self._width = width
        self._height = height
        self._font_height = font_height
        self._renderer = get_renderer()
        self._duration = max(0.0, float(duration))
        if isinstance(padding, int):
            padding = (padding, padding)
        self._padding = (
            max(0, int(padding[0])),
            max(0, int(padding[1])),
        )
        self._margin = (
            max(0, int(margin))
            if margin is not None
            else max(4, height // 24)
        )
        self._alpha = max(0.0, min(1.0, float(alpha)))
        self._thickness = max(1, int(thickness))
        self._clock = clock or time.monotonic

        self._text: str | None = None
        self._expires_at = 0.0
        self._is_update = False

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    @property
    def text(self) -> str | None:
        return self._text

    @property
    def is_update(self) -> bool:
        return self._is_update

    @property
    def visible(self) -> bool:
        return self._visible_at(self._clock())

    def _visible_at(self, now: float) -> bool:
        """Check visibility against an explicit timestamp.

        Kept separate from the ``visible`` property so ``decorate`` can
        evaluate the clock only once per frame.
        """

        if not self._text:
            return False
        if self._expires_at <= 0:
            # Zero means "sticky" (duration 0 in set_text).
            return True
        return now < self._expires_at

    def set_text(self, text: str, duration: float | None = None) -> None:
        """Show ``text`` and (re)start its display timer.

        ``duration=None`` uses the toast's default; ``duration=0`` keeps the
        message visible until it is replaced or cleared.
        """

        text = "" if text is None else str(text).strip()
        if not text:
            self.clear()
            return

        self._text = text
        if duration is None:
            duration = self._duration
        duration = max(0.0, float(duration))
        self._expires_at = (
            self._clock() + duration if duration > 0 else 0.0
        )
        self._is_update = True

    def clear(self) -> None:
        self._text = None
        self._expires_at = 0.0
        self._is_update = False

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def _text_width(self, text: str) -> int:
        return self._renderer.measure(tr(text), self._font_height)[0]

    def _fit_text(self, text: str, max_width: int) -> str:
        """Translate and truncate ``text`` to fit ``max_width`` pixels."""

        text = tr(text)
        if max_width <= 0 or not text:
            return ""
        if self._text_width(text) <= max_width:
            return text

        # Shorten one character at a time; proportional glyph widths make a
        # character-count estimate unreliable.
        suffix = ".."
        while text:
            text = text[:-1]
            if self._text_width(text + suffix) <= max_width:
                return text + suffix
        return ""

    def decorate(self, frame: np.ndarray, rotate: int = 0) -> None:
        """Draw the toast if it is still visible.

        ``rotate`` follows the preview rotation; the scratch buffer is
        rotated rather than the frame, and the result is discarded if the
        rotated resolution does not match.
        """

        now = self._clock()
        self._is_update = False
        if not self._visible_at(now):
            return

        frame_height, frame_width = frame.shape[:2]
        pad_x, pad_y = self._padding
        max_text_width = max(
            1, frame_width - 2 * (pad_x + self._margin)
        )
        text = self._fit_text(self._text or "", max_text_width)
        if not text:
            return

        text_width, text_height = self._renderer.measure(
            text, self._font_height
        )

        box_width = text_width + 2 * pad_x
        box_height = text_height + 2 * pad_y
        x0 = max(0, (frame_width - box_width) // 2)
        x1 = min(frame_width - 1, x0 + box_width)
        y1 = max(
            0,
            min(frame_height - 1, frame_height - self._margin),
        )
        y0 = max(0, y1 - box_height)
        baseline = min(
            frame_height - 1, y0 + pad_y + text_height
        )
        text_x = max(0, (frame_width - text_width) // 2)

        # Render into a black scratch buffer so the finished box can be
        # composited through a mask; drawing directly would paint a solid
        # rectangle over the preview and defeat the translucency.
        sketch = np.zeros_like(frame)
        cv2.rectangle(
            sketch, (x0, y0), (x1, y1), _BACKGROUND, -1, cv2.LINE_AA
        )
        cv2.rectangle(
            sketch,
            (x0, y0),
            (x1, y1),
            _ACCENT,
            self._thickness,
            cv2.LINE_AA,
        )
        cv2.line(
            sketch,
            (x0 + 1, y0 + 1),
            (x0 + 1, y1 - 1),
            _ACCENT,
            2,
            cv2.LINE_AA,
        )
        self._renderer.draw(
            sketch,
            text,
            (text_x, baseline),
            self._font_height,
            _TEXT_COLOR,
            self._thickness,
        )

        if rotate:
            rotated = np.rot90(sketch, -rotate // 90)
            if rotated.shape[:2] == frame.shape[:2]:
                sketch = rotated

        alpha = self._alpha
        if self._expires_at > 0:
            # Linear fade over the final 250 ms; the clamp prevents a
            # negative multiplier if the frame arrives slightly late.
            remaining = self._expires_at - now
            alpha *= max(0.0, min(1.0, remaining / 0.25))

        # Non-black pixels are exactly the toast; masking avoids tinting
        # the rest of the scene when ``alpha`` is applied.
        mask = np.any(sketch != 0, axis=2)
        if not mask.any():
            return
        if alpha >= 1.0:
            frame[mask] = sketch[mask]
        else:
            blended = cv2.addWeighted(
                sketch, alpha, frame, 1.0 - alpha, 0
            )
            frame[mask] = blended[mask]
