"""RGB/luma histogram overlay.

The widget computes a 256-bin histogram on every frame, resamples it to the
available width and draws the three channels additively so overlaps show the
mixed colour.  Rendering is vectorised (one filled polygon per channel), which
is considerably cheaper on a Raspberry Pi than one rectangle per bin.

Optional constructor arguments control the display: linear/log scaling,
RGB/luma channels, opacity, a dark backdrop, temporal smoothing, grid and
border.  The defaults keep the full-frame layout of the original widget.
"""

from __future__ import annotations

import cv2
import numpy as np

_CHANNEL_COLORS = (
    (255, 0, 0),  # blue (BGR)
    (0, 255, 0),  # green
    (0, 0, 255),  # red
)
_LUMA_COLOR = (255, 255, 255)
_GRID_COLOR = (70, 70, 70)
_BORDER_COLOR = (120, 120, 120)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


class Histogram:
    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        padding: tuple = (0, 0, 0, 0),
        *,
        scale: str = "linear",
        alpha: float = 0.8,
        background: tuple[int, int, int] | None = (0, 0, 0),
        background_alpha: float = 0.25,
        channel_mode: str = "rgb",
        smoothing: float = 0.0,
        grid: bool = False,
        border: bool = True,
    ):
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if isinstance(padding, int):
            padding = (padding,) * 4
        if len(padding) != 4 or any(value < 0 for value in padding):
            raise ValueError(
                "padding must be four non-negative values "
                "(left, top, right, bottom)"
            )

        scale = str(scale).lower()
        if scale not in {"linear", "log"}:
            raise ValueError("scale must be 'linear' or 'log'")
        channel_mode = str(channel_mode).lower()
        if channel_mode not in {"rgb", "luma"}:
            raise ValueError("channel_mode must be 'rgb' or 'luma'")

        self._width = width
        self._height = height
        self._padding = tuple(int(value) for value in padding)
        self._scale = scale
        self._alpha = _clamp(alpha)
        self._background = (
            tuple(background) if background is not None else None
        )
        self._background_alpha = _clamp(background_alpha)
        self._channel_mode = channel_mode
        self._smoothing = _clamp(smoothing, 0.0, 0.99)
        self._grid = bool(grid)
        self._border = bool(border)

        self._data_list: list[np.ndarray] = []
        self._min = 0.0
        self._max = 0.0

    # ------------------------------------------------------------------
    # Histogram data
    # ------------------------------------------------------------------
    @staticmethod
    def _as_bgr(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.shape[2] == 4:
            return frame[:, :, :3]
        return frame

    def _channel_histograms(self, frame: np.ndarray) -> list[np.ndarray]:
        if self._channel_mode == "luma":
            if frame.ndim == 2:
                gray = frame
            else:
                gray = cv2.cvtColor(
                    self._as_bgr(frame), cv2.COLOR_BGR2GRAY
                )
            images = [gray]
            indexes = [0]
        else:
            bgr = self._as_bgr(frame)
            images = [bgr, bgr, bgr]
            indexes = [0, 1, 2]

        return [
            cv2.calcHist(
                [image], [index], None, [256], [0, 256]
            ).flatten().astype(np.float64)
            for image, index in zip(images, indexes)
        ]

    @staticmethod
    def _resample(channel: np.ndarray, size: int) -> np.ndarray:
        """Average the 256 bins into ``size`` buckets covering the full range."""

        if len(channel) <= size:
            return channel
        boundaries = (np.arange(size) * len(channel)) // size
        sums = np.add.reduceat(channel, boundaries)
        counts = np.diff(np.append(boundaries, len(channel)))
        return sums / counts

    def _frame_calc(self, frame: np.ndarray) -> None:
        frame_height, frame_width = frame.shape[:2]
        pad_left, _, pad_right, _ = self._padding
        horizontal = max(1, frame_width - pad_left - pad_right)
        bin_count = max(1, min(256, horizontal))

        resampled = [
            self._resample(channel, bin_count)
            for channel in self._channel_histograms(frame)
        ]

        if (
            self._smoothing > 0
            and self._data_list
            and len(self._data_list) == len(resampled)
            and all(
                previous.shape == current.shape
                for previous, current in zip(self._data_list, resampled)
            )
        ):
            self._data_list = [
                previous * self._smoothing
                + current * (1.0 - self._smoothing)
                for previous, current in zip(self._data_list, resampled)
            ]
        else:
            self._data_list = resampled

        data = np.concatenate(self._data_list) if self._data_list else None
        if data is None or data.size == 0:
            self._min = self._max = 0.0
        else:
            self._min = float(np.min(data))
            self._max = float(np.max(data))

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _draw_background(
        self, frame: np.ndarray, left: int, top: int, right: int, bottom: int
    ) -> None:
        if self._background is None or self._background_alpha <= 0:
            return
        region = frame[top:bottom, left:right]
        if self._background == (0, 0, 0):
            # Blending with black is equivalent to scaling towards zero and
            # avoids a full panel allocation.
            cv2.convertScaleAbs(
                region, region, alpha=1.0 - self._background_alpha
            )
            return
        panel = np.empty_like(region)
        panel[:] = self._background
        cv2.addWeighted(
            panel,
            self._background_alpha,
            region,
            1.0 - self._background_alpha,
            0,
            region,
        )

    @staticmethod
    def _draw_grid(
        frame: np.ndarray, left: int, top: int, right: int, bottom: int
    ) -> None:
        width = right - left
        height = bottom - top
        for fraction in (0.25, 0.5, 0.75):
            x = left + int(width * fraction)
            y = top + int(height * fraction)
            cv2.line(
                frame, (x, top), (x, bottom - 1), _GRID_COLOR, 1, cv2.LINE_AA
            )
            cv2.line(
                frame, (left, y), (right - 1, y), _GRID_COLOR, 1, cv2.LINE_AA
            )

    def _bar_heights(self, plot_height: int) -> list[np.ndarray]:
        if not self._data_list:
            return []

        if self._scale == "log":
            transformed = [np.log1p(channel) for channel in self._data_list]
            low = 0.0
            high = max(float(np.max(channel)) for channel in transformed)
        else:
            transformed = self._data_list
            low = min(float(np.min(channel)) for channel in transformed)
            high = max(float(np.max(channel)) for channel in transformed)

        span = high - low
        if span <= 0:
            return []

        return [
            np.clip(
                (channel - low) / span * plot_height, 0, plot_height
            ).astype(np.int32)
            for channel in transformed
        ]

    def _build_bars(
        self,
        frame: np.ndarray,
        left: int,
        top: int,
        right: int,
        bottom: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        overlay = np.zeros_like(frame)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        plot_width = right - left
        plot_height = bottom - top
        heights = self._bar_heights(plot_height)
        if not heights:
            return overlay, mask

        bin_count = len(heights[0])
        xs = left + (np.arange(bin_count) + 0.5) * plot_width / bin_count
        baseline = bottom - 1
        layer = np.zeros_like(frame)
        for index, channel_heights in enumerate(heights):
            points = np.column_stack(
                (
                    np.concatenate((xs, [right - 1, left])),
                    np.concatenate(
                        (
                            baseline - channel_heights,
                            [baseline, baseline],
                        )
                    ),
                )
            ).astype(np.int32)

            if self._channel_mode == "luma":
                color = _LUMA_COLOR
            else:
                color = _CHANNEL_COLORS[index % len(_CHANNEL_COLORS)]

            layer.fill(0)
            cv2.fillPoly(layer, [points], color, cv2.LINE_AA)
            cv2.fillPoly(mask, [points], 255)
            cv2.add(overlay, layer, overlay)
        return overlay, mask

    def _compose(
        self, frame: np.ndarray, overlay: np.ndarray, mask: np.ndarray
    ) -> None:
        if cv2.countNonZero(mask) == 0:
            return
        if self._alpha >= 1.0:
            cv2.copyTo(overlay, mask, frame)
        else:
            blended = cv2.addWeighted(
                overlay, self._alpha, frame, 1.0 - self._alpha, 0
            )
            cv2.copyTo(blended, mask, frame)

    def decorate(self, frame: np.ndarray, rotate: int = 0) -> None:
        if frame is None or frame.size == 0:
            return
        if frame.ndim not in (2, 3):
            return

        grayscale = frame.ndim == 2
        if grayscale:
            canvas = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            canvas = frame[:, :, :3]
        else:
            canvas = frame

        self._frame_calc(frame)
        frame_height, frame_width = canvas.shape[:2]
        pad_left, pad_top, pad_right, pad_bottom = self._padding

        left = max(0, min(pad_left, frame_width - 1))
        top = max(0, min(pad_top, frame_height - 1))
        right = max(left + 1, min(frame_width, frame_width - pad_right))
        bottom = max(top + 1, min(frame_height, frame_height - pad_bottom))
        if right - left <= 0 or bottom - top <= 0:
            return

        self._draw_background(canvas, left, top, right, bottom)
        if self._grid:
            self._draw_grid(canvas, left, top, right, bottom)

        overlay, mask = self._build_bars(
            canvas, left, top, right, bottom
        )
        if rotate:
            rotation_count = (-rotate // 90) % 4
            rotated = np.rot90(overlay, rotation_count)
            rotated_mask = np.rot90(mask, rotation_count)
            if rotated.shape[:2] == canvas.shape[:2]:
                overlay = rotated
                mask = rotated_mask
        self._compose(canvas, overlay, mask)

        if self._border:
            cv2.rectangle(
                canvas,
                (left, top),
                (right - 1, bottom - 1),
                _BORDER_COLOR,
                1,
                cv2.LINE_AA,
            )

        if grayscale:
            frame[:] = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
