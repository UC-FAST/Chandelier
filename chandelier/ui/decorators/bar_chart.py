"""Scrolling bar chart overlay."""

from __future__ import annotations

from math import ceil, inf

import cv2
import numpy as np


class BarChart:
    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        max_size: int = 320,
        scale: float = 0,
        color: tuple = (0, 0, 255),
        thickness: int = 1,
        fill: bool = False,
        alpha: float = 1.0,
    ):
        self._step = max(1, ceil(width / max_size))
        self._max_size = max_size
        self._height = height
        self._width = width
        self._scale = scale
        self._color = color
        self._thickness = thickness
        self._fill = fill
        self._alpha = max(0.0, min(1.0, alpha))
        self._max = -inf
        self._min = inf
        self._data_list: list[float] = []

    @property
    def data_list(self) -> list[float]:
        return self._data_list

    @data_list.setter
    def data_list(self, values: list[float]) -> None:
        if len(values) > self._max_size:
            step = max(1, len(values) // self._max_size)
            self._data_list = [
                sum(values[i : i + step]) / step
                for i in range(0, len(values), step)
            ][: self._max_size]
        else:
            self._data_list = list(values)

        if self._data_list:
            self._min = min(self._data_list)
            self._max = max(self._data_list)

    def add_data(self, value: float) -> "BarChart":
        if len(self._data_list) == self._max_size:
            removed = self._data_list.pop(0)
            if not self._scale:
                if removed == self._max:
                    self._max = max(self._data_list) if self._data_list else -inf
                elif removed == self._min:
                    self._min = min(self._data_list) if self._data_list else inf

        self._data_list.append(value)
        if not self._scale:
            self._max = max(self._max, value)
            self._min = min(self._min, value)
        return self

    def _normalize_value(self, value: float) -> float:
        if self._scale:
            return min(100.0, max(0.0, value / self._scale * 100.0))
        if self._max - self._min < 1e-6:
            return 50.0
        normalized = 10 + 80 * (value - self._min) / (self._max - self._min)
        return min(90.0, max(10.0, normalized))

    def decorate(self, frame: np.ndarray, rotate: int = 0) -> None:
        if not self._data_list:
            return

        overlay = np.zeros_like(frame)
        base_y = min(self._height, frame.shape[0])

        values = np.asarray(self._data_list, dtype=np.float64)
        if self._scale:
            normalized = np.clip(
                values / self._scale * 100.0, 0.0, 100.0
            )
        elif self._max - self._min < 1e-6:
            normalized = np.full(values.shape, 50.0)
        else:
            normalized = np.clip(
                10.0 + 80.0 * (values - self._min) / (self._max - self._min),
                10.0,
                90.0,
            )
        heights = (base_y * normalized / 100.0).astype(np.int32)

        if self._fill:
            # One staircase polygon is much cheaper than hundreds of
            # per-column rectangles at 30 fps.
            count = len(heights)
            xs1 = np.arange(count, dtype=np.int32) * self._step
            xs2 = xs1 + self._step
            tops = base_y - heights
            points = np.empty((2 * count + 2, 2), dtype=np.int32)
            points[0:-2:2, 0] = xs1
            points[1:-2:2, 0] = xs2
            points[0:-2:2, 1] = tops
            points[1:-2:2, 1] = tops
            points[-2] = (xs2[-1], base_y)
            points[-1] = (xs1[0], base_y)
            cv2.fillPoly(overlay, [points], self._color, cv2.LINE_AA)
        else:
            for index, bar_height in enumerate(heights):
                top_y = base_y - int(bar_height)
                x1 = index * self._step
                cv2.rectangle(
                    overlay,
                    (x1, top_y - self._thickness),
                    (x1 + self._step, top_y),
                    self._color,
                    -1,
                )

        if self._alpha < 1.0:
            cv2.convertScaleAbs(overlay, overlay, alpha=self._alpha)

        rotation_count = rotate // 90
        if rotation_count % 4:
            overlay = np.rot90(overlay, -rotation_count)
        cv2.add(frame, overlay, frame)
