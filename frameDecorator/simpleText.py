from collections.abc import Iterable
from math import ceil
import cv2
import numpy as np


class SimpleText:
    def __init__(
        self,
        func_list: list,
        height: int,
        padding: tuple = (5, 5, 5, 5),
        font_height: int = 24,
        color: tuple = (255, 255, 255),
        thickness: float = 1
    ):

        if height <= 0:
            raise ValueError("Height must be positive")
        if any(p < 0 for p in padding):
            raise ValueError("Padding values must be non-negative")
        if not func_list:
            raise ValueError("Function list cannot be empty")

        self.__font_height = font_height
        self.__font_size = cv2.getFontScaleFromHeight(
            cv2.FONT_ITALIC, self.__font_height)
        self.__height = height
        self.__padding = padding
        self.__color = color
        self.__thickness = thickness
        self.__func_list = tuple(func_list)
        self.__index = 0

    @property
    def current_page(self):
        return self.__index

    @property
    def total_pages(self):
        return len(self.__func_list)

    def next_page(self):
        self.__index = min(self.__index + 1, self.total_pages - 1)

    def previous_page(self):
        self.__index = max(self.__index - 1, 0)

    def set_page(self, page: int):
        if 0 <= page < self.total_pages:
            self.__index = page
        else:
            raise IndexError(
                f"Page index out of range [0, {self.total_pages - 1}]")

    def decorate(self, frame, rotate=0):
        widget = self.__func_list[self.__index]()
        if not widget:
            return frame

        sketch = np.zeros_like(frame, dtype=np.uint8)
        _, top_padding,  _, bottom_padding = self.__padding
        available_height = self.__height - top_padding - bottom_padding
        text_count = len(widget)

        if text_count == 1:
            step = 0
        else:
            total_text_height = text_count * self.__font_height
            step = ceil((available_height - total_text_height) / (text_count - 1))

        left_padding = self.__padding[0]
        y_pos = top_padding

        for key, value in widget.items():

            if isinstance(value, str):
                text = key.format(value)
            elif isinstance(value, Iterable) and not isinstance(value, str):
                text = key.format(*value)
            else:
                text = key.format(value)

            cv2.putText(
                sketch,
                text,
                (left_padding, y_pos),
                cv2.FONT_ITALIC,
                self.__font_size,
                self.__color,
                self.__thickness,
            )
            y_pos += self.__font_height + step

        if rotate != 0:
            rotate_times = (-rotate // 90) % 4
            if rotate_times:
                sketch = np.rot90(sketch, rotate_times)

        cv2.addWeighted(sketch, 1, frame, 1, 0, frame)
        return frame
