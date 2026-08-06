from math import ceil

import cv2
import numpy as np


class DialogBox:
    def __init__(
            self,
            width: int,
            height: int,
            options: list|None = None,
            title: str|None = None,
            padding: tuple = (0, 0, 0, 0),
            column_count: int = 1,
            show_index: bool = False,
            font_height: int = 12,
            color: tuple = (255, 255, 255),
            thickness: int = 1
    ):

        self.__options = options
        self.__column_count = column_count
        self.__title = title
        self.__height = height
        self.__width = width
        self.__font_height = font_height
        self.__padding = padding
        if self.__options:
            if show_index:
                self.__option_list = tuple(["{}: {}".format(i + 1, x) for i, x in enumerate(options)])
            else:
                self.__option_list = tuple(options)
            self.__vertical_step = self.__vertical_step_calc()
            self.__horizontal_step = self.__horizontal_step_calc()
        else:
            self.__option_list = None

        self.__show_index = show_index
        self.__font_size = cv2.getFontScaleFromHeight(cv2.FONT_ITALIC, font_height)
        self.__color = color
        self.__thickness = thickness
        self.__index = 0

    @property
    def options(self):
        return self.__options

    @options.setter
    def options(self, options):
        self.__options = options
        if self.__options:
            if self.__show_index:
                self.__option_list = tuple(["{}: {}".format(i + 1, x) for i, x in enumerate(options)])
            else:
                self.__option_list = tuple(options)
            self.__vertical_step = self.__vertical_step_calc()
            self.__horizontal_step = self.__horizontal_step_calc()
            self.__index = 0
        else:
            self.__option_list = None

    @property
    def title(self):
        return self.__title

    @title.setter
    def title(self, t):
        if t:
            self.__title = t
            if not self.__title:
                self.__vertical_step = self.__vertical_step_calc()
                self.__horizontal_step = self.__horizontal_step_calc()

        elif not t:
            self.__title = None
            if self.__title:
                self.__vertical_step = self.__vertical_step_calc()
                self.__horizontal_step = self.__horizontal_step_calc()

    def __vertical_step_calc(self):
        lineCount = ceil(len(self.__option_list) / self.__column_count)
        if self.__title:
            lineCount += 1
        spaceCount = lineCount
        heightTotal = self.__height - self.__padding[1] - self.__padding[3]
        spaceTotal = heightTotal - lineCount * self.__font_height
        return int(spaceTotal / spaceCount)

    def __horizontal_step_calc(self):
        return int((self.__width - self.__padding[0] - self.__padding[2]) / self.__column_count)

    def set_index(self, index: int):
        if not self.__option_list or index < len(self.__option_list):
            raise IndexError
        self.__index = index

    def get_current_index(self):
        if self.__options:
            return self.__index
        raise IndexError

    def option_up(self):
        if self.__index - self.__column_count < 0:
            return
        else:
            self.__index -= self.__column_count

    def option_down(self):
        if (len(self.__option_list) - self.__column_count) <= self.__index < len(self.__option_list):
            return
        else:
            self.__index += self.__column_count

    def option_right(self):
        self.__index = self.__index if self.__index == len(self.__option_list) - 1 else self.__index + 1

    def option_left(self):
        self.__index = 0 if self.__index == 0 else self.__index - 1

    def decorate(self, frame, rotate=0):
        sketch = np.zeros(frame.shape, np.uint8)
        vertical_offset = self.__padding[1] + self.__font_height
        if self.__title:
            vertical_offset += self.__vertical_step + self.__font_height
            total_width = self.__width - self.__padding[0] - self.__padding[2]
            x = (total_width - len(self.__title) * self.__font_height) // 2

            cv2.putText(
                sketch,
                self.__title,
                (
                    self.__padding[0] + x,
                    self.__font_height + self.__padding[1]
                ),
                cv2.FONT_ITALIC, self.__font_size,
                self.__color,
                self.__thickness
            )
        current_index = 0
        for i in range(ceil(len(self.__option_list) / self.__column_count)):
            for j in range(self.__column_count):
                try:
                    coordinate = (
                        j * self.__horizontal_step + self.__padding[0],
                        i * (self.__font_height + self.__vertical_step) + vertical_offset
                    )
                    if current_index == self.__index:
                        cv2.rectangle(sketch,
                                      (coordinate[0], coordinate[1] - self.__font_height),
                                      (coordinate[0] + self.__horizontal_step, coordinate[1]),
                                      color=(255, 0, 0),
                                      thickness=-self.__thickness
                                      )

                    cv2.putText(
                        sketch,
                        self.__option_list[i * self.__column_count + j],
                        coordinate,
                        cv2.FONT_ITALIC, self.__font_size,
                        self.__color,
                        self.__thickness
                    )
                    current_index += 1
                except IndexError:
                    break
        if rotate:
            sketch = np.rot90(sketch, -rotate // 90)

        cv2.addWeighted(sketch, 1, frame, 0.3, 0, frame)
