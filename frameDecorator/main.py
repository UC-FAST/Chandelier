from typing import Tuple

import cv2
import numpy as np


class main:
    def __init__(self, width=320, height=240, fontHeight=12):
        self.__width = width
        self.__height = height
        self.__cursor = 0
        self.__text = (None, None, None, None)
        self.__font_scale = cv2.getFontScaleFromHeight(cv2.FONT_ITALIC, fontHeight)
        self.__padding = self.__width // 40

    def decorate(self, frame):
        sketch = np.zeros(frame.shape, np.uint8)
        cv2.rectangle(
            sketch,
            (self.__padding, self.__height // 2 - self.__height // 12 // 2),
            (self.__padding + self.__height // 12, self.__height // 2 + self.__height // 24),
            (255, 0, 0),
            -1
        )
        for i in range(4):
            if i == self.__cursor:
                cv2.rectangle(sketch, (i * 76 + 12, 202), (i * 76 + 80, 232), (255, 0, 0), -1)
                cv2.putText(
                    sketch, self.__text[i],
                    (i * 76 + 12, 232),
                    cv2.FONT_ITALIC,
                    self.__font_scale,
                    (255, 255, 255)
                )
            else:
                cv2.rectangle(sketch, (i * 76 + 12, 202), (i * 76 + 80, 232), (255, 255, 255), -1)
                cv2.putText(
                    sketch, self.__text[i],
                    (i * 76 + 12, 232),
                    cv2.FONT_ITALIC,
                    self.__font_scale,
                    (255, 0, 0)
                )
        cv2.addWeighted(sketch, 1, frame, 1, 0, frame)

    @property
    def bottom_bar_text(self) -> Tuple[str | None, str | None, str | None, str | None]:
        return self.__text

    @bottom_bar_text.setter
    def bottom_bar_text(self, text: Tuple[str, str, str, str]):
        self.__text = text

    def next_cursor(self):
        if self.__cursor == 3:
            self.__cursor = 0
        else:
            self.__cursor += 1
