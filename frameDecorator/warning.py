import cv2
import numpy as np

from .colors import Colors


class Warining:
    def __init__(self, width=128, height=128, font_height: int = 12):
        self.__width = width
        self.__height = height
        self.__font_height = font_height
        self.__scale = cv2.getFontScaleFromHeight(cv2.FONT_ITALIC, self.__font_height)

        self.__sketch = np.zeros((self.__width, self.__height, 3), np.uint8)
        cv2.rectangle(self.__sketch, (0, 0), (self.__width, self.__height), Colors.darkred.value, -1)
        cv2.rectangle(
            self.__sketch,
            (self.__width // 10, self.__height // 10),
            (int(self.__width * 0.9), int(self.__height * 0.9)),
            Colors.darkblue.value,
            -1
        )

    def decorate(self, text: str, rotate=0):
        text_list = text.split()
        space_count = len(text_list) - 1
        space_total = (self.__height * 0.8) - len(text_list) * self.__font_height
        try:
            space = int(space_total // space_count)
        except ZeroDivisionError:
            space = 0
        for index, _ in enumerate(text_list):
            cv2.putText(
                self.__sketch,
                _,
                (self.__width // 10, (index + 1) * self.__font_height + index * space + int(self.__height // 10)),
                cv2.FONT_ITALIC, self.__scale,
                (255, 255, 255),
                1
            )
        return np.rot90(self.__sketch, -rotate // 90)
