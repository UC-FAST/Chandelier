import cv2
import numpy as np


class Toast:
    def __init__(
            self,
            width=320,
            height=240,
            font_height=12,
    ):
        self.__width = width
        self.__height = height
        self.__font_height = font_height
        self.__font_scale = cv2.getFontScaleFromHeight(cv2.FONT_ITALIC, self.__font_height)
        self.__text = None
        self.__offset_right = 0
        self.__offset_left = self.__width
        self.__is_update = False

    @property
    def isUpdate(self):
        return self.__is_update

    def set_text(self, text: str):
        if not self.__text or len(text) != len(self.__text):
            self.__offset_right = (self.__width - len(text) * self.__font_height) // 2 - self.__font_height
            self.__offset_left = (self.__width + len(text) * self.__font_height) // 2 + self.__font_height
        self.__text = text
        self.__is_update = True

    def decorate(self, frame, rotate=0):
        if self.__text:
            sketch = np.zeros(frame.shape, np.uint8)
            cv2.rectangle(
                sketch,
                (self.__offset_right, int(self.__height * 0.96 - self.__font_height)),
                (self.__offset_left, int(self.__height * 0.96)),
                (255, 0, 0),
                -1
            )
            cv2.putText(
                sketch,
                self.__text,
                (
                    (self.__width - self.__font_height * len(self.__text)) // 2,
                    int(self.__height * 0.96)
                ),
                cv2.FONT_ITALIC,
                self.__font_scale,
                (255, 255, 255)
            )
            if rotate:
                sketch = np.rot90(sketch, -rotate // 90)
            cv2.addWeighted(sketch, 1, frame, 1, 0, frame)
            self.__is_update = False
