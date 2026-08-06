import cv2
import numpy as np


class DirectionIndicator:
    def __init__(self, width=128, height=128, color=(0, 0, 255), during=None):
        self.__width = width
        self.__height = height
        self.__during = during
        self.__color = color
        self.__count = self.__during
        self.__sketch = np.zeros((self.__width, self.__height, 3), np.uint8)
        polygon = np.array(
            (
                (self.__width // 12, self.__height // 4),
                (self.__width // 6, self.__height // 12),
                (self.__width // 4, self.__height // 4),
                (self.__width // 5, self.__height // 4), m
                (self.__width // 5, self.__height - self.__height // 12),
                (self.__width // 8, self.__height - self.__height // 12),
                (self.__width // 8, self.__height // 4)
            )
        )
        cv2.fillConvexPoly(self.__sketch, polygon, self.__color)

    def trigger(self):
        self.__count = 0

    def decorate(self, frame, rotate):
        if self.__count == self.__during:
            return
        sketch = np.rot90(self.__sketch, -rotate // 90)
        cv2.addWeighted(sketch, 1, frame, 1, 0, frame)
        self.__count += 1
