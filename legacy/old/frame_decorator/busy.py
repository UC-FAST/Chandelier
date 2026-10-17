import cv2


class Busy:
    def __init__(self, width=320, height=240, color=(0, 0, 255)):
        self.__width = width
        self.__height = height
        self.__color = color

    def decorate(self, frame, rotate=0):
        if rotate == 0:
            cv2.rectangle(
                frame,
                (self.__width - self.__width // 16, 0),
                (self.__width, self.__height // 16),
                color=self.__color,
                thickness=-1
            )
        elif rotate == 90:
            cv2.rectangle(
                frame,
                (self.__width - self.__width // 16, self.__height - self.__height // 16),
                (self.__width, self.__height),
                color=self.__color,
                thickness=-1
            )
        elif rotate == 180:
            cv2.rectangle(
                frame,
                (0, self.__height - self.__height // 16),
                (self.__width // 16, self.__height),
                color=self.__color,
                thickness=-1
            )
        elif rotate == 270:
            cv2.rectangle(
                frame,
                (0, 0),
                (self.__width // 16, self.__height // 16),
                color=self.__color,
                thickness=-1
            )
