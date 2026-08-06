import cv2
import numpy as np


class WaterMark:
    def __init__(self, width=128, height=128, font_height=0):
        self.__width = width
        self.__height = height
        if font_height:
            self.__font_height = font_height
            self.__font_size = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_DUPLEX, self.__font_height)
        else:
            self.__font_height = self.__height // 35
            self.__font_size = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_DUPLEX, self.__font_height)

    def decorate(self, frame):
        circle_coordinate = (
            int(self.__width / 20.6) + self.__font_height // 2,
            self.__height - (int(self.__height / 39.3) + self.__font_height // 2)
        )
        sketch = np.zeros(frame.shape, frame.dtype)
        cv2.circle(sketch, circle_coordinate, int(self.__font_height / 2.8), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(sketch, circle_coordinate, int(self.__font_height / 1.8), (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(sketch,
                    "CEEE",
                    (
                        circle_coordinate[0] + self.__font_height // 2 + self.__width // 70,
                        circle_coordinate[1] + self.__font_height // 2
                    ),
                    cv2.FONT_HERSHEY_DUPLEX,
                    self.__font_size,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                    )
        cv2.addWeighted(sketch, 0.6, frame, 1, 0, frame)
