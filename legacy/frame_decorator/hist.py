from math import inf, floor

import cv2
import numpy as np


class Hist:
    def __init__(self, width=320, height=240, padding=(0, 0, 0, 0), thickness=1, fill=False):
        self.__width = width
        self.__padding = padding
        horizontal = width - self.__padding[0] - self.__padding[2]
        self.__maxSize = 256 if horizontal > 256 else horizontal
        self.__step = floor(horizontal / self.__maxSize)
        self.__height = height
        self.__thickness = thickness
        self.__fill = fill
        self.__max = -inf
        self.__min = inf
        self.__dataList = [[], [], []]

    def __frameCalc(self, frame):
        self.__max = -inf
        self.__min = inf
        frameDataList = (
            [i[0] for i in cv2.calcHist([frame], [0], None, [self.__width], [0, 255])],
            [i[0] for i in cv2.calcHist([frame], [1], None, [self.__width], [0, 255])],
            [i[0] for i in cv2.calcHist([frame], [2], None, [self.__width], [0, 255])]
        )

        minListSize = min(len(frameDataList[0]), len(frameDataList[1]), len(frameDataList[2]))
        if minListSize > self.__maxSize:
            tempb, tempg, tempr = list(), list(), list()
            step = minListSize // self.__maxSize
            for i in range(0, minListSize, step):
                sumb, sumg, sumr = 0, 0, 0
                for j in range(step):
                    sumb += frameDataList[0][i + j]
                    sumg += frameDataList[1][i + j]
                    sumr += frameDataList[2][i + j]
                tempb.append(sumb / step)
                tempg.append(sumg / step)
                tempr.append(sumr / step)
            self.__dataList[0], self.__dataList[1], self.__dataList[2] = tempb, tempg, tempr
        else:
            self.__dataList[0], self.__dataList[1], self.__dataList[2] = frameDataList[0], frameDataList[1], \
                                                                         frameDataList[2]

        self.__min = min(min(self.__dataList[0]), min(self.__dataList[1]), min(self.__dataList[2]))
        self.__max = max(max(self.__dataList[0]), max(self.__dataList[1]), max(self.__dataList[2]))

    def decorate(self, frame, rotate=0):
        self.__frameCalc(frame)
        sketch = np.zeros(frame.shape, np.uint8)
        for index, i in enumerate(zip(self.__dataList[0], self.__dataList[1], self.__dataList[2])):
            if self.__max == self.__min:
                b, g, r = 0, 0, 0
            else:
                b = 0 + (100 - 0) / (self.__max - self.__min) * (i[0] - self.__min)
                g = 0 + (100 - 0) / (self.__max - self.__min) * (i[1] - self.__min)
                r = 0 + (100 - 0) / (self.__max - self.__min) * (i[2] - self.__min)

            data = (b, g, r)
            color = [0, 0, 0]
            for indey, _ in enumerate(data):
                color[indey] = 255
                if self.__fill:
                    cv2.rectangle(sketch,
                                  (
                                      self.__padding[0] + index * self.__step,
                                      int(
                                          self.__height - (
                                                  _ / 100 * (self.__height - self.__padding[1] - self.__padding[3])
                                          ) - self.__padding[3])
                                  ),
                                  (int((index + 1) * self.__step + self.__padding[0]),
                                   self.__height - self.__padding[3]),
                                  color=color,
                                  thickness=-1)
                else:
                    cv2.rectangle(sketch,
                                  (self.__padding[0] + index * self.__step,
                                   int(
                                       self.__height - (_ / 100 * (
                                               self.__height - self.__padding[1] - self.__padding[3])) -
                                       self.__padding[3] - self.__thickness
                                   )
                                   ),
                                  (
                                      int((index + 1) * self.__step + self.__padding[0]),
                                      int(
                                          self.__height - (_ / 100 * (
                                                  self.__height - self.__padding[1] - self.__padding[3])) -
                                          self.__padding[3]
                                      )
                                  ),
                                  color=color,
                                  thickness=1)
                color[indey] = 0
            sketch = np.rot90(sketch, -rotate // 90)
            cv2.addWeighted(sketch, 1, frame, 1, 0, frame)
