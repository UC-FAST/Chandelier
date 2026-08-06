import os

import cv2
import numpy as np


class Hdr:
    def __init__(self, timeList: list, imageList: list, correction=False):
        self.__exposureTimeList, self.__imageList = np.array(timeList, dtype=np.float32), imageList
        if correction:
            alignMTB = cv2.createAlignMTB()
            alignMTB.process(self.__imageList, self.__imageList)

    def calibrateDebevec(self):
        return cv2.createCalibrateDebevec().process(
            self.__imageList,
            self.__exposureTimeList
        )

    def mergeDebevec(self):
        return cv2.createMergeDebevec().process(
            self.__imageList,
            self.__exposureTimeList,
            self.calibrateDebevec()
        )

    def exposureFusion(self):
        mergeMertens = cv2.createMergeMertens()
        exposureFusion = mergeMertens.process(self.__imageList)
        return exposureFusion * 255

    def tonemapDrago(self):
        tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
        return tonemapDrago.process(self.mergeDebevec()) * 255

    def tonemapMantiuk(self):
        tonemapMantiuk = cv2.createTonemapMantiuk(2, 0.85, 1.2)
        return tonemapMantiuk.process(self.mergeDebevec()) * 255

    def tonemapReinhard(self):
        tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
        return tonemapReinhard.process(self.mergeDebevec()) * 255


if __name__ == '__main__':

    frameList = list()
    for i in os.listdir('../temp'):
        frameList.append(cv2.imread(os.path.join('../temp', i)))
    h = Hdr(frameList)
    h.exposureFusion('1.png')
