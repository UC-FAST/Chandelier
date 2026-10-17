from math import inf, floor
import cv2
import numpy as np


class Hist2:
    def __init__(self, width=320, height=240, padding=(0, 0, 0, 0)):
        self.__width = width
        self.__padding = padding  # (left, top, right, bottom)
        horizontal = width - padding[0] - padding[2]
        self.__maxSize = min(256, horizontal)
        self.__step = max(1, floor(horizontal / self.__maxSize))
        self.__height = height
        self.__max = -inf
        self.__min = inf
        self.__data_list = [np.array([]), np.array(
            []), np.array([])]

    def __frame_calc(self, frame):
        hist_channels = [
            cv2.calcHist([frame], [i], None, [256], [0, 255]).flatten()
            for i in range(3)
        ]

        if len(hist_channels[0]) > self.__maxSize:
            step = len(hist_channels[0]) // self.__maxSize
            new_len = self.__maxSize * step
            self.__data_list = [
                ch[:new_len].reshape(self.__maxSize, step).mean(axis=1)
                for ch in hist_channels
            ]
        else:
            self.__data_list = [ch[:self.__maxSize] for ch in hist_channels]

        all_data = np.concatenate(self.__data_list)
        self.__min = np.min(all_data)
        self.__max = np.max(all_data)

    def decorate(self, frame, rotate=0):
        self.__frame_calc(frame)
        sketch = np.zeros((self.__height, self.__width, 3), dtype=np.uint8)

        pad_left, pad_top, pad_right, pad_bottom = self.__padding
        plot_height = self.__height - pad_top - pad_bottom
        base_Y = self.__height - pad_bottom

        if self.__max <= self.__min:
            pass
        else:

            norm_range = 100.0 if plot_height > 0 else 0
            norm_data = [
                (ch - self.__min) / (self.__max - self.__min) * norm_range
                for ch in self.__data_list
            ]

            bar_heights = [
                (ch * plot_height /
                 100).astype(int) if norm_range > 0 else np.zeros_like(ch)
                for ch in norm_data
            ]

            for i in range(len(self.__data_list[0])):
                x_start = pad_left + i * self.__step
                x_end = x_start + self.__step

                for ch_index, height in enumerate([h[i] for h in bar_heights]):
                    if height <= 0:
                        continue

                    y_start = base_Y - height
                    color = [0, 0, 0]
                    color[ch_index] = 255
                   
                    cv2.rectangle(
                        sketch,
                        (x_start, y_start),
                        (x_end, base_Y),
                        color,
                        -1
                    )

        
        if rotate != 0:
            rotate_times = rotate // 90 % 4  # 标准化旋转角度
            if rotate_times != 0:
                sketch = np.rot90(sketch, rotate_times)

        cv2.add(sketch, frame, frame)
