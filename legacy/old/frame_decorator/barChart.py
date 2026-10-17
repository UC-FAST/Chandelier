from math import ceil, inf
import cv2
import numpy as np


class BarChart:
    def __init__(self, width=320, height=240, maxSize=320, scale=0, color=(0, 0, 255), thickness=1, fill=False,
                 alpha: float = 1):
        # 计算柱形步长（宽度）
        self.__step = max(1, ceil(width / maxSize))  # 确保步长至少为1
        self.__max_size = maxSize
        self.__height = height
        self.__width = width
        self.__scale = scale
        self.__color = color
        self.__thickness = thickness
        self.__fill = fill
        
        # 限制alpha在[0,1]范围内
        self.__alpha = max(0.0, min(1.0, alpha))
        
        # 数据统计
        self.__max = -inf
        self.__min = inf
        self.__data_list = []

    @property
    def data_list(self):
        return self.__data_list

    @data_list.setter
    def data_list(self, datas):
        # 数据降采样处理
        if len(datas) > self.__max_size:
            step = max(1, len(datas) // self.__max_size)
            self.__data_list = [
                sum(datas[i:i+step]) / step
                for i in range(0, len(datas), step)
            ][:self.__max_size]
        else:
            self.__data_list = datas.copy()  # 避免外部修改影响
        
        # 更新极值
        if self.__data_list:
            self.__min = min(self.__data_list)
            self.__max = max(self.__data_list)

    def add_data(self, data):
        # 数据队列管理
        if len(self.__data_list) == self.__max_size:
            removed = self.__data_list.pop(0)
            # 动态更新极值
            if not self.__scale:
                if removed == self.__max:
                    self.__max = max(self.__data_list) if self.__data_list else -inf
                elif removed == self.__min:
                    self.__min = min(self.__data_list) if self.__data_list else inf
        
        self.__data_list.append(data)
        
        # 更新极值
        if not self.__scale:
            if data > self.__max:
                self.__max = data
            if data < self.__min:
                self.__min = data
        return self

    def __normalize_value(self, value):
        """归一化数据值到[0,100]范围"""
        if self.__scale:
            return min(100.0, max(0.0, (value / self.__scale) * 100.0))
        
        if self.__max - self.__min < 1e-6:  # 处理除零情况
            return 50.0  # 默认中间值
        
        # 线性映射到[10,90]范围（保留边界空间）
        normalized = 10 + 80 * (value - self.__min) / (self.__max - self.__min)
        return min(90.0, max(10.0, normalized))

    def decorate(self, frame, rotate=0):
        # 创建透明图层
        overlay = np.zeros_like(frame)
        
        # 计算旋转角度
        rotation_count = rotate // 90
        rotated = rotation_count % 4 != 0
        
        # 预计算绘图参数
        bar_width = self.__step
        base_y = self.__height
        
        for idx, data in enumerate(self.__data_list):
            # 计算柱形高度
            norm_val = self.__normalize_value(data)
            bar_height = int(base_y * norm_val / 100.0)
            top_y = base_y - bar_height
            
            # 计算柱形位置
            x1 = idx * bar_width
            x2 = x1 + bar_width
            
            # 绘制柱形
            if self.__fill:
                cv2.rectangle(
                    overlay, 
                    (x1, top_y), 
                    (x2, base_y), 
                    self.__color, 
                    -1  # 填充模式
                )
            else:
                # 绘制顶部横条（厚度由thickness控制）
                cv2.rectangle(
                    overlay,
                    (x1, top_y - self.__thickness),
                    (x2, top_y),
                    self.__color,
                    -1
                )
        
        # 应用透明度
        if self.__alpha < 1.0:
            # 创建透明混合层
            blended = np.zeros_like(frame)
            cv2.addWeighted(overlay, self.__alpha, blended, 1 - self.__alpha, 0, blended)
            overlay = blended
        
        # 应用旋转
        if rotated:
            overlay = np.rot90(overlay, -rotation_count)
        
        # 叠加到原图
        cv2.add(frame, overlay, frame)