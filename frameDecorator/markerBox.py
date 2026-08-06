import cv2


class MarkerBox:
    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        color: tuple[int, int, int] = (0, 0, 255),
        size: tuple[int, int] = (10, 10),
        font_height: int = 12
    ):
        self.__width = width
        self.__height = height
        self.__color = color
        self.__size = size
        self.__font_height = font_height
        self.__font_scale = cv2.getFontScaleFromHeight(
            cv2.FONT_ITALIC,
            self.__font_height
        )

    def decorate(self, frame, coordinate: tuple[int, int], text: str):
        print(coordinate)
        """
        以中心点绘制方框，自动判断文字放框上方/下方
        :param frame: opencv图像 ndarray
        :param center_xy: (cx, cy) 中心坐标
        :param text: 需要绘制的标签文字
        :param rotate: 预留旋转参数（暂未实现，按需扩展）
        :return: 绘制完成的图像（原图就地修改，同时返回）
        """
        if coordinate == None:
            coordinate=(1,1)
        else:
            coordinate=int(coordinate[0]),int(coordinate[1])
        h_half, w_half = self.__size[0]//2, self.__size[1] // 2

        # 计算矩形对角坐标
        x1, y1 = coordinate[0] - w_half, coordinate[1] - h_half
        x2, y2 = coordinate[0] + w_half, coordinate[1] + h_half

        # 边界裁剪，防止框越界
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.__width - 1, x2)
        y2 = min(self.__height-1, y2)

        # 绘制方框
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.__color, thickness=1)

        thickness = 1
        text_color = (255, 255, 255)  # 白色文字

        # 获取文字包围盒
        (tw, th), baseline = cv2.getTextSize(
            text,
            cv2.FONT_ITALIC,
            self.__font_scale,
            thickness=thickness
        )

        # 自动判断文字位置：框靠近图像顶部，则文字放框下方；否则放上方
        padding = 3
        if y1 < th + padding * 2:
            # 空间不足，文字放在框下沿
            text_y = y2 + th + padding
            bg_y1 = y2
            bg_y2 = y2 + th + 2 * padding
        else:
            # 文字放在框上沿
            text_y = y1 - padding
            bg_y1 = y1 - th - 2 * padding
            bg_y2 = y1

        # 文字水平居中对齐方框
        text_x = coordinate[0] - tw // 2
        bg_x1 = text_x - padding
        bg_x2 = text_x + tw + padding

        # 文字背景矩形（和边框同色）
        bg_x1 = max(0, bg_x1)
        bg_y1 = max(0, bg_y1)
        bg_x2 = min(self.__width - 1, bg_x2)
        bg_y2 = min(self.__height - 1, bg_y2)
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), self.__color, -1)

        # 绘制白色文字
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            cv2.FONT_ITALIC,
            self.__font_scale,
            text_color,
            thickness
        )

        return frame
