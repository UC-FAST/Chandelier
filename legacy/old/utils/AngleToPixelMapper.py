'''
我现在在使用python做一个声源定位可视化项目，已经能获取声源坐标azi和ele，现在要把这个坐标结合显示在屏幕上的摄像头画面结合
屏幕分辨率是320*240,输入的azi范围为0~359，ele范围为0~90，
#摄像头的视场角未知，能映射在屏幕上的角度也未知，所以需要帮忙留下调试参数
函数名称为aziel_to_screen_spherical，指定第一个参数为azi，第二个参数为ele
azi=0，ele为ele_const_w对应屏幕坐标(1,120)，
azi=180，ele为ele_const_w对应屏幕坐标(320,120)，
azi=90，ele为ele_const_h对应屏幕坐标(160,1)，
azi=270，ele为ele_const_h对应屏幕坐标(160,240)，
ele为90时对应屏幕坐标(160,120)
'''


import math


import math

def aziel_to_screen_spherical(azi, ele, width=320, height=240):
    """
    将球面坐标(方位角azi, 仰角ele)转换为屏幕坐标

    参数:
        azi: 方位角 0~359度
        ele: 仰角 0~90度

    返回:
        (x, y): 屏幕坐标
    """
    # 屏幕分辨率

    # 中心点坐标
    center_x = width / 2  # 160
    center_y = height / 2  # 120
    ele_const_w = 65
    ele_const_h = 68
    if ele == 90:
        return (center_x, center_y)
    
    # 根据给定的映射关系，计算半径
    # 当ele=ele_const_w时，水平最大半径为160（从中心到边缘）
    # 当ele=ele_const_h时，垂直最大半径为119（从中心到边缘）

    # 使用非线性映射代替线性插值
    # 将ele映射到0~1范围，使用正弦函数使变化更自然
    if ele <= ele_const_w:
        # ele从0到ele_const_w，使用正弦映射使起始和结束更平滑
        t = ele / ele_const_w  # 0~1
        # 使用正弦映射，让中间区域变化更明显，两端更平缓
        radius_h = 160 * math.sin(t * math.pi / 2)
    else:
        # ele从ele_const_w到90，使用余弦映射平滑降到0
        t = (ele - ele_const_w) / (90 - ele_const_w)  # 0~1
        radius_h = 160 * math.cos(t * math.pi / 2)

    # 计算垂直方向的半径（随仰角变化）
    if ele <= ele_const_h:
        t = ele / ele_const_h  # 0~1
        radius_v = 119 * math.sin(t * math.pi / 2)
    else:
        t = (ele - ele_const_h) / (90 - ele_const_h)  # 0~1
        radius_v = 119 * math.cos(t * math.pi / 2)

    # 使用椭圆的半径，根据方位角计算x和y偏移
    # 将azi转换为弧度
    azi_rad = math.radians(azi)

    # 根据方位角计算偏移量
    # 注意：azi=0在右侧，azi=90在上方
    offset_x = radius_h * math.cos(azi_rad)
    offset_y = -radius_v * math.sin(azi_rad)  # 负号因为屏幕y轴向下

    # 计算最终的屏幕坐标
    x = center_x - offset_x
    y = center_y + offset_y

    # 确保坐标在屏幕范围内
    x = max(1, min(width, int(round(x))))
    y = max(1, min(height, int(round(y))))

    print(x, y)
    return (x, y)
