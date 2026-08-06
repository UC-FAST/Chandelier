import math

import numpy as np


def aziel_to_screen_spherical(azi, ele, 
                                 screen_width=320, 
                                 screen_height=240,
                                 ele_center=90,      # 映射到屏幕中心的ele值
                                 ele_edge=30,         # 映射到屏幕边缘的ele值  
                                 mapping_type='linear'):  # 'linear', 'cos', 'sqrt'
    """
    将声源球坐标转换为屏幕坐标（支持灵活ele范围）
    
    参数:
    azi: 方位角 (0-360°)
        阵列12点: 90°
        阵列6点: 270°
        阵列9点: 360° (或0°)
        阵列3点: 180°
    ele: 仰角 (任意范围，根据ele_center和ele_edge设置)
    screen_width: 屏幕宽度 (默认320)
    screen_height: 屏幕高度 (默认240)
    ele_center: 映射到屏幕中心的ele值 (默认90)
    ele_edge: 映射到屏幕边缘的ele值 (默认0)
    mapping_type: 映射类型
        'linear' - 线性映射
        'cos' - 余弦映射（边缘变化更快）
        'sqrt' - 平方根映射（中心更敏感）
    
    返回:
    (x, y): 屏幕坐标，原点在左上角
    """
    
    # 屏幕中心
    cx = screen_width / 2
    cy = screen_height / 2
    
    # 如果ele正好等于中心值，直接返回屏幕中心
    if abs(ele - ele_center) < 0.001:
        return (cx, cy)
    
    # 计算标准化因子 [0, 1]，0=中心，1=边缘
    if ele_center > ele_edge:
        # 正常情况：ele越大越靠近中心
        ele_range = ele_center - ele_edge
        normalized = (ele_center - ele) / ele_range
    else:
        # 反转情况：ele越小越靠近中心
        ele_range = ele_edge - ele_center
        normalized = (ele - ele_center) / ele_range
    
    # 限制在[0, 1]范围内
    normalized = max(0.0, min(1.0, normalized))
    
    # 根据映射类型计算半径因子
    if mapping_type == 'linear':
        radius_factor = normalized
    elif mapping_type == 'cos':
        # 余弦映射：在中心区域变化较慢，边缘变化较快
        radius_factor = math.sin(normalized * math.pi / 2)
    elif mapping_type == 'sqrt':
        # 平方根映射：在中心区域更敏感
        radius_factor = math.sqrt(normalized)
    else:
        # 默认线性映射
        radius_factor = normalized
    
    # 如果radius_factor为0，说明在中心
    if radius_factor < 0.001:
        return (cx, cy)
    
    # 计算实际半径
    max_radius = min(screen_width, screen_height) / 2
    radius = radius_factor * max_radius
    
    # 角度映射：azi到屏幕方向
    screen_angle_deg = azi - 160
    screen_angle_rad = math.radians(screen_angle_deg)
    
    # 计算偏移
    dx = radius * math.cos(screen_angle_rad)
    dy = radius * math.sin(screen_angle_rad)  
    
    # 最终坐标
    x = cx + dx
    y = cy + dy
    
    # 确保坐标在屏幕范围内
    x = max(0, min(screen_width, x))
    y = max(0, min(screen_height, y))
    
    return int(x), int(y)