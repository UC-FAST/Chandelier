# CEEE-树莓派相机 软件

开发中，尚未实现所有预定功能

# 运行环境

系统版本(已测试)：

2022-01-28-raspios-bullseye-armhf
2022-01-28-raspios-bullseye-arm64-lite

Python版本：3.9.2

# 准备工作：

```
sudo apt update
sudo apt install python3-libcamera python3-pyqt5 python3-kms++ libatlas-base-dev raspberrypi-ui-mods ffmpeg git python3-pip libcap-dev
```

在raspi-config的Interface Options选项中开启SPI和I2C，Legacy Camera选择'No'

```
cd ~
git clone https://github.com/UC-FAST/ceee-cam-software.git
cd ceee-cam-software/
pip3 install -r requirements-py39.txt
```

_____________
设置开机自启动

* 按需修改cam.service
* 执行

```
sudo cp cam.service /etc/systemd/system/
sudo systemctl daemon-reload
```

* 检查是否能正常执行

```
sudo systemctl start cam
```

若无报错，进入下一步

```
sudo systemctl enable cam.service
```

____________
可选的swap分区扩容

```
sudo dd if=/dev/zero of=/swap bs=1k count=2048000
sudo mkswap /swap
sudo swapon /swap
```

编辑/etc/fstab，加入一行：

```
/swap swap swap defaults 0 0
```

______
启动

```sudo systemctl start cam```

或

```python3 startup.py```

# 使用方法(简明版)

## circle按钮

短按：拍照、确认

长按：录像

## triangle按钮

短按：菜单、返回

长按：返回至camera

## rectangle按钮

短按：旋转显示方向

长按：截图

## cross按钮

短按：熄屏

长按：关机

# 如果需要二次开发(简明版)

frameDecorator是关界面渲染的相关组件

逻辑部分需要继承controlledEnd.ControlledEnd，随后放入universalControl.UniversalControl的列表类型的参数controlledEndList中

# 如果需要二次开发(编辑中)
## 目录结构
* components：一些不便分类的组件
* controlledEnd：提供与交互相关的逻辑实现
* frameDecorator：帧渲染工具
* tuning：相机的配置文件目录
* utils：一些工具

具体说明查看相应目录内readme文件

## universalControl.py
在本项目中，实现交互的单元为controlledEnd.ControlledEnd。需要继承ControlledEnd类实现从交互到逻辑的控制。
universalControl.UniversalControl则是控制ControlledEnd的最上层逻辑。

```
UniversalControl(
    self, 
    lcd: screen.Lcd, 
    controlledEndList: List[controlledEnd.ControlledEnd]
)
```
UniversalControl类的构造函数

参数：

lcd：screen.Lcd类变量

controlledEndList：由controlledEnd.ControlledEnd的子类构成的列表

```
UniversalControl.mainLoop(self)
```
主循环，将ControlledEnd.mainloop返回的图像序列送入显示队列



camConfig.json编写指南

{
    "ControlledEnd1":{
        "0":{
            "title":"Enter menu title here",
            "Options":[  #List your Options
                {
                    "id":
                    "content"
                    "type": menu, irq, msg, option, bool, num,numeral
                },
                ...
            ]
        }
    },
    "ControlledEnd2":{

    }
}
