# configLoader.ConfigLoader

json.load的脱裤子放屁版，唯一的用处是实现了只读？

```
configLoader.ConfigLoader(self, configPath)
```

类的构造函数

参数：

configPath：json文件路径
_____

# galleryBrowser.GalleryBrowser

一个带读缓存的图库浏览工具

```
galleryBrowser.GalleryBrowser(
    self, 
    pictPath='./pict', 
    width=None, 
    height=None
):
```

类的构造函数

参数：

pictPath：图片文件路径

width,height：输出图像大小，为None时保留原尺寸

```
GalleryBrowser.refreshPictList(self)
```

更新图文件列表，图片文件路径内不包含图片时抛出FileNotFoundError

```
GalleryBrowser.next(self)
GalleryBrowser.previous(self)
```

下一张/上一张图片

```
GalleryBrowser.getPict(self)
```

返回当前图片

```
GalleryBrowser.delete(self)
```

删除当前图片

```
GalleryBrowser.getPictName(self)
```

返回当前图片路径
_____

# led

灯

```
led.toggleState(led)
led.on(led)
led.off(led)
```

状态翻转/开灯/关灯

```
led.green 
led.blue
```

绿灯、蓝灯
_____

# max17048.Max17048

max17048的驱动

```
max17048.Max17048(self)
```

类的构造函数

```
Max17048.getBat(self)
```

返回当前电压

```
Max17048.version(self)
```

返回VERSION寄存器数据

```
Max17048.soc(self)
```

返回State of Charge
_____
# picam2.Cam
基于Picamera2的相机管理类，
Picamera2基于libcamera实现

参考链接

Piacamera2:

[Picamera2](https://github.com/raspberrypi/picamera2)

libcamera:

[libcamera](https://libcamera.org/api-html/control__ids_8h.html)

[libcamera_github](https://github.com/raspberrypi/libcamera)

```
picam2.Cam(
    self, 
    verbose_console=None, 
    tuning=None
)
```
类的构造函数

参数：

verbose_console：日志输出级别，
0：仅将错误和关键消息打印到控制台。
1：将所有信息、错误和关键消息打印到控制台。
2：将所有调试、信息、错误和关键消息打印到控制台。

tuning：tuning配置，见[raspberry-pi-camera-guide](https://datasheets.raspberrypi.com/camera/raspberry-pi-camera-guide.pdf)

```
Cam.zoom(self, zoom)
```
数码变焦

参数：

zoom：缩放系数,大于等于1

```
Cam.setAwbMode(self, mode)
```
设置自动白平衡模式
参数：

mode：白平衡模式，可选
(
    "Auto",
    "Incandescent",
    "Tungsten",
    "Fluorescent",
    "Indoor",
    "Daylight",
    "Cloudy"
)

```
Cam.brightness(self,brt)
```
亮度设置

参数：

brt：亮度，范围-1~1


_____
# screen.Lcd
ST7735R屏幕驱动，
在spi通信频率为17500000Hz、传入的图像为numpy数组时至少可达到30fps的显示帧率。（好用到哭）

```
screen.Lcd(
    self, 
    width=128, 
    height=128, 
    ScanDir=ScanDir.R2L_D2U
)
```
类的初始化函数

参数：

width：屏幕宽度

height：屏幕高度

ScanDir：刷新方向，见[screen.ScanDir](#screen.ScanDir)

```
Lcd.backlight(self, state: bool)
```
背光开关

参数：

state：背光控制
```
Lcd.reset(self)
```
屏幕重置

```
Lcd.init(self)
```
屏幕初始化

```
Lcd.clear(self)
```
清屏

```
showImage(self, image)
```
显示一帧图像

参数：

image：图像数组

------

## <span id="screen.ScanDir">screen.ScanDir</span>
设置屏幕的刷新方向

```ScanDir.L2R_U2D```

从左到右，从上到下

```ScanDir.L2R_D2U```

从左到右，从下到上

```ScanDir.R2L_U2D```

从右到左，从上到下

```ScanDir.R2L_D2U```

从右到左，从下到上

```ScanDir.U2D_L2R```

从上到下，从左到右

```ScanDir.U2D_R2L```

从上到下，从右到左

```ScanDir.D2U_L2R```

从下到上，从左到右

```ScanDir.D2U_R2L```

从下到上，从右到左
