# controlledEnd.ControlledEnd

把它抽象理解成操作系统里的独立进程吧

```
ControlledEnd(self, _id)
```

ControlledEnd类的构造函数

参数：

_id：类的唯一标识符

```
centerPressAction(self)
upPressAction(self)
downPressAction(self) 
leftPressAction(self)  
rightPressAction(self)   
circlePressAction(self)     
trianglePressAction(self)
```

抽象方法，对应按钮按下时的回调函数

```
centerReleaseAction(self)
upReleaseAction(self)
downReleaseAction(self)
leftReleaseAction(self)
rightReleaseAction(self)
circleReleaseAction(self)
triangleReleaseAction(self)
```

虚方法，对应按钮弹起时的回调函数

```   
direction(self, direction)
```

抽象方法，显示方向改变时的回调函数

参数：

direction：显示角度

```
msgReceiver(self, sender, msg):
```

抽象方法，消息接收函数

参数：

sender：消息发送者的id

msg：消息内容

```
onExit(self)
onEnter(self)
```

虚方法，ControlledEnd组件上下文切换时调用

```
mainLoop(self)
```
抽象方法，必须返回一个可迭代的图像序列

```
id
```

只读属性，返回类的唯一标识符_id

```
_msgSender(self, sender: str, receiver: str, msg)
```

消息发送函数，在UniversalControl初始化时被初始化

参数：

sender：发送者，通常赋值为self._id

receiver：接收者，通常赋值为接收对象的_id

msg：消息内容

```
_irq(self, _id: str)
```

用于退出并拉起另一个ControlledEnd,在UniversalControl初始化时被初始化

参数：

_id：被拉起对象的_id


