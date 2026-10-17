import queue
import typing
from time import sleep

import numpy as np

import controlledEnd
import frameDecorator
from components import mediaBrowser


class GalleryControlledEnd(controlledEnd.ControlledEnd, mediaBrowser.MediaBrowser):
    def __init__(self, _id='GalleryControlledEnd', width=320, height=240, pictPath='./pict/'):
        controlledEnd.ControlledEnd.__init__(self, _id)
        pictPath = pictPath if pictPath.endswith('/') else pictPath + '/'
        mediaBrowser.MediaBrowser.__init__(self, pictPath, width, height)
        self.__width, self.__height = width, height
        self.__option: typing.Dict[typing.Dict] = None
        self.__frame_list = queue.Queue()
        self.__busy = frameDecorator.Busy()
        self.__rotate = 0
        self.__current_frame = np.zeros((self.__width, self.__height, 3), np.uint8)
        self.__hist = frameDecorator.Hist2()
        self.__raw_frame = None
        self.__from = None
        self.__delete = False
        self.__empty = False
    

        self.__decorator = frameDecorator.SimpleText(
            [self.__worker1],
            height=self.__height,
            padding=(10, 20, 0, 0),
            font_height=10,
            color=frameDecorator.Colors.gold.value
        )
        self.__simpleTextEnable = False

    def __worker1(self):
        return {
            "{}": self.getCurrentFileName(),
        }

    def __findOptionByID(self, target):
        for key, value in self.__option.items():
            if 'options' in value.keys():
                for j in value['options']:
                    if 'id' in j.keys() and 'value' in j.keys():
                        if j['id'] == target:
                            return j['value']
        raise LookupError('target')


    def center_press_action(self):
        if self.__empty:
            self.__empty = False
            self._irq(self.__from)
            return
        if self.isPlaying():
            self.togglePlayPause()

    def up_press_action(self):
        if self.__empty:
            self.__empty = False
            self._irq(self.__from)
            return
        self.previous()
        self.__refreshFrame()
        sleep(0.2)

    def down_press_action(self):
        if self.__empty:
            self.__empty = False
            self._irq(self.__from)
            return
        self.next()
        self.__refreshFrame()
        sleep(0.2)

    def left_press_action(self):
        if self.__empty:
            self.__empty = False
            self._irq(self.__from)
            return
        self.previous()
        self.__refreshFrame()
        sleep(0.2)

    def right_press_action(self):
        if self.__empty:
            self.__empty = False
            self._irq(self.__from)
            return
        self.next()
        self.__refreshFrame()
        sleep(0.2)

    def __addHist(self):
        if self.__findOptionByID("show hist"):
            self.__hist.decorate(self.__current_frame)
        else:
            self.__current_frame = self.__raw_frame.copy()

    def circle_press_action(self):
        if self.__empty:
            self.__empty = False
            self._irq(self.__from)
            return
        self.__simpleTextEnable = not self.__simpleTextEnable
        self.__refreshFrame()

    def cross_press_action(self):
        if self.__empty:
            self.__empty = False
            self._irq(self.__from)
            return
        self._irq('CameraControlledEnd')

    def crossLongPressAction(self):
        if self.__empty:
            self.__empty = False
            self._irq(self.__from)
            return
        self._irq('CameraControlledEnd')

    def square_press_action(self):
        if self.__empty:
            self.__empty = False
            self._irq(self.__from)
            return
        self._irq('MenuControlledEnd')

    def shutter_press_action(self):
        pass
    
    def rotary_encoder_clockwise(self):
        pass

    def rotary_encoder_counter_clockwise(self):
        pass

    def rotary_encoder_select(self):
        pass
    
    def msg_receiver(self, sender, msg):
        if sender == 'MenuControlledEnd':
            self.__option = msg[1]
            if msg[0] == 'delete':
                self.__delete = True
                self.deleteCurrent()
                self.__refreshFrame()
            elif msg[0] == 'update':
                self.update()

    def update(self):
        pass

    def __refreshFrame(self):
        if self.__current_frame is not None:
            self.__busy.decorate(self.__current_frame)
            self.__frame_list.put(self.__current_frame)
        try:
            self.__raw_frame = self.getCurrentFrame()
        except FileExistsError:
            self.__empty = True
            self._irq("CameraControlledEnd")
            return
        self.__current_frame = self.__raw_frame.copy()
        self.__addHist()
        self.__frame_list.put(self.__current_frame)

    def main_loop(self):
        while True:
            pict = self.__frame_list.get(block=True)
            if self.__simpleTextEnable:
                self.__decorator.decorate(pict)
            yield np.rot90(pict, -self.__rotate // 90)


    def on_exit(self):
        self.__frame_list.put(frameDecorator.Warining().decorate("Empty"))

    def on_enter(self, lastID):
        try:
            if not self.__delete:
                self.__from = lastID
                self.refreshMediaList()
            self.__delete = False
            self._msg_sender(self._id, "MenuControlledEnd", self._id)
            self.__refreshFrame()
        except FileExistsError:
            self.__empty = True
            self.__frame_list.put(frameDecorator.Warining().decorate("Empty"))
