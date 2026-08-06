from dataclasses import dataclass
import inspect
import json
import logging
import os
import time
import typing

import cv2
import numpy

import frameDecorator
from components import max17048, picam2, led
from utils import SlidingWindowFilter, configLoader
from . import controlledEnd
from utils import Logger, LogMsg, singleton


@singleton
@dataclass
class CameraControlledEndState:
    zoom: float = 1
    brightness: float = 0
    show_hist: bool = False
    is_busy: bool = False
    mfassist: bool = False
    decorate_enable: bool = False
    zoom_hold: bool = False
    bright_hold: bool = False
    record_timestamp: float | None = None


class CameraControlledEnd(controlledEnd.ControlledEnd, picam2.Cam):
    def __init__(self, _id='CameraControlledEnd', verbose_console: int = logging.INFO, tuning_file_path=None):
        controlledEnd.ControlledEnd.__init__(self, _id)
        if tuning_file_path:
            with open(tuning_file_path, 'r') as f:
                tuning = json.load(f)
        else:
            tuning = None
        picam2.Cam.__init__(
            self,
            verbose_console=verbose_console,
            tuning=tuning
        )

        self.__config = configLoader.ConfigLoader('./config.json')
        self.__bar_chart = frameDecorator.BarChart(
            self.__config['screen']['width'],
            self.__config['screen']['height'],
            fill=True,
            alpha=0.7
        )
        self.__toast = frameDecorator.Toast()
        self.__decorator = frameDecorator.SimpleText(
            [self.__worker2, ],
            height=self.__config['screen']['height'],
            padding=(10, 20, 0, 0),
            font_height=10,
            color=frameDecorator.Colors.gold.value
        )
        self.__busy = frameDecorator.Busy(
            self.__config['screen']['width'],
            self.__config['screen']['height']
        )
        self.__hist = frameDecorator.Hist2()
        
        
        self.__option: None | typing.Dict = None
        self.__filter = SlidingWindowFilter(10)

        self.__state = CameraControlledEndState()

        self.__logger = Logger()
        self.__logger.info(
            LogMsg(
                content=f'Camera init finished',
                module=self.__module__,
                filename=os.path.basename(os.path.abspath(__file__)),
                currentframe=inspect.currentframe(),
                pid=os.getpid()
            )
        )

    

    def __worker2(self):
        if self.metadata:
            return {
                "EPTime {}": self.metadata['ExposureTime'],
                'FocusFoM {}': self.frame_quality,
                'FrameDur {}': self.metadata['FrameDuration'],
                'AnGain {}': round(self.metadata['AnalogueGain'], 2),
                'DigGain {}': round(self.metadata['DigitalGain'], 2),
                'Lux {}': round(self.metadata['Lux'], 2),
                'ClrTemp {}': self.metadata['ColourTemperature'],
                "FPS {}": round(self.frame_per_second, 1),
                "FocusFoM {}": int(self.__filter.calc())
            }

    def __find_option_by_ID(self, target):
        if self.__option is None:
            self.__logger.fatal(
                LogMsg(
                    content='Param self.__option is None',
                    module=self.__module__,
                    filename=os.path.basename(os.path.abspath(__file__)),
                    currentframe=inspect.currentframe(),
                    pid=os.getpid()
                )
            )
            raise RuntimeError()
        for key, value in self.__option.items():
            if 'options' in value.keys():
                for j in value['options']:
                    if 'id' in j.keys() and 'value' in j.keys():
                        if j['id'] == target:
                            return j['value']
        raise LookupError(target)

    def up_press_action(self):
        if self.__state.decorate_enable:
            self.__decorator.previous_page()
            time.sleep(0.3)
        else:
            self.__state.bright_hold = True
            if self.__state.brightness + 0.01 > 1:
                self.__state.brightness = 1
            else:
                self.__state.brightness += 0.01
            self.__toast.set_text(f"BRT {int(self.__state.brightness * 100)}")

            self.brightness(self.__state.brightness)
            time.sleep(0.05)

    def up_release_action(self):
        self.__state.bright_hold = False

    def down_press_action(self):
        if self.__state.decorate_enable:
            self.__decorator.next_page()
            time.sleep(0.3)
        else:
            self.__state.bright_hold = True
            if self.__state.brightness - 0.01 < -1:
                self.__state.brightness = -1
            else:
                self.__state.brightness -= 0.01
            self.__toast.set_text(f"BRT {int(self.__state.brightness * 100)}")
            self.brightness(self.__state.brightness)
            time.sleep(0.05)

    def down_release_action(self):
        self.__state.bright_hold = False

    def left_press_action(self):
        self.__state.zoom_hold = True
        if self.__state.zoom - 0.05 < 1:
            self.__state.zoom = 1
        else:
            self.__state.zoom -= 0.2
        self.__toast.set_text(f"X {round(self.__state.zoom, 1)}")
        self.zoom(self.__state.zoom)
        time.sleep(0.05)

    def left_release_action(self):
        self.__state.zoom_hold = False

    def right_press_action(self):
        self.__state.zoom_hold = True
        self.__state.zoom += 0.2
        self.__toast.set_text(f"X {round(self.__state.zoom, 1)}")
        self.zoom(self.__state.zoom)
        time.sleep(0.05)

    def right_release_action(self):
        self.__state.zoom_hold = False

    def shutter_press_action(self):
        if self.__state.record_timestamp is not None:
            self.stop_recording()
            led.off(led.blue)
            self.__state.record_timestamp = None
        else:
            try:
                width, height = tuple(
                    self.__find_option_by_ID('resolution')['value'])
            except ValueError:
                width, height = 0, 0

            delay = self.__find_option_by_ID('delay')
            for i in range(delay):
                if delay - i <= 3:
                    led.toggle_state(led.green)
                    time.sleep(0.5)
                    led.toggle_state(led.green)
                    time.sleep(0.5)
                else:
                    led.toggle_state(led.green)
                    time.sleep(1)
            self.__state.is_busy = True
            led.on(led.green)
            self.__toast.set_text("Processing")
            file_path = os.path.join(
                self.__config['camera']['path'], f"{int(time.time())}"
            )
            fmat = self.__find_option_by_ID('pict format')
            self.save_frame(
                filePath=file_path,
                fmat=fmat,
                width=int(width),
                height=int(height),
                saveMetadata=self.__find_option_by_ID("save metadata"),
                saveRaw=self.__find_option_by_ID("dng enable")
            )
            if self.__find_option_by_ID('watermark'):
                frame = cv2.imread(f'{file_path}.{fmat}')
                frameDecorator.WaterMark(
                    int(width), int(height)).decorate(frame)
                cv2.imwrite(f'{file_path}.{fmat}', frame)

            led.off(led.green)
            self.__state.is_busy = False

    def shutterLongPressAction(self):
        if self.__state.is_busy:
            return
        if self.__state.record_timestamp is None:
            try:
                width, height = tuple(
                    self.__find_option_by_ID('resolution').split('x')
                )
            except ValueError:
                width, height = 0, 0
            self.start_recording(
                int(width), int(height),
                '{}'.format(
                    os.path.join(
                        self.__config['camera']['video_path'],
                        str(int(time.time())) + '.mp4'
                    )
                )
            )
            led.on(led.blue)
            self.__state.record_timestamp = time.time()
        else:
            self.stop_recording()
            led.off(led.blue)
            self.__state.record_timestamp = None

    def square_press_action(self):
        if self.__state.record_timestamp is None and not self.__state.is_busy:
            self._irq('MenuControlledEnd')

    def circle_press_action(self):
        self.__state.decorate_enable = not self.__state.decorate_enable

    def cross_press_action(self):
        pass

    def __expose_setting(self):
        if self.__find_option_by_ID('auto expose'):
            self.set_AE_enable(
                True
            )
            self.set_AE_constraint_mode(
                self.__find_option_by_ID('constraint mode')['value']
            )
            self.set_AE_exposureMode(
                self.__find_option_by_ID('exposure mode')['value']
            )
            self.set_AE_metering_mode(
                self.__find_option_by_ID('metering mode')['value']
            )
            self.set_AE_flicker_mode(
                self.__find_option_by_ID('flicker mode')['value']
            )
            self.set_AE_flicker_period(
                self.__find_option_by_ID('flicker period')['value']
            )
        else:
            self.set_AE_enable(
                False
            )

            self.set_manual_exposure(
                self.__find_option_by_ID('exposure time'),
                self.__find_option_by_ID('analogue gain')
            )

    def __AWB_setting(self):
        if self.__find_option_by_ID('awb'):
            self.set_AWB_enable(
                True
            )
            self.set_AWB_mode(
                self.__find_option_by_ID('awb mode')['value']
            )
        else:
            self.set_AWB_enable(
                False
            )
            red, blue = self.__find_option_by_ID(
                'red gain'), self.__find_option_by_ID('blue gain')
            self.set_colour_gains(red, blue)

    def msg_receiver(self, sender, msg):
        self.__option = msg[1]
        self.load_settings()

    def load_settings(self):
        self.__expose_setting()
        self.__AWB_setting()
        self.__state.mfassist = self.__find_option_by_ID('mf assist')
        self.__show_hist = self.__find_option_by_ID('show hist')

    def center_press_action(self):
        pass

    def rotary_encoder_clockwise(self):
        pass

    def rotary_encoder_counter_clockwise(self):
        pass

    def rotary_encoder_select(self):
        pass
        # self.__main.nextCursor()

    def on_enter(self, lastID):
        if not os.path.exists(self.__config['camera']['path']) or not os.path.isdir(self.__config['camera']['path']):
            os.mkdir(self.__config['camera']['path'])
        self._msg_sender(self._id, 'MenuControlledEnd', self._id)
        self.load_settings()

    def active(self):
        self.start()

    def inactive(self):
        self.stop()

    def main_loop(self):
        for index, frame in enumerate(self.preview()):
            self.__filter.addData(self.frame_quality)
            self.__bar_chart.add_data(int(self.__filter.calc()))

            if self.__state.mfassist:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(
                    blurred,
                    threshold1=70,
                    threshold2=400
                )
                if edges.any():
                    colorfulEdges = numpy.zeros(
                        (edges.shape[0], edges.shape[1], 3),
                        dtype=numpy.uint8
                    )

                    if index % 3 == 0:
                        colorfulEdges[edges != 0] = (0, 0, 255)
                    elif index % 3 == 1:
                        colorfulEdges[edges != 0] = (255, 0, 0)
                    else:
                        colorfulEdges[edges != 0] = (0, 255, 0)

            if self.__state.decorate_enable and not self.__state.zoom_hold and self.__state.record_timestamp is None:
                self.__bar_chart.decorate(frame)
                self.__decorator.decorate(frame)
            if self.__state.is_busy:
                self.__busy.decorate(frame)

            if self.__state.record_timestamp is not None and not self.__state.zoom_hold:
                millis = (time.time() - self.__state.record_timestamp) * 1000
                seconds, milliseconds = divmod(int(millis), 1000)
                minutes, seconds = divmod(int(seconds), 60)
                hours, minutes = divmod(int(minutes), 60)
                self.__toast.set_text(
                    f'{hours}:{minutes}:{seconds}:{milliseconds}')

            

            if self.__state.zoom_hold:
                self.__toast.decorate(frame)
            if self.__state.bright_hold:
                self.__toast.decorate(frame)
            if self.__toast.isUpdate:
                self.__toast.decorate(frame)
            if self.__state.show_hist:
                self.__hist.decorate(frame)

            if self.__state.mfassist and edges.any():
                frame = cv2.addWeighted(frame, 1, colorfulEdges, 1.0, 0)
            yield frame
