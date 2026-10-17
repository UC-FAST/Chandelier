from datetime import datetime
import re
import subprocess
import time

import numpy
import psutil

from components import bq32002, ina230, max17048
import frameDecorator
from . import ControlledEnd


class SystemMonitor(ControlledEnd):
    def __init__(self, _id='SystemMonitor'):
        ControlledEnd.__init__(self, _id)
        self._id = _id
        self._irq = None
        self._msgSender = None
        self.__decorator = frameDecorator.SimpleText(
            [
                self.__hardwareReport,
                self.__powerReport,
                self.__iwconfig
            ],
            height=240,
            padding=(10, 20, 0, 0),
            font_height=10,
            color=frameDecorator.Colors.gold.value
        )
        self.__m = max17048.MAX17048()
        self.__i = ina230.INA230()
        self.__b = bq32002.BQ32002()

    def __hardwareReport(self):
        try:
            rtc = datetime.fromtimestamp(self.__b.read_time())
        except ValueError:
            rtc = None
        return {
            "CPU {}%": round(psutil.cpu_percent(interval=0.3), 2),
            "MEM {}%": round((psutil.virtual_memory().used / psutil.virtual_memory().total) * 100, 2),
            "DISK {}%": psutil.disk_usage('/').percent,
            "TEMP {} C": psutil.sensors_temperatures()['cpu_thermal'][0].current,
            'System Time {}': datetime.now(),
            'RTC Time {}': rtc,
        }

    def __powerReport(self):
        try:
            bat = round(self.__m.get_bat(), 2)
            percent = self.__m.get_battery_percent(bat)
        except IOError:
            bat = None
            percent = None

        try:
            busVoltage = round(self.__i.read_voltage(), 2)
        except IOError:
            busVoltage = None

        try:
            shuntVoltage = round(self.__i.read_shunt_voltage()*1000, 3)
        except IOError:
            shuntVoltage = None

        try:
            current = round(self.__i.read_current(), 2)*1000
        except IOError:
            current = None

        try:
            power = round(self.__i.read_power(), 2)
        except IOError:
            power = None
        time.sleep(0.3)

        return {
            "BAT {}v {}%": (bat, percent),
            "Bus Voltage {}V": busVoltage,
            "Shunt Voltage {}mV": shuntVoltage,
            "Current {}mA": current,
            "Power {}W": power
        }

    def __iwconfig(self):
        result = subprocess.run(
            ["ifconfig"], capture_output=True
        ).stdout.decode()
        pattern = 'wlan0:.*inet (.*)  netmask'
        try:
            ip = re.findall(
                pattern=pattern, string=result, flags=re.DOTALL)[0]
        except IndexError:
            ip = None

        result = subprocess.run(
            ["iwconfig"], capture_output=True
        ).stdout.decode()

        pattern = '''
                (\w+)\s+           
                IEEE\s.*?\s+      
                ESSID:"([^"]+)"     
                .*?                
                Frequency:([\d.]+\sGHz) 
                .*?                
                Link\sQuality=([\d/]+)   
                .*?                
                Signal\slevel=([-\d]+\sdBm)  
            '''
        matches = re.findall(pattern, result, re.VERBOSE | re.DOTALL)
        if matches:
            interface, essid, freq, quality, level = matches[0]
        else:
            interface, essid, freq, quality, level = None, None, None, None, None

        return {
            'Interface {}': interface,
            'ESSID {}': essid,
            'IP {}':ip,
            'Frequency {}': freq,
            'Link Quality {}': quality,
            'Signal Level {}': level
        }
    

    def up_release_action(self):
        self.__decorator.previous_page()

    def down_press_action(self):
        self.__decorator.next_page()

    def cross_press_action(self):
        self._irq("CameraControlledEnd")

    def rotary_encoder_clockwise(self):
        self.__decorator.previous_page()

    def rotary_encoder_counter_clockwise(self):
        self.__decorator.next_page()

    def rotary_encoder_select(self):
        pass

    def msg_sender(self, func):
        self._msgSender = func

    def irq(self, func):
        self._irq = func

    def main_loop(self):
        frame = numpy.zeros((240, 320, 3), dtype=numpy.uint8)
        self.__decorator.decorate(frame)
        yield frame

    @property
    def id(self):
        return self._id
