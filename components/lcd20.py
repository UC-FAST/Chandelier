
import logging
import time

import cv2
import numpy

import gpiozero
import spidev

from utils import ConfigLoader, singleton, LogMsg, Logger


@singleton
class Lcd:
    def __init__(self, width=320, height=240, spi_freq=80000000):
        self.width, self.height = width, height
        self.__config = ConfigLoader('./config.json')
        self.RST_PIN = gpiozero.DigitalOutputDevice(
            self.__config['pin']['rst'],
            active_high=True,
            initial_value=False
        )
        self.DC_PIN = gpiozero.DigitalOutputDevice(
            self.__config['pin']['dc'],
            active_high=True,
            initial_value=False
        )
        self.BL_PIN = gpiozero.DigitalOutputDevice(
            self.__config['pin']['bl'],
            active_high=True,
            initial_value=False
        )
        self.backlight(True)
        self.SPI = spidev.SpiDev(0, 0)
        if self.SPI != None:
            self.SPI.max_speed_hz = spi_freq
            self.SPI.mode = 0b00

    def backlight(self, state: bool):
        self.__digital_write(self.BL_PIN, state)

    def __digital_write(self, pin, value: bool):
        if value:
            pin.on()
        else:
            pin.off()

    def module_exit(self):
        logging.debug("spi end")
        if self.SPI != None:
            self.SPI.close()

        logging.debug("gpio cleanup...")
        self.__digital_write(self.RST_PIN, True)
        self.__digital_write(self.DC_PIN, False)
        self.BL_PIN.close()
        time.sleep(0.001)

    def __SPI_write_byte(self, data):
        if self.SPI != None:
            self.SPI.writebytes(data)

    def command(self, cmd):
        self.__digital_write(self.DC_PIN, False)
        self.__SPI_write_byte([cmd])

    def data(self, val):
        self.__digital_write(self.DC_PIN, True)
        self.__SPI_write_byte([val])

    def reset(self):
        """Reset the display"""
        self.__digital_write(self.RST_PIN, True)
        time.sleep(0.01)
        self.__digital_write(self.RST_PIN, False)
        time.sleep(0.01)
        self.__digital_write(self.RST_PIN, True)
        time.sleep(0.01)

    def init(self):
        """Initialize dispaly"""
        self.reset()
        self.command(0x36)
        self.data(0x00)
        self.command(0x3A)
        self.data(0x05)
        self.command(0x21)
        self.command(0x2A)
        self.data(0x00)
        self.data(0x00)
        self.data(0x01)
        self.data(0x3F)
        self.command(0x2B)
        self.data(0x00)
        self.data(0x00)
        self.data(0x00)
        self.data(0xEF)
        self.command(0xB2)
        self.data(0x0C)
        self.data(0x0C)
        self.data(0x00)
        self.data(0x33)
        self.data(0x33)
        self.command(0xB7)
        self.data(0x35)
        self.command(0xBB)
        self.data(0x1F)
        self.command(0xC0)
        self.data(0x2C)
        self.command(0xC2)
        self.data(0x01)
        self.command(0xC3)
        self.data(0x12)
        self.command(0xC4)
        self.data(0x20)
        self.command(0xC6)
        self.data(0x0F)
        self.command(0xD0)
        self.data(0xA4)
        self.data(0xA1)
        self.command(0xE0)
        self.data(0xD0)
        self.data(0x08)
        self.data(0x11)
        self.data(0x08)
        self.data(0x0C)
        self.data(0x15)
        self.data(0x39)
        self.data(0x33)
        self.data(0x50)
        self.data(0x36)
        self.data(0x13)
        self.data(0x14)
        self.data(0x29)
        self.data(0x2D)
        self.command(0xE1)
        self.data(0xD0)
        self.data(0x08)
        self.data(0x10)
        self.data(0x08)
        self.data(0x06)
        self.data(0x06)
        self.data(0x39)
        self.data(0x44)
        self.data(0x51)
        self.data(0x0B)
        self.data(0x16)
        self.data(0x14)
        self.data(0x2F)
        self.data(0x31)
        self.command(0x21)
        self.command(0x11)
        self.command(0x29)

    def __set_windows(self, Xstart, Ystart, Xend, Yend):
        # set the X coordinates
        self.command(0x2A)
        # Set the horizontal starting point to the high octet
        self.data(Xstart >> 8)
        # Set the horizontal starting point to the low octet
        self.data(Xstart & 0xff)
        self.data(Xend >> 8)  # Set the horizontal end to the high octet
        self.data((Xend - 1) & 0xff)  # Set the horizontal end to the low octet

        # set the Y coordinates
        self.command(0x2B)
        self.data(Ystart >> 8)
        self.data((Ystart & 0xff))
        self.data(Yend >> 8)
        self.data((Yend - 1) & 0xff)

        self.command(0x2C)

    def show_image(self, img):
        try:
            if img is None:
                logging.error("Image is None")
                return
            try:
                imheight, imwidth, _ = img.shape
                if imwidth != self.width or imheight != self.height:
                    img = cv2.resize(img, (self.width, self.height))
            except ValueError:
                logging.error("Image shape is not valid")
                return
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pix = numpy.zeros((self.height, self.width, 2), dtype=numpy.uint8)
            # RGB888 >> RGB565
            pix[..., [0]] = numpy.add(numpy.bitwise_and(
                img[..., [0]], 0xF8), numpy.right_shift(img[..., [1]], 5))
            pix[..., [1]] = numpy.add(numpy.bitwise_and(numpy.left_shift(
                img[..., [1]], 3), 0xE0), numpy.right_shift(img[..., [2]], 3))
            pix = pix.flatten().tolist()
            self.command(0x36)
            self.data(0x70)
            self.__set_windows(0, 0, self.width, self.height)
            self.__digital_write(self.DC_PIN, True)
            for i in range(0, len(pix), 4096):
                self.__SPI_write_byte(pix[i:i+4096])
        except Exception:
            self.RST_PIN.close()
            self.DC_PIN.close()
            self.BL_PIN.close()

    def clear(self):
        """Clear contents of image buffer"""
        _buffer = [0xff]*(self.width * self.height * 2)
        self.__set_windows(0, 0, self.height, self.width)
        self.__digital_write(self.DC_PIN, True)
        for i in range(0, len(_buffer), 4096):
            self.__SPI_write_byte(_buffer[i:i+4096])


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    lcd = Lcd()
    lcd.init()
    img = cv2.imread('1.jpg')
    lcd.show_image(img)
    time.sleep(5)
    lcd.clear()
    lcd.module_exit()
    exit(0)
