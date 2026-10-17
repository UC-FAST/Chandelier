"""BQ32002 real-time clock driver."""

from __future__ import annotations

import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class BQ32002:
    """Minimal BQ32002 driver using SMBus."""

    REGISTERS = {
        "SECONDS": 0x00,
        "MINUTES": 0x01,
        "HOURS": 0x02,
        "DAY": 0x03,
        "DATE": 0x04,
        "MONTH": 0x05,
        "YEAR": 0x06,
        "CALIBRATION": 0x07,
        "CONTROL": 0x0E,
        "STATUS": 0x0F,
    }
    STOP_BIT = 0x80
    HOUR_12_24_FLAG = 0x02
    OSC_DISABLE = 0x08
    CAL_SIGN = 0x80

    def __init__(self, bus_number: int = 1, address: int = 0x68):
        import smbus2

        self._bus = smbus2.SMBus(bus_number)
        self._address = address
        self.configure_device()
        logger.info(
            "BQ32002 initialised on I2C bus %s at %#x",
            bus_number,
            address,
        )

    def configure_device(self) -> None:
        """Ensure the clock runs in 24-hour mode with the oscillator enabled."""

        seconds = self._bus.read_byte_data(
            self._address, self.REGISTERS["SECONDS"]
        )
        if seconds & self.STOP_BIT:
            self._bus.write_byte_data(
                self._address,
                self.REGISTERS["SECONDS"],
                seconds & ~self.STOP_BIT,
            )

        status = self._bus.read_byte_data(
            self._address, self.REGISTERS["STATUS"]
        )
        if status & self.HOUR_12_24_FLAG:
            self._bus.write_byte_data(
                self._address,
                self.REGISTERS["STATUS"],
                status & ~self.HOUR_12_24_FLAG,
            )

        control = self._bus.read_byte_data(
            self._address, self.REGISTERS["CONTROL"]
        )
        if control & self.OSC_DISABLE:
            self._bus.write_byte_data(
                self._address,
                self.REGISTERS["CONTROL"],
                control & ~self.OSC_DISABLE,
            )

    def read_time(self) -> float:
        """Return the RTC time as a Unix timestamp."""

        data = self._bus.read_i2c_block_data(
            self._address, self.REGISTERS["SECONDS"], 7
        )
        seconds = self._bcd_to_dec(data[0] & 0x7F)
        minutes = self._bcd_to_dec(data[1])
        hours = self._bcd_to_dec(data[2] & 0x3F)
        day = self._bcd_to_dec(data[3])  # noqa: F841 - kept for clarity
        date = self._bcd_to_dec(data[4])
        month = self._bcd_to_dec(data[5] & 0x1F)
        year = self._bcd_to_dec(data[6]) + 2000

        value = datetime(year, month, date, hours, minutes, seconds)
        logger.debug("BQ32002 read time %s", value)
        return value.timestamp()

    def write_time(self, timestamp: float) -> None:
        """Set the RTC from a Unix timestamp."""

        value = datetime.fromtimestamp(timestamp)
        data = [
            self._dec_to_bcd(value.second) & 0x7F,
            self._dec_to_bcd(value.minute),
            self._dec_to_bcd(value.hour) & 0x3F,
            self._dec_to_bcd(value.isoweekday()),
            self._dec_to_bcd(value.day),
            self._dec_to_bcd(value.month),
            self._dec_to_bcd(value.year % 100),
        ]
        self._bus.write_i2c_block_data(
            self._address, self.REGISTERS["SECONDS"], data
        )
        logger.info("BQ32002 wrote time %s", value)

    @property
    def calibration(self) -> tuple[int, bool]:
        value = self._bus.read_byte_data(
            self._address, self.REGISTERS["CALIBRATION"]
        )
        negative = bool(value & self.CAL_SIGN)
        calibration = value & 0x1F
        if negative:
            return calibration - 32, True
        return calibration, False

    @calibration.setter
    def calibration(self, value: int) -> None:
        if not -16 <= value <= 15:
            raise ValueError("calibration must be between -16 and +15")
        if value < 0:
            raw = (abs(value) | self.CAL_SIGN) ^ 0x20
        else:
            raw = value
        self._bus.write_byte_data(
            self._address, self.REGISTERS["CALIBRATION"], raw
        )
        logger.info("BQ32002 calibration set to %s", value)

    @staticmethod
    def _bcd_to_dec(value: int) -> int:
        return (value // 16) * 10 + (value & 0x0F)

    @staticmethod
    def _dec_to_bcd(value: int) -> int:
        return (value // 10) << 4 | (value % 10)

    def close(self) -> None:
        self._bus.close()


if __name__ == "__main__":
    rtc = BQ32002()
    rtc.write_time(time.time())
    time.sleep(3)
    print(f"RTC reports time: {rtc.read_time()}")
    print(f"Calibration: {rtc.calibration}")
